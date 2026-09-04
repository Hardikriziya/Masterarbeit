"""
NMC_train_Transformer.py - Train CNN+Transformer classifier on NMC BatteryML-derived features.

How to run:
    python NMC_train_Transformer.py --content_dir ./content_NMC --output_dir ./checkpoints_clf_transformer
"""

import os
import argparse
import joblib
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from NMC_dataset_loader import build_dataloaders, N_CLASSES, N_INPUT

# Global perf knobs (set once at import / start of main)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Loss
class OrdinalLoss(nn.Module):
    """
    Takes raw logits (B, K) — same interface as CrossEntropyLoss.
    Converts to cumulative probabilities internally using softmax.

    P(y > k) = sum_{j=k+1}^{K-1} softmax(logits)_j
    """
    def __init__(self, n_classes: int = 5, reduction: str = 'mean'):
        super().__init__()
        self.K = n_classes
        self.reduction = reduction
        self.register_buffer("thresholds", torch.arange(n_classes - 1), persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)                       # (B, K)
        cum = 1.0 - torch.cumsum(probs, dim=1)[:, :-1]              # (B, K-1)
        labels = (targets.unsqueeze(1) > self.thresholds).float()  # (B, K-1)
        loss = F.binary_cross_entropy(cum.clamp(1e-7, 1 - 1e-7), labels, reduction='none')
        loss = loss.sum(dim=1)
        return loss.mean() if self.reduction == 'mean' else loss.sum()


def predict_cls(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1).argmax(dim=-1)


def ordinal_predict(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    cum = 1.0 - torch.cumsum(probs, dim=1)[:, :-1]
    return (cum > 0.5).sum(dim=1).long()

# Model: CNN + Transformer
def _broadcast(value, length: int, name: str) -> list:
    """
    Accept either a single value (repeated for every layer) or a list/tuple
    with exactly `length` entries (one per layer, e.g. filters that grow
    64->128).
    """
    if isinstance(value, (list, tuple)):
        if len(value) != length:
            raise ValueError(
                f"{name} has {len(value)} entries but *_layers={length}; "
                f"either pass one value per layer or a single scalar."
            )
        return list(value)
    return [value] * length

def _sinusoidal_positional_encoding(seq_len: int, d_model: int, device, dtype) -> torch.Tensor:
    """
    Standard sin/cos positional encoding, computed fresh per forward call.
    Self-attention has no inherent notion of cycle order (unlike an
    RNN, which processes cycles sequentially), so this is added to the
    fused per-cycle features before the encoder. Returns shape
    (1, seq_len, d_model), ready to add to the input.
    """
    position = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(seq_len, d_model, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe.unsqueeze(0).to(dtype)


class BatteryRULClassifierTransformer(nn.Module):
    """
    Per-cycle CNN over the dq curve -> fuse with projected summary
    features -> Transformer encoder over cycles -> classification head.

    """

    def __init__(
        self,
        cnn_filters=32,
        kernel_size=3,
        cnn_layers: int = 1,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 64,
        summary_feats: int = 24,   # overwritten at runtime from summary_scaler.n_features_in_
        n_classes: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")

        cnn_filters_list = _broadcast(cnn_filters, cnn_layers, "cnn_filters")
        kernel_list = _broadcast(kernel_size, cnn_layers, "kernel_size")

        # ---- CNN over the per-cycle dq curve --------------------------
        layers = []
        in_ch = 1
        for i in range(cnn_layers):
            out_ch = cnn_filters_list[i]
            k = kernel_list[i]
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(out_ch),
                nn.ELU(),
                nn.Dropout(dropout),
            ]
            if i < cnn_layers - 1:
                layers.append(nn.MaxPool1d(2))
            in_ch = out_ch
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.cnn = nn.Sequential(*layers)
        cnn_out_dim = cnn_filters_list[-1]

        # ---- Project summary scalars into the same dim as CNN output --
        self.summary_proj = nn.Sequential(
            nn.Linear(summary_feats, cnn_out_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # ---- Sequence model over cycles: Transformer encoder -----------
        fusion_dim = cnn_out_dim + cnn_out_dim
        self.d_model = d_model
        self.input_proj = nn.Linear(fusion_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, dq: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        B, T, C, F_ = dq.shape

        dq_feat = self.cnn(dq.reshape(B * T, C, F_)).squeeze(-1).reshape(B, T, -1)
        summary_feat = self.summary_proj(summary)

        fused = self.drop(torch.cat([dq_feat, summary_feat], dim=-1))
        x = self.input_proj(fused)

        pe = _sinusoidal_positional_encoding(T, self.d_model, x.device, x.dtype)
        x = x + pe

        x = self.encoder(x)                # (B, T, d_model)
        pooled = self.drop(x.mean(dim=1))  # mean-pool across cycles -> (B, d_model)

        return self.head(pooled)



# Train one epoch (AMP + fewer syncs)
def train_epoch(model, loader, criterion, pred_fn, optimizer, device, scaler, use_amp, amp_dtype):
    model.train()
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    total = 0

    for batch in loader:
        dq = batch["dq"].to(device, non_blocking=True)
        summary = batch["summary"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(dq, summary)

      
        loss = criterion(out.float(), labels)

        if use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            # scaler.unscale_(optimizer)  # uncomment if you re-enable grad clipping
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        bs = labels.size(0)
        # Keep everything on-device; only sync once at the end of the epoch.
        total_loss += loss.detach() * bs
        correct += (pred_fn(out.detach()) == labels).sum()
        total += bs

    return (total_loss / total).item(), (correct / total).item()



# Evaluate (AMP + batched tensor accumulation instead of per-batch .cpu())
@torch.no_grad()
def evaluate(model, loader, criterion, pred_fn, device, use_amp, amp_dtype):
    model.eval()
    total_loss = torch.zeros((), device=device)
    total = 0
    pred_chunks = []
    true_chunks = []

    for batch in loader:
        dq = batch["dq"].to(device, non_blocking=True)
        summary = batch["summary"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(dq, summary)

        loss = criterion(out.float(), labels)

        bs = labels.size(0)
        total_loss += loss.detach() * bs
        total += bs
        pred_chunks.append(pred_fn(out.detach()))
        true_chunks.append(labels)

    all_pred = torch.cat(pred_chunks).cpu().numpy()
    all_true = torch.cat(true_chunks).cpu().numpy()
    return (total_loss / total).item(), float((all_pred == all_true).mean()), all_pred, all_true



# Main
def train(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    print("\nLoading data...")
    train_loader, val_loader, test_loader, scalers = build_dataloaders(
        content_dir=args.content_dir,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        val_ratio=0.1,
        num_workers=args.num_workers,
    )

    dq_scaler, summary_scaler = scalers
    joblib.dump(dq_scaler, os.path.join(args.output_dir, "dq_scaler.pkl"))
    joblib.dump(summary_scaler, os.path.join(args.output_dir, "summary_scaler.pkl"))

    summary_feats = summary_scaler.n_features_in_
    print(f"summary_feats: {summary_feats}")

    model = BatteryRULClassifierTransformer(
        cnn_filters=args.cnn_filters,
        kernel_size=args.kernel_size,
        cnn_layers=args.cnn_layers,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        summary_feats=summary_feats,
        n_classes=N_CLASSES,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}\n")

    # ── Layer-by-layer summary (run on the raw model, before torch.compile:
    # forward hooks on a compiled model can be unreliable) ─────────────────
    print("Model Summary:")
    print("=" * 80)
    print(f"  {'Layer':<38} {'Output Shape':<25} {'Params':>10}")
    print("-" * 80)

    _handles = []
    _summary = []

    def _extract_shape(out):
        """
        Best-effort shape extraction for the hook below. TransformerEncoder
        returns a plain tensor, so this branch is mostly here for symmetry
        with the GRU/LSTM versions (where nn.GRU/nn.LSTM return a tuple),
        but it's kept general in case a wrapped submodule ever does too.
        """
        if isinstance(out, torch.Tensor):
            return tuple(out.shape)
        if isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
            return tuple(out[0].shape)
        return "?"

    def _hook(module, inp, out):
       
        if isinstance(module, nn.MultiheadAttention):
            n = sum(p.numel() for p in module.parameters())
            label = module.__class__.__name__
        elif len(list(module.children())) == 0:          # leaf modules
            n = sum(p.numel() for p in module.parameters())
            label = module.__class__.__name__
        else:
            
            own_params = sum(p.numel() for p in module.parameters(recurse=False))
            if own_params == 0:
                return
            n = own_params
            label = f"{module.__class__.__name__} (own params)"
        shape = _extract_shape(out)
        _summary.append((label, shape, n))

    for m in model.modules():
        _handles.append(m.register_forward_hook(_hook))

    # V_BINS inferred from the fitted dq_scaler (it was fit on
    # (N, 1*V_BINS)-shaped data) rather than imported from an external
    # module, so this stays correct even if that module's constant drifts.
    _V = dq_scaler.n_features_in_
    _dummy_dq = torch.zeros(1, N_INPUT, 1, _V, device=device)
    _dummy_summary = torch.zeros(1, N_INPUT, summary_feats, device=device)
    with torch.no_grad():
        model(_dummy_dq, _dummy_summary)

    for h in _handles:
        h.remove()

    for name, shape, n in _summary:
        print(f"  {name:<38} {str(shape):<25} {n:>10,}")

    print("=" * 80)
    print(f"  {'Total trainable parameters':<38} {'':25} {n_params:>10,}")
    print("=" * 80 + "\n")

    if args.compile:
        model = torch.compile(model)

    # ── Loss function ─────────────────────────────────────────────────────
    use_ordinal = args.loss == "ordinal"
    if use_ordinal:
        criterion = OrdinalLoss(n_classes=N_CLASSES).to(device)
        pred_fn = predict_cls
        print("Loss: OrdinalLoss (N-1 sigmoid thresholds)\n")
    else:
        criterion = nn.CrossEntropyLoss()
        pred_fn = lambda logits: logits.argmax(dim=1)
        print("Loss: CrossEntropyLoss\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=args.lr * 0.01,
    )

    best_val_acc = 0.0
    header = f"{'Epoch':>5} | {'TrLoss':>8} | {'TrAcc':>7} | {'VaLoss':>8} | {'VaAcc':>7} | {'LR':>8}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, criterion, pred_fn, optimizer, device, scaler, use_amp, amp_dtype
        )
        va_loss, va_acc, _, _ = evaluate(
            model, val_loader, criterion, pred_fn, device, use_amp, amp_dtype
        )
        scheduler.step(va_acc)

        lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:5d} | {tr_loss:8.4f} | {tr_acc:7.4f} | "
              f"{va_loss:8.4f} | {va_acc:7.4f} | {lr:.2e}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_clf.pt"))

    # ---- Test ----
    print("\n" + "=" * 60)
    model.load_state_dict(
        torch.load(os.path.join(args.output_dir, "best_clf.pt"),
                   map_location=device, weights_only=True)
    )
    te_loss, te_acc, pred, true = evaluate(
        model, test_loader, criterion, pred_fn, device, use_amp, amp_dtype
    )

    print(f"\nTest Loss: {te_loss:.4f}  Accuracy: {te_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        true, pred,
        labels=list(range(N_CLASSES)),
        target_names=["RUL>400", "RUL>300", "RUL>200", "RUL>100", "RUL<100"],
        zero_division=0,
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(true, pred))

    np.save(os.path.join(args.output_dir, "clf_pred.npy"), pred)
    np.save(os.path.join(args.output_dir, "clf_true.npy"), true)
    print(f"\nSaved to {args.output_dir}/")

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir", default=r"D:\TU\Master_Thesis\Raw_Data\Battery_Data\Plots\NMC_Selected_Features_1000\Stanford_2\Stanford_2")
    parser.add_argument("--output_dir", default=r"D:\TU\Master_Thesis\Raw_Data\Battery_Data\NMC_Tranformer\Results_Transformer\Stanford_2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cnn_filters", type=int, default=32)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--cnn_layers", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dim_feedforward", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--loss", default="ordinal", choices=["cross_entropy", "ordinal"],
                         help="Loss function: ordinal or cross_entropy")
    parser.add_argument("--compile", action="store_true",
                         help="Wrap the model with torch.compile() (PyTorch 2.x, CUDA/CPU).")
    args = parser.parse_args()

    train(args)