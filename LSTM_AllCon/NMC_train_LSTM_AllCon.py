import os
import argparse
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from NMC_dataset_loader import build_dataloaders, N_CLASSES, N_INPUT

# -------------------------------------------------------------------------
# Global perf knobs (set once at import / start of main)
# -------------------------------------------------------------------------
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# -------------------------------------------------------------------------
# Loss
# -------------------------------------------------------------------------
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
        probs = torch.softmax(logits, dim=1)                     # (B, K)
        cum = 1.0 - torch.cumsum(probs, dim=1)[:, :-1]            # (B, K-1)
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


# -------------------------------------------------------------------------
# Model: CNN + LSTM
# -------------------------------------------------------------------------
def _broadcast(value, length: int, name: str) -> list:
    """
    Accept either a single value (repeated for every layer) or a list/tuple
    with exactly `length` entries (one per layer, e.g. filters that grow
    64->128, or LSTM units that taper 128->64).
    """
    if isinstance(value, (list, tuple)):
        if len(value) != length:
            raise ValueError(
                f"{name} has {len(value)} entries but *_layers={length}; "
                f"either pass one value per layer or a single scalar."
            )
        return list(value)
    return [value] * length


class BatteryRULClassifierLSTM(nn.Module):
    """
    Per-cycle CNN over the dq curve -> fuse with projected summary
    features -> LSTM over cycles -> classification head.

    Examples from the table:
        C1: cnn_filters=32,        kernel_size=3,      cnn_layers=1, lstm_units=32,       lstm_layers=1
        C3: cnn_filters=[64,128],  kernel_size=3,      cnn_layers=2, lstm_units=64,        lstm_layers=1
        C4: cnn_filters=[64,128],  kernel_size=[5,3],  cnn_layers=2, lstm_units=128,       lstm_layers=1
        C5: cnn_filters=[64,128],  kernel_size=[5,3],  cnn_layers=2, lstm_units=[128,64],  lstm_layers=2

    When lstm_units is uniform across layers (C1-C4), a single nn.LSTM with
    num_layers>1 is used (fused cuDNN kernel, fastest path). When lstm_units
    tapers (C5: 128->64), nn.LSTM cannot represent that directly — a single
    LSTM module only ever has one hidden_size shared by all its layers — so
    instead a stack of single-layer bidirectional LSTMs is built manually,
    each layer's output (hidden_size*2, due to bidirectionality) feeding
    into the next layer's input.
    """

    def __init__(
        self,
        cnn_filters=32,
        kernel_size=3,
        cnn_layers: int = 1,
        lstm_units=32,
        lstm_layers: int = 1,
        summary_feats: int = 24,   # overwritten at runtime from summary_scaler.n_features_in_
        n_classes: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()

        cnn_filters_list = _broadcast(cnn_filters, cnn_layers, "cnn_filters")
        kernel_list = _broadcast(kernel_size, cnn_layers, "kernel_size")
        lstm_units_list = _broadcast(lstm_units, lstm_layers, "lstm_units")

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

        # ---- Sequence model over cycles --------------------------------
        fusion_dim = cnn_out_dim + cnn_out_dim
        self.uniform_lstm = len(set(lstm_units_list)) == 1

        if self.uniform_lstm:
            self.lstm = nn.LSTM(
                input_size=fusion_dim,
                hidden_size=lstm_units_list[0],
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if lstm_layers > 1 else 0.0,
            )
        else:
            self.lstm_stack = nn.ModuleList()
            self.lstm_stack_drop = nn.ModuleList()
            in_size = fusion_dim
            for i, units in enumerate(lstm_units_list):
                self.lstm_stack.append(nn.LSTM(
                    input_size=in_size, hidden_size=units,
                    num_layers=1, batch_first=True, bidirectional=True,
                ))
                self.lstm_stack_drop.append(
                    nn.Dropout(dropout) if i < lstm_layers - 1 else nn.Identity()
                )
                in_size = units * 2

        final_hidden = lstm_units_list[-1]
        self.drop = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(final_hidden * 2, 32), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, dq: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        B, T, C, F_ = dq.shape

        dq_feat = self.cnn(dq.reshape(B * T, C, F_)).squeeze(-1).reshape(B, T, -1)
        summary_feat = self.summary_proj(summary)

        fused = self.drop(torch.cat([dq_feat, summary_feat], dim=-1))

        if self.uniform_lstm:
            out, (h_n, c_n) = self.lstm(fused)
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            x = fused
            h_n = None
            for layer, drop in zip(self.lstm_stack, self.lstm_stack_drop):
                x, (h_n, c_n) = layer(x)   # h_n: (2, B, hidden) — single layer, bidirectional
                x = drop(x)
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)

        h_last = self.drop(h_last)
        return self.head(h_last)


# -------------------------------------------------------------------------
# Train one epoch (AMP + fewer syncs)
# -------------------------------------------------------------------------
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

        # Loss computed outside autocast, in fp32: OrdinalLoss uses
        # F.binary_cross_entropy internally, which PyTorch refuses to run
        # under autocast (it's flagged unsafe in reduced precision). Only
        # the model forward pass needs/benefits from autocast anyway.
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


# -------------------------------------------------------------------------
# Evaluate (AMP + batched tensor accumulation instead of per-batch .cpu())
# -------------------------------------------------------------------------
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


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def train(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Prefer bf16 when available (no GradScaler needed, more stable);
    # fall back to fp16 + GradScaler otherwise.
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

    model = BatteryRULClassifierLSTM(
        cnn_filters=args.cnn_filters,
        kernel_size=args.kernel_size,
        cnn_layers=args.cnn_layers,
        lstm_units=args.lstm_units,
        lstm_layers=args.lstm_layers,
        summary_feats=summary_feats,
        n_classes=N_CLASSES,
        dropout=args.dropout,
    ).to(device)

    if args.compile:
        # Graph-compiles the model into fused kernels. First batch/epoch
        # will be slower (compilation), subsequent ones faster.
        model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}\n")

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
    print(classification_report(true, pred,
          target_names=["RUL>400", "RUL>300", "RUL>200", "RUL>100", "RUL<100"]))
    print("Confusion Matrix:")
    print(confusion_matrix(true, pred))

    np.save(os.path.join(args.output_dir, "clf_pred.npy"), pred)
    np.save(os.path.join(args.output_dir, "clf_true.npy"), true)
    print(f"\nSaved to {args.output_dir}/")


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir", default="./content")
    parser.add_argument("--output_dir", default="./checkpoints_clf")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cnn_filters", type=int, default=32)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--cnn_layers", type=int, default=1)
    parser.add_argument("--lstm_units", type=int, default=32)
    parser.add_argument("--lstm_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=4,
                         help="Increase from the original default of 0 to overlap "
                              "data loading with GPU compute.")
    parser.add_argument("--loss", default="ordinal", choices=["cross_entropy", "ordinal"],
                         help="Loss function: ordinal or cross_entropy")
    parser.add_argument("--compile", action="store_true",
                         help="Wrap the model with torch.compile() (PyTorch 2.x, CUDA/CPU).")
    args = parser.parse_args()

    train(args)
