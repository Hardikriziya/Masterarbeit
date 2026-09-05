"""
NMC_run_LSTM_AllCon.py - Train all 5 CNN-LSTM configs from the table
back-to-back and report a comparison table (val/test accuracy, param count).

Data is loaded ONCE and reused across all 5 configs (same train/val/test
split, same fitted scalers) so the comparison is apples-to-apples — only
the model architecture changes between runs.

Each config gets its own output subfolder:
    <output_dir>/C1/best_clf.pt, dq_scaler.pkl, summary_scaler.pkl, ...
    <output_dir>/C2/...
    ...
and a final summary is printed and saved to:
    <output_dir>/NMC_lstm_results.csv

How to run:
    python NMC_run_LSTM_AllCon.py --content_dir ./content_NMC --output_dir ./NMC_lstm
"""

import argparse
import csv
import os

import joblib
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import classification_report, confusion_matrix

from NMC_dataset_loader import N_CLASSES, N_INPUT, build_dataloaders
from NMC_train_LSTM_AllCon import (
    BatteryRULClassifierLSTM,
    OrdinalLoss,
    evaluate,
    predict_cls,
    train_epoch,
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# The 5 CNN-LSTM configs, 
# CNN Filters/Kernel/LSTM Units may be a single int (same value on every
# layer) or a list (one value per layer, for the ones that grow/shrink).

CONFIGS = {
    "C1": dict(cnn_filters=32,        kernel_size=3,      cnn_layers=1, lstm_units=32,        lstm_layers=1, dropout=0.2),
    "C2": dict(cnn_filters=64,        kernel_size=3,      cnn_layers=1, lstm_units=64,        lstm_layers=1, dropout=0.3),
    "C3": dict(cnn_filters=[64, 128], kernel_size=3,      cnn_layers=2, lstm_units=64,        lstm_layers=1, dropout=0.3),
    "C4": dict(cnn_filters=[64, 128], kernel_size=[5, 3], cnn_layers=2, lstm_units=128,       lstm_layers=1, dropout=0.4),
    "C5": dict(cnn_filters=[64, 128], kernel_size=[5, 3], cnn_layers=2, lstm_units=[128, 64], lstm_layers=2, dropout=0.4),
}


def run_one_config(
    name: str,
    cfg: dict,
    args,
    device: torch.device,
    train_loader, val_loader, test_loader,
    summary_feats: int,
    use_amp: bool, amp_dtype, scaler,
) -> dict:
    """Train + evaluate one config, returning a results dict for the summary table."""
    print("\n" + "=" * 70)
    print(f"Config {name}: {cfg}")
    print("=" * 70)

    cfg_output_dir = os.path.join(args.output_dir, name)
    os.makedirs(cfg_output_dir, exist_ok=True)

    model = BatteryRULClassifierLSTM(
        **cfg,
        summary_feats=summary_feats,
        n_classes=N_CLASSES,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    if args.loss == "ordinal":
        criterion = OrdinalLoss(n_classes=N_CLASSES).to(device)
        pred_fn = predict_cls
    else:
        criterion = nn.CrossEntropyLoss()
        pred_fn = lambda logits: logits.argmax(dim=1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=args.lr * 0.01,
    )

    best_path = os.path.join(cfg_output_dir, "best_clf.pt")
    best_val_acc = -1.0

    header = f"{'Epoch':>5} | {'TrLoss':>8} | {'TrAcc':>7} | {'VaLoss':>8} | {'VaAcc':>7} | {'LR':>8}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, criterion, pred_fn, optimizer, device,
            scaler, use_amp, amp_dtype,
        )
        va_loss, va_acc, _, _ = evaluate(
            model, val_loader, criterion, pred_fn, device, use_amp, amp_dtype,
        )
        scheduler.step(va_acc)

        lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:5d} | {tr_loss:8.4f} | {tr_acc:7.4f} | "
              f"{va_loss:8.4f} | {va_acc:7.4f} | {lr:.2e}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_path)

    # ---- Test with the best checkpoint for this config ----
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    te_loss, te_acc, pred, true = evaluate(
        model, test_loader, criterion, pred_fn, device, use_amp, amp_dtype,
    )

    print(f"\n[{name}] Test Loss: {te_loss:.4f}  Test Accuracy: {te_acc:.4f}")
    print(classification_report(
        true, pred, labels=list(range(N_CLASSES)),
        target_names=["RUL>400", "RUL>300", "RUL>200", "RUL>100", "RUL<100"],
        zero_division=0,
    ))
    print(confusion_matrix(true, pred))

    np.save(os.path.join(cfg_output_dir, "clf_pred.npy"), pred)
    np.save(os.path.join(cfg_output_dir, "clf_true.npy"), true)

    return {
        "config": name,
        "cnn_filters": cfg["cnn_filters"],
        "kernel_size": cfg["kernel_size"],
        "cnn_layers": cfg["cnn_layers"],
        "lstm_units": cfg["lstm_units"],
        "lstm_layers": cfg["lstm_layers"],
        "dropout": cfg["dropout"],
        "n_params": n_params,
        "best_val_acc": round(best_val_acc, 4),
        "test_acc": round(te_acc, 4),
        "test_loss": round(te_loss, 4),
    }


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    # ---- Load data ONCE, reuse the same split + scalers for every config ----
    print("\nLoading data (shared across all configs)...")
    train_loader, val_loader, test_loader, scalers = build_dataloaders(
        content_dir=args.content_dir,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    dq_scaler, summary_scaler = scalers
    joblib.dump(dq_scaler, os.path.join(args.output_dir, "dq_scaler.pkl"))
    joblib.dump(summary_scaler, os.path.join(args.output_dir, "summary_scaler.pkl"))
    summary_feats = summary_scaler.n_features_in_
    print(f"summary_feats: {summary_feats}")

    results = []
    for name, cfg in CONFIGS.items():
        result = run_one_config(
            name, cfg, args, device,
            train_loader, val_loader, test_loader,
            summary_feats, use_amp, amp_dtype, scaler,
        )
        results.append(result)

    # ---- Summary table ----
    results.sort(key=lambda r: r["test_acc"], reverse=True)

    print("\n" + "=" * 100)
    print("CNN-LSTM SWEEP SUMMARY (sorted by test accuracy)")
    print("=" * 100)
    header = f"{'Config':<8} {'Params':>10} {'BestValAcc':>11} {'TestAcc':>9} {'TestLoss':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['config']:<8} {r['n_params']:>10,} {r['best_val_acc']:>11.4f} "
              f"{r['test_acc']:>9.4f} {r['test_loss']:>9.4f}")

    csv_path = os.path.join(args.output_dir, "NMC_lstm_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved summary to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir", default="./content_NMC")
    parser.add_argument("--output_dir", default="./NMC_lstm")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=600)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss", default="ordinal", choices=["cross_entropy", "ordinal"])
    args = parser.parse_args()

    main(args)
