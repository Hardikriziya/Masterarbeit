"""
NMC_dataset_loader.py - BatteryML RUL classification dataset.

"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


# Configuration
EOL_fraction = 0.80
V_bins = 1000
ref_cycle = 9          # reference cycle index used for the qdlin baseline curve

N_EARLY = 8
N_RANDOM = 8
N_INPUT = N_EARLY + N_RANDOM
N_CLASSES = 5

# Summary feature keys (24 scalar features)
SUMMARY_KEYS = (
    "Qd", "Qc", "c_t", "dc_t", "t80_soc",
    "dqdv_min", "dqdv_avg", "dqdv_slope_max", "dqdv_slope_min",
    "dqdv_peak_width", "dqdv_peak_voltage",
    "dvdq_min", "dvdq_avg", "dvdq_slope_max", "dvdq_slope_min",
    "tmax", "tavg", "IR", "discharge_V_kurtosis",
    "log_std_Qd", "log_std_Qc", "log_std_Id", "log_std_Ic",
    "qdlin_mean",
)

# 24 summary features + 4 positional-encoding values 
N_SUMMARY = len(SUMMARY_KEYS) + 4


def rul_to_class(rul: float) -> int:
    """Map remaining useful life to one of five classes."""
    if rul > 400: return 0
    if rul > 300: return 1
    if rul > 200: return 2
    if rul > 100: return 3
    return 4


def find_eol_idx(qd: np.ndarray, fraction: float = EOL_fraction) -> int:
    """
    Return the array index of EOL.

    Initial capacity = maximum valid Qd among the first up-to-10 valid cycles
    (10 is a fixed reference-window size, independent of `ref_cycle`, which
    is used elsewhere for the qdlin baseline curve).
    EOL = first later cycle with Qd < fraction * initial_capacity.
    If no crossing occurs, use the minimum-Qd cycle after the reference
    period. If fewer than 2 valid cycles exist, keep everything.
    """
    qd = np.asarray(qd, dtype=np.float32).reshape(-1)
    valid_idx = np.flatnonzero(np.isfinite(qd) & (qd > 0))

    if valid_idx.size < 2:
        return len(qd)

    n_ref = min(10, valid_idx.size)
    q_ini = float(qd[valid_idx[:n_ref]].max())
    if q_ini <= 0:
        return len(qd)

    q_eol = fraction * q_ini
    ref_end = valid_idx[n_ref - 1]

    later = valid_idx[valid_idx > ref_end]
    if later.size == 0:
        return len(qd)

    crossing = later[qd[later] < q_eol]
    if crossing.size:
        return int(crossing[0])

    return int(later[np.argmin(qd[later])])


# NPZ loading
def get_array(data, key: str, dtype=np.float32) -> np.ndarray:
    """Read a 1-D array; return an empty array if the key is missing."""
    if key not in data.files:
        return np.empty(0, dtype=dtype)
    return np.asarray(data[key], dtype=dtype).reshape(-1)


def load_npz_cell(path: str) -> dict:
    """
    Load one cell, determine 80%-capacity EOL, and truncate post-EOL data.
    Curves are kept at their native per-cycle length (no resampling) -
    length mismatches are handled later via truncate/zero-pad, same as
    code_1.
    """
    with np.load(path, allow_pickle=True) as d:
        qd = get_array(d, "qd")
        qc = get_array(d, "qc")
        cycle_index = get_array(d, "cycle_index", np.int32)

        qdlin_field = d["qdlin"]
        qdlin_raw = (np.asarray(qdlin_field, dtype=np.float32)
                     if qdlin_field.dtype != object
                     else np.asarray(qdlin_field, dtype=object))

        dqdv_raw = (np.asarray(d["dqdv"], dtype=np.float32)
                    if "dqdv" in d.files else np.zeros((0,), dtype=np.float32))

        eol_idx = find_eol_idx(qd)
        n_keep = min(eol_idx + 1, len(qd))

        if eol_idx < len(qd) and cycle_index.size and eol_idx < cycle_index.size:
            cycle_life = int(cycle_index[eol_idx])
        elif eol_idx < len(qd):
            cycle_life = int(eol_idx + 1)
        else:
            cycle_life = int(cycle_index[-1]) if cycle_index.size else int(len(qd))

        summary = {k: get_array(d, k)[:n_keep] for k in SUMMARY_KEYS}
        summary["Qd"] = qd[:n_keep]
        summary["Qc"] = qc[:n_keep]

        curves = [np.asarray(x, dtype=np.float32).reshape(-1)
                  for x in qdlin_raw[:n_keep]]

        return {
            "cycle_life": cycle_life,
            "cycle_index": cycle_index[:n_keep],
            "summary": summary,
            "qdlin": curves,
            "dqdv": [x for x in dqdv_raw[:n_keep]] if dqdv_raw.ndim > 1 else [],
        }


def load_all_npz(content_dir: str) -> dict:
    """Recursively load all .npz cell files."""
    root = Path(content_dir)
    cells = {}

    for path in sorted(root.rglob("*.npz")):
        if path.name.startswith("._"):
            continue

        cell_id = "__".join(path.relative_to(root).with_suffix("").parts)
        try:
            cells[cell_id] = load_npz_cell(str(path))
        except Exception as exc:
            print(f"WARNING: skipped {path.name}: {exc}")

    print(f"Loaded {len(cells)} cells")
    return cells


# ---------------------------------------------------------------------
# Sample indexing (per-cell, class-balanced — mirrors code_1)
# ---------------------------------------------------------------------
def build_sample_index(
    cells: dict,
    cell_ids: list,
    n_samples: int = 500,
    seed: int = 42,
) -> list:
    """
    Build a class-balanced sample index.

    `n_samples // N_CLASSES` windows per class are drawn INDEPENDENTLY
    FROM EACH CELL (using a per-cell RNG), matching code_1. Total dataset
    size therefore scales with the number of cells, not a single global
    cap.

    Returns a list of lightweight index tuples:
        (cell_id, window_start, label, rul)
    """
    index = []
    n_per_class = max(1, n_samples // N_CLASSES)

    for i, cid in enumerate(cell_ids):
        if cid not in cells:
            continue
        cell_rng = np.random.default_rng(seed + i + 3)

        cell = cells[cid]
        cycle_life = int(cell["cycle_life"])
        cycle_index = np.asarray(cell.get("cycle_index", []), dtype=np.int32).reshape(-1)
        n_cyc = min(len(cell["summary"]["Qd"]), len(cell["qdlin"]))
        if cycle_index.size:
            n_cyc = min(n_cyc, cycle_index.size)

        if n_cyc < N_INPUT:
            continue

        max_start = n_cyc - N_RANDOM
        class_starts = {c: [] for c in range(N_CLASSES)}

        for start in range(N_EARLY, max_start + 1):
            if start + N_RANDOM > n_cyc - 4:   # 4-cycle safety margin
                continue
            end_cycle = int(cycle_index[start + N_RANDOM - 1])
            rul = max(0, cycle_life - end_cycle)
            label = rul_to_class(rul)
            class_starts[label].append(start)

        for label, starts_list in class_starts.items():
            if not starts_list:
                continue
            pick = cell_rng.choice(
                starts_list, size=min(n_per_class, len(starts_list)), replace=False
            )
            for s in pick:
                end_cycle = int(cycle_index[int(s) + N_RANDOM - 1])
                index.append((cid, int(s), label, max(0, cycle_life - end_cycle)))

    return index


# ---------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------
def safe_value(arr: np.ndarray, idx: int) -> float:
    """Return arr[idx] when valid, otherwise 0."""
    return float(arr[idx]) if idx < len(arr) and np.isfinite(arr[idx]) else 0.0


def summary_row(cell: dict, cycle_idx: int, d_pos: int = 5) -> np.ndarray:
    """
    Build scalar features for one cycle: the summary scalars plus a
    4-value sinusoidal positional encoding of the real cycle number
    (matches code_1 exactly, and keeps the row length = N_SUMMARY = 28).
    """
    summary = cell["summary"]
    cycle_index = np.asarray(cell["cycle_index"], dtype=np.int32)

    if cycle_index.size and cycle_idx < cycle_index.size:
        cycle_num = max(1, int(cycle_index[cycle_idx]))
    else:
        cycle_num = cycle_idx + 1

    scalar = np.array(
        [safe_value(summary[k], cycle_idx) for k in SUMMARY_KEYS],
        dtype=np.float32,
    )

    pe = np.asarray([
        np.sin(cycle_num / 3000 ** (2 * i / d_pos)) if i % 2 == 0 else
        np.cos(cycle_num / 3000 ** ((2 * i - 1) / d_pos))
        for i in range(1, d_pos)
    ], dtype=np.float32)

    return np.concatenate([scalar, pe])


class CellCache:
    """
    Pre-builds all cycle arrays for a cell once.
    Curves are truncated/zero-padded to V_bins (never resampled), same
    as code_1.
    """

    def __init__(self, cell: dict):
        n_cyc = len(cell["qdlin"])
        ref_idx = min(ref_cycle, n_cyc - 1)
        ref_curve = np.asarray(cell["qdlin"][ref_idx], dtype=np.float32).reshape(-1)

        dq_arr = np.zeros((n_cyc, V_bins), dtype=np.float32)
        for c in range(n_cyc):
            q = np.asarray(cell["qdlin"][c], dtype=np.float32).reshape(-1)
            length = min(len(q), len(ref_curve), V_bins)
            dq_arr[c, :length] = q[:length] - ref_curve[:length]

        self.dq = dq_arr[:, None, :]   # (n_cyc, 1, V_bins)
        self.summary = np.stack(
            [summary_row(cell, c) for c in range(n_cyc)]
        ).astype(np.float32)


def build_sample(
    cell: dict,
    cache: CellCache,
    start: int,
    dq_scaler: Optional[StandardScaler],
    summary_scaler: Optional[StandardScaler],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create one N_INPUT-cycle model input (first N_EARLY + N_RANDOM window)."""
    cycle_ids = np.r_[np.arange(N_EARLY), np.arange(start, start + N_RANDOM)]

    dq = cache.dq[cycle_ids].copy()
    summary = cache.summary[cycle_ids].copy()

    if dq_scaler is not None:
        t, c, f = dq.shape
        dq = dq_scaler.transform(dq.reshape(t, c * f)).reshape(t, c, f)
        summary = summary_scaler.transform(summary)

    return (
        torch.from_numpy(dq.astype(np.float32)),
        torch.from_numpy(summary.astype(np.float32)),
    )


# ---------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------
class BMLBatteryClsDataset(Dataset):
    def __init__(
        self,
        cells: dict,
        cell_ids: list,
        n_samples: int = 500,
        scalers: Optional[Tuple[StandardScaler, StandardScaler]] = None,
        seed: int = 42,
        fit_scaler: bool = False,
    ):
        self.cells = cells
        self.index = build_sample_index(cells, cell_ids, n_samples, seed)

        if not self.index:
            raise ValueError("No valid samples could be constructed.")

        # Build caches only for cells actually used by this split.
        used_cells = {cid for cid, _, _, _ in self.index}
        self.cache = {cid: CellCache(cells[cid]) for cid in used_cells}

        print(f"Samples: {len(self.index)}")
        labels = np.array([x[2] for x in self.index])
        for c in range(N_CLASSES):
            print(f"  Class {c}: {(labels == c).sum()}")

        if scalers is not None:
            self.dq_scaler, self.summary_scaler = scalers
        elif fit_scaler:
            self.dq_scaler, self.summary_scaler = self.fit_scalers(seed)
        else:
            self.dq_scaler = self.summary_scaler = None

    def fit_scalers(self, seed: int, max_fit: int = 10000):
        """Fit scalers using only a random subset of THIS dataset (train only)."""
        rng = np.random.default_rng(seed)
        chosen = rng.choice(
            len(self.index),
            size=min(max_fit, len(self.index)),
            replace=False,
        )

        dq_rows, summary_rows = [], []

        for i in chosen:
            cid, start, _, _ = self.index[i]
            dq, summary = build_sample(
                self.cells[cid], self.cache[cid], start, None, None
            )
            dq_rows.append(dq.numpy())
            summary_rows.append(summary.numpy())

        dq = np.concatenate(dq_rows, axis=0)
        summary = np.concatenate(summary_rows, axis=0)

        dq_scaler = StandardScaler().fit(dq.reshape(dq.shape[0], -1))
        summary_scaler = StandardScaler().fit(summary)

        return dq_scaler, summary_scaler

    def get_scalers(self):
        return self.dq_scaler, self.summary_scaler

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        cid, start, label, rul = self.index[idx]
        dq, summary = build_sample(
            self.cells[cid],
            self.cache[cid],
            start,
            self.dq_scaler,
            self.summary_scaler,
        )

        return {
            "dq": dq,
            "summary": summary,
            "label": torch.tensor(label, dtype=torch.long),
            "rul": torch.tensor(rul, dtype=torch.float32),
        }


# DataLoader construction
def build_dataloaders(
    content_dir: str,
    batch_size: int = 32,
    n_samples: int = 600,
    val_ratio: float = 0.20,
    num_workers: int = 0,
    seed: int = 42,
):
    cells = load_all_npz(content_dir)

    ids = list(cells)
    if len(ids) < 3:
        raise ValueError("At least three cells are required for train/val/test splitting.")

    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    n_val = max(1, int(len(ids) * val_ratio))
    val_ids = ids[:n_val]
    test_ids = ids[n_val:2 * n_val]
    train_ids = ids[2 * n_val:]

    print(
        f"Cell split -> Train: {len(train_ids)}, "
        f"Val: {len(val_ids)}, Test: {len(test_ids)}"
    )

    train_ds = BMLBatteryClsDataset(
        cells, train_ids, n_samples=n_samples,
        seed=seed, fit_scaler=True,
    )
    scalers = train_ds.get_scalers()

    val_ds = BMLBatteryClsDataset(
        cells, val_ids, n_samples=n_samples,
        scalers=scalers, seed=seed + 1,
    )
    test_ds = BMLBatteryClsDataset(
        cells, test_ids, n_samples=n_samples,
        scalers=scalers, seed=seed + 2,
    )

    loader_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        DataLoader(train_ds, shuffle=True, **loader_args),
        DataLoader(val_ds, shuffle=False, **loader_args),
        DataLoader(test_ds, shuffle=False, **loader_args),
        scalers,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=600)
    parser.add_argument("--val_ratio", type=float, default=0.20)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_loader, val_loader, test_loader, scalers = build_dataloaders(
        args.content_dir,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    for name, loader in (
        ("Train", train_loader),
        ("Val", val_loader),
        ("Test", test_loader),
    ):
        batch = next(iter(loader))
        print(
            f"{name}: {len(loader.dataset)} samples | "
            f"dq={tuple(batch['dq'].shape)} | "
            f"summary={tuple(batch['summary'].shape)} | "
            f"label={tuple(batch['label'].shape)}"
        )

    print(
        f"Scaler dimensions -> dq: {scalers[0].n_features_in_}, "
        f"summary: {scalers[1].n_features_in_}"
    )


if __name__ == "__main__":
    main()
