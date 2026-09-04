"""
Preprocessing of NMC dataset — whole-dataset feature extraction + SOH/correlation analysis.

"""

import os
import re
import zipfile
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.interpolate import interp1d


class Preprocessing_NMC:

    feature_names = [
        # --- capacity features ---
        "qd", "qc",

        # --- time features ---
        "c_t", "dc_t", "t80_soc",

        # --- dQ/dV features ---
        "dqdv_max", "dqdv_min", "dqdv_avg",
        "dqdv_slope_max", "dqdv_slope_min",
        "dqdv_area", "dqdv_peak_width", "dqdv_peak_voltage",

        # --- dV/dQ features ---
        "dvdq_max", "dvdq_min", "dvdq_avg",
        "dvdq_slope_max", "dvdq_slope_min",

        # --- temperature and internal-resistance features ---
        "tmax", "tavg", "IR",

        # --- shape feature ---
        "discharge_V_kurtosis",

        # --- noise descriptors ---
        "log_std_Qd", "log_std_Qc", "log_std_Id", "log_std_Ic",

        # --- curve-derived scalar descriptor ---
        "qdlin",
    ]

    feature_labels = {
        "qd": "Discharge capacity, qd (Ah)",
        "qc": "Charge capacity, qc (Ah)",
        "c_t": "Charge time, c_t (s)",
        "dc_t": "Discharge time, dc_t (s)",
        "t80_soc": "Time to 80% discharge capacity (s)",
        "dqdv_min": "Minimum discharge dQ/dV",
        "dqdv_avg": "Average discharge dQ/dV",
        "dqdv_slope_max": "Maximum slope of discharge dQ/dV",
        "dqdv_slope_min": "Minimum slope of discharge dQ/dV",
        "dqdv_area": "Area under the discharge dQ/dV curve",
        "dqdv_peak_width": "Main dQ/dV peak width (V)",
        "dqdv_peak_voltage": "Voltage of the main dQ/dV peak (V)",
        "dvdq_max": "Maximum discharge dV/dQ",
        "dvdq_min": "Minimum discharge dV/dQ",
        "dvdq_avg": "Average discharge dV/dQ",
        "dvdq_slope_max": "Maximum slope of discharge dV/dQ",
        "dvdq_slope_min": "Minimum slope of discharge dV/dQ",
        "tmax": "Maximum cycle temperature (°C)",
        "tavg": "Average cycle temperature (°C)",
        "IR": "Average internal resistance (Ω)",
        "discharge_V_kurtosis": "Excess kurtosis of discharge voltage",
        "log_std_Qd": "20·log10(std(discharge capacity))",
        "log_std_Qc": "20·log10(std(charge capacity))",
        "log_std_Id": "20·log10(std(discharge current))",
        "log_std_Ic": "20·log10(std(charge current))",
        "qdlin": "Mean interpolated discharge capacity over voltage bins",
    }

    def __init__(self, zip_path, cycle_indices=None, save_dir="plots", max_cells=None):
        self.zip_path = zip_path
        self.zip_name = os.path.splitext(os.path.basename(zip_path))[0]
        self.cycle_indices = cycle_indices
        self.save_dir = save_dir
        self.max_cells = max_cells

        self.all_cycle_data = []
        self.cell_names = []
        self.original_cell_indices = []
        self.virtual_paths = []
        self.rows = []
        self.cell_npz_records = []

    def _save_fig(self, fig, filename):
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {path}")

    @staticmethod
    def _finite(x):
        x = np.asarray(x, dtype=float)
        return x[np.isfinite(x)]

    @staticmethod
    def _trapezoid(y, x):
        """Compatibility helper for NumPy 1.x and 2.x."""
        if hasattr(np, "trapezoid"):
            return np.trapezoid(y, x)
        return np.trapz(y, x) 
    
    def read_data(self):
        """
        Load all cells from the ZIP file. Use max_cells for testing/debugging.
        """
        with zipfile.ZipFile(self.zip_path, "r") as z:
            pkl_files = sorted([f for f in z.namelist() if f.endswith(".pkl")])
            if self.max_cells is not None:
                pkl_files = pkl_files[: int(self.max_cells)]

            print(f"Found {len(pkl_files)} PKL files to process.")

            for original_idx, pkl_name in enumerate(pkl_files):
                try:
                    with z.open(pkl_name) as f:
                        raw_cell = pickle.load(f)
                    cell_cycles, cell_name = self.organize_cell(raw_cell)
                    self.all_cycle_data.append(cell_cycles)
                    self.cell_names.append(cell_name)
                    self.original_cell_indices.append(original_idx)
                    self.virtual_paths.append(Path(pkl_name))
                    print(f"Loaded cell {original_idx}: {cell_name} | cycles={len(cell_cycles)}")
                except Exception as exc:
                    print(f"Skipping {pkl_name} due to error: {exc}")

        print(f"Loaded {len(self.all_cycle_data)} cells successfully.")

    def organize_cell(self, raw_cell):
        cell_id = raw_cell.get("cell_id", "unknown")
        cycle_data = raw_cell.get("cycle_data", [])
        cycles_iter = cycle_data.items() if isinstance(cycle_data, dict) else enumerate(cycle_data)

        cell_cycles = []
        for cycle_index, cycle in cycles_iter:
            temp = cycle.get(
                "temperature_in_C",
                cycle.get("temperature", cycle.get("temperature_in_Celsius", [])),
            )
            ir = cycle.get(
                "internal_resistance_in_ohm",
                cycle.get("internal_resistance", cycle.get("IR", [])),
            )

            cell_cycles.append({
                "Cycle_No": int(cycle_index),
                "Voltage": np.asarray(cycle.get("voltage_in_V", []), float),
                "Current": np.asarray(cycle.get("current_in_A", []), float),
                "Time": np.asarray(cycle.get("time_in_s", []), float),
                "Charge_Capacity": np.asarray(cycle.get("charge_capacity_in_Ah", []), float),
                "Discharge_Capacity": np.asarray(cycle.get("discharge_capacity_in_Ah", []), float),
                "Temperature": np.asarray(temp, float),
                "Internal_Resistance": np.asarray(ir, float),
                "Cell_Name": cell_id,
            })

        return cell_cycles, cell_id

    def process_cell_data(self):
        """
        Extract scalar features for all loaded cells and retain the full
        cycle-wise arrays required for one compressed NPZ file per cell.
        """
        if not self.all_cycle_data:
            raise RuntimeError("No data loaded. Run read_data() first.")

        selected = (
            set(self.cycle_indices)
            if self.cycle_indices is not None
            else None
        )

        self.rows = []
        self.cell_npz_records = []

        for internal_cell_index, cell_data in enumerate(self.all_cycle_data):
            cell_name = self.cell_names[internal_cell_index]
            original_cell_index = self.original_cell_indices[
                internal_cell_index
            ]
            virtual_path = self.virtual_paths[internal_cell_index]
            valid_count = 0

            cell_record = {
                "cell_id": cell_name,
                "virtual_path": virtual_path,
                "cell_index": original_cell_index,
                "cycle_index": [],
                "features": {
                    feature: [] for feature in self.feature_names
                },
                "qdlin": [],
                "dqdv": [],
            }

            for cycle in cell_data:
                cyc_no = cycle["Cycle_No"]

                if selected is not None and cyc_no not in selected:
                    continue

                V = cycle["Voltage"]
                I = cycle["Current"]
                Time = cycle["Time"]
                Qc = cycle["Charge_Capacity"]
                Qd = cycle["Discharge_Capacity"]
                Temp = cycle["Temperature"]
                IR = cycle["Internal_Resistance"]

                if V.size == 0 or I.size == 0 or Time.size == 0:
                    continue

                if Time.size != I.size or V.size != I.size:
                    continue

                charge_time = Time[I > 0]
                discharge_time = Time[I < 0]

                if not self._is_complete(
                    charge_time,
                    discharge_time,
                    Qd,
                    Qc,
                ):
                    continue

                (
                    Ic, Vc, Qcc, Tc, Tempc, IRc
                ) = self._clean_charge_data(
                    V, I, Qc, Time, Temp, IR
                )

                (
                    Id, Vd, Qdd, Td, Tempd, IRd
                ) = self._clean_discharge_data(
                    V, I, Qd, Time, Temp, IR
                )

                if Qcc.size == 0 or Qdd.size == 0:
                    continue

                dis_deriv = self.calculate_derivatives(Vd, Qdd)
                dis_deriv_inv = self.calculate_derivatives_dVdQ(
                    Vd, Qdd
                )
                qdlin_curve = self.interpolate_capacity_over_voltage(
                    Vd, Qdd
                )

                feats = self._compute_cycle_features(
                    Ic, Vc, Tc, Qcc, Tempc, IRc,
                    Id, Vd, Td, Qdd, Tempd, IRd,
                    dis_deriv, dis_deriv_inv,
                    qdlin_curve,
                )

                row = {
                    "Cell_Index": original_cell_index,
                    "Cell_Name": cell_name,
                    "Cycle": cyc_no,
                }
                row.update(feats)
                self.rows.append(row)

                cell_record["cycle_index"].append(cyc_no)

                for feature_name in self.feature_names:
                    cell_record["features"][feature_name].append(
                        feats.get(feature_name, np.nan)
                    )

                cell_record["qdlin"].append(
                    self._fixed_curve(qdlin_curve, length=400)
                )

                if (
                    dis_deriv is not None
                    and "dQdV" in dis_deriv
                ):
                    dqdv_curve = dis_deriv["dQdV"]
                else:
                    dqdv_curve = np.full(
                        400,
                        np.nan,
                        dtype=np.float32,
                    )

                cell_record["dqdv"].append(
                    self._fixed_curve(dqdv_curve, length=400)
                )

                valid_count += 1

            if valid_count:
                self.cell_npz_records.append(cell_record)

            print(
                f"Processed cell {original_cell_index} "
                f"({cell_name}) | valid cycles={valid_count}"
            )

        print(
            "Processing complete. "
            f"Total rows/cycles extracted: {len(self.rows)}"
        )

    @staticmethod
    def _is_complete(charge_time, discharge_time, dq_raw, cq_raw):
        if charge_time.size == 0 or discharge_time.size == 0:
            return False
        qd_max = float(np.nanmax(dq_raw)) if dq_raw.size else 0.0
        qc_max = float(np.nanmax(cq_raw)) if cq_raw.size else 0.0
        return not (qd_max <= 0.0 and qc_max <= 0.0)

    @staticmethod
    def _align_optional_signal(x, mask):
        x = np.asarray(x, float)
        if x.size == mask.size:
            return x[mask]
        if x.size == 1:
            return x.copy()
        return np.array([], dtype=float)

    @classmethod
    def _clean_charge_data(cls, V, I, Q, Time, Temp=None, IR=None):
        mask = (I > 0) & (V >= 2.70) & (V <= 4.00)
        return (
            I[mask], V[mask], Q[mask], Time[mask],
            cls._align_optional_signal(Temp if Temp is not None else [], mask),
            cls._align_optional_signal(IR if IR is not None else [], mask),
        )

    @classmethod
    def _clean_discharge_data(cls, V, I, Q, Time, Temp=None, IR=None):
        mask = (I < 0) & (V >= 2.70) & (V <= 4.00)
        return (
            I[mask], V[mask], Q[mask], Time[mask],
            cls._align_optional_signal(Temp if Temp is not None else [], mask),
            cls._align_optional_signal(IR if IR is not None else [], mask),
        )

    @staticmethod
    def calculate_derivatives(V, Q):
        """Calculate discharge dQ/dV curve."""
        V = np.asarray(V, float)
        Q = np.asarray(Q, float)
        mask = np.isfinite(V) & np.isfinite(Q)
        V, Q = V[mask], Q[mask]

        if V.size < 20:
            return None

        idx = np.argsort(V)
        V_u, ui = np.unique(V[idx], return_index=True)
        Q_u = Q[idx][ui]

        if V_u.size < 20 or np.nanmax(V_u) == np.nanmin(V_u):
            return None

        V_lin = np.linspace(V_u.min(), V_u.max(), 400)
        kind = "cubic" if V_u.size >= 4 else "linear"
        Q_lin = interp1d(V_u, Q_u, kind=kind, bounds_error=False, fill_value="extrapolate")(V_lin)

        window_length = 31
        if Q_lin.size <= window_length:
            window_length = Q_lin.size if Q_lin.size % 2 == 1 else Q_lin.size - 1
        if window_length < 5:
            return None

        polyorder = min(3, window_length - 2)
        dQdV = savgol_filter(
            Q_lin,
            window_length=window_length,
            polyorder=polyorder,
            deriv=1,
            delta=V_lin[1] - V_lin[0],
        )

        return {"Voltage_V_dQdV": V_lin, "dQdV": dQdV}

    @staticmethod
    def calculate_derivatives_dVdQ(V, Q):
        """
        Calculate discharge dV/dQ curve.
        """
        V = np.asarray(V, float)
        Q = np.asarray(Q, float)
        mask = np.isfinite(V) & np.isfinite(Q)
        V, Q = V[mask], Q[mask]

        if Q.size < 20:
            return None

        idx = np.argsort(Q)
        Q_u, ui = np.unique(Q[idx], return_index=True)
        V_u = V[idx][ui]

        if Q_u.size < 20 or np.nanmax(Q_u) == np.nanmin(Q_u):
            return None

        Q_lin = np.linspace(Q_u.min(), Q_u.max(), 400)
        kind = "cubic" if Q_u.size >= 4 else "linear"
        V_lin_curve = interp1d(Q_u, V_u, kind=kind, bounds_error=False, fill_value="extrapolate")(Q_lin)

        window_length = 31
        if V_lin_curve.size <= window_length:
            window_length = V_lin_curve.size if V_lin_curve.size % 2 == 1 else V_lin_curve.size - 1
        if window_length < 5:
            return None

        dVdQ = savgol_filter(
            V_lin_curve,
            window_length=window_length,
            polyorder=min(3, window_length - 2),
            deriv=1,
            delta=Q_lin[1] - Q_lin[0],
        )

        return {"Capacity_Ah_dVdQ": Q_lin, "dVdQ": dVdQ}

    @staticmethod
    def interpolate_capacity_over_voltage(V, Q, n_bins=400):
        V = np.asarray(V, dtype=float)
        Q = np.asarray(Q, dtype=float)
        mask = np.isfinite(V) & np.isfinite(Q)
        V, Q = V[mask], Q[mask]

        if V.size < 20:
            return np.array([], dtype=float)

        idx = np.argsort(V)
        V_u, ui = np.unique(V[idx], return_index=True)
        Q_u = Q[idx][ui]

        if V_u.size < 20 or np.nanmax(V_u) == np.nanmin(V_u):
            return np.array([], dtype=float)

        V_lin = np.linspace(V_u.min(), V_u.max(), n_bins)
        kind = "cubic" if V_u.size >= 4 else "linear"
        return interp1d(V_u, Q_u, kind=kind, bounds_error=False, fill_value="extrapolate")(V_lin)

    @staticmethod
    def _time_to_fraction_of_capacity(Td, Qdd, fraction=0.8):
        Td = np.asarray(Td, float)
        Qdd = np.asarray(Qdd, float)
        mask = np.isfinite(Td) & np.isfinite(Qdd)
        Td, Qdd = Td[mask], Qdd[mask]

        if Td.size < 2 or Qdd.size < 2:
            return np.nan

        idx = np.argsort(Qdd)
        Qdd_s, Td_s = Qdd[idx], Td[idx]
        Qdd_u, ui = np.unique(Qdd_s, return_index=True)
        Td_u = Td_s[ui]

        if Qdd_u.size < 2:
            return np.nan

        target_q = fraction * np.nanmax(Qdd_u)
        if target_q < Qdd_u.min() or target_q > Qdd_u.max():
            return np.nan

        t_at_target = np.interp(target_q, Qdd_u, Td_u)
        return float(t_at_target - Td_u[0])

    @staticmethod
    def _peak_features(x_axis, y_curve):
        x_axis = np.asarray(x_axis, float)
        y_curve = np.asarray(y_curve, float)
        mask = np.isfinite(x_axis) & np.isfinite(y_curve)
        x_axis, y_curve = x_axis[mask], y_curve[mask]

        if y_curve.size < 5:
            return np.nan, np.nan, np.nan

        y_abs = np.abs(y_curve)
        peaks, _ = find_peaks(y_abs)
        if peaks.size == 0:
            return np.nan, np.nan, np.nan

        main_peak = peaks[np.argmax(y_abs[peaks])]
        widths, _, left_ips, right_ips = peak_widths(y_abs, [main_peak], rel_height=0.5)
        dx = np.mean(np.diff(x_axis)) if x_axis.size > 1 else 1.0

        peak_width_val = float(widths[0] * abs(dx))
        peak_x = float(x_axis[main_peak])
        left_dist = main_peak - left_ips[0]
        right_dist = right_ips[0] - main_peak
        symmetry = float(left_dist / right_dist) if right_dist > 0 else np.nan

        return peak_width_val, peak_x, symmetry

    @classmethod
    def _safe_kurtosis(cls, x):
        """
        Excess kurtosis. Normal distribution gives approximately 0.
        """
        x = cls._finite(x)
        if x.size < 4:
            return np.nan
        std = np.std(x)
        if not np.isfinite(std) or std <= 0:
            return np.nan
        z = (x - np.mean(x)) / std
        return float(np.mean(z ** 4) - 3.0)

    @classmethod
    def _shape_features(cls, x, prefix):
        return {
            f"{prefix}_kurtosis": cls._safe_kurtosis(x),
        }
    
    @classmethod
    def _compute_cycle_features(
        cls,
        Ic, Vc, Tc, Qcc, Tempc, IRc,
        Id, Vd, Td, Qdd, Tempd, IRd,
        dis_deriv, dis_deriv_inv,
        qdlin_curve,
    ):
        """Compute only the features listed in ``feature_names``."""
        feats = {name: np.nan for name in cls.feature_names}

        def _finite(x):
            return cls._finite(x)

        def _mean_or_nan(x):
            x = _finite(x)
            return float(np.mean(x)) if x.size else np.nan

        def _max_or_nan(x):
            x = _finite(x)
            return float(np.max(x)) if x.size else np.nan

        def _log_std_db(x):
            x = _finite(x)
            if x.size <= 1:
                return np.nan
            standard_deviation = float(np.std(x))
            if not np.isfinite(standard_deviation) or standard_deviation <= 0:
                return np.nan
            return float(20.0 * np.log10(standard_deviation))

        # ---- capacity features ----
        Qdd_f = _finite(Qdd)
        Qcc_f = _finite(Qcc)
        feats["qd"] = float(np.max(Qdd_f)) if Qdd_f.size else np.nan
        feats["qc"] = float(np.max(Qcc_f)) if Qcc_f.size else np.nan

        # ---- time features ----
        Tc_f = _finite(Tc)
        Td_f = _finite(Td)
        feats["c_t"] = float(Tc_f[-1] - Tc_f[0]) if Tc_f.size > 1 else np.nan
        feats["dc_t"] = float(Td_f[-1] - Td_f[0]) if Td_f.size > 1 else np.nan
        feats["t80_soc"] = cls._time_to_fraction_of_capacity(Td, Qdd, fraction=0.8)

        # ---- temperature and internal resistance ----
        temperature_parts = [_finite(Tempc), _finite(Tempd)]
        temperature_parts = [part for part in temperature_parts if part.size]
        T_all = np.concatenate(temperature_parts) if temperature_parts else np.array([], dtype=float)

        resistance_parts = [_finite(IRc), _finite(IRd)]
        resistance_parts = [part for part in resistance_parts if part.size]
        IR_all = np.concatenate(resistance_parts) if resistance_parts else np.array([], dtype=float)

        feats["tmax"] = _max_or_nan(T_all)
        feats["tavg"] = _mean_or_nan(T_all)
        feats["IR"] = _mean_or_nan(IR_all)

        # ---- discharge-voltage shape feature ----
        feats["discharge_V_kurtosis"] = cls._safe_kurtosis(Vd)

        # ---- dQ/dV features ----
        if dis_deriv is not None and "dQdV" in dis_deriv:
            V_lin = np.asarray(dis_deriv.get("Voltage_V_dQdV", []), dtype=float)
            dQdV = np.asarray(dis_deriv.get("dQdV", []), dtype=float)
            valid = np.isfinite(V_lin) & np.isfinite(dQdV)
            V_lin, dQdV = V_lin[valid], dQdV[valid]

            if dQdV.size:
                feats["dqdv_max"] = float(np.max(dQdV))
                feats["dqdv_min"] = float(np.min(dQdV))
                feats["dqdv_avg"] = float(np.mean(dQdV))

            if dQdV.size > 1 and V_lin.size == dQdV.size:
                dqdv_gradient = np.gradient(dQdV, V_lin)
                feats["dqdv_slope_max"] = float(np.max(dqdv_gradient))
                feats["dqdv_slope_min"] = float(np.min(dqdv_gradient))
                feats["dqdv_area"] = float(cls._trapezoid(dQdV, V_lin))

                peak_width, peak_voltage, _ = cls._peak_features(V_lin, dQdV)
                feats["dqdv_peak_width"] = peak_width
                feats["dqdv_peak_voltage"] = peak_voltage

        # ---- dV/dQ features ----
        if dis_deriv_inv is not None and "dVdQ" in dis_deriv_inv:
            Q_lin = np.asarray(dis_deriv_inv.get("Capacity_Ah_dVdQ", []), dtype=float)
            dVdQ = np.asarray(dis_deriv_inv.get("dVdQ", []), dtype=float)
            valid = np.isfinite(Q_lin) & np.isfinite(dVdQ)
            Q_lin, dVdQ = Q_lin[valid], dVdQ[valid]

            if dVdQ.size:
                feats["dvdq_max"] = float(np.max(dVdQ))
                feats["dvdq_min"] = float(np.min(dVdQ))
                feats["dvdq_avg"] = float(np.mean(dVdQ))

            if dVdQ.size > 1 and Q_lin.size == dVdQ.size:
                dvdq_gradient = np.gradient(dVdQ, Q_lin)
                feats["dvdq_slope_max"] = float(np.max(dvdq_gradient))
                feats["dvdq_slope_min"] = float(np.min(dvdq_gradient))

        # ---- noise descriptors ----
        feats["log_std_Qd"] = _log_std_db(Qdd)
        feats["log_std_Qc"] = _log_std_db(Qcc)
        feats["log_std_Id"] = _log_std_db(Id)
        feats["log_std_Ic"] = _log_std_db(Ic)

        # ---- qdlin scalar descriptor ----
        qdlin_f = _finite(qdlin_curve)
        feats["qdlin"] = float(np.mean(qdlin_f)) if qdlin_f.size else np.nan

        return feats

    @staticmethod
    def _fixed_curve(values, length=400):
        """
        Return a fixed-length float32 curve.

        Curves shorter than ``length`` are linearly resampled. Missing curves
        are represented by NaN and later interpolated cycle-wise where possible.
        """
        values = np.asarray(values, dtype=float).reshape(-1)
        values = values[np.isfinite(values)]

        if values.size == 0:
            return np.full(length, np.nan, dtype=np.float32)

        if values.size == length:
            return values.astype(np.float32)

        old_axis = np.linspace(0.0, 1.0, values.size)
        new_axis = np.linspace(0.0, 1.0, length)

        return np.interp(
            new_axis,
            old_axis,
            values,
        ).astype(np.float32)

    @staticmethod
    def _interp_nan(values):
        """
        Fill NaN/inf values by linear interpolation along the cycle axis.

        If every value is missing, zeros are returned. A single finite value
        is propagated through the array.
        """
        values = np.asarray(values, dtype=np.float32).copy()
        values[~np.isfinite(values)] = np.nan

        if values.size == 0:
            return values

        valid = np.isfinite(values)

        if not valid.any():
            return np.zeros_like(values, dtype=np.float32)

        if valid.sum() == 1:
            values[:] = values[valid][0]
            return values.astype(np.float32)

        positions = np.arange(values.size)
        values[~valid] = np.interp(
            positions[~valid],
            positions[valid],
            values[valid],
        )

        return values.astype(np.float32)

    @classmethod
    def _interp_nan_matrix(cls, matrix):
        """
        Fill missing values independently in every curve position across
        the cycle axis.
        """
        matrix = np.asarray(matrix, dtype=np.float32)

        if matrix.ndim != 2:
            raise ValueError(
                "Expected a two-dimensional cycle-by-curve matrix."
            )

        output = matrix.copy()

        for column_index in range(output.shape[1]):
            output[:, column_index] = cls._interp_nan(
                output[:, column_index]
            )

        return output.astype(np.float32)

    def get_feature_dataframe(self):
        if not self.rows:
            return pd.DataFrame()
        df = pd.DataFrame(self.rows)
        cols = ["Cell_Index", "Cell_Name", "Cycle"] + self.feature_names
        return df[[c for c in cols if c in df.columns]]

    def add_soh_target(self, df, n_ref_cycles=10):
        if "qd" not in df.columns:
            raise KeyError("Column 'qd' not found. Cannot compute SOH.")
        if "Cell_Index" not in df.columns:
            raise KeyError("Column 'Cell_Index' not found. Cannot compute cell-wise SOH.")

        df = df.copy().sort_values(["Cell_Index", "Cycle"]).reset_index(drop=True)
        df["SOH"] = np.nan
        df["Q_ref"] = np.nan

        for cell_id, g in df.groupby("Cell_Index", sort=False):
            qd = g["qd"].to_numpy(dtype=float)
            finite_qd = qd[np.isfinite(qd)]
            if finite_qd.size == 0:
                continue
            n_use = min(n_ref_cycles, finite_qd.size)
            ref = float(np.max(finite_qd[:n_use]))
            if not np.isfinite(ref) or ref <= 0:
                continue
            idx = g.index
            df.loc[idx, "Q_ref"] = ref
            df.loc[idx, "SOH"] = df.loc[idx, "qd"] / ref

        print("\nSOH calculation completed cell-by-cell.")
        print(df[["Cell_Index", "Cycle", "qd", "Q_ref", "SOH"]].head(20))
        print("\nSOH statistics across all cells:")
        print(df["SOH"].describe())
        return df

    @staticmethod
    def _clean_numeric_for_correlation(df, target="SOH"):
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        numeric_df = numeric_df.dropna(axis=1, how="all")

        keep_cols = []
        dropped_constant = []
        for col in numeric_df.columns:
            if col == target:
                keep_cols.append(col)
                continue
            n_unique = numeric_df[col].dropna().nunique()
            if n_unique > 1:
                keep_cols.append(col)
            else:
                dropped_constant.append(col)

        numeric_df = numeric_df[keep_cols]
        numeric_df = numeric_df.dropna(axis=0, how="any")
        return numeric_df, dropped_constant

    @classmethod
    def compute_spearman_with_soh(cls, df, target="SOH", min_pair_rows=3):
        """
        Calculate Spearman correlation between each selected feature and SOH.

        """
        if target not in df.columns:
            raise KeyError(f"Target column '{target}' not found in dataframe.")

        correlations = {}
        for feature in cls.feature_names:
            if feature not in df.columns:
                continue

            pair = df[[feature, target]].apply(pd.to_numeric, errors="coerce")
            pair = pair.replace([np.inf, -np.inf], np.nan).dropna()

            if len(pair) < min_pair_rows:
                continue
            if pair[feature].nunique() <= 1 or pair[target].nunique() <= 1:
                continue

            correlations[feature] = pair[feature].corr(
                pair[target], method="spearman"
            )

        spearman_corr = pd.Series(correlations, name="Spearman", dtype=float)
        if spearman_corr.empty:
            raise ValueError("No valid Spearman correlations with SOH were found.")

        order = spearman_corr.abs().sort_values(ascending=False).index
        return spearman_corr.reindex(order)


    @classmethod
    def compute_per_cell_correlation_summary(cls, df, target="SOH", min_rows=8):
        """
        For each cell, compute feature-to-SOH correlations.
        Returns:
          per_cell_corr: rows = cells, columns = feature correlations
          corr_stats: mean/std/min/max correlation for each feature across cells
        """
        records = []
        for cell_id, g in df.groupby("Cell_Index"):
            numeric_df, _ = cls._clean_numeric_for_correlation(g, target=target)
            if target not in numeric_df.columns or len(numeric_df) < min_rows:
                continue
            if numeric_df[target].nunique() <= 1:
                continue
            corr = numeric_df.corr(method="spearman")[target].drop(labels=[target], errors="ignore")
            rec = {"Cell_Index": cell_id, "n_rows": len(numeric_df)}
            rec.update(corr.to_dict())
            records.append(rec)

        per_cell_corr = pd.DataFrame(records)
        if per_cell_corr.empty:
            return per_cell_corr, pd.DataFrame()

        feature_cols = [c for c in per_cell_corr.columns if c not in ["Cell_Index", "n_rows"]]
        corr_stats = pd.DataFrame({
            "mean_spearman": per_cell_corr[feature_cols].mean(skipna=True),
            "std_spearman": per_cell_corr[feature_cols].std(skipna=True),
            "min_spearman": per_cell_corr[feature_cols].min(skipna=True),
            "max_spearman": per_cell_corr[feature_cols].max(skipna=True),
            "valid_cells": per_cell_corr[feature_cols].notna().sum(),
        }).sort_values("mean_spearman", key=lambda s: s.abs(), ascending=False)

        return per_cell_corr, corr_stats

    @classmethod
    def compute_feature_correlation_matrix(cls, df, target="SOH"):
        """
        Compute one Spearman correlation matrix containing:

        - every available feature listed in ``feature_names``
        - SOH as the only target column

        """
        if target not in df.columns:
            raise KeyError(
                f"Target column '{target}' not found in dataframe."
            )

        selected_columns = [
            feature
            for feature in cls.feature_names
            if feature in df.columns
        ]
        selected_columns.append(target)

        numeric_df = df[selected_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        numeric_df = numeric_df.replace(
            [np.inf, -np.inf],
            np.nan,
        )
        numeric_df = numeric_df.dropna(
            axis=1,
            how="all",
        )
        numeric_df = numeric_df.loc[
            :,
            numeric_df.nunique(dropna=True) > 1,
        ]

        if target not in numeric_df.columns:
            raise ValueError(
                f"Target '{target}' has no usable variation."
            )

        return numeric_df.corr(
            method="spearman",
            min_periods=3,
        )

    def plot_feature_correlation_matrix(
        self,
        corr_matrix,
        title_suffix="Across All NMC Cells",
        save=True,
        show=True,
    ):
        """
        Plot lower triangle of the Spearman matrix containing all selected features plus SOH.

        """
        if corr_matrix is None or corr_matrix.empty:
            print("Spearman correlation matrix is empty.")
            return

        corr_df = corr_matrix.copy()
        corr_values = corr_df.to_numpy(dtype=float)

        # Hide diagonal and upper triangle.
        plot_values = corr_values.copy()
        upper_indices = np.triu_indices_from(plot_values, k=0)
        plot_values[upper_indices] = np.nan
        plot_values[~np.isfinite(plot_values)] = np.nan
        plot_masked = np.ma.masked_invalid(plot_values)

        n_variables = len(corr_df.columns)
        figure_size = max(12, 0.65 * n_variables)

        fig, ax = plt.subplots(
            figsize=(figure_size, figure_size)
        )
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        cmap = plt.colormaps["RdBu_r"].copy()
        cmap.set_bad(color="white", alpha=1.0)

        image = ax.imshow(
            plot_masked,
            cmap=cmap,
            vmin=-1,
            vmax=1,
            interpolation="nearest",
            aspect="equal",
        )

        ax.set_xticks(np.arange(n_variables))
        ax.set_xticklabels(
            corr_df.columns,
            rotation=90,
            fontsize=8,
        )

        ax.set_yticks(np.arange(n_variables))
        ax.set_yticklabels(
            corr_df.index,
            fontsize=8,
        )

        for row_index in range(n_variables):
            for column_index in range(row_index):
                value = corr_values[row_index, column_index]

                if not np.isfinite(value):
                    continue

                text_color = (
                    "white"
                    if abs(value) >= 0.60
                    else "black"
                )

                ax.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=text_color,
                )

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.tick_params(
            axis="both",
            which="both",
            length=0,
        )
        ax.grid(False)
        ax.set_xlim(-0.5, n_variables - 0.5)
        ax.set_ylim(n_variables - 0.5, -0.5)

        colorbar = fig.colorbar(
            image,
            ax=ax,
            shrink=0.85,
            pad=0.02,
        )
        colorbar.set_label(
            "Spearman correlation",
            fontsize=11,
        )
        colorbar.outline.set_visible(False)

        ax.set_title(
            "feature to feature Spearman Correlation Matrix "
            "(Features + SOH)\n"
            f"{title_suffix}",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )

        plt.tight_layout()

        if save:
            safe_suffix = re.sub(
                r"[^A-Za-z0-9_.-]+",
                "_",
                str(title_suffix),
            ).strip("_")

            self._save_fig(
                fig,
                f"features_SOH_Matrix"
                f"{safe_suffix}.png",
            )

        if show:
            plt.show()
        else:
            plt.close(fig)



def find_zip_files(folder_path, recursive=True):
    """
    Find every ZIP file inside a folder, optionally including subfolders.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {folder}")

    pattern = "**/*.zip" if recursive else "*.zip"
    return sorted(path for path in folder.glob(pattern) if path.is_file())


def process_all_zip_files(
    input_folder,
    output_folder,
    cycle_indices=None,
    max_cells_per_zip=None,
    n_ref_cycles=10,
    recursive=True,
):
    """
    Process all ZIP files one by one and combine their extracted features.
    """
    zip_files = find_zip_files(input_folder, recursive=recursive)
    if not zip_files:
        raise FileNotFoundError(f"No ZIP files found in: {input_folder}")

    print(f"Found {len(zip_files)} ZIP files:")
    for path in zip_files:
        print(f"  - {path}")

    os.makedirs(output_folder, exist_ok=True)
    all_frames = []
    global_cell_offset = 0

    for zip_number, zip_path in enumerate(zip_files, start=1):
        print("\n" + "=" * 80)
        print(f"Processing ZIP {zip_number}/{len(zip_files)}: {zip_path.name}")
        print("=" * 80)

        dataset_output_folder = os.path.join(
            output_folder,
            zip_path.stem,
        )
        os.makedirs(dataset_output_folder, exist_ok=True)

        proc = Preprocessing_NMC(
            zip_path=str(zip_path),
            cycle_indices=cycle_indices,
            save_dir=dataset_output_folder,
            max_cells=max_cells_per_zip,
        )

        try:
            proc.read_data()
            proc.process_cell_data()
            df = proc.get_feature_dataframe()

            if df.empty:
                print(f"No valid rows extracted from {zip_path.name}; skipping.")
                continue

            df.insert(0, "Dataset", proc.zip_name)
            df = proc.add_soh_target(df, n_ref_cycles=n_ref_cycles)

            # Preserve local index and create a unique index across all ZIP files.
            df["Local_Cell_Index"] = df["Cell_Index"]
            local_cells = sorted(df["Cell_Index"].dropna().unique())
            global_map = {
                local_cell: global_cell_offset + position
                for position, local_cell in enumerate(local_cells)
            }
            df["Global_Cell_Index"] = df["Cell_Index"].map(global_map)
            global_cell_offset += len(local_cells)

            ordered = [
                "Dataset", "Global_Cell_Index", "Local_Cell_Index",
                "Cell_Name", "Cycle"
            ] + proc.feature_names + ["Q_ref", "SOH"]
            df = df[[c for c in ordered if c in df.columns]]

            
            # Individual-dataset lower-triangular Spearman matrix.
            dataset_feature_corr = proc.compute_feature_correlation_matrix(
                df,
                target="SOH",
            )
            if not dataset_feature_corr.empty:
                
                proc.plot_feature_correlation_matrix(
                    dataset_feature_corr,
                    title_suffix=proc.zip_name,
                    save=True,
                    show=False,
                )
            else:
                print(
                    f"No valid feature-to-feature Spearman matrix for "
                    f"{proc.zip_name}."
                )

            all_frames.append(df)

        except Exception as exc:
            print(f"Failed to process {zip_path.name}: {exc}")

    if not all_frames:
        raise RuntimeError("No valid feature data were extracted from any ZIP file.")

    combined_df = pd.concat(all_frames, ignore_index=True, sort=False)
 
    return combined_df


def main():
    # Folder that contains all ZIP files.
    input_folder = (
        r"D:\TU\Master_Thesis\Raw_Data\Battery_Data\Raw_BML_NMC"
    )

    # Folder used for the  final figures.
    output_folder = (
        r"D:\TU\Master_Thesis\Raw_Data\Battery_Data"
        r"\Plots\NMC_2"
    )

    cycle_indices = None          # None = use all cycles
    max_cells_per_zip = None      # None = use all cells in each ZIP
    n_ref_cycles = 10
    recursive_zip_search = True   # Also search ZIP files in subfolders

    combined_df = process_all_zip_files(
        input_folder=input_folder,
        output_folder=output_folder,
        cycle_indices=cycle_indices,
        max_cells_per_zip=max_cells_per_zip,
        n_ref_cycles=n_ref_cycles,
        recursive=recursive_zip_search,
    )

    print("\nCombined folder-level dataset:")
    print(combined_df.head())
    print(f"ZIP datasets: {combined_df['Dataset'].nunique()}")
    print(f"Cells: {combined_df['Global_Cell_Index'].nunique()}")
    print(f"Valid cycles: {len(combined_df)}")

    # This object is used for its correlation and plotting methods.
    plot_proc = Preprocessing_NMC(
        zip_path="whole_folder.zip",
        save_dir=output_folder,
    )
    feature_corr = plot_proc.compute_feature_correlation_matrix(
        combined_df,
        target="SOH",
    )

if __name__ == "__main__":
    main()
