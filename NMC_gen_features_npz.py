"""
Processing of NMC dataset feature extraction and saved into .npz.

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
        "dqdv_min", "dqdv_avg",
        "dqdv_slope_max", "dqdv_slope_min",
        "dqdv_peak_width", "dqdv_peak_voltage",

        # --- dV/dQ features ---
        "dvdq_min", "dvdq_avg",
        "dvdq_slope_max", "dvdq_slope_min",

        # --- temperature and internal-resistance features(if available, otherwise dummy values) ---
        "tmax", "tavg", "IR",

        # --- shape feature ---
        "discharge_V_kurtosis",

        # --- noise descriptors ---
        "log_std_Qd", "log_std_Qc", "log_std_Id", "log_std_Ic",

        # --- curve-derived scalar descriptor ---
        "qdlin_mean",
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
        "dqdv_peak_width": "Main dQ/dV peak width (V)",
        "dqdv_peak_voltage": "Voltage of the main dQ/dV peak (V)",
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
        "qdlin_mean": "Mean interpolated discharge capacity over voltage bins",
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

    @staticmethod
    def _finite(x):
        x = np.asarray(x, dtype=float)
        return x[np.isfinite(x)]

    @staticmethod
    def _trapezoid(y, x):
        """
        Compatibility helper for NumPy 1.x and 2.x.
        """
        if hasattr(np, "trapezoid"):
            return np.trapezoid(y, x)
        return np.trapz(y, x) 
    
    @staticmethod
    def _average_duplicate_x(x, y):
        """
        Sort x and average all y-values that have the same x-value.

        """
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)

        # Ensure x and y have the same number of samples.
        n = min(x.size, y.size)
        x = x[:n]
        y = y[:n]

        # Remove NaN and infinite samples.
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]

        if x.size == 0:
            return (
                np.array([], dtype=float),
                np.array([], dtype=float),
            )

        # Sort samples by x.
        order = np.argsort(x)
        x = x[order]
        y = y[order]

        # Find unique x-values and identify the group of every sample.
        unique_x, inverse = np.unique(
            x,
            return_inverse=True,
        )

        # Sum y-values and count samples in every duplicate-x group.
        y_sum = np.zeros(unique_x.size, dtype=np.float64)
        counts = np.zeros(unique_x.size, dtype=np.int64)

        np.add.at(y_sum, inverse, y)
        np.add.at(counts, inverse, 1)

        averaged_y = y_sum / counts

        return unique_x, averaged_y
    
    def read_data(self):
        """
        Load all cells from the ZIP file.
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

                if not self._is_complete(charge_time, discharge_time, Qd, Qc):
                    continue

                Ic, Vc, Qcc, Tc, Tempc, IRc = self._clean_charge_data( V, I, Qc, Time, Temp, IR)
                Id, Vd, Qdd, Td, Tempd, IRd = self._clean_discharge_data(V, I, Qd, Time, Temp, IR)

                if Qcc.size == 0 or Qdd.size == 0:
                    continue

                dis_dQdV = self.calculate_derivatives_dQdV(Vd, Qdd)
                dis_dVdQ = self.calculate_derivatives_dVdQ(Vd, Qdd)
                qdlin_curve = self.interpolate_capacity_over_voltage(Vd, Qdd)

                feats = self._compute_cycle_features(
                    Ic, Vc, Tc, Qcc, Tempc, IRc,
                    Id, Vd, Td, Qdd, Tempd, IRd,
                    dis_dQdV, dis_dVdQ,
                    qdlin_curve)

                row = {"Cell_Index": original_cell_index,
                       "Cell_Name": cell_name,
                       "Cycle": cyc_no,
                    }
                row.update(feats)
                self.rows.append(row)

                cell_record["cycle_index"].append(cyc_no)

                for feature_name in self.feature_names:
                    cell_record["features"][feature_name].append(feats.get(feature_name, np.nan))

                cell_record["qdlin"].append(self._fixed_curve(qdlin_curve, length=1000))

                if (dis_dQdV is not None and "dQdV" in dis_dQdV):
                    dqdv_curve = dis_dQdV["dQdV"]
                else:
                    dqdv_curve = np.full(1000, np.nan, dtype=np.float32,)

                cell_record["dqdv"].append(self._fixed_curve(dqdv_curve, length=1000))
                valid_count += 1

            if valid_count:
                self.cell_npz_records.append(cell_record)

            print(
                f"Processed cell {original_cell_index} "
                f"({cell_name}) | valid cycles={valid_count}"
            )

        print("Processing complete. "
            f"Total rows/cycles extracted: {len(self.rows)}"
        )

    @staticmethod
    def _is_complete(charge_time, discharge_time, dq_raw, cq_raw):
        if charge_time.size == 0 or discharge_time.size == 0:
            return False

        dq = dq_raw[np.isfinite(dq_raw)]
        cq = cq_raw[np.isfinite(cq_raw)]

        qd_max = np.max(dq) if dq.size else 0.0
        qc_max = np.max(cq) if cq.size else 0.0

        return not (qd_max <= 0.0 or qc_max <= 0.0)

    @staticmethod
    def _process_optional_values(x, mask):
        x = np.asarray(x, float)
        if x.size == mask.size:
            return x[mask]
        if x.size == 1:
            return x.copy()
        return np.array([0,0], dtype=float)

    @classmethod
    def _clean_charge_data(cls, V, I, Q, Time, Temp=None, IR=None):
        mask = (I > 0) & (V >= 2.70) & (V <= 4.00)
        return (
            I[mask], V[mask], Q[mask], Time[mask],
            cls._process_optional_values(Temp if Temp is not None else [], mask),
            cls._process_optional_values(IR if IR is not None else [], mask),
        )

    @classmethod
    def _clean_discharge_data(cls, V, I, Q, Time, Temp=None, IR=None):
        mask = (I < 0) & (V >= 2.70) & (V <= 4.00)
        return (
            I[mask], V[mask], Q[mask], Time[mask],
            cls._process_optional_values(Temp if Temp is not None else [], mask),
            cls._process_optional_values(IR if IR is not None else [], mask),
        )

    @staticmethod
    def calculate_derivatives_dQdV(V, Q):
        """
        Calculate discharge dQ/dV curve.
        """
        V = np.asarray(V, float)
        Q = np.asarray(Q, float)
        mask = np.isfinite(V) & np.isfinite(Q)
        V, Q = V[mask], Q[mask]

        V_u, Q_u = Preprocessing_NMC._average_duplicate_x(V, Q)

        if V_u.size < 20 or np.nanmax(V_u) == np.nanmin(V_u):
            return None

        V_lin = np.linspace(V_u.min(), V_u.max(), 1000)
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
        
        Q_u, V_u = Preprocessing_NMC._average_duplicate_x(Q, V)

        if Q_u.size < 20 or np.nanmax(Q_u) == np.nanmin(Q_u):
            return None

        Q_lin = np.linspace(Q_u.min(), Q_u.max(), 1000)
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
    def interpolate_capacity_over_voltage(V, Q, n_bins=1000):
        V = np.asarray(V, dtype=float)
        Q = np.asarray(Q, dtype=float)
        mask = np.isfinite(V) & np.isfinite(Q)
        V, Q = V[mask], Q[mask]

        V_u, Q_u = Preprocessing_NMC._average_duplicate_x(V, Q)

        if V_u.size < 20 or np.nanmax(V_u) == np.nanmin(V_u):
            return np.array([], dtype=float)

        V_lin = np.linspace(V_u.min(), V_u.max(), n_bins)
        kind = "cubic" if V_u.size >= 4 else "linear"
        return interp1d(V_u, Q_u, kind=kind, bounds_error=False, fill_value="extrapolate")(V_lin)

    @staticmethod
    def _time_to_fraction_of_capacity(Td, Qdd, fraction=0.80):
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
            return np.nan, np.nan

        y_abs = np.abs(y_curve)
        peaks, _ = find_peaks(y_abs)
        if peaks.size == 0:
            return np.nan, np.nan

        main_peak = peaks[np.argmax(y_abs[peaks])]
        widths, _, _, _ = peak_widths(y_abs, [main_peak], rel_height=0.5)
        dx = np.mean(np.diff(x_axis)) if x_axis.size > 1 else 1.0

        dqdv_peak_width = float(widths[0] * abs(dx))
        dqdv_peak_voltage = float(x_axis[main_peak])
        
        return dqdv_peak_width, dqdv_peak_voltage

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
        return {f"{prefix}_kurtosis": cls._safe_kurtosis(x)}
    
    @classmethod
    def _compute_cycle_features(
        cls,
        Ic, Vc, Tc, Qcc, Tempc, IRc,
        Id, Vd, Td, Qdd, Tempd, IRd,
        dis_dQdV, dis_dVdQ,
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
        feats["t80_soc"] = cls._time_to_fraction_of_capacity(Td, Qdd, fraction=0.80)

        # ---- temperature and internal resistance ----
        temperature = [_finite(Tempc), _finite(Tempd)]
        temperature = [part for part in temperature if part.size]
        T_all = np.concatenate(temperature) if temperature else np.array([], dtype=float)

        resistance = [_finite(IRc), _finite(IRd)]
        resistance = [part for part in resistance if part.size]
        IR_all = np.concatenate(resistance) if resistance else np.array([], dtype=float)

        feats["tmax"] = _max_or_nan(T_all)
        feats["tavg"] = _mean_or_nan(T_all)
        feats["IR"] = _mean_or_nan(IR_all)

        # ---- discharge-voltage shape feature ----
        feats["discharge_V_kurtosis"] = cls._safe_kurtosis(Vd)

        # ---- dQ/dV features ----
        if dis_dQdV is not None and "dQdV" in dis_dQdV:
            V_lin = np.asarray(dis_dQdV.get("Voltage_V_dQdV", []), dtype=float)
            dQdV = np.asarray(dis_dQdV.get("dQdV", []), dtype=float)
            valid = np.isfinite(V_lin) & np.isfinite(dQdV)
            V_lin, dQdV = V_lin[valid], dQdV[valid]

            if dQdV.size:
                feats["dqdv_min"] = float(np.min(dQdV))
                feats["dqdv_avg"] = float(np.mean(dQdV))

            if dQdV.size > 1 and V_lin.size == dQdV.size:
                dqdv_gradient = np.gradient(dQdV, V_lin)
                feats["dqdv_slope_max"] = float(np.max(dqdv_gradient))
                feats["dqdv_slope_min"] = float(np.min(dqdv_gradient))

                peak_width, peak_voltage = cls._peak_features(V_lin, dQdV)
                feats["dqdv_peak_width"] = peak_width
                feats["dqdv_peak_voltage"] = peak_voltage

        # ---- dV/dQ features ----
        if dis_dVdQ is not None and "dVdQ" in dis_dVdQ:
            Q_lin = np.asarray(dis_dVdQ.get("Capacity_Ah_dVdQ", []), dtype=float)
            dVdQ = np.asarray(dis_dVdQ.get("dVdQ", []), dtype=float)
            valid = np.isfinite(Q_lin) & np.isfinite(dVdQ)
            Q_lin, dVdQ = Q_lin[valid], dVdQ[valid]

            if dVdQ.size:
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
        feats["qdlin_mean"] = float(np.mean(qdlin_f)) if qdlin_f.size else np.nan

        return feats

    @staticmethod
    def _fixed_curve(values, length=1000):
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

    @staticmethod
    def _find_eol(qd, cycle_index, eol_fraction=0.80, ref_cycles=9):
        """
        Determine observed end-of-life from discharge-capacity retention.

        """
        qd = np.asarray(qd, dtype=np.float32).reshape(-1)
        cycle_index = np.asarray(cycle_index, dtype=np.int32).reshape(-1)

        if qd.size != cycle_index.size:
            raise ValueError("qd and cycle_index must have the same length.")
        if qd.size == 0:
            return {
                "q_ref": np.nan,
                "eol_threshold": np.nan,
                "eol_index": -1,
                "cycle_life": 0,
                "n_keep": 0,
                "eol_reached": False,
                "retention_at_last": np.nan,
            }

        valid = np.isfinite(qd) & (qd > 0)
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size == 0:
            return {
                "q_ref": np.nan,
                "eol_threshold": np.nan,
                "eol_index": -1,
                "cycle_life": int(cycle_index[-1]),
                "n_keep": int(qd.size),
                "eol_reached": False,
                "retention_at_last": np.nan,
            }

        ref_idx = valid_idx[: min(int(ref_cycles), valid_idx.size)]
        q_ref = float(np.max(qd[ref_idx]))
        q_threshold = float(eol_fraction * q_ref)
        ref_end = int(ref_idx[-1])

        eol_index = -1
        for idx in valid_idx:
            if idx > ref_end and qd[idx] < q_threshold:
                eol_index = int(idx)
                break

        if eol_index >= 0:
            n_keep = eol_index + 1
            cycle_life = int(cycle_index[eol_index])
            retention = float(qd[eol_index] / q_ref)
            eol_reached = True
        else:
            n_keep = int(qd.size)
            cycle_life = int(cycle_index[-1])
            last_valid_idx = int(valid_idx[-1])
            retention = float(qd[last_valid_idx] / q_ref)
            eol_reached = False

        return {
            "q_ref": q_ref,
            "eol_threshold": q_threshold,
            "eol_index": eol_index,
            "cycle_life": cycle_life,
            "n_keep": n_keep,
            "eol_reached": eol_reached,
            "retention_at_last": retention,
        }

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
            output[:, column_index] = cls._interp_nan(output[:, column_index])

        return output.astype(np.float32)

    def save_individual_cell_npz(
        self,
        out_dir,
        eol_fraction=0.80,
        ref_cycles=9,
    ):
        """Save one compressed NPZ file per cell, truncated at observed EOL."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []

        for record in self.cell_npz_records:
            full_cycle_index = np.asarray(record["cycle_index"], dtype=np.int32)
            if full_cycle_index.size == 0:
                continue

            qd_raw = np.asarray(
                record["features"]["qd"], dtype=np.float32
            )[: full_cycle_index.size]
            qd_for_eol = self._interp_nan(qd_raw)
            eol = self._find_eol(
                qd_for_eol,
                full_cycle_index,
                eol_fraction=eol_fraction,
                ref_cycles=ref_cycles,
            )
            n_keep = int(eol["n_keep"])
            cycle_index = full_cycle_index[:n_keep]

            if eol["eol_reached"]:
                print(
                    f"  {record['cell_id']}: observed EOL at cycle "
                    f"{eol['cycle_life']} "
                    f"({100.0 * eol['retention_at_last']:.1f}% retention)."
                )
            else:
                print(
                    f"  {record['cell_id']}: EOL not observed; keeping all "
                    f"{n_keep} cycles (last retention "
                    f"{100.0 * eol['retention_at_last']:.1f}%)."
                )

            virtual_path = Path(record["virtual_path"])
            raw_cell_id = record.get("cell_id") or virtual_path.stem
            cell_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw_cell_id))
            rel_parent = virtual_path.parent
            cell_out_dir = out_dir / rel_parent
            cell_out_dir.mkdir(parents=True, exist_ok=True)
            out_path = cell_out_dir / f"{cell_id}.npz"

            feature_arrays = {}
            for feature_name in self.feature_names:
                values = np.asarray(
                    record["features"][feature_name], dtype=np.float32
                )[:n_keep]

                if feature_name in {"IR", "tmax", "tavg"} and not np.isfinite(values).any():
                    values = np.zeros(n_keep, dtype=np.float32)
                else:
                    values = self._interp_nan(values)

                feature_arrays[feature_name] = values

            qdlin_matrix = np.stack(record["qdlin"], axis=0).astype(np.float32)[:n_keep]
            dqdv_matrix = np.stack(record["dqdv"], axis=0).astype(np.float32)[:n_keep]
            qdlin_matrix = self._interp_nan_matrix(qdlin_matrix)
            dqdv_matrix = self._interp_nan_matrix(dqdv_matrix)

            payload = {
                "source_file": np.asarray(f"{self.zip_path}!/{virtual_path}", dtype=str),
                "dataset": np.asarray(self.zip_name, dtype=str),
                "cell_id": np.asarray(cell_id, dtype=str),
                "cell_index": np.asarray(record["cell_index"], dtype=np.int32),
                "cycle_life": np.asarray(eol["cycle_life"], dtype=np.int32),
                "cycle_index": cycle_index,
                "eol_reached": np.asarray(eol["eol_reached"], dtype=np.bool_),
                "eol_index": np.asarray(eol["eol_index"], dtype=np.int32),
                "eol_fraction": np.asarray(eol_fraction, dtype=np.float32),
                "eol_threshold": np.asarray(eol["eol_threshold"], dtype=np.float32),
                **feature_arrays,
                "qdlin": qdlin_matrix,
                "dqdv": dqdv_matrix,
            }

            np.savez_compressed(out_path, **payload)
            saved_paths.append(str(out_path))
            print(f"Saved cell NPZ to: {out_path}")

        print(
            f"Saved {len(saved_paths)} individual cell NPZ files "
            f"for dataset {self.zip_name}."
        )
        return saved_paths

    def get_feature_dataframe(self):
        if not self.rows:
            return pd.DataFrame()
        df = pd.DataFrame(self.rows)
        cols = ["Cell_Index", "Cell_Name", "Cycle"] + self.feature_names
        return df[[c for c in cols if c in df.columns]]

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
    ref_cycles=10,
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
            ] + proc.feature_names 
            df = df[[c for c in ordered if c in df.columns]]
            proc.save_individual_cell_npz(
                out_dir=dataset_output_folder,
                eol_fraction=0.80,
                ref_cycles=ref_cycles,
            )
            all_frames.append(df)

        except Exception as exc:
            print(f"Failed to process {zip_path.name}: {exc}")

    if not all_frames:
        raise RuntimeError("No valid feature data were extracted from any ZIP file.")

    combined_df = pd.concat(all_frames, ignore_index=True, sort=False)
    return combined_df

def main():
    
    input_folder = r"D:\TU\Master_Thesis\Raw_Data\Battery_Data\Raw_BML_NMC"
    output_folder = r"D:\TU\Master_Thesis\Raw_Data\Battery_Data\Plots\NMC_Selected_Features_1000"
    cycle_indices = None          # None = use all cycles
    max_cells_per_zip = None      # None = use all cells in each ZIP
    ref_cycles = 9
    recursive_zip_search = True   # Also search ZIP files in subfolders

    combined_df = process_all_zip_files(
        input_folder=input_folder,
        output_folder=output_folder,
        cycle_indices=cycle_indices,
        max_cells_per_zip=max_cells_per_zip,
        ref_cycles=ref_cycles,
        recursive=recursive_zip_search,
    )

    print("\nCombined folder-level dataset:")
    print(combined_df.head())
    print(f"ZIP datasets: {combined_df['Dataset'].nunique()}")
    print(f"Cells: {combined_df['Global_Cell_Index'].nunique()}")
    print(f"Valid cycles: {len(combined_df)}")

if __name__ == "__main__":
    main()
