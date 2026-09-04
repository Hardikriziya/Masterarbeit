"""
Preprocessing of the NMC

"""
import os
import zipfile
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


class Preprocessing_NMC:

    feature_names = [
        # --- capacity features ---
        "Qd", "Qc",

        # --- time features ---
        "c_t", "dc_t", "t80_soc",

        # --- dQ/dV features ---
        "dqdv_min", "dqdv_avg",
        "dqdv_slope_max", "dqdv_slope_min",
        "dqdv_peak_width", "dqdv_peak_voltage",

        # --- dV/dQ features ---
        "dvdq_min", "dvdq_avg",
        "dvdq_slope_max", "dvdq_slope_min",

        # --- shape feature ---
        "discharge_V_kurtosis",

        # --- noise descriptors ---
        "log_std_Qd", "log_std_Qc", "log_std_Id", "log_std_Ic",

        # --- curve-derived scalar descriptor ---
        "qdlin",
    ]

    def __init__(self, zip_path, cell_index=None, cycle_indices=None, save_dir="plots"):
        self.zip_path      = zip_path
        self.zip_name       = os.path.splitext(os.path.basename(zip_path))[0]
        self.nominal_capacity = None
        self.cell_index    = cell_index
        self.cycle_indices = cycle_indices
        self.save_dir      = save_dir

        self.all_cycle_data = []
        self.num_cycl_per_cell    = []
        self.capacity_per_cell    = []
        self.all_charge_dQ        = []
        self.all_discharge_dQ     = []

        self.cycle_cache    = []   
        self.feature_data   = []   
        self.feature_cycles = []  

    def _save_fig(self, fig, filename):
        """
        Save a figure into self.save_dir, creating the folder if needed.
        """
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {path}")

    def read_data(self, max_cells: int = None):
        """
        Load Only the selected cell_index from the ZIP file.

        """
        if self.cell_index is None:
            selected_index = 0
        else:
            selected_index = int(self.cell_index)

        with zipfile.ZipFile(self.zip_path, "r") as z:
            pkl_files = sorted([f for f in z.namelist() if f.endswith(".pkl")])
            print(f"Found {len(pkl_files)} PKL files.")

            if selected_index < 0 or selected_index >= len(pkl_files):
                raise IndexError(
                    f"cell_index {selected_index} is out of range. "
                    f"Available indices: 0 to {len(pkl_files) - 1}"
                )

            selected_file = pkl_files[selected_index]
            print(f"Loading only selected cell_index={selected_index}: {selected_file}")

            with z.open(selected_file) as f:
                raw_cell = pickle.load(f)

            cell_cycles = self.organize_cell(raw_cell)
            self.all_cycle_data = [cell_cycles]

        n = len(self.all_cycle_data)
        self.num_cycl_per_cell = [0]  * n
        self.capacity_per_cell = [[]] * n
        self.all_charge_dQ     = [[]] * n
        self.all_discharge_dQ  = [[]] * n

        self.cycle_cache    = [dict() for _ in range(n)]
        self.feature_data   = [{name: [] for name in self.feature_names} for _ in range(n)]
        self.feature_cycles = [[] for _ in range(n)]

        self.original_cell_index = selected_index

        print(f"Processed {n} selected cell internally as index 0.")

    def organize_cell(self, raw_cell: dict) -> list:
        cell_id = raw_cell.get("cell_id", "unknown")
        print(f"  Organizing cell: {cell_id}")

        if self.nominal_capacity is None:
            self.nominal_capacity = raw_cell.get("nominal_capacity_in_Ah")

        cycle_data = raw_cell.get("cycle_data", [])
        if isinstance(cycle_data, dict):
            cycles_iter = cycle_data.items()
        else:
            cycles_iter = enumerate(cycle_data)

        cell_cycles = []
        for cycle_index, cycle in cycles_iter:
            cell_cycles.append({
                "Cycle_No":            int(cycle_index),
                "Voltage":             np.asarray(cycle.get("voltage_in_V",             []), float),
                "Current":             np.asarray(cycle.get("current_in_A",             []), float),
                "Time":                np.asarray(cycle.get("time_in_s",                []), float),
                "Charge_Capacity":     np.asarray(cycle.get("charge_capacity_in_Ah",    []), float),
                "Discharge_Capacity":  np.asarray(cycle.get("discharge_capacity_in_Ah", []), float),
                "Cell_Name":           cell_id,
            })

        return cell_cycles

    def process_cell_data(self):
        """
        Process only the selected cell.

        """
        target = 0

        for idx, cell_data in enumerate(self.all_cycle_data):
            cell_name = cell_data[0]["Cell_Name"] if cell_data else f"cell_{idx}"
            self.num_cycl_per_cell[idx] = len(cell_data)

            is_target = True
            print(f"  Processing selected cell fully: {cell_name} ({len(cell_data)} cycles)…")

            capacity_list     = []
            charge_dQ_list    = []
            discharge_dQ_list = []

            for cycle in cell_data:
                V    = cycle["Voltage"]
                I    = cycle["Current"]
                Time = cycle["Time"]
                Qc   = cycle["Charge_Capacity"]
                Qd   = cycle["Discharge_Capacity"]

                if V.size == 0:
                    continue

                charge_time    = Time[I > 0] if Time.size else Time
                discharge_time = Time[I < 0] if Time.size else Time
                if not self._is_complete(charge_time, discharge_time, Qd, Qc):
                    continue

                Ic, Vc, Qcc, Tc = self._clean_charge_data(V, I, Qc, Time)
                Id, Vd, Qdd, Td = self._clean_discharge_data(V, I, Qd, Time)

                if Qcc.size > 0 and Qdd.size > 0:
                    cycle_capacity = (np.max(Qcc) + np.max(Qdd)) / 2
                    capacity_list.append(cycle_capacity)
                else:
                    continue

                if is_target:
                    chg_deriv = self.calculate_derivatives(Vc, Qcc)
                    dis_deriv = self.calculate_derivatives(Vd, Qdd)

                    if chg_deriv:
                        charge_dQ_list.append(chg_deriv)
                    if dis_deriv:
                        discharge_dQ_list.append(dis_deriv)

                    feats = self._compute_cycle_features(Ic, Vc, Tc, Qcc, Id, Vd, Td, Qdd, dis_deriv)

                    cyc_no = cycle["Cycle_No"]
                    self.cycle_cache[idx][cyc_no] = {
                        "Vc": Vc, "Ic": Ic, "Tc": Tc, "Qcc": Qcc,
                        "Vd": Vd, "Id": Id, "Td": Td, "Qdd": Qdd,
                        "chg_deriv": chg_deriv, "dis_deriv": dis_deriv,
                    }

                    for name in self.feature_names:
                        self.feature_data[idx][name].append(feats.get(name, np.nan))
                    self.feature_cycles[idx].append(cyc_no)

            self.capacity_per_cell[idx] = capacity_list
            if is_target:
                self.all_charge_dQ[idx]    = charge_dQ_list
                self.all_discharge_dQ[idx] = discharge_dQ_list

        print("Processing complete.")

    @staticmethod
    def _is_complete(charge_time, discharge_time, dq_raw, cq_raw) -> bool:
        if charge_time.size == 0 or discharge_time.size == 0:
            return False

        qd_max = float(np.nanmax(dq_raw)) if dq_raw.size else 0.0
        qc_max = float(np.nanmax(cq_raw)) if cq_raw.size else 0.0

        return not (qd_max <= 0.0 and qc_max <= 0.0)
    
    @staticmethod
    def _clean_charge_data(V, I, Q, T=None):
        mask = (I > 0) & (V >= 2.70) & (V <= 4.00)
        if T is not None:
            return I[mask], V[mask], Q[mask], T[mask]
        return I[mask], V[mask], Q[mask]

    @staticmethod
    def _clean_discharge_data(V, I, Q, T=None):
        mask = (I < 0) & (V >= 2.70) & (V <= 4.00)
        if T is not None:
            return I[mask], V[mask], Q[mask], T[mask]
        return I[mask], V[mask], Q[mask]

    @staticmethod
    def calculate_derivatives(V, Q):
        """
        Interpolated dQ/dV and dV/dQ (smoother).
        """
        if len(V) < 20:
            return None
        V, Q = np.asarray(V, float), np.asarray(Q, float)

        idx  = np.argsort(V)
        V_u, ui = np.unique(V[idx], return_index=True)
        Q_u     = Q[idx][ui]
        if len(V_u) < 20:
            return None
        V_lin  = np.linspace(V_u.min(), V_u.max(), 400)
        kind   = 'cubic' if len(V_u) >= 4 else 'linear'
        Q_lin  = interp1d(V_u, Q_u, kind=kind)(V_lin)
        dQdV   = savgol_filter(Q_lin, 31, 3, deriv=1,
                               delta=V_lin[1] - V_lin[0])

        idx2   = np.argsort(Q_u)
        Q_s2, V_s2 = Q_u[idx2], V_u[idx2]
        Q_u2, ui2  = np.unique(Q_s2, return_index=True)
        V_u2       = V_s2[ui2]
        if len(Q_u2) < 20:
            return None
        Q_lin2 = np.linspace(Q_u2.min(), Q_u2.max(), 400)
        kind2  = 'cubic' if len(Q_u2) >= 4 else 'linear'
        V_lin2 = interp1d(Q_u2, V_u2, kind=kind2)(Q_lin2)
        dVdQ   = savgol_filter(V_lin2, 31, 3, deriv=1,
                               delta=Q_lin2[1] - Q_lin2[0])

        return {"Voltage_V_dQdV": V_lin,  "dQdV": dQdV,
                "Capacity_Q_dVdQ": Q_lin2, "dVdQ": dVdQ}

    @staticmethod
    def compute_derivatives(V, Q):
        V, Q = np.asarray(V, float), np.asarray(Q, float)

        idx = np.argsort(V)
        V_u, ui = np.unique(V[idx], return_index=True)
        Q_u     = Q[idx][ui]
        if len(V_u) < 20:
            return None

        dQ, dV = np.diff(Q_u), np.diff(V_u)
        mask   = dV != 0
        dQdV   = savgol_filter((dQ / dV)[mask], 31, 3, mode='nearest')

        idx2   = np.argsort(Q_u)
        Q_s2, V_s2 = Q_u[idx2], V_u[idx2]
        Q_u2, ui2  = np.unique(Q_s2, return_index=True)
        V_u2       = V_s2[ui2]
        if len(Q_u2) < 20:
            return None

        dV2, dQ2 = np.diff(V_u2), np.diff(Q_u2)
        mask2    = dQ2 != 0
        dVdQ     = savgol_filter((dV2 / dQ2)[mask2], 31, 3, mode='nearest')

        return {"Voltage_V_dQdV": V_u[:-1][mask],   "dQdV": dQdV,
                "Capacity_Q_dVdQ": Q_u2[:-1][mask2], "dVdQ": dVdQ}

    @staticmethod
    def _safe_kurtosis(x):
        """
        Excess kurtosis of a 1D signal. Normal distribution gives ~0.
        """
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]

        if x.size < 4:
            return np.nan

        std = np.std(x)
        if std <= 0:
            return np.nan

        z = (x - np.mean(x)) / std
        return float(np.mean(z ** 4) - 3.0)

    @classmethod
    def _shape_features(cls, x, prefix):
        """
        Return kurtosis for a signal.
        """
        return {
            f"{prefix}_kurtosis": cls._safe_kurtosis(x), 
        }

    @staticmethod
    def _compute_cycle_features(Ic, Vc, Tc, Qcc, Id, Vd, Td, Qdd, dis_deriv):
        """
        Compute all scalar cycle features that can be obtained from the
        voltage, current, time, capacity, dQ/dV and dV/dQ signals.
        """
        feats = {}
        feats["Qd"]   = float(np.max(Qdd)) if Qdd.size else np.nan
        feats["Qc"]   = float(np.max(Qcc)) if Qcc.size else np.nan
        feats["c_t"]  = float(Tc[-1] - Tc[0]) if Tc.size > 1 else np.nan
        feats["dc_t"] = float(Td[-1] - Td[0]) if Td.size > 1 else np.nan

        # Time required during charge to reach 80% of the cycle charge capacity.
        feats["t80_soc"] = np.nan
        if Qcc.size > 1 and Tc.size == Qcc.size:
            q80 = 0.80 * float(np.nanmax(Qcc))
            hit = np.flatnonzero(Qcc >= q80)
            if hit.size:
                feats["t80_soc"] = float(Tc[hit[0]] - Tc[0])

        # Defaults for derivative-derived features.
        for name in (
            "dqdv_min", "dqdv_avg", "dqdv_max", "dqdv_std",
            "dqdv_slope_max", "dqdv_slope_min", "dqdv_area",
            "dqdv_peak_width", "dqdv_peak_voltage",
            "dvdq_max", "dvdq_min", "dvdq_avg", "dvdq_std",
            "dvdq_slope_max", "dvdq_slope_min",
        ):
            feats[name] = np.nan

        if dis_deriv is not None:
            V_lin = np.asarray(dis_deriv["Voltage_V_dQdV"], float)
            dQdV = np.asarray(dis_deriv["dQdV"], float)
            Q_lin = np.asarray(dis_deriv["Capacity_Q_dVdQ"], float)
            dVdQ = np.asarray(dis_deriv["dVdQ"], float)

            valid_dqdv = np.isfinite(V_lin) & np.isfinite(dQdV)
            V_f, dQdV_f = V_lin[valid_dqdv], dQdV[valid_dqdv]
            if dQdV_f.size:
                feats["dqdv_min"] = float(np.min(dQdV_f))
                feats["dqdv_avg"] = float(np.mean(dQdV_f))
                feats["dqdv_max"] = float(np.max(dQdV_f))
                feats["dqdv_std"] = float(np.std(dQdV_f))
                feats["dqdv_area"] = float(np.trapezoid(np.abs(dQdV_f), V_f))

                if dQdV_f.size > 1:
                    grad = np.gradient(dQdV_f, V_f)
                    feats["dqdv_slope_max"] = float(np.max(grad))
                    feats["dqdv_slope_min"] = float(np.min(grad))

                # Dominant dQ/dV peak and full width at half maximum (FWHM).
                amplitude = np.abs(dQdV_f)
                peak_idx = int(np.argmax(amplitude))
                feats["dqdv_peak_voltage"] = float(V_f[peak_idx])
                half_height = 0.5 * amplitude[peak_idx]
                above = amplitude >= half_height
                left = peak_idx
                right = peak_idx
                while left > 0 and above[left - 1]:
                    left -= 1
                while right < above.size - 1 and above[right + 1]:
                    right += 1
                feats["dqdv_peak_width"] = float(V_f[right] - V_f[left])

            valid_dvdq = np.isfinite(Q_lin) & np.isfinite(dVdQ)
            Q_f, dVdQ_f = Q_lin[valid_dvdq], dVdQ[valid_dvdq]
            if dVdQ_f.size:
                feats["dvdq_max"] = float(np.max(dVdQ_f))
                feats["dvdq_min"] = float(np.min(dVdQ_f))
                feats["dvdq_avg"] = float(np.mean(dVdQ_f))
                feats["dvdq_std"] = float(np.std(dVdQ_f))
                if dVdQ_f.size > 1:
                    grad = np.gradient(dVdQ_f, Q_f)
                    feats["dvdq_slope_max"] = float(np.max(grad))
                    feats["dvdq_slope_min"] = float(np.min(grad))

        def _log_std(x):
            if x.size > 1:
                std = np.std(x)
                return float(np.log(std)) if std > 0 else np.nan
            return np.nan

        feats["log_std_Qd"] = _log_std(Qdd)
        feats["log_std_Qc"] = _log_std(Qcc)
        feats["log_std_Id"] = _log_std(Id)
        feats["log_std_Ic"] = _log_std(Ic)

        feats["discharge_V_mean"] = float(np.mean(Vd)) if Vd.size else np.nan
        feats["discharge_V_std"] = float(np.std(Vd)) if Vd.size else np.nan
        feats["discharge_V_kurtosis"] = Preprocessing_NMC._safe_kurtosis(Vd)

        # Slope of the best straight-line fit Qd = slope * Vd + intercept.
        feats["qdlin"] = np.nan
        valid_line = np.isfinite(Vd) & np.isfinite(Qdd)
        if np.count_nonzero(valid_line) >= 2 and np.ptp(Vd[valid_line]) > 0:
            feats["qdlin"] = float(np.polyfit(Vd[valid_line], Qdd[valid_line], 1)[0])

        return feats

    def compute_discharge_dvdq_stats(self):
        cell_index = 0
        cache = self.cycle_cache[cell_index]
        cycle_indices = self.cycle_indices or []

        results = {
            "cycle": [],
            "discharge_dvdq_max": [],
            "discharge_dvdq_min": [],
            "discharge_dvdq_mean": [],
            "discharge_dvdq_std": [],
        }

        for i in cycle_indices:
            c = cache.get(i)
            if c is None or c["dis_deriv"] is None:
                continue

            dvdq = c["dis_deriv"]["dVdQ"]

            if dvdq.size == 0:
                continue

            results["cycle"].append(i)
            results["discharge_dvdq_max"].append(np.max(dvdq))
            results["discharge_dvdq_min"].append(np.min(dvdq))
            results["discharge_dvdq_mean"].append(np.mean(dvdq))
            results["discharge_dvdq_std"].append(np.std(dvdq))

        return results
    
    def plot_differential_curves(self, save=True):
        fig, axs = plt.subplots(2, 2, figsize=(18, 10))
        fig.patch.set_facecolor("white")
        for ax in axs.flat:
            ax.set_facecolor("white")

        cell_index    = 0
        cycle_indices = self.cycle_indices or []
        cache = self.cycle_cache[cell_index]

        cell_name = (self.all_cycle_data[cell_index][0]["Cell_Name"]
                     if self.all_cycle_data[cell_index] else "Unknown")

        print(f"Plotting differential curves for {cell_name}  "
              f"| cached cycles: {len(cache)}")

        for i in cycle_indices:
            c = cache.get(i)
            if c is None:
                continue
            d = c["dis_deriv"]
            if d:
                axs[0, 0].plot(d["Capacity_Q_dVdQ"], d["dVdQ"], label=f"Cycle {i}")
                axs[0, 1].plot(d["Voltage_V_dQdV"],  d["dQdV"], label=f"Cycle {i}")
            ch = c["chg_deriv"]
            if ch:
                axs[1, 0].plot(ch["Capacity_Q_dVdQ"], ch["dVdQ"], label=f"Cycle {i}")
                axs[1, 1].plot(ch["Voltage_V_dQdV"],  ch["dQdV"], label=f"Cycle {i}")

        axs[0, 0].set_title("Discharge dV/dQ for Cell {}".format(cell_name))
        axs[0, 0].set_xlabel("Capacity (Ah)")
        #axs[0,0].legend(title="Cycle",fontsize=8,title_fontsize=9,loc="best", ncol=3)
        axs[0, 0].set_ylabel("dV/dQ")
        axs[0, 0].grid()

        axs[0, 1].set_title("Discharge dQ/dV for Cell {}".format(cell_name))
        axs[0, 1].set_xlabel("Voltage (V)")
        #axs[0,1].legend(title="Cycle",fontsize=8,title_fontsize=9,loc="best", ncol=3)
        axs[0, 1].set_ylabel("dQ/dV")
        axs[0, 1].grid()

        axs[1, 0].set_title("Charge dV/dQ for Cell {}".format(cell_name))
        axs[1, 0].set_xlabel("Capacity (Ah)")
        axs[1,0].legend(title="Cycle",fontsize=8,title_fontsize=9,loc="best",  ncol=3 )
        axs[1, 0].set_ylabel("dV/dQ")
        axs[1, 0].grid()

        axs[1, 1].set_title("Charge dQ/dV for Cell {}".format(cell_name))
        axs[1, 1].set_xlabel("Voltage (V)")
        #axs[1,1].legend(title="Cycle",fontsize=8,title_fontsize=9,loc="best", ncol=3)
        axs[1, 1].set_ylabel("dQ/dV")
        axs[1, 1].grid()

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.25)

        if save:
            self._save_fig(fig, f"{cell_name}_differential_curves.png")

        plt.show()

    def plot_discharge_dvdq_stats(self, save=True):
        stats = self.compute_discharge_dvdq_stats()

        cyc_nos = np.array(stats["cycle"])

        if cyc_nos.size == 0:
            print("No valid discharge dV/dQ data found.")
            return

        cell_name = self.all_cycle_data[0][0]["Cell_Name"]

        cmap = mpl.colormaps["summer"]
        norm = plt.Normalize(vmin=cyc_nos.min(), vmax=cyc_nos.max())

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor("white")
        axs = axs.flatten()

        features = [
            ("discharge_dvdq_max", "Discharge dV/dQ Max"),
            ("discharge_dvdq_min", "Discharge dV/dQ Min"),
            ("discharge_dvdq_mean", "Discharge dV/dQ Mean"),
            ("discharge_dvdq_std", "Discharge dV/dQ Std"),
        ]

        for ax, (key, title) in zip(axs, features):
            y = np.array(stats[key])

            ax.scatter(
                cyc_nos, y,
                c=cyc_nos,
                cmap=cmap,
                norm=norm,
                marker="D",
                s=20,
                edgecolors="black",
                linewidths=0.8,
                zorder=3
            )

            ax.set_title(f"{title} — {cell_name}")
            ax.set_xlabel("Cycle Number")
            ax.set_ylabel(title)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.set_facecolor("white")

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                                pad=0.02, fraction=0.046)
            cbar.set_label("Cycle Number", fontsize=9)

        fig.suptitle(f"Discharge dV/dQ Statistics vs Cycle — {cell_name}", fontsize=14)
        plt.tight_layout()

        if save:
            self._save_fig(fig, f"{cell_name}_discharge_dvdq_stats.png")

        plt.show()
    
    @staticmethod
    def _scatter_subplot(ax, fig, x, y, norm, cmap, title, xlabel, ylabel):
        """
        Draw a single diamond scatter plot on `ax` with its own colourbar.
        """
        ax.scatter(x, y, c=x, cmap=cmap, norm=norm,
                   marker="D", s=35, edgecolors="black", linewidths=0.8, zorder=3)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_facecolor("white")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                            pad=0.02, fraction=0.046)
        cbar.set_label("Cycle Number", fontsize=9)

    def plot_voltage_mean_std(self, save=True):
        cell_index    = 0
        cycle_indices = self.cycle_indices or []
        cache = self.cycle_cache[cell_index]

        if cell_index >= len(self.all_cycle_data):
            print(f"Cell index {cell_index} out of range."); return

        cell_data = self.all_cycle_data[cell_index]
        cell_name = cell_data[0]["Cell_Name"] if cell_data else "Unknown"

        chg_cyc, chg_mean, chg_std = [], [], []
        dis_cyc, dis_mean, dis_std = [], [], []

        for i in cycle_indices:
            c = cache.get(i)
            if c is None:
                continue

            Vc, Vd = c["Vc"], c["Vd"]

            if Vc.size > 0:
                chg_cyc.append(i)
                chg_mean.append(np.mean(Vc))
                chg_std.append(np.std(Vc))
            if Vd.size > 0:
                dis_cyc.append(i)
                dis_mean.append(np.mean(Vd))
                dis_std.append(np.std(Vd))

        print(f"Plotting voltage mean/std for {cell_name}  "
              f"| charge cycles: {len(chg_cyc)}  "
              f"| discharge cycles: {len(dis_cyc)}")

        chg_cyc  = np.array(chg_cyc);   dis_cyc  = np.array(dis_cyc)
        chg_mean = np.array(chg_mean);  dis_mean = np.array(dis_mean)
        chg_std  = np.array(chg_std);   dis_std  = np.array(dis_std)

        cmap       = mpl.colormaps["summer"]
        norm_chg = plt.Normalize(chg_cyc.min(), chg_cyc.max()) if chg_cyc.size else plt.Normalize(0, 1)
        norm_dis = plt.Normalize(dis_cyc.min(), dis_cyc.max()) if dis_cyc.size else plt.Normalize(0, 1)

        fig, axs = plt.subplots(2, 2, figsize=(14, 9))
        fig.patch.set_facecolor("white")

        self._scatter_subplot(axs[0, 0], fig, dis_cyc, dis_mean, norm_dis, cmap,
                              f"Discharge Voltage Mean — {cell_name}",
                              "Cycle Number", "Voltage Mean (V)")
        
        self._scatter_subplot(axs[1, 0], fig, chg_cyc, chg_mean, norm_chg, cmap,
                              f"Charge Voltage Mean — {cell_name}",
                              "Cycle Number", "Voltage Mean (V)")
        
        self._scatter_subplot(axs[0, 1], fig, dis_cyc, dis_std, norm_dis, cmap,
                              f"Discharge Voltage Std — {cell_name}",
                              "Cycle Number", "Voltage Std (V)")
        
        self._scatter_subplot(axs[1, 1], fig, chg_cyc, chg_std, norm_chg, cmap,
                              f"Charge Voltage Std — {cell_name}",
                              "Cycle Number", "Voltage Std (V)")

        fig.suptitle(f"Voltage Mean & Std vs Cycle — {cell_name}", fontsize=14)
        plt.tight_layout()

        if save:
            self._save_fig(fig, f"{cell_name}_voltage_mean_std.png")

        plt.show()

    def plot_voltage_stats(self, save=True):
       
        self.plot_voltage_mean_std(save=save)

    def plot_feature_scatter(self, features=None, save=True):
        """
        Scatter-plot any subset of the cached per-cycle features vs cycle number.

        features : list[str] or None
            Any combination of Preprocessing_NMC feature_names., e.g.
                proc.plot_feature_scatter(features=["Qd", "Qc", "log_std_Qd"])
            Pass None (default) to plot all available features.
        """
        cell_index = 0

        if features is None:
            features = self.feature_names
        else:
            unknown = [f for f in features if f not in self.feature_names]
            if unknown:
                raise ValueError(
                    f"Unknown feature(s): {unknown}. Available: {self.feature_names}"
                )

        cell_data = self.all_cycle_data[cell_index]
        cell_name = cell_data[0]["Cell_Name"] if cell_data else "Unknown"
        all_cycl = np.array(self.feature_cycles[cell_index])
        
        if self.cycle_indices is not None:
            selected_cycles = np.array(self.cycle_indices)
            mask = np.isin(all_cycl, selected_cycles)
        else:
            mask = np.ones_like(all_cycl, dtype=bool)

        cyc_nos = all_cycl[mask]

        if cyc_nos.size == 0:
            print("No feature data available — did you run process_cell_data()?")
            return

        n = len(features)
        ncols = 2
        nrows = int(np.ceil(n / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows))
        fig.patch.set_facecolor("white")
        axs = np.atleast_1d(axs).flatten()

        cmap = mpl.colormaps["summer"]
        norm = plt.Normalize(vmin=cyc_nos.min(), vmax=cyc_nos.max())

        for ax, feat in zip(axs, features):
            y = np.array(self.feature_data[cell_index][feat])
            y = y[mask]
            ax.scatter(cyc_nos, y, c=cyc_nos, cmap=cmap, norm=norm,
                       marker="D", s=35, edgecolors="black", linewidths=0.8, zorder=3)
            ax.set_title(f"{feat} — {cell_name}")
            ax.set_xlabel("Cycle Number")
            ax.set_ylabel(feat)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.set_facecolor("white")

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                                pad=0.02, fraction=0.046)
            cbar.set_label("Cycle Number", fontsize=9)

        for ax in axs[len(features):]:
            ax.axis("off")

        fig.suptitle(f"Cycle Features — {cell_name}", fontsize=14)
        plt.tight_layout()

        if save:
            self._save_fig(fig, f"{cell_name}_features_{'_'.join(features)}.png")

        plt.show()

    def plot_capacity_trends(self, save=True):
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        max_cycles = max(self.num_cycl_per_cell)

        for i, cap_list in enumerate(self.capacity_per_cell):
            if len(cap_list) < 5:
                continue

            ref = max(cap_list[:10])
            soh = np.array([(c / ref) * 100 for c in cap_list])
            soh = savgol_filter(soh, window_length=21, polyorder=3, mode='nearest')

            color = plt.cm.viridis(self.num_cycl_per_cell[i] / max_cycles)
            ax.plot(range(len(soh)), soh, color=color, linewidth=1.4)

        sm = plt.cm.ScalarMappable(
            cmap='viridis',
            norm=plt.Normalize(vmin=0, vmax=max_cycles)
        )
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Number of Cycles')

        ax.set_xlabel("Cycle Number")
        ax.set_ylabel("Normalized Capacity (SoH)")
        ax.set_title(f"Capacity Trends (SoH) for ({self.zip_name}) ")
        ax.grid()

        if save:
            self._save_fig(fig, f"{self.zip_name}_capacity_trends.png")

        plt.show()

    def plot_number_of_cycles(self, save=True):
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        sorted_cycles = sorted(self.num_cycl_per_cell)
        cell_indices  = range(1, len(sorted_cycles) + 1)

        norm   = plt.Normalize(vmin=min(sorted_cycles), vmax=max(sorted_cycles))
        colors = plt.cm.viridis(norm(sorted_cycles))

        ax.bar(cell_indices, sorted_cycles, color=colors)

        ax.set_xlabel("Cell Index")
        ax.set_ylabel("Number of Cycles")
        ax.set_title(f"Number of Cycles per Cell for ({self.zip_name}) ")
        ax.grid(axis='y')

        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array(sorted_cycles)
        plt.colorbar(sm, ax=ax, label='Number of Cycles')

        if save:
            self._save_fig(fig, f"{self.zip_name}_number_of_cycles.png")

        plt.show()

    def plot_Q_vs_V_and_I_vs_time(self, save=True):
        

        cell_index    = 0
        cycle_indices = self.cycle_indices or []
        cache = self.cycle_cache[cell_index]

        if cell_index >= len(self.all_cycle_data):
            print(f"Cell index {cell_index} out of range."); return

        cell_data = self.all_cycle_data[cell_index]
        cell_name = cell_data[0]["Cell_Name"] if cell_data else "Unknown"
        fig, axs = plt.subplots(2, 2, figsize=(14, 8))

        fig.patch.set_facecolor("white")
        for ax in axs.flat:
            ax.set_facecolor("white")

        for i in cycle_indices:
            c = cache.get(i)
            if c is None:
                continue

            Vc, Ic, Tc, Qcc = c["Vc"], c["Ic"], c["Tc"], c["Qcc"]
            Vd, Id, Td, Qdd = c["Vd"], c["Id"], c["Td"], c["Qdd"]

            # real cached time, offset to start at 0 (no more re-derivation)
            t_c = (Tc - Tc[0]) if Tc.size > 0 else np.array([])
            t_d = (Td - Td[0]) if Td.size > 0 else np.array([])

            label = f"Cycle {i}"
            if Qdd.size > 0: axs[0, 0].plot(Qdd, Vd, label=label)
            if t_d.size > 0: axs[0, 1].plot(t_d, Id, label=label)
            if Qcc.size > 0: axs[1, 0].plot(Qcc, Vc, label=label)
            if t_c.size > 0: axs[1, 1].plot(t_c, Ic, label=label)
            

        axs[0,0].set_title(f"Discharge Capacity vs Voltage ({cell_name})")
        axs[0,0].set_xlabel("Discharge Capacity (Ah)")
        #axs[0,0].legend(title="Cycle",fontsize=8,title_fontsize=9,loc="best", ncol=3)
        axs[0,0].set_ylabel("Voltage (V)")
        axs[0,0].grid()

        axs[0,1].set_title(f"Discharging Current vs Time ({cell_name})")
        axs[0,1].set_xlabel("Time (s)")
        #axs[0,1].legend(title="Cycle",fontsize=8,title_fontsize=9,loc="best", ncol=3)
        axs[0,1].set_ylabel("Current (A)")
        axs[0,1].grid()

        axs[1,0].set_title(f"Charge Capacity vs Voltage ({cell_name})")
        axs[1,0].set_xlabel("Charge Capacity (Ah)")
        #axs[1,0].legend(title="Cycle",fontsize=8,title_fontsize=9,loc="best", ncol=3)
        axs[1,0].set_ylabel("Voltage (V)")
        axs[1,0].grid()

        axs[1,1].set_title(f"Charging Current vs Time ({cell_name})")
        axs[1,1].set_xlabel("Time (s)")
        #axs[1,1].legend(title="Cycle",fontsize=8,title_fontsize=9,loc="best", ncol=3)
        axs[1,1].set_ylabel("Current (A)")
        axs[1,1].grid()

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save:
            self._save_fig(fig, f"{cell_name}_Q_vs_V_and_I_vs_time.png")

        plt.show()


def main():
    zip_path = r"D:\TU\Master_Thesis\Raw_Data\Battery_Data\Raw_BML_NMC\SDU.zip"

    proc = Preprocessing_NMC(
        zip_path     = zip_path,
        cell_index   = 15,
        cycle_indices= [0,10,20,30,40,50,60,70,80,90,100,
                        150,200,250,300,350,400,450,500,
                        550,600,650,700,750,800,850,
                        900,950,1000,1050,1100,1150,1200,1250,1300],
        save_dir     = r"D:\TU\Master_Thesis\Raw_Data\Battery_Data\Plots\SDU\features_plot"
    )
    proc.read_data()
    proc.process_cell_data()          

    #proc.plot_capacity_trends()
    #proc.plot_number_of_cycles()
    #proc.plot_Q_vs_V_and_I_vs_time()
    proc.plot_differential_curves()
    #proc.plot_voltage_mean_std(),
    proc.plot_feature_scatter(features=["Qd", "Qc", "log_std_Qd", "log_std_Qc"])
    proc.plot_feature_scatter(features=["dqdv_slope_max", "dqdv_slope_min", "dqdv_min", "dqdv_avg"])
    proc.plot_feature_scatter(features=["discharge_V_kurtosis"])
    proc.plot_discharge_dvdq_stats()
    proc.plot_feature_scatter(features=["c_t", "dc_t", "t80_soc", "dqdv_peak_width"])
    proc.plot_feature_scatter(features=["dqdv_peak_voltage", "dvdq_slope_max", "dvdq_slope_min", "qdlin"])
    
    # or plot everything:
    #proc.plot_feature_scatter()

if __name__ == "__main__":
    main()