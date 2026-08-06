import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
from scipy.interpolate import interp1d
from collections import defaultdict
from matplotlib import cm, colors
import colorsys
import os

# =========================  User Configuration Section  =========================
# (Cleaned all non-standard spaces)
config = {
    "files": [
        "data/biNBs/purewater/004/merged_plot_size_and_distance/merged_2.csv",
        "data/biNBs/NaOH/002/merged_plot_size_and_distance/merged_2.csv",
        "data/biNBs/NaOH/004/merged_plot_size_and_distance/merged_2.csv",
        "data/biNBs/NaCl/002/merged_plot_size_and_distance/merged_2.csv",
        "data/biNBs/NaCl/003/merged_plot_size_and_distance/merged_2.csv",
        "data/biNBs/HCl/002/merged_plot_size_and_distance/merged_2.csv",
        "data/biNBs/HCl/003/merged_plot_size_and_distance/merged_2.csv",
        "data/biNBs/HCl/004/merged_plot_size_and_distance/merged_2.csv",
    ],
    "start_time_ns": [10.000, 14.000, 40.000, 8.000, 10.000, 28.000, 18.000, 32.000],
    "end_time_ns":   [64.000, 62.000, 88.000, 64.000, 66.000, 64.000, 54.000, 68.000],
    "size_threshold": 20,      # Small < 20; Large >= 20
    "fig_fmt": ["pdf", "png"],
    "dpi": 300,
}
# ==============================================================================

# --- NEW: Nature Plotting Style Settings ---
def set_nature_style():
    """Apply 'Nature' like plotting style."""
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 16, 
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12,
        'lines.linewidth': 2.0,
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
    })
# --------------------------------

def extract_system_label(path):
    for p in path.split(os.sep):
        if p in {"purewater", "NaOH", "NaCl", "HCl"}:
            return p
    return "Unknown"

def load_data(fpath, start_t, end_t):
    """
    MODIFIED:
    This is now the sole data loading function.
    If end_t is None (though not in this script), end time is ignored.
    """
    df = pd.read_csv(fpath)
    df.columns = [c.strip() for c in df.columns]
    df["Time(ns)"] = pd.to_numeric(df["Time(ns)"], errors="coerce")
    mask = df["Time(ns)"] >= start_t
    if end_t is not None:
        mask &= df["Time(ns)"] <= end_t
    return df[mask].copy().reset_index(drop=True)

def pick_small_bubble(df, threshold):
    last = df.iloc[-1]
    b1, b2 = last["Bubble1"], last["Bubble2"]
    if b1 < threshold:
        return df["Bubble1"].values
    elif b2 < threshold:
        return df["Bubble2"].values
    else:
        return df["Bubble1"].values

def pick_large_bubble(df, threshold):
    """Select large bubble: The sequence with size >= threshold at the last frame."""
    last = df.iloc[-1]
    b1, b2 = last["Bubble1"], last["Bubble2"]
    if b1 >= threshold:
        return df["Bubble1"].values
    elif b2 >= threshold:
        return df["Bubble2"].values
    else:
        # Fallback to Bubble1 if both are small
        return df["Bubble1"].values

def normalize_time_size(t, size):
    t_new = t - t[0]
    ratio = size / size[0]
    return t_new, size, ratio

# ========================= System Mean Evolution =========================
def _group_mean_evolution(data_list, kind: str):
    """
    kind: 'real' or 'ratio'
    Returns {system: (t_common, y_mean, y_std)}
    """
    key = "size" if kind == "real" else kind

    groups = defaultdict(list)
    for d in data_list:
        groups[d["system"]].append((d["t_norm"], d[key]))

    result = {}
    for sys, curves in groups.items():
        # Filter out empty data
        valid_curves = [c for c in curves if len(c[0]) > 1 and len(c[1]) > 1]
        if not valid_curves:
            continue
            
        t_min = max(c[0].min() for c in valid_curves)
        t_max = min(c[0].max() for c in valid_curves)
        
        if t_min >= t_max:
            continue
            
        t_common = np.linspace(t_min, t_max, 200)
        y_mat = []
        for t, y in valid_curves:
            f = interp1d(t, y, kind="linear", bounds_error=False, fill_value=np.nan)
            y_mat.append(f(t_common))
            
        y_mat = np.array(y_mat)
        
        y_mean = np.nanmean(y_mat, axis=0)
        if np.all(np.isnan(y_mean)):
            continue
            
        y_std  = np.nanstd(y_mat, axis=0)
        result[sys] = (t_common, y_mean, y_std)
    return result
    
# ---------- Colors (MODIFIED) ----------
def make_value_gradient(systems):
    base_cmap = cm.get_cmap("tab10")
    gradients = {}

    # --- MODIFIED: Fixed mapping based on user specification ---
    # purewater: Red (tab10 index 3)
    # NaOH:      Green (tab10 index 2)
    # NaCl:      Orange (tab10 index 1)
    # HCl:       Blue (tab10 index 0)
    color_map_indices = {
        "HCl": 0,        # Blue
        "NaCl": 1,       # Orange
        "NaOH": 2,       # Green
        "purewater": 3,  # Red
    }

    assigned_indices = set(color_map_indices.values())
    next_available_idx = 4 

    for sys in systems:
        if sys in color_map_indices:
            idx = color_map_indices[sys]
        else:
            while next_available_idx in assigned_indices and next_available_idx < 10:
                next_available_idx += 1
            if next_available_idx < 10:
                idx = next_available_idx
                assigned_indices.add(idx)
                next_available_idx += 1
            else:
                idx = (len(assigned_indices) % 10) 

        rgb = colors.to_rgb(base_cmap(idx))
        h, s, v = colorsys.rgb_to_hsv(*rgb)
        gradients[sys] = [
            colors.to_hex(colorsys.hsv_to_rgb(h, s, v0))
            for v0 in np.linspace(0.9, 0.4, 10)
        ]
    return gradients

# ---------- Plotting Generic (MODIFIED) ----------
def plot_evolution_generic(all_data, kind, prefix):
    fig, ax = plt.subplots(figsize=(6, 4))
    systems = sorted({d["system"] for d in all_data})
    
    if not systems:
        print(f"Warning: No data to plot for {prefix}size_evolution_{kind}")
        plt.close(fig)
        return

    gradients = make_value_gradient(systems)
    sys_counter = {s: 0 for s in systems}

    for d in all_data:
        sys = d["system"]
        t, size, ratio = d["t_norm"], d["size"], d["ratio"]
        lab = f"{sys}_{d['idx']}"
        col = gradients[sys][sys_counter[sys] % 10]
        sys_counter[sys] += 1
        y = size if kind == "real" else ratio
        ax.plot(t, y, label=lab, color=col, lw=1.2)

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Bubble size (atoms)" if kind == "real" else "Relative size")
    # --- MODIFIED: Removed Title ---
    
    ax.legend(ncol=2, loc='best')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for fmt in config["fig_fmt"]:
        fig.savefig(f"{prefix}size_evolution_{kind}.{fmt}", dpi=config["dpi"])
    plt.close(fig)

def plot_distance_stats_generic(all_data, prefix):
    from collections import defaultdict
    groups = defaultdict(list)
    for d in all_data:
        groups[d["system"]].append(d)

    if not groups:
        print(f"Warning: No data to plot for {prefix}distance_stats")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    xlabs, means, stds, cols = [], [], [], []
    sorted_systems = sorted(groups.keys())
    gradients = make_value_gradient(sorted_systems)

    for sys in sorted_systems:
        grad = gradients[sys]
        for k, d in enumerate(groups[sys]):
            dist = d["distance"]
            if dist is None or len(dist) == 0:
                continue
            xlabs.append(f"{sys}_{d['idx']}")
            means.append(dist.mean())
            stds.append(dist.std())
            cols.append(grad[k % 10])
            
    if not xlabs:
        print(f"Warning: No valid distance data to plot for {prefix}distance_stats")
        plt.close(fig)
        return

    xpos = np.arange(len(xlabs))
    ax.bar(xpos, means, yerr=stds, capsize=3, color=cols, edgecolor="k", width=0.7)
    ax.set_xticks(xpos)
    ax.set_xticklabels(xlabs, rotation=45, ha="right")
    ax.set_ylabel("Average distance (Å)")
    # --- MODIFIED: Removed Title ---
    
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for fmt in config["fig_fmt"]:
        fig.savefig(f"{prefix}distance_stats.{fmt}", dpi=config["dpi"])
    plt.close(fig)

def plot_mean_evolution(data_list, kind, prefix):
    # --- NEW: Define vertical line positions ---
    vlines_spec = {
        "HCl": [10],
        "NaOH": [16],
        "purewater": [17],
        "NaCl": [31] 
    }
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    systems = sorted({d["system"] for d in data_list})
    if not systems:
        print(f"Warning: No data to plot for {prefix}size_evolution_mean_{kind}")
        plt.close(fig)
        return
        
    gradients = make_value_gradient(systems)
    mean_dict = _group_mean_evolution(data_list, kind)
    
    if not mean_dict:
        print(f"Warning: No valid mean data calculated for {prefix}size_evolution_mean_{kind}")
        plt.close(fig)
        return

    for sys, (t, ym, ys) in mean_dict.items():
        color = colors.to_rgb(gradients[sys][0])
        
        mask = ~np.isnan(ym)
        t_valid = t[mask]
        ym_valid = ym[mask]
        ys_valid = ys[mask]
        
        if len(t_valid) < 2:
            continue
            
        ax.plot(t_valid, ym, label=sys, color=color) # lw=2 set by style
        ax.fill_between(t_valid, ym_valid - ys_valid, ym_valid + ys_valid, color=color, alpha=0.2)
        
        # --- NEW: Plot vertical lines ---
        if sys in vlines_spec:
            for x_val in vlines_spec[sys]:
                ax.axvline(x=x_val, color=color, linestyle='--', linewidth=2.5) # 2.5 Bold

    ax.set_xlabel("Time (ns)")
    y_label = "Average bubble size (atoms)" if kind == "real" else "Average relative size"
    ax.set_ylabel(y_label)
    
    # --- MODIFIED: Removed Title ---
    
    ax.legend() 
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for fmt in config["fig_fmt"]:
        fig.savefig(f"{prefix}size_evolution_mean_{kind}.{fmt}", dpi=config["dpi"])
    plt.close(fig)
    
# ---------- Main Process (MODIFIED) ----------
def main():
    # --- NEW: Apply Nature Style ---
    set_nature_style()

    # 1. Small Bubbles (Logic preserved)
    small_data = []
    for idx, (fpath, st, ed) in enumerate(zip(config["files"],
                                              config["start_time_ns"],
                                              config["end_time_ns"])):
        # Use load_data (st, ed)
        df = load_data(fpath, st, ed) 
        if df.empty:
            print(f"Warning: {fpath} has no data in small bubble interval, skipping.")
            continue
        size_seq = pick_small_bubble(df, config["size_threshold"])
        t = df["Time(ns)"].values
        distance = df["Distance"].values
        t_norm, size_real, size_ratio = normalize_time_size(t, size_seq)
        system = extract_system_label(fpath)
        small_data.append({
            "system": system, "idx": idx,
            "t_norm": t_norm, "size": size_real, "ratio": size_ratio,
            "distance": distance,
        })

    # 2. Large Bubbles (MODIFIED: Also uses end_time_ns)
    large_data = []
    # --- MODIFIED: Added config["end_time_ns"] ---
    for idx, (fpath, st, ed) in enumerate(zip(config["files"],
                                              config["start_time_ns"],
                                              config["end_time_ns"])):
        # --- MODIFIED: Changed to call load_data(fpath, st, ed) ---
        df = load_data(fpath, st, ed) 
        if df.empty:
            print(f"Warning: {fpath} has no data in large bubble interval, skipping.")
            continue
        size_seq = pick_large_bubble(df, config["size_threshold"])
        t = df["Time(ns)"].values
        distance = df["Distance"].values
        t_norm, size_real, size_ratio = normalize_time_size(t, size_seq)
        system = extract_system_label(fpath)
        large_data.append({
            "system": system, "idx": idx,
            "t_norm": t_norm, "size": size_real, "ratio": size_ratio,
            "distance": distance,
        })

    # 3. Plotting (With empty check)
    if small_data:
        plot_evolution_generic(small_data, "real", "")
        plot_evolution_generic(small_data, "ratio", "")
        plot_distance_stats_generic(small_data, "")
    else:
        print("No small bubble data found. Skipping small bubble plots.")

    if large_data:
        plot_evolution_generic(large_data, "real", "large_")
        plot_evolution_generic(large_data, "ratio", "large_")
        plot_distance_stats_generic(large_data, "large_")
    else:
        print("No large bubble data found. Skipping large bubble plots.")
    
    # ===== NEW: Output System Mean Evolution =====
    if small_data:
        plot_mean_evolution(small_data, "real", "")
        plot_mean_evolution(small_data, "ratio", "")
    
    if large_data:
        plot_mean_evolution(large_data, "real", "large_")
        plot_mean_evolution(large_data, "ratio", "large_")  
    
    print("All figures saved (small + large) with Nature style and corrected time logic.")

if __name__ == "__main__":
    main()