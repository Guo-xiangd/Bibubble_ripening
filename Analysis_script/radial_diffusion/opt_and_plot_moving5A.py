#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate 1 ns moving-window MSD/MRD blocks, determine a common MSD fitting
window, and compute radial profiles of D_r and MRD.

Input block files are produced by compute_block_msd.py and contain:
    msd, mrd, count, timefrac, window_centers

D_r definition:
    D_r = 1/2 * d(MSD_r)/d(tau)

Unit conversion:
    slope unit A^2/ps
    1 A^2/ps = 10^-4 cm^2/s = 10 * 10^-5 cm^2/s
    therefore D_r in 10^-5 cm^2/s = slope / 2 * 10
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# ================= Must be consistent with compute_block_msd.py =================
GLOBAL_START_NS = 8
GLOBAL_END_NS = 38
BLOCK_SIZE_NS = 1
TIME_PER_FRAME_NS = 0.001
MAX_DELTA_T_NS = 0.1

R_PRIME_MIN = -14.0
R_PRIME_MAX = 6.0
WINDOW_THICKNESS = 5.0
SLIDE_STEP = 1.0
# ==============================================================================

MIN_START_PS = 5.0
MAX_END_PS = 100.0
MIN_WINDOW_LENGTH_PS = 30.0
MIN_FIT_POINTS = 5
MIN_D_FIT_R2 = 0.0
MRD_TAU_STRIDE_PS = 5.0

INPUT_DIR = "./temp_msd_data"
OUTPUT_DIR = "batch_diffusion_results"

OUTPUT_FILE_FINAL_STATS = os.path.join(OUTPUT_DIR, "Moving5A_Averaged_Diffusion_MRD_Stats.txt")
OUTPUT_FILE_FINAL_PLOT = os.path.join(OUTPUT_DIR, "Moving5A_Averaged_Diffusion_MRD_Profile.png")
OUTPUT_FILE_MRD_TAU_STATS = os.path.join(OUTPUT_DIR, "Moving5A_Averaged_MRD_By_Tau.txt")
OUTPUT_FILE_MRD_TAU_PLOT = os.path.join(OUTPUT_DIR, "Moving5A_Averaged_MRD_By_Tau.png")
OUTPUT_FILE_TIMEFRAC_STATS = os.path.join(OUTPUT_DIR, "Moving5A_Averaged_TimeFraction_By_Tau.txt")
OUTPUT_FILE_TIMEFRAC_PLOT = os.path.join(OUTPUT_DIR, "Moving5A_Averaged_TimeFraction_By_Tau.png")


def weighted_mean_std(values, weights, axis=0):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = ~np.isnan(values) & ~np.isnan(weights) & (weights > 0)
    safe_values = np.where(valid, values, 0.0)
    safe_weights = np.where(valid, weights, 0.0)

    weight_sum = np.sum(safe_weights, axis=axis)
    numerator = np.sum(safe_values * safe_weights, axis=axis)
    mean = np.divide(
        numerator,
        weight_sum,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=weight_sum > 0,
    )

    expanded_mean = np.expand_dims(mean, axis=axis)
    diff = np.where(valid, values - expanded_mean, 0.0)
    variance_num = np.sum(safe_weights * diff**2, axis=axis)
    variance = np.divide(
        variance_num,
        weight_sum,
        out=np.full_like(variance_num, np.nan, dtype=float),
        where=weight_sum > 0,
    )
    return mean, np.sqrt(variance), weight_sum


def weighted_mean_over_tau(values, weights, tau_mask):
    vals = values[:, :, tau_mask]
    wts = weights[:, :, tau_mask]
    valid = ~np.isnan(vals) & (wts > 0)
    numerator = np.sum(np.where(valid, vals * wts, 0.0), axis=2)
    denominator = np.sum(np.where(valid, wts, 0.0), axis=2)
    mean = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )
    return mean, denominator


def choose_tau_indices(tau_axis_ps, fit_mask):
    all_indices = np.where(fit_mask)[0]
    if all_indices.size == 0:
        return all_indices
    step = max(1, int(round(MRD_TAU_STRIDE_PS / (TIME_PER_FRAME_NS * 1000.0))))
    selected = all_indices[::step]
    if selected[-1] != all_indices[-1]:
        selected = np.append(selected, all_indices[-1])
    return selected


def load_block_tensors(block_starts, max_dt_frames):
    num_blocks = len(block_starts)
    default_centers = np.arange(R_PRIME_MIN, R_PRIME_MAX + 0.5 * SLIDE_STEP, SLIDE_STEP)
    window_centers = default_centers.copy()
    num_windows = len(window_centers)

    msd_tensor = np.full((num_blocks, num_windows, max_dt_frames), np.nan)
    mrd_tensor = np.full((num_blocks, num_windows, max_dt_frames), np.nan)
    count_tensor = np.zeros((num_blocks, num_windows, max_dt_frames), dtype=float)
    timefrac_tensor = np.full((num_blocks, num_windows, max_dt_frames), np.nan)

    for idx_b, t_start in enumerate(block_starts):
        npz_path = os.path.join(INPUT_DIR, f"msd_block_{t_start}.npz")
        if not os.path.exists(npz_path):
            print(f"[WARN] Missing block file: {npz_path}")
            continue

        data = np.load(npz_path)
        if "window_centers" in data:
            file_centers = data["window_centers"]
            if len(file_centers) != num_windows or not np.allclose(file_centers, window_centers):
                raise ValueError(f"Window centers in {npz_path} are inconsistent with opt_and_plot.py")

        msd_tensor[idx_b, :, :] = data["msd"]

        if "mrd" not in data:
            raise KeyError(f"{npz_path} does not contain 'mrd'. Please rerun the updated compute_block_msd.py")
        mrd_tensor[idx_b, :, :] = data["mrd"]

        if "count" not in data:
            raise KeyError(f"{npz_path} does not contain 'count'. Please rerun the updated compute_block_msd.py")
        count_tensor[idx_b, :, :] = data["count"]

        if "timefrac" in data:
            timefrac_tensor[idx_b, :, :] = data["timefrac"]

    return window_centers, msd_tensor, mrd_tensor, count_tensor, timefrac_tensor


def find_best_fit_window(msd_tensor, tau_axis_ps):
    valid_taus = tau_axis_ps[(tau_axis_ps >= MIN_START_PS) & (tau_axis_ps <= MAX_END_PS)]
    best_mean_r2 = -1.0
    best_window = (None, None)

    for start in valid_taus:
        for end in valid_taus:
            if end - start < MIN_WINDOW_LENGTH_PS:
                continue

            mask = (tau_axis_ps >= start) & (tau_axis_ps <= end)
            t_fit = tau_axis_ps[mask]
            r2_list = []

            for block_curve in msd_tensor:
                for msd_curve in block_curve[:, mask]:
                    valid = ~np.isnan(msd_curve)
                    if np.sum(valid) >= MIN_FIT_POINTS:
                        res = linregress(t_fit[valid], msd_curve[valid])
                        r2_list.append(res.rvalue**2)

            if r2_list:
                mean_r2 = float(np.mean(r2_list))
                if mean_r2 > best_mean_r2:
                    best_mean_r2 = mean_r2
                    best_window = (float(start), float(end))

    if best_window[0] is None:
        raise RuntimeError("No valid MSD fitting window was found.")
    return best_window, best_mean_r2


def calculate_d_tensor(msd_tensor, count_tensor, tau_axis_ps, fit_mask):
    num_blocks, num_windows, _ = msd_tensor.shape
    d_tensor = np.full((num_blocks, num_windows), np.nan)
    r2_tensor = np.full((num_blocks, num_windows), np.nan)
    fit_weight_tensor = np.zeros((num_blocks, num_windows), dtype=float)
    t_fit = tau_axis_ps[fit_mask]

    for idx_b in range(num_blocks):
        for idx_w in range(num_windows):
            msd_curve = msd_tensor[idx_b, idx_w, fit_mask]
            count_curve = count_tensor[idx_b, idx_w, fit_mask]
            valid = ~np.isnan(msd_curve) & (count_curve > 0)
            if np.sum(valid) < MIN_FIT_POINTS:
                continue

            res = linregress(t_fit[valid], msd_curve[valid])
            r2 = res.rvalue**2
            r2_tensor[idx_b, idx_w] = r2
            fit_weight_tensor[idx_b, idx_w] = np.sum(count_curve[valid])

            if r2 >= MIN_D_FIT_R2:
                d_tensor[idx_b, idx_w] = (res.slope / 2.0) * 10.0

    return d_tensor, r2_tensor, fit_weight_tensor


def write_summary_stats(
    path,
    window_centers,
    mean_dr,
    std_dr,
    mean_fit_r2,
    mean_mrd,
    std_mrd,
    mrd_sample_count,
    mean_timefrac,
    opt_start,
    opt_end,
    best_mean_r2,
    num_blocks,
):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Moving radial window thickness: {WINDOW_THICKNESS:.1f} A\n")
        f.write(f"# Sliding step: {SLIDE_STEP:.1f} A\n")
        f.write(f"# Optimized common D fitting window: {opt_start:.1f} to {opt_end:.1f} ps\n")
        f.write(f"# Mean window-search R^2: {best_mean_r2:.4f}\n")
        f.write(f"# 1 ns blocks count: {num_blocks}\n")
        f.write("# D_r unit: 10^-5 cm^2/s; MRD unit: A\n")
        f.write("# D_r is obtained from MSD_r using D_r = slope/2.\n")
        f.write("# MRD and TimeFraction are weighted averages over the same tau window used for D fitting.\n")
        f.write("# r_prime_center_A\tMean_Dr\tStd_Dr\tMean_D_fit_R2\tMean_MRD_A\tStd_MRD_A\tMRD_Samples\tMean_TimeFraction\n")
        for i, r_center in enumerate(window_centers):
            f.write(
                f"{r_center:8.2f}\t{mean_dr[i]:10.4e}\t{std_dr[i]:13.4e}\t"
                f"{mean_fit_r2[i]:10.4f}\t{mean_mrd[i]:10.4f}\t"
                f"{std_mrd[i]:10.4f}\t{mrd_sample_count[i]:12.0f}\t{mean_timefrac[i]:10.4f}\n"
            )


def write_profile_by_tau(path, window_centers, tau_axis_ps, tau_indices, mean_values, count_by_tau, quantity_name, unit):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Weighted {quantity_name} radial profiles at selected tau values inside the optimized D fitting window.\n")
        f.write(f"# Unit: {unit}\n")
        header = ["r_prime_center_A"]
        for tau_idx in tau_indices:
            tau = tau_axis_ps[tau_idx]
            header.extend([f"{quantity_name}_tau_{tau:.1f}ps", f"N_tau_{tau:.1f}ps"])
        f.write("\t".join(header) + "\n")

        for idx_w, r_center in enumerate(window_centers):
            row = [f"{r_center:.2f}"]
            for local_idx, _tau_idx in enumerate(tau_indices):
                row.append(f"{mean_values[idx_w, local_idx]:.6f}")
                row.append(f"{count_by_tau[idx_w, local_idx]:.0f}")
            f.write("\t".join(row) + "\n")


def plot_d_and_window_mrd(path, window_centers, mean_dr, std_dr, mean_mrd, opt_start, opt_end):
    plt.rcParams.update({"font.size": 14, "axes.linewidth": 1.5, "xtick.direction": "in", "ytick.direction": "in"})
    fig, ax1 = plt.subplots(figsize=(9, 6))

    valid_plot = ~np.isnan(mean_dr) & ~np.isnan(mean_mrd)
    x_plot = window_centers[valid_plot]
    y_dr = mean_dr[valid_plot]
    y_std = std_dr[valid_plot]
    y_mrd = mean_mrd[valid_plot]

    color_dr = "navy"
    line1 = ax1.plot(x_plot, y_dr, "-o", color=color_dr, linewidth=2.5, markersize=6, label=r"$D_r$")[0]
    ax1.fill_between(x_plot, y_dr - y_std, y_dr + y_std, color=color_dr, alpha=0.15)
    ax1.set_xlabel(r"Moving-window center $r'$ (A)", fontsize=16)
    ax1.set_ylabel(r"$D_r$ ($10^{-5}$ cm$^2$/s)", fontsize=16, color=color_dr)
    ax1.tick_params(axis="y", labelcolor=color_dr)
    ax1.axvline(x=0, color="black", linestyle="--", linewidth=2.0, zorder=0)
    ax1.axvspan(-WINDOW_THICKNESS / 2, WINDOW_THICKNESS / 2, color="forestgreen", alpha=0.08)
    ax1.grid(alpha=0.20)

    ax2 = ax1.twinx()
    color_mrd = "crimson"
    line2 = ax2.plot(x_plot, y_mrd, "--s", color=color_mrd, linewidth=2.0, markersize=5, label="MRD")[0]
    ax2.set_ylabel(f"MRD (A)\n[tau={opt_start:.0f}-{opt_end:.0f} ps]", fontsize=16, color=color_mrd)
    ax2.tick_params(axis="y", labelcolor=color_mrd)
    ax2.axhline(0, color="black", linestyle=":", alpha=0.5)

    lines = [line1, line2]
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_quantity_tau_profiles(path, window_centers, tau_axis_ps, tau_indices, mean_by_tau, ylabel, title):
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(tau_indices)))

    for local_idx, tau_idx in enumerate(tau_indices):
        y = mean_by_tau[:, local_idx]
        valid = ~np.isnan(y)
        if not np.any(valid):
            continue
        ax.plot(
            window_centers[valid],
            y[valid],
            "-o",
            linewidth=1.8,
            markersize=4,
            color=colors[local_idx],
            label=f"{tau_axis_ps[tau_idx]:.0f} ps",
        )

    ax.axvline(x=0, color="black", linestyle="--", linewidth=1.5)
    ax.axhline(y=0, color="black", linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"Moving-window center $r'$ (A)", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=14)
    ax.legend(title=r"$\tau$", ncol=2, fontsize=10, title_fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    block_starts = list(range(GLOBAL_START_NS, GLOBAL_END_NS, BLOCK_SIZE_NS))
    num_blocks = len(block_starts)
    print(f"Total 1 ns blocks to process: {num_blocks}")

    max_dt_frames = int(round(MAX_DELTA_T_NS / TIME_PER_FRAME_NS))
    tau_axis_ps = np.arange(1, max_dt_frames + 1) * (TIME_PER_FRAME_NS * 1000.0)

    window_centers, msd_tensor, mrd_tensor, count_tensor, timefrac_tensor = load_block_tensors(block_starts, max_dt_frames)

    (opt_start, opt_end), best_mean_r2 = find_best_fit_window(msd_tensor, tau_axis_ps)
    print(f"Optimal common fitting window: {opt_start:.1f}-{opt_end:.1f} ps (mean R^2={best_mean_r2:.4f})")

    fit_mask = (tau_axis_ps >= opt_start) & (tau_axis_ps <= opt_end)
    d_tensor, d_r2_tensor, d_weight_tensor = calculate_d_tensor(msd_tensor, count_tensor, tau_axis_ps, fit_mask)

    mean_dr, std_dr, _d_weight_sum = weighted_mean_std(d_tensor, d_weight_tensor, axis=0)
    mean_fit_r2, _std_fit_r2, _r2_weight_sum = weighted_mean_std(d_r2_tensor, d_weight_tensor, axis=0)

    mrd_window_tensor, mrd_window_weights = weighted_mean_over_tau(mrd_tensor, count_tensor, fit_mask)
    mean_mrd, std_mrd, mrd_sample_count = weighted_mean_std(mrd_window_tensor, mrd_window_weights, axis=0)

    timefrac_window_tensor, timefrac_window_weights = weighted_mean_over_tau(timefrac_tensor, count_tensor, fit_mask)
    mean_timefrac, _std_timefrac, _timefrac_count = weighted_mean_std(timefrac_window_tensor, timefrac_window_weights, axis=0)

    tau_indices = choose_tau_indices(tau_axis_ps, fit_mask)

    mrd_tau_values = np.transpose(mrd_tensor[:, :, tau_indices], (0, 2, 1))
    mrd_tau_weights = np.transpose(count_tensor[:, :, tau_indices], (0, 2, 1))
    mean_mrd_by_tau_tmp, _std_mrd_by_tau_tmp, count_by_tau_tmp = weighted_mean_std(mrd_tau_values, mrd_tau_weights, axis=0)
    mean_mrd_by_tau = mean_mrd_by_tau_tmp.T
    count_by_tau = count_by_tau_tmp.T

    timefrac_tau_values = np.transpose(timefrac_tensor[:, :, tau_indices], (0, 2, 1))
    timefrac_tau_weights = np.transpose(count_tensor[:, :, tau_indices], (0, 2, 1))
    mean_timefrac_by_tau_tmp, _std_timefrac_by_tau_tmp, count_timefrac_by_tau_tmp = weighted_mean_std(
        timefrac_tau_values,
        timefrac_tau_weights,
        axis=0,
    )
    mean_timefrac_by_tau = mean_timefrac_by_tau_tmp.T
    count_timefrac_by_tau = count_timefrac_by_tau_tmp.T

    write_summary_stats(
        OUTPUT_FILE_FINAL_STATS,
        window_centers,
        mean_dr,
        std_dr,
        mean_fit_r2,
        mean_mrd,
        std_mrd,
        mrd_sample_count,
        mean_timefrac,
        opt_start,
        opt_end,
        best_mean_r2,
        num_blocks,
    )

    write_profile_by_tau(
        OUTPUT_FILE_MRD_TAU_STATS,
        window_centers,
        tau_axis_ps,
        tau_indices,
        mean_mrd_by_tau,
        count_by_tau,
        quantity_name="MRD",
        unit="A",
    )

    write_profile_by_tau(
        OUTPUT_FILE_TIMEFRAC_STATS,
        window_centers,
        tau_axis_ps,
        tau_indices,
        mean_timefrac_by_tau,
        count_timefrac_by_tau,
        quantity_name="TimeFraction",
        unit="dimensionless",
    )

    plot_d_and_window_mrd(OUTPUT_FILE_FINAL_PLOT, window_centers, mean_dr, std_dr, mean_mrd, opt_start, opt_end)
    plot_quantity_tau_profiles(
        OUTPUT_FILE_MRD_TAU_PLOT,
        window_centers,
        tau_axis_ps,
        tau_indices,
        mean_mrd_by_tau,
        ylabel="Mean radial displacement (A)",
        title="Moving-window MRD profiles",
    )
    plot_quantity_tau_profiles(
        OUTPUT_FILE_TIMEFRAC_PLOT,
        window_centers,
        tau_axis_ps,
        tau_indices,
        mean_timefrac_by_tau,
        ylabel="Time fraction in initial 5 A window",
        title="Moving-window residence fraction profiles",
    )

    print(f"Results saved to {OUTPUT_DIR}")
    print(f"Summary: {OUTPUT_FILE_FINAL_STATS}")
    print(f"Main plot: {OUTPUT_FILE_FINAL_PLOT}")


if __name__ == "__main__":
    main()
