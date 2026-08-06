#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute radial-window MSD and MRD for one 1 ns trajectory block.

Key definition:
    r' = R_ref - r_abs

For each moving radial window centered at r'_c with width WINDOW_THICKNESS:
    Select all N2 molecules whose initial position r'(t0) is inside the window.
    Then compute, for each lag time tau:
        MSD_r(r'_c, tau) = < [r_abs(t0+tau) - r_abs(t0)]^2 >
        MRD_r(r'_c, tau) = < r'(t0+tau) - r'(t0) >

The molecule is not required to remain inside the window after t0. This avoids
survival bias and characterizes the short-time mobility of molecules initially
located in a given radial region.

Outputs per block:
    msd_block_<start_ns>.npz with arrays:
        msd          shape = (num_windows, max_dt_frames), unit A^2
        mrd          shape = (num_windows, max_dt_frames), unit A
        count        shape = (num_windows, max_dt_frames), number of samples
        timefrac     shape = (num_windows, max_dt_frames), fraction of time staying in same window
        window_centers
"""

import argparse
import os
import re
from math import sqrt

import numpy as np

# ================= User configuration =================
BUBBLE_ID = "bubble2"
N2_ATOM_TYPE = 3

BLOCK_SIZE_NS = 1
TIME_PER_FRAME_NS = 0.001
MAX_DELTA_T_NS = 0.1

R_PRIME_MIN = -14.0
R_PRIME_MAX = 6.0
WINDOW_THICKNESS = 5.0
SLIDE_STEP = 1.0

PATH_TEMPLATE_TRAJ = "data/biNBs/NaCl/002/{TIME_RANGE}/COM_shift/centered_{BUBBLE_ID}.lammpstrj"
PATH_TEMPLATE_LOG = "data/plot/biBNBs/salt/002_ripening/ion_distribution_for_several_timeblock/1ns_block/{TIME_RANGE}/results/{BUBBLE_ID}_block{BLOCK_NUM}/log.txt"
# ======================================================


def map_ns_to_chunk_and_block(t_ns):
    chunk_start = int(np.floor(t_ns / 8)) * 8
    time_range_str = f"{chunk_start}-{chunk_start + 8}ns"
    block_num = int(t_ns - chunk_start) + 1
    return time_range_str, block_num


def extract_r_from_log(t_ns):
    time_range_str, block_num = map_ns_to_chunk_and_block(t_ns)
    path = PATH_TEMPLATE_LOG.format(
        TIME_RANGE=time_range_str,
        BUBBLE_ID=BUBBLE_ID,
        BLOCK_NUM=block_num,
    )
    if not os.path.exists(path):
        print(f"[WARN] Missing log file: {path}; using default R_ref=11.0 A")
        return 11.0

    regex = re.compile(r"(?m)^interface_position\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$")
    try:
        with open(path, "r", encoding="utf-8") as f:
            matches = regex.findall(f.read())
        if matches:
            return float(matches[-1])
    except Exception as exc:
        print(f"[WARN] Failed to parse R_ref from {path}: {exc}; using default R_ref=11.0 A")
    return 11.0


def read_lammps_frame(f, n2_atom_type):
    line = f.readline()
    if not line:
        return None

    while line and not line.startswith("ITEM: TIMESTEP"):
        line = f.readline()
    if not line:
        return None
    f.readline()  # timestep value

    line = f.readline()
    if not line.startswith("ITEM: NUMBER OF ATOMS"):
        return None
    try:
        natoms = int(f.readline().strip())
    except ValueError:
        return None

    line = f.readline()
    if not line.startswith("ITEM: BOX BOUNDS"):
        return None
    f.readline()
    f.readline()
    f.readline()

    header_line = f.readline().strip()
    if not header_line.startswith("ITEM: ATOMS"):
        return None
    cols = header_line.split()

    try:
        id_idx = cols.index("id") - 2
        type_idx = cols.index("type") - 2
        x_idx = next(i for i, c in enumerate(cols) if c in ["x", "xu", "xs"]) - 2
        y_idx = next(i for i, c in enumerate(cols) if c in ["y", "yu", "ys"]) - 2
        z_idx = next(i for i, c in enumerate(cols) if c in ["z", "zu", "zs"]) - 2
    except (StopIteration, ValueError):
        print("[ERROR] Cannot find id/type/x/y/z columns in trajectory header.")
        return None

    atoms_pos = {}
    for _ in range(natoms):
        parts = f.readline().split()
        if not parts:
            continue
        try:
            if int(parts[type_idx]) == n2_atom_type:
                atoms_pos[int(parts[id_idx])] = (
                    float(parts[x_idx]),
                    float(parts[y_idx]),
                    float(parts[z_idx]),
                )
        except Exception:
            continue
    return atoms_pos


def calculate_molecular_r_abs_proxy(atoms_pos):
    """
    Convert atom positions to a molecular radial coordinate proxy.
    For each N2 molecule, only one N atom is used as the molecular proxy,
    consistent with the previous implementation.
    """
    mol_r_map = {}
    seen_mols = set()
    for atom_id, pos in atoms_pos.items():
        mol_id = (atom_id + 1) // 2
        if mol_id in seen_mols:
            continue
        mol_r_map[mol_id] = sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2)
        seen_mols.add(mol_id)
    return mol_r_map


def skip_to_block_start(f, skip_frames_needed):
    frames_skipped = 0
    while frames_skipped < skip_frames_needed:
        line = f.readline()
        if not line:
            break
        if "ITEM: TIMESTEP" in line:
            frames_skipped += 1


def compute_single_block_msd_mrd(t_start_ns, r_ref, window_centers):
    frames_per_block = int(round(BLOCK_SIZE_NS / TIME_PER_FRAME_NS))
    max_dt_frames = int(round(MAX_DELTA_T_NS / TIME_PER_FRAME_NS))
    dt_frames_array = np.arange(1, max_dt_frames + 1)
    num_windows = len(window_centers)

    time_range_str, _ = map_ns_to_chunk_and_block(t_start_ns)
    traj_path = PATH_TEMPLATE_TRAJ.format(
        TIME_RANGE=time_range_str,
        BUBBLE_ID=BUBBLE_ID,
    )

    msd_block = np.full((num_windows, max_dt_frames), np.nan)
    mrd_block = np.full((num_windows, max_dt_frames), np.nan)
    count_block = np.zeros((num_windows, max_dt_frames), dtype=np.int64)
    timefrac_block = np.full((num_windows, max_dt_frames), np.nan)

    if not os.path.exists(traj_path):
        print(f"[ERROR] Trajectory file does not exist: {traj_path}")
        return msd_block, mrd_block, count_block, timefrac_block

    chunk_start = int(np.floor(t_start_ns / 8)) * 8
    skip_frames_needed = int(round((t_start_ns - chunk_start) / TIME_PER_FRAME_NS))

    all_mol_ids = set()
    with open(traj_path, "r") as f:
        skip_to_block_start(f, skip_frames_needed)
        for _ in range(frames_per_block):
            pos = read_lammps_frame(f, N2_ATOM_TYPE)
            if pos is None:
                break
            all_mol_ids.update(calculate_molecular_r_abs_proxy(pos).keys())

    sorted_mol_ids = sorted(all_mol_ids)
    if not sorted_mol_ids:
        print(f"[ERROR] No N2 molecules found in {t_start_ns}-{t_start_ns + BLOCK_SIZE_NS} ns block.")
        return msd_block, mrd_block, count_block, timefrac_block

    mol_id_to_col_idx = {mid: i for i, mid in enumerate(sorted_mol_ids)}
    r_abs_matrix = np.full((frames_per_block, len(sorted_mol_ids)), np.nan)

    with open(traj_path, "r") as f:
        skip_to_block_start(f, skip_frames_needed)
        for frame_idx in range(frames_per_block):
            pos = read_lammps_frame(f, N2_ATOM_TYPE)
            if pos is None:
                break
            mol_r_map = calculate_molecular_r_abs_proxy(pos)
            for mid, r_val in mol_r_map.items():
                if mid in mol_id_to_col_idx:
                    r_abs_matrix[frame_idx, mol_id_to_col_idx[mid]] = r_val

    valid_ratio = np.sum(~np.isnan(r_abs_matrix)) / r_abs_matrix.size
    print(f"  -> Coordinate matrix fill ratio: {valid_ratio * 100:.2f}% (molecules: {len(sorted_mol_ids)})")

    r_prime_matrix = r_ref - r_abs_matrix

    for w_idx, rc in enumerate(window_centers):
        w_min = rc - WINDOW_THICKNESS / 2.0
        w_max = rc + WINDOW_THICKNESS / 2.0

        in_window = (r_prime_matrix >= w_min) & (r_prime_matrix <= w_max)
        cum_in = np.cumsum(in_window.astype(np.int32), axis=0)

        for idx_dt, dt_f in enumerate(dt_frames_array):
            r_prime_t0 = r_prime_matrix[:-dt_f, :]
            r_prime_t1 = r_prime_matrix[dt_f:, :]
            r_abs_t0 = r_abs_matrix[:-dt_f, :]
            r_abs_t1 = r_abs_matrix[dt_f:, :]

            valid_mask = (
                ~np.isnan(r_prime_t0)
                & ~np.isnan(r_prime_t1)
                & ~np.isnan(r_abs_t0)
                & ~np.isnan(r_abs_t1)
            )
            started_valid = in_window[:-dt_f, :] & valid_mask
            count = int(np.sum(started_valid))

            if count <= 0:
                continue

            sq_disp = (r_abs_t1 - r_abs_t0) ** 2
            delta_r_prime = r_prime_t1 - r_prime_t0

            msd_block[w_idx, idx_dt] = float(np.mean(sq_disp[started_valid]))
            mrd_block[w_idx, idx_dt] = float(np.mean(delta_r_prime[started_valid]))
            count_block[w_idx, idx_dt] = count

            # Fraction of frames from t0 to t0+tau for which selected molecules remain in this same window.
            sum_in_window = cum_in[dt_f:, :] - cum_in[:-dt_f, :] + in_window[:-dt_f, :]
            frac_in_window = sum_in_window[started_valid] / (dt_f + 1.0)
            timefrac_block[w_idx, idx_dt] = float(np.mean(frac_in_window))

    return msd_block, mrd_block, count_block, timefrac_block


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_ns", type=int, required=True, help="Block start time in ns")
    parser.add_argument("--out_dir", type=str, default="temp_msd_data", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    r_ref = extract_r_from_log(args.start_ns)
    print(f"\n[{args.start_ns}-{args.start_ns + BLOCK_SIZE_NS} ns] R_ref = {r_ref:.4f} A")

    window_centers = np.arange(R_PRIME_MIN, R_PRIME_MAX + 0.5 * SLIDE_STEP, SLIDE_STEP)
    msd, mrd, count, timefrac = compute_single_block_msd_mrd(
        args.start_ns,
        r_ref,
        window_centers,
    )

    save_path = os.path.join(args.out_dir, f"msd_block_{args.start_ns}.npz")
    np.savez(
        save_path,
        msd=msd,
        mrd=mrd,
        count=count,
        timefrac=timefrac,
        window_centers=window_centers,
        window_thickness=np.array([WINDOW_THICKNESS], dtype=float),
        slide_step=np.array([SLIDE_STEP], dtype=float),
    )
    print(f"Saved moving-window MSD, MRD, count, and timefrac to {save_path}\n")
