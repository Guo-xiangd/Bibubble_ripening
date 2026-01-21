#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculate Local Radial Diffusion Coefficient D_r(r') (v3 - Added real-time progress bar and fixed time parsing)

This script calculates the Local Radial Diffusion Coefficient of Nitrogen molecules (N2)
near the bubble-liquid interface as a function of the relative position (r') to the interface.

It integrates two data sources:
1. LAMMPS Trajectory File: Used to read instantaneous absolute coordinates (x, y, z) of N2 molecules.
2. Interface Log File: Used to read the bubble interface position R(t) for each frame.

Calculation Logic (Ensemble Averaging Method):
1. Parse the time range string (e.g., "8-38") to calculate the total target frames.
2. Load the R(t) log until the total target frames are reached.
3. Load the trajectory and create a (N_frames_total, N_mols) matrix.
4. Iterate through all start times t0 (from 0 to t_max - delta_t):
    a. Iterate through all N2 molecules i.
    b. Calculate relative position at t0: r'(t0) = R(t0) - r_abs(i, t0)
    c. Assign the event to the corresponding radial bin based on r'(t0).
    d. Calculate relative position at t_end = t0 + delta_t:
       r'(t_end) = R(t_end) - r_abs(i, t_end)
    e. Calculate squared radial displacement: (delta_r')^2 = (r'(t_end) - r'(t0))^2
    f. Store (delta_r')^2 in the bin corresponding to r'(t0).
5. Iterate through all bins and calculate Mean Squared Displacement (MSD) = <(delta_r')^2>.
6. Calculate local diffusion coefficient using the Einstein relation: D_r = MSD / (2 * delta_t).
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import re
from math import sqrt
from typing import List, Dict, Tuple, Optional

# ==============================================================================
# --- 1. User Configuration Section ---
# (!!! Please modify this section based on your files and system !!!)
# ==============================================================================

# --- A. Analysis Scope and System Definition ---

# (Critical) Define the *total time range* (ns) to analyze.
# Example: "8-32" (24ns) or "8-38" (30ns)
TIME_RANGE_NS = "8-38"

# (Critical) Bubble ID (used for file matching)
BUBBLE_ID = "bubble2"

# (Critical) Type ID of N2 atoms in the trajectory file
N2_ATOM_TYPE = 3  # 1:H, 2:O, 3:N, 4:Na, 5:Cl

# --- B. Trajectory and Log File Path Templates ---

# (Critical) LAMMPS trajectory file path template
# {TIME_RANGE} will be replaced by "8-16ns", "16-24ns", etc.
# {BUBBLE_ID} will be replaced by "bubble2"
PATH_TEMPLATE_TRAJ = (
    "/data/HOME_BACKUP/xiangdang/gpumd/biNBs/NaCl/002/{TIME_RANGE}"
    "/COM_shift/centered_{BUBBLE_ID}.lammpstrj"
)

# (Critical) Interface log file path template
# {TIME_RANGE} as above, {BLOCK_NUM} will be replaced by 1, 2, ... 8
PATH_TEMPLATE_LOG = (
    "/data/HOME_BACKUP/xiangdang/gpumd/plot/biBNBs/salt/002_ripening"
    "/ion_distribution_for_several_timeblock/1ns_block/{TIME_RANGE}"
    "/results/{BUBBLE_ID}_block{BLOCK_NUM}/log.txt"
)

# --- C. System and Calculation Parameters ---

TIME_PER_FRAME_NS = 0.001   # Time per frame (ns)
FRAMES_PER_1NS_BLOCK = 1000 # Number of frames per 1ns block in R(t) log
BLOCKS_PER_8NS_FILE = 8     # Number of 1ns blocks per 8ns time segment

# (Critical) Time window for MSD calculation (ns)
DELTA_T_NS = 0.02

# Binning parameters for radial distance r' (Unit: Å)
# Coordinate system: Negative outside bubble, Positive inside bubble
R_PRIME_MIN = -14.0  # Minimum r' (Outside bubble)
R_PRIME_MAX = 6.0    # Maximum r' (Inside bubble)
BIN_SIZE = 1.0       # Width of each bin

# --- D. Output Settings ---
OUTPUT_DIR = "local_diffusion_analysis_Dr"
OUTPUT_FILE_STATS = os.path.join(OUTPUT_DIR, "local_diffusion_Dr_stats.txt")
OUTPUT_FILE_PLOT = os.path.join(OUTPUT_DIR, "local_diffusion_Dr_plot.png")


# ==============================================================================
# --- 2. Helper Functions: Parsing and Loading ---
# ==============================================================================

# (*** Placeholder ***)
# These functions will be overwritten by their "robust_" versions in the __main__ block
def parse_time_range(range_str: str) -> Tuple[List[str], int]:
    """ (Placeholder) This function will be replaced in __main__ """
    print("Error: parse_time_range was not correctly replaced!")
    sys.exit(1)

def load_all_interface_positions(time_chunks: List[str], log_template: str, bubble_id: str, total_expected_frames: int) -> np.ndarray:
    """ (Placeholder) This function will be replaced in __main__ """
    print("Error: load_all_interface_positions was not correctly replaced!")
    sys.exit(1)

def read_lammps_frame(f, n2_atom_type: int) -> Optional[Dict[int, Tuple[float, float, float]]]:
    """
    Reads a single frame from the LAMMPS trajectory.
    Returns a dictionary {atom_id: (x, y, z)} containing only N2 atoms.
    """
    line = f.readline()
    if not line: return None
    
    if "ITEM: TIMESTEP" not in line:
        # Fault tolerance: Handle non-standard lines appearing in the middle of the file
        print(f"\nWarning: Abnormal trajectory format, 'ITEM: TIMESTEP' not found. Skipping... Read: {line[:50]}...")
        # Try to read until TIMESTEP is found or EOF
        while True:
            line = f.readline()
            if not line: return None
            if "ITEM: TIMESTEP" in line: break
        # Found it, proceed
    
    f.readline() # timestep
    f.readline() # ITEM: NUMBER OF ATOMS
    try:
        natoms = int(f.readline())
    except ValueError:
        print("\nError: Failed to read 'ITEM: NUMBER OF ATOMS'. File might be corrupted.")
        return None
        
    f.readline() # ITEM: BOX BOUNDS
    f.readline() # box x
    f.readline() # box y
    f.readline() # box z
    header_line = f.readline().strip() # ITEM: ATOMS id type x y z
    
    if not header_line.startswith("ITEM: ATOMS"):
        print(f"\nTrajectory file format error, 'ITEM: ATOMS' not found. Read: {header_line}")
        return None
        
    # Dynamically determine column indices
    cols = header_line.split(" ")
    try:
        id_idx = cols.index("id") - 2     # Subtract "ITEM:" and "ATOMS"
        type_idx = cols.index("type") - 2
        x_idx = cols.index("x") - 2
        y_idx = cols.index("y") - 2
        z_idx = cols.index("z") - 2
    except ValueError as e:
        print(f"\nError: Trajectory ATOMS header missing required columns (id, type, x, y, z). Header: '{header_line}'")
        sys.exit(1)

    atoms_pos = {} # {atom_id: (x, y, z)}
    for _ in range(natoms):
        parts = f.readline().split()
        if not parts:
             print(f"\nWarning: Empty line encountered while reading {natoms} atoms. Frame might be incomplete.")
             continue # Skip empty line
        try:
            if int(parts[type_idx]) == n2_atom_type:
                atm_id = int(parts[id_idx])
                x = float(parts[x_idx])
                y = float(parts[y_idx])
                z = float(parts[z_idx])
                atoms_pos[atm_id] = (x, y, z)
        except IndexError:
             print(f"\nWarning: Incomplete line data: '{' '.join(parts)}'. Skipping atom.")
             continue
        except ValueError:
             print(f"\nWarning: Malformed line data: '{' '.join(parts)}'. Skipping atom.")
             continue
            
    return atoms_pos

def parse_mol_id(atom_id: int) -> int:
    """Convert N2 Atom ID to Molecule ID (Assuming 1,2 -> 1; 3,4 -> 2)"""
    return (atom_id + 1) // 2

def calculate_molecular_r_abs(atoms_pos: Dict[int, Tuple[float, float, float]]) -> Dict[int, float]:
    """
    Converts atom coordinate dict {atom_id: (x,y,z)} to
    molecule coordinate dict {mol_id: r_abs}.
    """
    mol_r_map = {}
    seen_mols = set()
    
    for atm_id, pos in atoms_pos.items():
        mol_id = parse_mol_id(atm_id)
        if mol_id in seen_mols:
            continue
        
        # Find the pair atom for N2
        if atm_id % 2 == 1: # Odd ID
            pair_id = atm_id + 1
        else: # Even ID
            pair_id = atm_id - 1
        
        if pair_id in atoms_pos:
            pos1 = np.array(pos)
            pos2 = np.array(atoms_pos[pair_id])
            
            # (Assume equal mass, take midpoint)
            mid_pos = (pos1 + pos2) / 2.0
            r_abs = sqrt(mid_pos[0]**2 + mid_pos[1]**2 + mid_pos[2]**2)
            mol_r_map[mol_id] = r_abs
            seen_mols.add(mol_id)
            
    return mol_r_map


# ==============================================================================
# --- 3. Core Calculation Logic (*** with Progress Bar ***) ---
# ==============================================================================

def main_calculation():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- 1. Parse config and load R(t) ---
    # `parse_time_range` and `load_all_interface_positions` will be
    # the robust_ versions defined in __main__
    time_chunks, n_frames_total = parse_time_range(TIME_RANGE_NS)
    
    R_all_frames = load_all_interface_positions(
        time_chunks, PATH_TEMPLATE_LOG, BUBBLE_ID, n_frames_total
    )
    
    # (Fix) Check if load_all_interface_positions returned early due to missing files
    if len(R_all_frames) < n_frames_total:
        print(f"!!! Warning: Loaded R(t) frames ({len(R_all_frames)}) do not match expected frames ({n_frames_total}).")
        print(f"!!! Please check if your log files are complete. Analyzing using the available {len(R_all_frames)} frames.")
        n_frames_total = len(R_all_frames) # Critical: Update total frames

    # --- 2. Load r_abs(t) for all N2 molecules ---
    print(f"--- 2. Pre-scanning trajectory to get all N2 molecule IDs... (Total {n_frames_total} frames) ---")
    all_mol_ids = set()
    frame_idx_counter = 0
    
    # (*** New ***) Progress bar settings
    update_interval_prescan = max(1, n_frames_total // 100) # Update every 1% or every frame

    for time_range in time_chunks:
        if frame_idx_counter >= n_frames_total: break
            
        traj_path = PATH_TEMPLATE_TRAJ.format(TIME_RANGE=time_range, BUBBLE_ID=BUBBLE_ID)
        if not os.path.exists(traj_path):
            print(f"\nError: Trajectory file not found: {traj_path}")
            sys.exit(1)
        
        print(f"  > Pre-scanning: {traj_path}")
        try:
            with open(traj_path, 'r') as f:
                while True:
                    if frame_idx_counter >= n_frames_total: break
                    atoms_pos = read_lammps_frame(f, N2_ATOM_TYPE)
                    if atoms_pos is None: break
                    
                    mol_r_map = calculate_molecular_r_abs(atoms_pos)
                    all_mol_ids.update(mol_r_map.keys())
                    frame_idx_counter += 1
                    
                    # (*** New ***) Pre-scan progress bar
                    if frame_idx_counter % update_interval_prescan == 0:
                        percent = (frame_idx_counter / n_frames_total) * 100
                        progress_str = f"    > Pre-scan progress: {percent:6.2f}% (Frame {frame_idx_counter}/{n_frames_total})".ljust(80)
                        print(f"\r{progress_str}", end='', flush=True)
        except Exception as e:
            print(f"\nCritical error reading {traj_path}: {e}")
            sys.exit(1)

    print() # (*** New ***) Newline after progress bar
    
    # (Fix) Check if trajectory frames and log frames *still* match
    if frame_idx_counter < n_frames_total:
        print(f"!!! Critical Warning: Total trajectory frames ({frame_idx_counter}) are fewer than (corrected) log frames ({n_frames_total})")
        print("  > Clipping R(t) array again to match trajectory length.")
        R_all_frames = R_all_frames[:frame_idx_counter]
        n_frames_total = frame_idx_counter
        
    sorted_mol_ids = sorted(list(all_mol_ids))
    mol_id_to_col_idx = {mid: i for i, mid in enumerate(sorted_mol_ids)}
    n_mols_total = len(sorted_mol_ids)
    
    if n_mols_total == 0:
        print(f"\n!!! Critical Error: No N2 atoms of type {N2_ATOM_TYPE} found in trajectory.")
        print("!!! Please check if `N2_ATOM_TYPE` configuration is correct.")
        sys.exit(1)
        
    print(f"  > Pre-scan complete. Found {n_mols_total} N2 molecules.")
    
    R_abs_all_frames = np.full((n_frames_total, n_mols_total), np.nan)
    
    print(f"--- 3. Loading N2 absolute radial positions r_abs(t)... (Total {n_frames_total} frames) ---")
    frame_idx_counter = 0
    
    # (*** New ***) Progress bar settings
    update_interval_load = max(1, n_frames_total // 100) # Update every 1% or every frame

    for time_range in time_chunks:
        if frame_idx_counter >= n_frames_total: break
        traj_path = PATH_TEMPLATE_TRAJ.format(TIME_RANGE=time_range, BUBBLE_ID=BUBBLE_ID)
        print(f"  > Loading: {traj_path}")
        try:
            with open(traj_path, 'r') as f:
                while True:
                    if frame_idx_counter >= n_frames_total: break
                    atoms_pos = read_lammps_frame(f, N2_ATOM_TYPE)
                    if atoms_pos is None: break
                    
                    mol_r_map = calculate_molecular_r_abs(atoms_pos)
                    
                    for mid, r_val in mol_r_map.items():
                        if mid in mol_id_to_col_idx: 
                            col_idx = mol_id_to_col_idx[mid]
                            R_abs_all_frames[frame_idx_counter, col_idx] = r_val
                    
                    frame_idx_counter += 1
                    
                    # (*** New ***) Loading progress bar
                    if frame_idx_counter % update_interval_load == 0:
                        percent = (frame_idx_counter / n_frames_total) * 100
                        progress_str = f"    > Load progress: {percent:6.2f}% (Frame {frame_idx_counter}/{n_frames_total})".ljust(80)
                        print(f"\r{progress_str}", end='', flush=True)
        except Exception as e:
            print(f"\nCritical error reading {traj_path}: {e}")
            sys.exit(1)

    print() # (*** New ***) Newline after progress bar
    print(f"  > r_abs(t) matrix (Shape: {R_abs_all_frames.shape}) loaded.")

    # --- 4. Calculate Local Diffusion Coefficient D_r(r') ---
    
    dt_frames = int(round(DELTA_T_NS / TIME_PER_FRAME_NS)) # Use round to avoid floating point issues
    if dt_frames <= 0:
        raise ValueError(f"DELTA_T_NS ({DELTA_T_NS}) is too small, smaller than one frame time ({TIME_PER_FRAME_NS})")
    
    n_calc_frames = n_frames_total - dt_frames
    if n_calc_frames <= 0:
        print(f"!!! Error: Total trajectory frames ({n_frames_total}) <= Delta_t frames ({dt_frames}).")
        print("!!! Cannot calculate diffusion. Please increase simulation time or decrease DELTA_T_NS.")
        sys.exit(1)

    print(f"--- 4. Calculating Local Radial Diffusion D_r(r')... ---")
    print(f"  > Diffusion time window (Delta_t): {DELTA_T_NS} ns ({dt_frames} frames)")
    print(f"  > Performing ensemble averaging over {n_calc_frames} time origins...")

    num_bins = int(round((R_PRIME_MAX - R_PRIME_MIN) / BIN_SIZE))
    r_prime_bins_edges = np.linspace(R_PRIME_MIN, R_PRIME_MAX, num_bins + 1)
    r_prime_bins_centers = (r_prime_bins_edges[:-1] + r_prime_bins_edges[1:]) / 2
    
    msd_data_per_bin = [[] for _ in range(num_bins)]
    
    # (*** New ***) Progress bar settings
    update_interval_calc = max(1, n_calc_frames // 100)
    
    # Loop t (start time)
    for t0 in range(n_calc_frames):
        t_end = t0 + dt_frames
        
        R_if_t0 = R_all_frames[t0]
        R_if_t_end = R_all_frames[t_end]
        
        # Vectorized operation for r_abs at t0 and t_end
        r_abs_t0_all_mols = R_abs_all_frames[t0, :]
        r_abs_t_end_all_mols = R_abs_all_frames[t_end, :]

        # (Filter) Must exist at both timestamps
        valid_mask = ~np.isnan(r_abs_t0_all_mols) & ~np.isnan(r_abs_t_end_all_mols)
        
        if not np.any(valid_mask):
            continue

        valid_r_abs_t0 = r_abs_t0_all_mols[valid_mask]
        valid_r_abs_t_end = r_abs_t_end_all_mols[valid_mask]

        # r' = R_interface(t) - r_abs(t)
        r_prime_t0 = R_if_t0 - valid_r_abs_t0
        
        # Determine bin for r'(t0)
        bin_indices = np.digitize(r_prime_t0, r_prime_bins_edges) - 1
        
        r_prime_t_end = R_if_t_end - valid_r_abs_t_end
        delta_r_prime_sq = (r_prime_t_end - r_prime_t0)**2

        # Loop m (molecule)
        for m_idx_valid in range(len(valid_r_abs_t0)):
            bin_idx = bin_indices[m_idx_valid]
            if 0 <= bin_idx < num_bins:
                msd_data_per_bin[bin_idx].append(delta_r_prime_sq[m_idx_valid])
        
        # (*** New ***) MSD Calculation Progress Bar
        if t0 % update_interval_calc == 0 or t0 == n_calc_frames - 1:
            percent = (t0 + 1) / n_calc_frames * 100
            progress_str = f"    > Calculating MSD: {percent:6.2f}% complete (Time Origin {t0+1}/{n_calc_frames})".ljust(80)
            print(f"\r{progress_str}", end='', flush=True)

    print() # (*** New ***) Newline after progress bar
    print("  > MSD ensemble averaging complete.")

    # --- 5. Post-processing and Saving ---
    print(f"--- 5. Calculating final D_r(r') and saving... ---")
    
    results_Dr = np.full(num_bins, np.nan)
    results_count = np.zeros(num_bins, dtype=int)
    
    for i in range(num_bins):
        data = msd_data_per_bin[i]
        count = len(data)
        results_count[i] = count
        
        if count > 0:
            msd_r_prime = np.mean(data)
            D_r = msd_r_prime / (2 * DELTA_T_NS)
            results_Dr[i] = D_r

    header = f"# Local Radial Diffusion Coefficient (D_r) vs. Interface Relative Position (r')\n"
    header += f"# Calculation based on trajectory {TIME_RANGE_NS} (ns), Bubble {BUBBLE_ID}\n"
    header += f"# Coordinate System: Negative outside bubble, Positive inside bubble\n"
    header += f"# D_r = <(r'(t+{DELTA_T_NS}) - r'(t))^2> / (2 * {DELTA_T_NS})\n"
    header += f"{'r_prime_center(A)':<20} {'D_r (A^2/ns)':<20} {'Event_Count':<15}"
    
    data_to_save = np.vstack([r_prime_bins_centers, results_Dr, results_count]).T
    np.savetxt(OUTPUT_FILE_STATS, data_to_save, header=header, fmt="%-19.6f %-19.8e %-14d")
    
    print(f"  > Statistics saved to: {OUTPUT_FILE_STATS}")

    # --- 6. Plotting ---
    try:
        plt.rcParams.update({'font.size': 12, 'axes.linewidth': 1.5, 'xtick.direction': 'in', 'ytick.direction': 'in'})
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        ax1.plot(r_prime_bins_centers, results_Dr, 'o-', color='crimson', label='$D_r$ (Radial Diffusion)')
        ax1.set_xlabel("Relative Radial Position, r' (Å) [Outside < 0 < Inside]", fontsize=14)
        ax1.set_ylabel("Radial Diffusion Coefficient, $D_r$ (Å²/ns)", fontsize=14)
        ax1.set_xlim(R_PRIME_MIN, R_PRIME_MAX)
        ax1.set_ylim(bottom=0)
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=2.5, label='Interface (r\' = 0)')
        
        ax2 = ax1.twinx()
        ax2.bar(r_prime_bins_centers, results_count, 
                width=BIN_SIZE*0.8, color='grey', alpha=0.3, 
                label='Event Count')
        ax2.set_ylabel("Event Count (log scale)", color='grey')
        # (*** Fix ***) Only set log scale if there is data
        if np.sum(results_count) > 0:
            ax2.set_yscale('log')
        
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left')
        
        fig.tight_layout()
        plt.savefig(OUTPUT_FILE_PLOT, dpi=300)
        print(f"  > Diffusion plot saved to: {OUTPUT_FILE_PLOT}")
        
    except ImportError:
        print("  > matplotlib not found, skipping plot.")
    except Exception as e:
        print(f"  > Error during plotting: {e}")
        print("  > (Possibly due to no data in any bins, or matplotlib backend issue)")

# ==============================================================================
# --- 7. Script Execution (*** Corrected Here ***) ---
# ==============================================================================

if __name__ == "__main__":
    
    # Check dependencies
    try:
        import numpy
        import matplotlib
    except ImportError as e:
        print(f"Error: Missing dependency {e.name}.", file=sys.stderr)
        print("Please install using 'pip install numpy matplotlib'.", file=sys.stderr)
        sys.exit(1)
        
    # (*** Fix ***) 
    # Only check if placeholder exists
    if "{TIME_RANGE}" not in PATH_TEMPLATE_TRAJ:
        print("!!! Config Error: Please check if `PATH_TEMPLATE_TRAJ` contains '{TIME_RANGE}' placeholder.")
        sys.exit(1)
    if "{TIME_RANGE}" not in PATH_TEMPLATE_LOG:
        print("!!! Config Error: Please check if `PATH_TEMPLATE_LOG` contains '{TIME_RANGE}' placeholder.")
        sys.exit(1)
    
    # (*** Replacement ***) 
    # Use corrected function to handle non-8ns integers
    
    # Replace parse_time_range
    def robust_parse_time_range(range_str: str) -> Tuple[List[str], int]:
        """
        (Corrected Version)
        Parses "8-38" into (["8-16ns", "16-24ns", "24-32ns", "32-40ns"], 30000)
        """
        try:
            start_ns, end_ns = map(int, range_str.split('-'))
        except ValueError:
            print(f"Error: TIME_RANGE_NS format incorrect ('{range_str}'). Should be 'Start-End' (e.g., '8-32').")
            sys.exit(1)
        
        total_duration_ns = end_ns - start_ns
        if total_duration_ns <= 0:
            print(f"Error: End time {end_ns} must be greater than start time {start_ns}.")
            sys.exit(1)
        # (*** Fix ***) Use round to avoid floating point precision issues (e.g. 0.001 * 30000)
        total_expected_frames = int(round(total_duration_ns / TIME_PER_FRAME_NS))

        chunks = []
        current_start = start_ns
        while current_start < end_ns: # As long as start time < target end time
            current_end = current_start + 8
            chunks.append(f"{current_start}-{current_end}ns")
            current_start = current_end
            
        print(f"--- Successfully parsed time range {range_str} (Total {total_duration_ns} ns) ---")
        print(f"--- Target total frames: {total_expected_frames} frames ---")
        print(f"--- Will read from {len(chunks)} 8ns file blocks: {chunks} ---")
        
        return chunks, total_expected_frames
    
    parse_time_range = robust_parse_time_range # Global replacement

    # Replace load_all_interface_positions
    def robust_load_R(time_chunks: List[str], log_template: str, bubble_id: str, total_expected_frames: int) -> np.ndarray:
        """
        (Corrected Version)
        Load R(t) from all log.txt files until total_expected_frames is reached.
        """
        all_interface_positions = []
        # Note: Please ensure your log file uses "Interface position" in English, 
        # or update this regex to match your file format.
        regex = re.compile(r"Interface position:\s*([-\d.]+)")
        
        print("--- 1. Loading interface positions R(t) for all time blocks... ---")
        
        # Get exact start and end time from TIME_RANGE_NS
        start_ns, end_ns = map(int, TIME_RANGE_NS.split('-'))

        for time_range in time_chunks:
            if len(all_interface_positions) >= total_expected_frames:
                print("  > Target frames reached, stopping load of subsequent 8ns blocks.")
                break
                
            print(f"  > Processing 8ns block: {time_range}")
            
            for i in range(1, BLOCKS_PER_8NS_FILE + 1):
                if len(all_interface_positions) >= total_expected_frames:
                    print(f"    > Target frames {total_expected_frames} reached. Stopping load at block {i}.")
                    break

                filepath = log_template.format(
                    TIME_RANGE=time_range,
                    BUBBLE_ID=bubble_id,
                    BLOCK_NUM=i
                )
                
                if not os.path.exists(filepath):
                    # Check if we are looking for a log outside total range
                    current_ns_loaded = (len(all_interface_positions) * TIME_PER_FRAME_NS)
                    
                    if (start_ns + current_ns_loaded) >= end_ns:
                         print(f"    > Note: {filepath} not found (Target {end_ns} ns reached, this is normal).")
                         break # Stop loading subsequent blocks for this 8ns chunk
                    else:
                        # Not loaded yet, but file missing
                        print(f"    > Error: Log file not found: {filepath}")
                        print(f"    > (Data missing before reaching target {end_ns} ns)")
                        sys.exit(f"    > Error: Log file not found: {filepath}")
                
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                    matches = regex.findall(content)
                    if not matches:
                        print(f"    > Error: 'Interface position:' not found in {filepath}")
                        sys.exit(f"    > Error: 'Interface position:' not found in {filepath}")
                    
                    r_interface = float(matches[-1])
                    
                    # (*** Critical Fix ***)
                    frames_needed = total_expected_frames - len(all_interface_positions)
                    frames_to_add = min(FRAMES_PER_1NS_BLOCK, frames_needed)
                    
                    all_interface_positions.extend([r_interface] * frames_to_add)
                    
                    if frames_to_add < FRAMES_PER_1NS_BLOCK:
                        print(f"    > (Block {i}) Loaded {frames_to_add} frames (Reached total frames {total_expected_frames}).")
                    
                except Exception as e:
                    print(f"    > Error: Failed to read or parse {filepath}: {e}")
                    sys.exit(f"    > Error: Failed to read or parse {filepath}: {e}")
        
        n_frames_loaded = len(all_interface_positions)
        print(f"  > Interface position R(t) loading complete. Total frames: {n_frames_loaded}.")
        
        if n_frames_loaded != total_expected_frames:
            print(f"!!! Critical Warning: Loaded R(t) frames ({n_frames_loaded}) do not match expected frames ({total_expected_frames}).")
            print(f"!!! Please check if your log files are complete. Analyzing using the available {n_frames_loaded} frames.")
            
        if n_frames_loaded == 0:
            print("Error: No interface positions loaded. Please check paths and configuration.")
            sys.exit(1)
            
        return np.array(all_interface_positions)

    # Replace the old buggy function
    load_all_interface_positions = robust_load_R

    # Run main program
    main_calculation()