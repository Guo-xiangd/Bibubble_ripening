#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive Analysis Script (Modified):
Combines radial distribution and angular analysis, supporting averaging over multiple 8ns time blocks.

Features:
1. (Modified) Reads interface positions R for *all* 1ns sub-blocks from *multiple* 8ns log templates.
2. (Modified) Reads *merged* CIP, SIP, 2SIP data files.
3. For every ion pair (Na-Cl pair) in every frame:
   a. Calculates midpoint position (Midpoint) and its distance to origin (r_mid).
   b. Calculates angle between (Origin->Midpoint) vector and (Na->Cl) vector.
   c. Converts angle to acute angle in [0, 90] range (acute_angle).
   d. Stores (frame_idx, r_mid, acute_angle).
4. For all collected data points, finds corresponding R based on frame_idx (from total list of blocks), calculates r_prime = r_mid - R.
5. Bins data by r_prime.
6. (Modified) Calculates for each r_prime bin:
   a. Average ion pair count (Total Pairs / Total Frames) (Total Frames is sum of all 8ns blocks).
   b. Mean Angle and Standard Deviation of ion pair angles (based on merged data).
7. Plots two figures (representing average over all time periods).
8. Saves plot data to .txt files.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import re # For extracting numbers from logs
import time
from collections import defaultdict
from math import sqrt, acos, degrees

# ==============================================================================
# --- 1. User Configuration ---
# (Please modify according to your file paths and system parameters)
# ==============================================================================

# --- A. File Paths ---

# (New) List of Log File Templates
# !!! Each template corresponds to an 8ns time period !!!
# The script will read all R values in the order of this list.
LOG_FILE_TEMPLATES = [
    "data/plot/biBNBs/salt/003_ripening/ion_distribution_for_several_timeblock/1ns_block/0-8ns/results/bubble1_block{X}/log.txt",
    "data/plot/biBNBs/salt/003_ripening/ion_distribution_for_several_timeblock/1ns_block/8-16ns/results/bubble1_block{X}/log.txt",
     "data/plot/biBNBs/salt/003_ripening/ion_distribution_for_several_timeblock/1ns_block/16-24ns/results/bubble1_block{X}/log.txt",
     "data/plot/biBNBs/salt/003_ripening/ion_distribution_for_several_timeblock/1ns_block/24-32ns/results/bubble1_block{X}/log.txt",
     "data/plot/biBNBs/salt/003_ripening/ion_distribution_for_several_timeblock/1ns_block/32-40ns/results/bubble1_block{X}/log.txt",
]

# (Modified) Mode Data Files
# !!! These files must be *already merged* !!!
# !!! They should contain *all* frames corresponding to the LOG_FILE_TEMPLATES list above !!!
FILE_CONFIG = {
    'CIP': {
        'filename': "../CIP/003/Bubble1/combined.txt", # <--- Path: Merged CIP file
        'block_size': 9,
        'na_offsets': [1, 2, 3],
        'cl_offsets': [5, 6, 7]
    },
    'SIP': {
        'filename': "../SIP/003/Bubble1/combined.txt", # <--- Path: Merged SIP file
        'block_size': 16,
        'na_offsets': [1, 2, 3],
        'cl_offsets': [9, 10, 11]
    },
    '2SIP': {
        'filename': "../2SIP/003/Bubble1/combined.txt", # <--- Path: Merged 2SIP file
        'block_size': 22,
        'na_offsets': [1, 2, 3],
        'cl_offsets': [13, 14, 15]
    }
}

# --- B. System and Binning Parameters ---

FRAMES_PER_8NS_BLOCK = 8000
NUM_8NS_BLOCKS = len(LOG_FILE_TEMPLATES)

# (Modified) Total frames is now automatically calculated
TOTAL_FRAMES = FRAMES_PER_8NS_BLOCK * NUM_8NS_BLOCKS

# Frames per 1ns sub-block (for matching R values)
FRAMES_PER_BLOCK = 1000 # This should not be changed

# Binning parameters for radial distance r' (Unit: Angstrom)
R_PRIME_MIN = -15.0  # Min r' (Inside interface)
R_PRIME_MAX = 15.0   # Max r' (Outside interface)
BIN_SIZE = 1         # Bin width (Modified to 1 as per code)

# --- C. Output Settings ---
OUTPUT_DIR = "./003/combined_radial_angle_results_MERGED_B1"

# ==============================================================================
# --- 2. Core Functions ---
# ==============================================================================

def load_interface_positions(template_list, num_sub_blocks=8):
    """
    (Modified) Reads *all* interface positions R from *multiple* 8ns templates.
    """
    all_interface_positions = [] # (New) Total list for all R values
    print("Reading interface positions for all time periods...")
    
    # Match the machine-readable English key-value record.
    regex = r"(?m)^interface_position\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"

    # (New) Iterate through each 8ns time period template
    for template_idx, template in enumerate(template_list):
        print(f"--- Processing 8ns block {template_idx + 1}/{len(template_list)} (Template: ...{template[-80:]}) ---")
        
        # (Original) Iterate through 8 1ns sub-blocks in this 8ns block
        for i in range(1, num_sub_blocks + 1):
            filepath = template.format(X=i)
            if not os.path.exists(filepath):
                print(f"  > Error: Log file not found {filepath}")
                sys.exit(f"  > Error: Log file not found {filepath}")
            
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
            
                matches = re.findall(regex, content)
                if not matches:
                    print(f"  > Error: Could not find 'interface_position=' in {filepath}")
                    sys.exit(f"  > Error: Could not find 'interface_position=' in {filepath}")
            
                # Assume the last match is the one needed
                r_interface = float(matches[-1])
                
                # (New) Add R value to *total list*
                all_interface_positions.append(r_interface)
                print(f"  > Successfully read Block {i} (R={r_interface:.4f} Å) from template {template_idx+1}")
                
            except Exception as e:
                print(f"  > Error: Reading or parsing {filepath}: {e}")
                sys.exit(f"  > Error: Reading or parsing {filepath}: {e}")
    
    # (New) Check if total number of R values read is correct
    expected_num = len(template_list) * num_sub_blocks
    if len(all_interface_positions) != expected_num:
        print(f"Error: Expected {expected_num} interface positions (from {len(template_list)} templates), read {len(all_interface_positions)}.")
        sys.exit("Error: Mismatch in number of interface positions read.")
        
    print(f"\nAll interface positions read. Total {len(all_interface_positions)} (from {len(template_list)} 8ns blocks).")
    return np.array(all_interface_positions)


def calculate_acute_angle(na_pos, cl_pos):
    """
    Calculates angle between Na->Cl vector R and Origin->Midpoint vector M.
    Returns acute angle in [0, 90] and midpoint radius.
    """
    na_pos = np.array(na_pos)
    cl_pos = np.array(cl_pos)

    # 1. Vector R (Na -> Cl)
    R_vec = cl_pos - na_pos
    
    # 2. Vector M (Origin -> Midpoint)
    M_vec = (na_pos + cl_pos) / 2.0

    # 3. Check zero vectors
    R_norm = np.linalg.norm(R_vec)
    M_norm = np.linalg.norm(M_vec) # Distance from midpoint to origin

    if R_norm == 0 or M_norm == 0:
        return np.nan, M_norm

    # 4. Calculate cosine of angle
    dot_product = np.dot(R_vec, M_vec)
    cos_theta = dot_product / (R_norm * M_norm)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # 5. Calculate angle (degrees)
    theta_deg = np.degrees(np.arccos(cos_theta))

    # 6. (New Requirement) Convert to acute angle [0, 90]
    if theta_deg > 90.0:
        acute_theta_deg = 180.0 - theta_deg
    else:
        acute_theta_deg = theta_deg

    return acute_theta_deg, M_norm


def parse_files_and_collect_data(file_config, total_frames):
    """
    Parses all mode files and collects (frame_idx, midpoint_r, acute_angle).
    """
    data_by_mode = defaultdict(list)
    
    for mode, config in file_config.items():
        filename = config['filename']
        block_size = config['block_size']
        na_offsets = config['na_offsets']
        cl_offsets = config['cl_offsets']
        
        print(f"--- Processing {mode} mode (file: {filename}) ---")
        
        try:
            with open(filename, 'r') as f:
                header_line = f.readline()
                if not header_line:
                    print(f"  > Warning: File {filename} is empty.")
                    continue
                
                num_cols = len(header_line.split())
                if num_cols <= 1:
                    print(f"  > Warning: Data columns not found in {filename}.")
                    continue

                # Assume first column is TIMESTEP
                num_pairs = (num_cols - 1) // block_size
                if num_pairs == 0:
                    print(f"  > Warning: Cannot determine number of ion pairs in {filename} (block_size={block_size}).")
                    continue
                
                print(f"  > Detected {num_pairs} pairs/line.")
                
                frame_idx = 0
                for line in f:
                    if frame_idx >= total_frames:
                        break # Ensure total frames not exceeded
                        
                    parts = line.split()
                    if len(parts) < num_cols:
                        continue  # Skip empty or malformed lines

                    for i in range(num_pairs):
                        base_idx = 1 + i * block_size
                        
                        try:
                            # Check filler -1 (based on logic in angle_distri.py)
                            if float(parts[base_idx + na_offsets[0]]) == -1:
                                continue

                            na_pos = [float(parts[base_idx + offset]) for offset in na_offsets]
                            cl_pos = [float(parts[base_idx + offset]) for offset in cl_offsets]

                            # Calculate angle and midpoint radius
                            angle, mid_r = calculate_acute_angle(na_pos, cl_pos)

                            if not np.isnan(angle):
                                # Store raw data
                                data_by_mode[mode].append((frame_idx, mid_r, angle))

                        except (ValueError, IndexError):
                            # End of line or non-number encountered
                            break # Stop processing subsequent pairs in this line
                    
                    frame_idx += 1
                
                print(f"  > {mode} processing complete. Total {frame_idx} frames (expected {total_frames}), collected {len(data_by_mode[mode])} valid pairs.")
                if frame_idx < total_frames:
                    print(f"  > WARNING: File {filename} only has {frame_idx} frames, but configured TOTAL_FRAMES is {total_frames}.")
                    print(f"  > PLEASE CHECK if file {filename} is complete, and if count of LOG_FILE_TEMPLATES is correct.")
                    # Decide whether to exit
                    # sys.exit("Data file frame count mismatch!")

        except FileNotFoundError:
            print(f"  > Error: File {filename} not found.")
        except Exception as e:
            print(f"  > Error processing {filename}: {e}")

    return data_by_mode


def bin_collected_data(data_by_mode, interface_R_blocks, r_prime_bins_edges, frames_per_block):
    """
    Bins collected (frame, r, angle) data into r' = r - R bins.
    """
    num_bins = len(r_prime_bins_edges) - 1
    
    # 1. Prepare data structures
    pair_counts_per_bin = {}
    angles_per_bin = {} # Stores all angles in each bin
    
    for mode in data_by_mode.keys():
        pair_counts_per_bin[mode] = np.zeros(num_bins, dtype=int)
        # Create a list of lists, each sublist corresponds to a bin
        angles_per_bin[mode] = [[] for _ in range(num_bins)]

    print("\n--- Binning by r' = r - R ---")

    # 2. Iterate through all collected data points
    for mode, data_list in data_by_mode.items():
        print(f"  > Binning {mode} (Total {len(data_list)} points)...")
        for (frame_idx, r_mid, acute_angle) in data_list:
            
            # Determine current R
            block_idx = frame_idx // frames_per_block # Key: e.g. 8500 // 1000 = 8
            if block_idx >= len(interface_R_blocks):
                # Should not happen, implies R list is too short
                print(f"  > Critical Error: block_idx {block_idx} for frame_idx {frame_idx} exceeds R list range (length {len(interface_R_blocks)})")
                block_idx = len(interface_R_blocks) - 1 # Fallback
            
            R = interface_R_blocks[block_idx] # Key: Get R[8] from total list
            
            # Calculate r'
            r_prime = r_mid - R
            
            # Find bin for r'
            bin_index = np.digitize(r_prime, r_prime_bins_edges) - 1
            
            # Check if in valid bin range [0, num_bins-1]
            if 0 <= bin_index < num_bins:
                # 1. Increment bin count
                pair_counts_per_bin[mode][bin_index] += 1
                # 2. Add angle to bin list
                angles_per_bin[mode][bin_index].append(acute_angle)
        print(f"  > {mode} binning complete.")

    return pair_counts_per_bin, angles_per_bin

def calculate_bin_volumes(r_prime_bins_edges, avg_R_interface):
    """
    Calculates precise spherical shell volume V = 4/3 * pi * (r_outer^3 - r_inner^3) for each r' bin.
    Uses avg_R_interface to convert r' back to absolute radius r.
    """
    print(f"\n--- Calculating Shell Volumes (Based on R_avg = {avg_R_interface:.4f} Å) ---")
    
    # 1. Convert r' bin edges to absolute r edges
    #    r = r' + R_avg
    r_inner_abs = r_prime_bins_edges[:-1] + avg_R_interface
    r_outer_abs = r_prime_bins_edges[1:] + avg_R_interface
    
    # 2. (Critical) Handle r < 0 case
    #    If a bin boundary is inside origin (e.g. r' = -15, R_avg = 12),
    #    its physical radius must be clipped to 0.
    r_inner_abs = np.clip(r_inner_abs, 0, None)
    r_outer_abs = np.clip(r_outer_abs, 0, None)
    
    # 3. Calculate volume V = 4/3 * pi * (r_outer^3 - r_inner^3)
    volumes = (4.0 / 3.0) * np.pi * (np.power(r_outer_abs, 3) - np.power(r_inner_abs, 3))
    
    # 4. Ensure volume is not negative due to float precision
    volumes = np.where(volumes < 0, 0, volumes)
    
    return volumes

def process_binned_angles(angles_per_bin, r_prime_bins_centers):
    """
    Calculates mean angle and standard deviation for each bin.
    """
    num_bins = len(r_prime_bins_centers)
    mean_angles = {}
    std_angles = {}
    
    for mode, bin_lists in angles_per_bin.items():
        mean_angles[mode] = np.full(num_bins, np.nan)
        std_angles[mode] = np.full(num_bins, np.nan)
        
        for i in range(num_bins):
            if len(bin_lists[i]) > 0:
                mean_angles[mode][i] = np.mean(bin_lists[i])
                std_angles[mode][i] = np.std(bin_lists[i])
                
    return mean_angles, std_angles


def plot_distribution(plot_data, bins_centers, title, ylabel, filename_base, avg_R_interface):
    """
    Plot and save distribution chart (Generic)
    """
    print(f"Generating chart: {title}")
    
    plt.figure(figsize=(10, 6))
    
    # Plot interface position (Black, Bold, Dashed)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=2.5, 
                label=f'Avg. Interface (R={avg_R_interface:.2f} Å)')
    
    colors = {"CIP": "red", "SIP": "blue", "2SIP": "green"}
    linestyles = {"CIP": "--", "SIP": ":", "2SIP": "-."}

    # Plot data
    for label, data in plot_data.items():
        plt.plot(bins_centers, data, 
                 label=label, 
                 color=colors.get(label, 'grey'), 
                 linestyle=linestyles.get(label, '-'),
                 linewidth=2)

    # Set chart properties
    plt.xlabel("Corrected Radial Distance (r' = r - R) [Å]", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.xlim(R_PRIME_MIN, R_PRIME_MAX)
    
    # Automatically set Y axis lower limit to 0 (if appropriate)
    all_values = np.concatenate([d[np.isfinite(d)] for d in plot_data.values()])
    if len(all_values) > 0:
        min_val = np.min(all_values)
        max_val = np.max(all_values)
        if "Angle" in ylabel:
             plt.ylim(0, 100) # Angle range [0, 90]
        elif min_val >= 0:
            plt.ylim(0, max_val * 1.15)
    
    # Save chart
    plt.savefig(f"{filename_base}.png", dpi=300)
    plt.close()
    print(f"  > Chart saved: {filename_base}.png")


def save_data_file(filename, bins_centers, mean_data_dict, std_data_dict=None):
    """
    Save data to .txt file.
    """
    try:
        header_parts = ["r_prime_center"]
        data_columns = [bins_centers]
        
        for mode in sorted(mean_data_dict.keys()):
            header_parts.append(f"{mode}_mean")
            data_columns.append(mean_data_dict[mode])
            
            if std_data_dict and mode in std_data_dict:
                header_parts.append(f"{mode}_std")
                data_columns.append(std_data_dict[mode])

        header = " ".join(header_parts)
        data_to_save = np.vstack(data_columns).T
        
        np.savetxt(filename, data_to_save, header=header, fmt="%.8e")
        print(f"  > Data saved: {filename}")
        
    except Exception as e:
        print(f"  > Warning: Failed to save data file {filename}: {e}")


# ==============================================================================
# --- 3. Main Execution Logic ---
# ==============================================================================

def main():
    start_time = time.time()
    # (Modified) Use new output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Interface Position R
    # (Modified) 'num_blocks' is now the *total* number of 1ns sub-blocks (e.g. 16, 24...)
    num_blocks = TOTAL_FRAMES // FRAMES_PER_BLOCK
    
    # (New) Sanity check for configuration match
    expected_sub_blocks = NUM_8NS_BLOCKS * (FRAMES_PER_8NS_BLOCK // FRAMES_PER_BLOCK)
    if num_blocks != expected_sub_blocks:
        print("!!! Critical Configuration Error !!!")
        print(f"LOG_FILE_TEMPLATES list contains {NUM_8NS_BLOCKS} templates.")
        print(f"Each 8ns block has {FRAMES_PER_8NS_BLOCK} frames, each 1ns sub-block has {FRAMES_PER_BLOCK} frames.")
        print(f"Therefore, script expects {expected_sub_blocks} total 1ns sub-blocks (R values).")
        print(f"However, TOTAL_FRAMES ({TOTAL_FRAMES}) / FRAMES_PER_BLOCK ({FRAMES_PER_BLOCK}) = {num_blocks}.")
        print("Please check FRAMES_PER_8NS_BLOCK and FRAMES_PER_BLOCK values.")
        sys.exit("Configuration mismatch!")

    # (Modified) Call modified function with template *list*
    # `num_sub_blocks=8` means each template contains 8 {X}
    interface_R_blocks = load_interface_positions(LOG_FILE_TEMPLATES, num_sub_blocks=8)
    
    # (Modified) avg_R_interface is now the average of *all* R values
    avg_R_interface = np.mean(interface_R_blocks)
    print(f"\nTotal Average Interface Position R_avg = {avg_R_interface:.4f} Å (Based on {len(interface_R_blocks)} 1ns blocks)")
    
    # 2. Define binning for r' (No change)
    num_bins = int((R_PRIME_MAX - R_PRIME_MIN) / BIN_SIZE)
    r_prime_bins_edges = np.linspace(R_PRIME_MIN, R_PRIME_MAX, num_bins + 1)
    r_prime_bins_centers = (r_prime_bins_edges[:-1] + r_prime_bins_edges[1:]) / 2

    # 2.5. Calculate volume for each bin
    bin_volumes = calculate_bin_volumes(r_prime_bins_edges, avg_R_interface)
    # (Prepare for division, prevent division by zero)
    volumes_with_fallback = np.where(bin_volumes == 0, np.nan, bin_volumes)
    
    # 3. Parse all files (No change)
    # (Runs until TOTAL_FRAMES, handling merged large files)
    data_by_mode = parse_files_and_collect_data(FILE_CONFIG, TOTAL_FRAMES)
    
    # 4. Bin by r' = r - R (No change)
    # (Automatically matches 0-N frame_idx to 0-M block_idx and R value)
    pair_counts_per_bin, angles_per_bin = bin_collected_data(
        data_by_mode, 
        interface_R_blocks, 
        r_prime_bins_edges, 
        FRAMES_PER_BLOCK
    )

    # 5. Process Plotting and Saving
    
    # --- Figure 1: Ion Pair Count Distribution ---
    # (Modified) Normalization uses larger TOTAL_FRAMES, which *is* the average
    avg_pair_counts_per_frame = {
        mode: counts / TOTAL_FRAMES
        for mode, counts in pair_counts_per_bin.items()
    }
    
    number_density = {
        mode: avg_pair_counts_per_frame[mode] / volumes_with_fallback
        for mode in avg_pair_counts_per_frame
    }
    plot_distribution(
        plot_data=number_density,
        bins_centers=r_prime_bins_centers,
        title="Average Ion Pair Count Distribution (Merged, relative to interface)",
        ylabel="Average Number Density (pairs/frame/Å³)",
        filename_base=os.path.join(OUTPUT_DIR, "pair_count_radial_dist_MERGED"),
        avg_R_interface=avg_R_interface
    )
    
    save_data_file(
        filename=os.path.join(OUTPUT_DIR, "pair_count_radial_dist_data_MERGED.txt"),
        bins_centers=r_prime_bins_centers,
        mean_data_dict=number_density,
        std_data_dict=None
    )

    # --- Figure 2: Ion Pair Angle Distribution ---
    # (Modified) Automatically average all data points in merged bins
    mean_angles, std_angles = process_binned_angles(angles_per_bin, r_prime_bins_centers)

    plot_distribution(
        plot_data=mean_angles,
        bins_centers=r_prime_bins_centers,
        title="Average Pair Angle Distribution (Merged, relative to interface)",
        ylabel="Average Acute Angle [0-90 degrees]",
        filename_base=os.path.join(OUTPUT_DIR, "pair_angle_radial_dist_MERGED"),
        avg_R_interface=avg_R_interface
    )
    
    save_data_file(
        filename=os.path.join(OUTPUT_DIR, "pair_angle_radial_dist_data_MERGED.txt"),
        bins_centers=r_prime_bins_centers,
        mean_data_dict=mean_angles,
        std_data_dict=std_angles
    )

    end_time = time.time()
    print(f"\nAnalysis Complete. Total Time: {end_time - start_time:.2f} seconds.")
    print(f"All output files saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    # Check dependencies
    try:
        import numpy
        import matplotlib
    except ImportError:
        print("Error: Missing dependencies (numpy or matplotlib).")
        print("Please install using 'pip install numpy matplotlib'.")
        sys.exit(1)
        
    main()
