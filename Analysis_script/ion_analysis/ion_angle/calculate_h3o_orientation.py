#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calculate the radial orientation distribution of hydronium (H3O+) ions relative to the bubble interface.

This script works by parsing pre-calculated ion position files and interface position log files.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict
import glob

# ==================== Utility Functions ====================

def calculate_angle_between_vectors(v1, v2):
    """Calculate the angle between two vectors (0-180 degrees)."""
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm == 0 or v2_norm == 0:
        return np.nan
    
    cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return angle

def parse_log_file(log_path):
    """
    Parse the interface position from a single log file.
    Uses regex to find 'Interface Position: X.XXXX'
    """
    # Updated regex to match English
    pattern = re.compile(r"Interface Position:\s*(\d+\.\d+)")
    
    if not os.path.exists(log_path):
        print(f"Warning: Log file does not exist, skipping: {log_path}")
        return None

    with open(log_path, 'r') as f:
        content = f.read()
        match = pattern.search(content)
        if match:
            return float(match.group(1))
        
    print(f"Warning: Could not find 'Interface Position:' in {log_path}")
    return None

def determine_files_to_process(start_ns, end_ns, time_step_ns):
    """
    Determine which time range files need to be processed based on the input start and end times.
    """
    start_file_index = int(np.floor(start_ns / time_step_ns))
    end_file_index = int(np.ceil(end_ns / time_step_ns)) - 1
    
    if start_file_index > end_file_index:
         end_file_index = start_file_index

    time_ranges = []
    for i in range(start_file_index, end_file_index + 1):
        range_start = i * time_step_ns
        range_end = (i + 1) * time_step_ns
        time_ranges.append(f"{range_start:.0f}-{range_end:.0f}ns")
        
    return time_ranges

def load_interface_map(time_ranges, log_path_template, bubble_name, block_step_ns):
    """
    Load all relevant log files and build a {global_ns: interface_pos} mapping table.
    """
    interface_map = {}
    blocks_per_file = int(args.time_step_ns / block_step_ns)
    
    print("--- Loading Interface Position Logs ---")
    
    for time_range in time_ranges:
        file_start_ns = float(time_range.split('-')[0])
        
        for i in range(blocks_per_file):
            block_index = i + 1
            current_block_start_ns = file_start_ns + (i * block_step_ns)
            
            log_path = log_path_template.format(
                time_range=time_range,
                bubble_name=bubble_name,
                block_index=block_index
            )
            
            interface_pos = parse_log_file(log_path)
            if interface_pos is not None:
                interface_map[current_block_start_ns] = interface_pos
                print(f"  {log_path} -> Mapped time {current_block_start_ns:.1f} ns = {interface_pos:.3f} Å")
                
    print(f"Successfully loaded {len(interface_map)} interface position data points.")
    print("---------------------------------")
    return interface_map

def process_ion_file(ion_file_path, interface_map, all_data_by_bin, dist_bin_edges,
                     dist_bin_centers, start_ns, end_ns, block_step_ns, time_range):
    """
    Process a single H3O+ ion file.
    """
    
    file_start_ns = float(time_range.split('-')[0])
    row_time_step = 0.001 # 1 row = 0.001 ns
    
    print(f"\nProcessing ion file: {ion_file_path}")
    
    if not os.path.exists(ion_file_path):
        print(f"Error: Ion file does not exist: {ion_file_path}")
        return 0

    processed_ions = 0
    with open(ion_file_path, 'r') as f:
        lines = f.readlines()
        
    for row_index, line in enumerate(lines):
        if line.startswith("#") or line.strip() == "":
            continue
            
        current_time_ns = file_start_ns + (row_index * row_time_step)
        
        if not (start_ns <= current_time_ns < end_ns):
            continue
            
        block_time_key = np.floor(current_time_ns / block_step_ns) * block_step_ns
        interface_pos = interface_map.get(block_time_key)
        
        if interface_pos is None:
            continue
            
        parts = line.split()
        if len(parts) < 18: # N_ions, MD_step and at least 1 H3O+ (16 columns)
            continue
            
        try:
            n_ions = int(parts[0])
            
            for i in range(n_ions):
                # ========================================================
                # [Modify] H3O+ format parsing (16 columns)
                # ========================================================
                # Each H3O+ ion has 16 columns: O(4) + H1(4) + H2(4) + H3(4)
                offset = 2 + i * 16
                if offset + 15 >= len(parts): # Check if enough columns exist
                    break # Incomplete row data

                # O position (columns 2, 3, 4 at offset)
                o_pos = np.array([float(parts[offset+1]), float(parts[offset+2]), float(parts[offset+3])])
                
                # H1 position (columns 6, 7, 8 at offset)
                h1_pos = np.array([float(parts[offset+5]), float(parts[offset+6]), float(parts[offset+7])])
                
                # H2 position (columns 10, 11, 12 at offset)
                h2_pos = np.array([float(parts[offset+9]), float(parts[offset+10]), float(parts[offset+11])])
                
                # H3 position (columns 14, 15, 16 at offset)
                h3_pos = np.array([float(parts[offset+13]), float(parts[offset+14]), float(parts[offset+15])])

                # ========================================================
                # [Modify] H3O+ Angle Definition
                # ========================================================
                
                # 1. Calculate geometric center of three hydrogens
                h_centroid = (h1_pos + h2_pos + h3_pos) / 3.0
                
                # 2. Calculate relative distance (r) - based on O position
                dist_to_center = np.linalg.norm(o_pos)
                relative_dist = interface_pos - dist_to_center
                
                # 3. Calculate orientation angle
                # Vector 1: H_centroid -> O
                vec_h_centroid_o = o_pos - h_centroid
                # Vector 2: O -> Center (0,0,0)
                vec_o_center = -o_pos 
                
                angle = calculate_angle_between_vectors(vec_h_centroid_o, vec_o_center)
                
                # ========================================================
                
                if np.isnan(angle):
                    continue
                    
                # 4. Store in bin
                dist_bin_index = np.digitize(relative_dist, dist_bin_edges) - 1
                
                if 0 <= dist_bin_index < len(dist_bin_centers):
                    bin_key = dist_bin_centers[dist_bin_index]
                    all_data_by_bin[bin_key].append(angle)
                    processed_ions += 1
                    
        except ValueError as e:
            print(f"Warning: Failed to parse row (Line {row_index+1}): {e}")
            continue
            
    print(f"File {time_range} processed, collected {processed_ions} ion data points.")
    return processed_ions

# ==================== Main Program ====================

def main():
    parser = argparse.ArgumentParser(description="Calculate H3O+ Ion Radial Orientation Distribution")
    parser.add_argument("--start_ns", type=float, required=True, help="Total analysis start time (e.g., 14.0)")
    parser.add_argument("--end_ns", type=float, required=True, help="Total analysis end time (e.g., 30.0)")
    parser.add_argument("--bubble_name", type=str, required=True, help="Bubble name (e.g., bubble2)")
    
    parser.add_argument("--ion_path_template", type=str, required=True, 
                        help="H3O+ ion file path template, using {time_range} and {bubble_name} placeholders. "
                             "Example: '/.../findion/{bubble_name}/h3o_..._{bubble_name}_{time_range}.txt'")
    
    parser.add_argument("--log_path_template", type=str, required=True, 
                        help="Log file path template, using {time_range}, {bubble_name}, {block_index} placeholders. "
                             "Example: '/.../{time_range}/results/{bubble_name}_block{block_index}'")
    
    parser.add_argument("--time_step_ns", type=float, default=8.0, help="Time span per ion file (e.g., 8.0)")
    parser.add_argument("--block_step_ns", type=float, default=1.0, help="Time span per Log file (e.g., 1.0)")
    
    parser.add_argument("--dist_min_r", type=float, default=-10.0, help="Radial distribution min distance")
    parser.add_argument("--dist_max_r", type=float, default=10.0, help="Radial distribution max distance")
    parser.add_argument("--dist_bin", type=float, default=0.2, help="Radial distribution bin width")
    
    parser.add_argument("--output_prefix", type=str, default="h3o_orientation_results", 
                        help="Output file prefix (e.g., 'bubble2_14-30ns')")

    global args
    args = parser.parse_args()
    
    # 1. Determine files to process
    time_ranges = determine_files_to_process(args.start_ns, args.end_ns, args.time_step_ns)
    if not time_ranges:
        print(f"Error: No matching files found in range {args.start_ns}-{args.end_ns} ns.")
        return
        
    print(f"Analysis Time Range: {args.start_ns} ns to {args.end_ns} ns")
    print(f"Will process the following file periods: {time_ranges}")
    
    # 2. Load all log files, build interface position map
    try:
        interface_map = load_interface_map(time_ranges, args.log_path_template, args.bubble_name, args.block_step_ns)
    except Exception as e:
        print(f"Error loading Log files: {e}")
        return

    if not interface_map:
        print("Error: Failed to load any interface position data. Please check log_path_template and file content.")
        return
        
    # 3. Initialize Bins
    dist_bin_edges = np.arange(args.dist_min_r, args.dist_max_r + args.dist_bin, args.dist_bin)
    dist_bin_centers = dist_bin_edges[:-1] + args.dist_bin / 2
    
    all_data_by_bin = defaultdict(list)
    for bin_center in dist_bin_centers:
        all_data_by_bin[bin_center]
        
    # 4. Loop to process all ion files
    total_ions = 0
    for time_range in time_ranges:
        ion_file_path = args.ion_path_template.format(
            time_range=time_range,
            bubble_name=args.bubble_name
        )
        
        total_ions += process_ion_file(
            ion_file_path, interface_map, all_data_by_bin,
            dist_bin_edges, dist_bin_centers,
            args.start_ns, args.end_ns, args.block_step_ns, time_range
        )
        
    print(f"\nTotal processing complete. Collected {total_ions} H3O+ ion data points.")
    if total_ions == 0:
        print("No data collected, cannot generate results. Please check your time range and file paths.")
        return
        
    # 5. Calculate statistics
    print("Calculating statistics...")
    results = []
    for bin_center in dist_bin_centers:
        angles_list = all_data_by_bin[bin_center]
        counts = len(angles_list)
        
        if counts > 0:
            mean = np.mean(angles_list)
            std = np.std(angles_list)
        else:
            mean = np.nan
            std = np.nan
            
        results.append((bin_center, mean, std, counts))
        
    results_array = np.array(results)
    
    # 6. Save data text file
    txt_filename = f"{args.output_prefix}_mean_angle_vs_dist.txt"
    header = (f"# H3O+ Mean Orientation vs. Distance ({args.start_ns}-{args.end_ns} ns)\n"
              f"# Bubble: {args.bubble_name}\n"
              f"# Angle: (H_centroid->O vector) vs (O->Center vector)\n"
              f"# r > 0 is inside bubble, r < 0 is outside bubble\n"
              f"# Dist_from_Interface(A)\tMean_Angle(deg)\tStd_Dev_Angle(deg)\tRaw_Counts")
              
    np.savetxt(txt_filename, results_array, header=header, fmt="%.6f", comments='')
    print(f"Data file saved: {txt_filename}")

    # 7. Plot curve
    png_filename = f"{args.output_prefix}_mean_angle_vs_dist.png"
    plt.figure(figsize=(10, 6))
    
    valid_data = results_array[~np.isnan(results_array[:, 1])]
    
    plt.errorbar(valid_data[:, 0], valid_data[:, 1], yerr=valid_data[:, 2],
                 marker='o', markersize=4, linestyle='-', capsize=5,
                 label=f"{args.bubble_name} ({args.start_ns}-{args.end_ns} ns)")
                 
    plt.axvline(0, color='k', linestyle='--', label="Interface (r=0)")
    plt.xlabel("Distance from Interface (Å)")
    plt.ylabel("Mean H3O⁺ Orientation Angle (deg)")
    plt.title(f"H3O⁺ Mean Orientation vs. Distance ({args.bubble_name})")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.ylim(0, 180)
    
    plt.savefig(png_filename, dpi=300)
    print(f"Chart file saved: {png_filename}")

if __name__ == "__main__":
    main()