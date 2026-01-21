#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import re
from collections import defaultdict

# ==================== Parameters & Constants ====================
# Atom Type Mapping (As per description)
# 1: H, 2: O, 3: N, 4: Na, 5: Cl
TYPE_H = '1'
TYPE_O = '2'

# ==================== Utility Functions ====================

def parse_interface_log(log_path):
    """Parse log file to get interface position."""
    if not os.path.exists(log_path):
        return None
    
    # Updated regex pattern: "Interface Position: X.XX"
    pattern = re.compile(r"Interface Position:\s*(\d+\.\d+)")
    try:
        with open(log_path, 'r') as f:
            content = f.read()
            match = pattern.search(content)
            if match:
                return float(match.group(1))
    except Exception as e:
        print(f"Warning: Log Parsing Error {log_path}: {e}")
    return None

def get_interface_pos(global_time_ns, file_start_ns, log_template):
    """
    Calculate local block index based on global time and read interface position.
    Assumes each block corresponds to 1ns.
    """
    # Calculate time difference relative to current file start
    rel_time = global_time_ns - file_start_ns
    # Calculate block index (1-based), e.g., 0.5ns -> block 1, 1.5ns -> block 2
    block_idx = int(rel_time) + 1
    
    # Clamp block_idx between 1-8 (Assuming 8ns file has 8 blocks)
    block_idx = max(1, min(8, block_idx))
    
    log_path = log_template.format(block_idx=block_idx)
    return parse_interface_log(log_path)

def calculate_angle(v1, v2):
    """Calculate angle between two vectors (in degrees)."""
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    cos_theta = dot / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def calc_hb_for_frame(u, oh_positions, box, r_cut=3.5, angle_cut=140):
    """
    Calculate HB count for a single frame.
    
        """
    n_hb_as_donor = 0
    n_hb_as_acceptor = 0
    
    all_oxygens = u.select_atoms(f"type {TYPE_O}")
    
    # ================= Case 1: OH- as Donor =================
    # OH- (O_d - H_d) ... O_a (Any other oxygen)
    
    for oh in oh_positions:
        o_donor_pos = oh['O']
        h_donor_pos = oh['H']
        
        # 1. Find all O atoms within r_cut of O_donor
        potential_acceptors = all_oxygens.select_atoms(f"point {o_donor_pos[0]} {o_donor_pos[1]} {o_donor_pos[2]} {r_cut}")
        
        for o_acc in potential_acceptors:
            # Exclude self
            if np.linalg.norm(o_acc.position - o_donor_pos) < 0.1: 
                continue
                
            # Vector 1: O_donor -> H_donor (Covalent)
            # Vector 2: O_donor -> O_acceptor (HB direction approx)
            # Rigorous Definition: Angle (O_d - H ... O_a) > 140
            
            vec_h_o_donor = o_donor_pos - h_donor_pos # Vector from H to O_donor
            vec_h_o_acc   = o_acc.position - h_donor_pos # Vector from H to O_acceptor
            
            # Calculate angle at H
            # Note: calculate_angle returns the angle. If O-H...O is linear, H-O_d and H-O_a angle should be ~180.
            # Criterion: angle > 140.
            
            ang_val = calculate_angle(vec_h_o_donor, vec_h_o_acc)
            
            if ang_val > angle_cut:
                n_hb_as_donor += 1

    # ================= Case 2: OH- as Acceptor =================
    # O_d (Any water/OH) - H_d ... O_a (Oxygen of OH-)
    
    for oh in oh_positions:
        o_acc_pos = oh['O']
        
        # 1. Find all O atoms within r_cut of OH- O (Potential Donors)
        potential_donors = all_oxygens.select_atoms(f"point {o_acc_pos[0]} {o_acc_pos[1]} {o_acc_pos[2]} {r_cut}")
        
        for o_donor in potential_donors:
            if np.linalg.norm(o_donor.position - o_acc_pos) < 0.1:
                continue
            
            # 2. Find H atoms covalently bonded to this Donor O (dist < 1.2A)
            bonded_hs = u.select_atoms(f"type {TYPE_H} and point {o_donor.position[0]} {o_donor.position[1]} {o_donor.position[2]} 1.2")
            
            for h_atom in bonded_hs:
                vec_h_o_donor = o_donor.position - h_atom.position
                vec_h_o_acc   = o_acc_pos - h_atom.position
                
                ang_val = calculate_angle(vec_h_o_donor, vec_h_o_acc)
                
                if ang_val > angle_cut:
                    n_hb_as_acceptor += 1
                    
    return n_hb_as_donor, n_hb_as_acceptor

def process_system(system_name, time_range, file_config, hb_params):
    """Core loop for processing a single system."""
    print(f"\n========== Processing System: {system_name} ==========")
    
    start_ns, end_ns = time_range
    print(f"Time Range: {start_ns} - {end_ns} ns")
    
    traj_template = file_config['traj']
    ion_template = file_config['ion']
    log_template = file_config['log']
    
    # Determine file segments (8ns per file)
    start_file_idx = int(start_ns // 8)
    end_file_idx = int(np.ceil(end_ns / 8)) - 1
    if end_file_idx < start_file_idx: end_file_idx = start_file_idx
    
    file_indices = range(start_file_idx, end_file_idx + 1)
    
    frame_results = [] # List of [time, donor_hb_norm, acceptor_hb_norm, count_oh]
    
    for f_idx in file_indices:
        f_start_time = f_idx * 8.0
        f_end_time = (f_idx + 1) * 8.0
        time_str = f"{int(f_start_time)}-{int(f_end_time)}ns"
        
        traj_path = traj_template.format(time_range=time_str)
        ion_path = ion_template.format(time_range=time_str)
        
        if not os.path.exists(traj_path) or not os.path.exists(ion_path):
            print(f"File missing, skipping: {time_str} (Path: {traj_path})")
            continue
            
        print(f"Reading file segment: {time_str}")
        
        try:
            u = mda.Universe(traj_path, format='LAMMPSDUMP', atom_style='id type x y z')
        except Exception as e:
            print(f"MDAnalysis Read Failed: {e}")
            continue
            
        with open(ion_path, 'r') as f:
            ion_lines = f.readlines()
            
        header_skipped = False
        traj_iter = u.trajectory.__iter__()
        
        for line_idx, line in enumerate(ion_lines):
            if not header_skipped:
                header_skipped = True
                continue
            
            parts = line.strip().split()
            if not parts: continue
            
            # Calculate current frame time
            frame_local_idx = line_idx - 1
            current_time = f_start_time + frame_local_idx * 0.001
            
            # Check if time is within requested range
            if current_time < start_ns or current_time >= end_ns:
                try:
                    next(traj_iter)
                except StopIteration:
                    break
                continue
            
            try:
                ts = next(traj_iter)
            except StopIteration:
                break
                
            # Get interface position
            current_log_template = log_template.replace("{time_range}", time_str)
            interface_pos = get_interface_pos(current_time, f_start_time, current_log_template)
            
            if interface_pos is None:
                continue
                
            try:
                n_ions = int(parts[0])
                valid_oh_list = []
                
                for i in range(n_ions):
                    base = 2 + i * 8
                    if base + 7 >= len(parts): break
                    
                    ox, oy, oz = float(parts[base+1]), float(parts[base+2]), float(parts[base+3])
                    hx, hy, hz = float(parts[base+5]), float(parts[base+6]), float(parts[base+7])
                    
                    # Interface Screening: Rel Dist = Interface - Ion Center
                    # Range [-2.5, 2.5]
                    dist_to_center = np.sqrt(ox**2 + oy**2 + oz**2)
                    rel_dist = interface_pos - dist_to_center
                    
                    if -2.5 <= rel_dist <= 2.5:
                        valid_oh_list.append({
                            'O': np.array([ox, oy, oz]),
                            'H': np.array([hx, hy, hz])
                        })
                
                n_valid_oh = len(valid_oh_list)
                if n_valid_oh == 0:
                    frame_results.append([current_time, 0.0, 0.0, 0])
                    continue
                
                # Calculate HB
                n_donor, n_acceptor = calc_hb_for_frame(u, valid_oh_list, ts.dimensions, 
                                                        r_cut=hb_params['r_cut'], 
                                                        angle_cut=hb_params['a_cut'])
                
                avg_donor = n_donor / n_valid_oh
                avg_acceptor = n_acceptor / n_valid_oh
                
                frame_results.append([current_time, avg_donor, avg_acceptor, n_valid_oh])
                
                if frame_local_idx % 500 == 0:
                    print(f"  Time: {current_time:.3f}ns, Region OH: {n_valid_oh}, AvgDonor: {avg_donor:.2f}")
                    
            except ValueError as e:
                print(f"  Parsing Error at time {current_time}: {e}")
                continue

    return np.array(frame_results)

def main():
    parser = argparse.ArgumentParser(description="Compare OH- HB Network with/without Na+")
    parser.add_argument("--save_path", default="./hb_results")
    
    # --- Specify time periods for two systems ---
    parser.add_argument("--nona_start", type=float, required=True, help="No Na System Start (ns)")
    parser.add_argument("--nona_end", type=float, required=True, help="No Na System End (ns)")
    
    parser.add_argument("--na_start", type=float, required=True, help="With Na System Start (ns)")
    parser.add_argument("--na_end", type=float, required=True, help="With Na System End (ns)")
    
    # Path templates for No Na
    parser.add_argument("--nona_traj", required=True)
    parser.add_argument("--nona_ion", required=True)
    parser.add_argument("--nona_log", required=True)
    
    # Path templates for With Na
    parser.add_argument("--na_traj", required=True)
    parser.add_argument("--na_ion", required=True)
    parser.add_argument("--na_log", required=True)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    hb_params = {'r_cut': 3.5, 'a_cut': 140}
    
    # Configure two systems with respective time ranges
    systems = {
        "Without_Na": {
            "traj": args.nona_traj,
            "ion": args.nona_ion,
            "log": args.nona_log,
            "range": (args.nona_start, args.nona_end)
        },
        "With_Na": {
            "traj": args.na_traj,
            "ion": args.na_ion,
            "log": args.na_log,
            "range": (args.na_start, args.na_end)
        }
    }
    
    results_summary = {}
    
    # Loop processing
    for sys_name, config in systems.items():
        data = process_system(sys_name, config["range"], config, hb_params)
        
        if len(data) > 0:
            # Save raw data
            txt_name = os.path.join(args.save_path, f"{sys_name}_hb_timeseries.txt")
            header = (f"Analysis Time Range: {config['range'][0]}-{config['range'][1]} ns\n"
                      "Time(ns)\tHB_Donor_Avg\tHB_Acceptor_Avg\tCount_OH_in_Region")
            np.savetxt(txt_name, data, header=header, fmt="%.4f")
            
            # Calculate stats
            mean_donor = np.mean(data[:, 1])
            std_donor = np.std(data[:, 1])
            mean_acc = np.mean(data[:, 2])
            std_acc = np.std(data[:, 2])
            
            results_summary[sys_name] = {
                'donor': (mean_donor, std_donor),
                'acceptor': (mean_acc, std_acc),
                'range_str': f"{config['range'][0]}-{config['range'][1]}ns"
            }
            print(f"Result {sys_name}: Donor={mean_donor:.3f}+-{std_donor:.3f}, Acc={mean_acc:.3f}+-{std_acc:.3f}")
        else:
            print(f"Result {sys_name}: No data found.")
            results_summary[sys_name] = None

    # ==================== Plotting ====================
    if results_summary["Without_Na"] and results_summary["With_Na"]:
        print("\nPlotting comparison...")
        
        labels = ['As Donor', 'As Acceptor']
        
        # Without Na Data
        nona_info = results_summary['Without_Na']
        means_nona = [nona_info['donor'][0], nona_info['acceptor'][0]]
        errs_nona = [nona_info['donor'][1], nona_info['acceptor'][1]]
        label_nona = f"Without Na ({nona_info['range_str']})"
        
        # With Na Data
        na_info = results_summary['With_Na']
        means_na = [na_info['donor'][0], na_info['acceptor'][0]]
        errs_na = [na_info['donor'][1], na_info['acceptor'][1]]
        label_na = f"With Na ({na_info['range_str']})"
        
        x = np.arange(len(labels))  
        width = 0.35  
        
        fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
        
        rects1 = ax.bar(x - width/2, means_nona, width, yerr=errs_nona, label=label_nona, capsize=5, color='#2BBBD8', alpha=0.8)
        rects2 = ax.bar(x + width/2, means_na, width, yerr=errs_na, label=label_na, capsize=5, color='#E76F51', alpha=0.8)
        
        ax.set_ylabel('Average HB Count per OH⁻')
        ax.set_title(f'OH⁻ Hydrogen Bond Network in Interface [-2.5, 2.5] Å\nComparison of Different Time Intervals')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_path, "comparison_hb_network.png"))
        print("Plotting complete.")

if __name__ == "__main__":
    main()