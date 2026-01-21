#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance
import re

# ==================== Parameters & Constants ====================
# Please verify atom types against your LAMMPS data file
TYPE_H = '1' 
TYPE_O = '2' 

# ==================== Utility Functions ====================

def parse_interface_log(log_path):
    if not os.path.exists(log_path): return None
    # Updated regex to match English log format
    pattern = re.compile(r"Interface Position:\s*(\d+\.\d+)")
    try:
        with open(log_path, 'r') as f:
            content = f.read()
            match = pattern.search(content)
            if match: return float(match.group(1))
    except: pass
    return None

def get_interface_pos(global_time_ns, file_start_ns, log_template):
    rel_time = global_time_ns - file_start_ns
    block_idx = int(rel_time) + 1
    block_idx = max(1, min(8, block_idx))
    log_path = log_template.format(block_idx=block_idx)
    return parse_interface_log(log_path)

def calculate_angle_array(vec1, vec2):
    norm1 = np.linalg.norm(vec1, axis=1)
    norm2 = np.linalg.norm(vec2, axis=1)
    dot = np.sum(vec1 * vec2, axis=1)
    
    mask = (norm1 > 0) & (norm2 > 0)
    angles = np.zeros(len(dot))
    
    cos_theta = dot[mask] / (norm1[mask] * norm2[mask])
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angles[mask] = np.degrees(np.arccos(cos_theta))
    return angles

def calc_water_hb_only(u, water_oxygens, shell_mask_water, r_cut=3.5, angle_cut=140):
    """
    Calculate hydrogen bonds between water molecules only (Water-Water).
    """
    water_O_coords = water_oxygens.positions
    all_H_atoms = u.select_atoms(f"type {TYPE_H}")
    all_H_coords = all_H_atoms.positions
    
    # [Step 1] Build global water O -> H topology
    pairs_wh = capped_distance(water_O_coords, all_H_coords, max_cutoff=1.2, return_distances=False)
    if len(pairs_wh) > 0:
        order = np.argsort(pairs_wh[:, 0])
        pairs_wh = pairs_wh[order]
        w_o_indices = pairs_wh[:, 0]
        w_h_indices = pairs_wh[:, 1]
        unique_o, split_indices = np.unique(w_o_indices, return_index=True)
        h_list_split = np.split(w_h_indices, split_indices[1:])
        w_h_map = {o_idx: h_idxs for o_idx, h_idxs in zip(unique_o, h_list_split)}
    else:
        w_h_map = {}

    indices_w_in_shell = np.where(shell_mask_water)[0]
    total_w_donor = 0
    total_w_acc = 0

    if len(indices_w_in_shell) == 0:
        return 0, 0

    target_w_indices = indices_w_in_shell
    target_w_pos = water_O_coords[target_w_indices]

    # [Step 2] Geometric Screening (Shell Water O <--> Global Water O)
    pairs = capped_distance(target_w_pos, water_O_coords, max_cutoff=r_cut, return_distances=False)
    
    if len(pairs) > 0:
        loc_idx = pairs[:, 0] # index in indices_w_in_shell
        glob_idx = pairs[:, 1] # global index
        
        # Exclude self-interaction
        real_w_idx = target_w_indices[loc_idx]
        mask_neq = real_w_idx != glob_idx
        
        if np.any(mask_neq):
            f_loc = loc_idx[mask_neq]
            f_glob = glob_idx[mask_neq]
            f_real_w = real_w_idx[mask_neq]
            
            # --- A. Water (Shell) as DONOR ---
            # Shell Water (O_d) - H ... Global Water (O_a)
            # Need to check H atoms of Shell Water
            count_d = 0
            for i in range(len(f_loc)):
                w_shell_idx = f_real_w[i] # Donor
                acc_idx = f_glob[i]       # Acceptor
                
                if w_shell_idx in w_h_map:
                    h_indices = w_h_map[w_shell_idx]
                    pos_od = water_O_coords[w_shell_idx]
                    pos_hs = all_H_coords[h_indices]
                    pos_oa = water_O_coords[acc_idx]
                    
                    v1 = pos_od - pos_hs
                    v2 = pos_oa - pos_hs
                    angs = calculate_angle_array(v1, v2)
                    count_d += np.sum(angs > angle_cut)
            total_w_donor += count_d
            
            # --- B. Water (Shell) as ACCEPTOR ---
            # Global Water (O_d) - H ... Shell Water (O_a)
            # Need to check H atoms of Global Water
            count_a = 0
            for i in range(len(f_loc)):
                donor_idx = f_glob[i]     # Donor (Global)
                acc_pos = target_w_pos[f_loc[i]] # Acceptor (Shell)
                
                if donor_idx in w_h_map:
                    h_indices = w_h_map[donor_idx]
                    pos_od = water_O_coords[donor_idx]
                    pos_hs = all_H_coords[h_indices]
                    
                    v1 = pos_od - pos_hs
                    v2 = acc_pos - pos_hs
                    angs = calculate_angle_array(v1, v2)
                    count_a += np.sum(angs > angle_cut)
            total_w_acc += count_a

    return total_w_donor, total_w_acc

def process_shell(system_name, time_range, file_config, bin_s, bin_e):
    """Process a single shell (Pure/Salt Water version)."""
    traj_template = file_config['traj']
    log_template = file_config['log']
    
    start_ns, end_ns = time_range
    start_file_idx = int(start_ns // 8)
    end_file_idx = int(np.ceil(end_ns / 8)) - 1
    if end_file_idx < start_file_idx: end_file_idx = start_file_idx
    
    results = [] 
    
    for f_idx in range(start_file_idx, end_file_idx + 1):
        f_start_time = f_idx * 8.0
        time_str = f"{int(f_start_time)}-{int((f_idx+1)*8.0)}ns"
        
        traj_path = traj_template.format(time_range=time_str)
        
        if not os.path.exists(traj_path): continue
        
        try:
            u = mda.Universe(traj_path, format='LAMMPSDUMP', atom_style='id type x y z')
        except: continue
            
        # Iterate trajectory directly, no ion file dependency
        for ts in u.trajectory:
            # Calculate current time (ps -> ns)
            # Assuming dump unit is ps (LAMMPS metal units usually ps)
            # and trajectory starts from 0 relative time
            current_time = f_start_time + ts.time / 1000.0
            
            if current_time < start_ns: continue
            if current_time >= end_ns: break
            
            # Interface and shell definition
            curr_log = log_template.replace("{time_range}", time_str)
            R_int = get_interface_pos(current_time, f_start_time, curr_log)
            
            if R_int is None: continue
            
            r_outer = R_int - bin_s
            r_inner = R_int - bin_e
            
            if r_inner < 0: r_inner = 0
            if r_outer < 0: r_outer = 0 
            if r_outer <= r_inner: continue 
            
            vol_shell = (4/3) * np.pi * (r_outer**3 - r_inner**3)
            
            # Select Water in Shell
            all_o = u.select_atoms(f"type {TYPE_O}")
            dists_o = np.linalg.norm(all_o.positions, axis=1)
            rels_o = R_int - dists_o
            w_in_shell_mask = (rels_o >= bin_s) & (rels_o <= bin_e)
            count_w = np.sum(w_in_shell_mask)
            
            # Calculate HB (Water Only)
            w_d, w_a = calc_water_hb_only(u, all_o, w_in_shell_mask)
            
            results.append([vol_shell, count_w, w_d, w_a])
            
            if ts.frame % 500 == 0:
                print(f"  Time {current_time:.2f}ns | Vol: {vol_shell:.1f} | W: {count_w}")
                
    return np.array(results)

def main():
    parser = argparse.ArgumentParser(description="Calculate Normalized Hydrogen Bonds for Water (Water-Water Only)")
    parser.add_argument("--bin_start", type=float, required=True, help="Shell start relative to interface")
    parser.add_argument("--bin_end", type=float, required=True, help="Shell end relative to interface")
    parser.add_argument("--save_path", default="./", help="Output directory")
    
    # System 1
    parser.add_argument("--sys1_name", required=True, help="Name of System 1")
    parser.add_argument("--sys1_start", type=float, required=True, help="Start time (ns)")
    parser.add_argument("--sys1_end", type=float, required=True, help="End time (ns)")
    parser.add_argument("--sys1_traj", required=True, help="Trajectory template")
    parser.add_argument("--sys1_log", required=True, help="Log template")
    
    # System 2
    parser.add_argument("--sys2_name", required=True, help="Name of System 2")
    parser.add_argument("--sys2_start", type=float, required=True, help="Start time (ns)")
    parser.add_argument("--sys2_end", type=float, required=True, help="End time (ns)")
    parser.add_argument("--sys2_traj", required=True, help="Trajectory template")
    parser.add_argument("--sys2_log", required=True, help="Log template")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
        
    systems = {
        args.sys1_name: {
            "traj": args.sys1_traj, "log": args.sys1_log,
            "range": (args.sys1_start, args.sys1_end)
        },
        args.sys2_name: {
            "traj": args.sys2_traj, "log": args.sys2_log,
            "range": (args.sys2_start, args.sys2_end)
        }
    }
    
    for sys_name, config in systems.items():
        print(f"Processing {sys_name} for bin [{args.bin_start}, {args.bin_end}]...")
        data = process_shell(sys_name, config["range"], config, args.bin_start, args.bin_end)
        
        out_file = os.path.join(args.save_path, f"{sys_name}_stats.txt")
        
        if len(data) > 0:
            # data columns: 0:Vol, 1:N_W, 2:W_D, 3:W_A
            avg_vol = np.mean(data[:, 0])
            
            mask_w = data[:, 1] > 0
            if np.any(mask_w):
                w_d_norm = np.mean(data[mask_w, 2] / data[mask_w, 1])
                w_a_norm = np.mean(data[mask_w, 3] / data[mask_w, 1])
            else:
                w_d_norm = 0.0; w_a_norm = 0.0
            
            with open(out_file, 'w') as f:
                f.write(f"BinStart: {args.bin_start}\n")
                f.write(f"BinEnd: {args.bin_end}\n")
                f.write(f"AvgVolume: {avg_vol}\n")
                f.write(f"Water_Donor_Norm: {w_d_norm}\n")
                f.write(f"Water_Acceptor_Norm: {w_a_norm}\n")
                f.write(f"Sum_N_W: {np.sum(data[:, 1])}\n")
            print(f"  -> Saved {out_file}")
        else:
            print(f"  -> No data found for {sys_name}")
            with open(out_file, 'w') as f:
                f.write(f"BinStart: {args.bin_start}\nBinEnd: {args.bin_end}\nAvgVolume: 0.0\n")
                f.write("Water_Donor_Norm: 0.0\nWater_Acceptor_Norm: 0.0\n")

if __name__ == "__main__":
    main()