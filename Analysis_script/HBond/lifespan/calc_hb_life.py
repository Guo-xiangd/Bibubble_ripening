#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance
import re

# ==================== Constant Definitions ====================
TYPE_H = '1'
TYPE_O = '2'

# ==================== Utility Functions ====================

def get_interface_pos(time_ns, start_ns, log_tpl):
    """
    Read the interface position from the log file.
    
    Args:
        time_ns: Current simulation time in ns.
        start_ns: Start time of the simulation in ns.
        log_tpl: Path template for the log file.
        
    Returns:
        float: Interface position if found, else None.
    """
    idx = int(time_ns - start_ns) + 1
    idx = max(1, min(8, idx))
    path = log_tpl.format(block_idx=idx)
    if not os.path.exists(path): return None
    try:
        with open(path, 'r') as f:
            # Note: Regex updated to match English output. 
            # Ensure your log files contain "Interface Position: <value>"
            m = re.search(r"Interface Position:\s*(\d+\.\d+)", f.read())
            return float(m.group(1)) if m else None
    except: return None

def check_single_hb_geom(d_pos, a_pos, h_pos, r_cut_sq=12.25, cos_cut=-0.766):
    """
    Check the geometry of a single hydrogen bond (Optimized).
    
    Criteria:
    1. Distance: r_DA < 3.5 Angstrom (r_cut_sq = 12.25)
    2. Angle: theta > 140 degrees
       cos(140) = -0.766. 
       Vectors defined as v1 = D - H, v2 = A - H.
       The angle at H should be close to 180 degrees, so cos(theta) should be < -0.766.
    """
    # Distance check (Donor - Acceptor)
    diff_da = d_pos - a_pos
    dist_sq = np.dot(diff_da, diff_da)
    if dist_sq > r_cut_sq: return False
    
    # Angle check
    v1 = d_pos - h_pos
    v2 = a_pos - h_pos
    
    # Fast cosine calculation
    dot = np.dot(v1, v2)
    norm_sq1 = np.dot(v1, v1)
    norm_sq2 = np.dot(v2, v2)
    
    # Avoid division by zero
    if norm_sq1 == 0 or norm_sq2 == 0: return False
    
    # cos_theta = dot / sqrt(n1*n2)
    val = dot / np.sqrt(norm_sq1 * norm_sq2)
    
    # If angle > 140 degrees, cos(angle) < -0.766
    if val < cos_cut: return True
    return False

def process_lifetime(sys_name, time_range, cfg, bin_s, bin_e):
    """
    Calculate Hydrogen Bond lifetimes using a Lagrangian (continuous existence) approach.
    """
    start_ns, end_ns = time_range
    traj_tpl, log_tpl = cfg['traj'], cfg['log']
    
    # Active HB Record
    # Key: (Type_Flag, d_id, a_id, h_id)
    # Type_Flag: 'Wat_D' (Donor in shell) or 'Wat_A' (Acceptor in shell)
    # Value: start_time
    active_hbs = {}
    
    # Results dictionary
    lifetimes = {'Wat_D': [], 'Wat_A': []}
    
    s_idx = int(start_ns // 8)
    e_idx = int(np.ceil(end_ns / 8)) - 1
    if e_idx < s_idx: e_idx = s_idx
    
    print(f"Analyzing {sys_name} from {start_ns} to {end_ns} ns...")

    for f_idx in range(s_idx, e_idx + 1):
        f_start = f_idx * 8.0
        time_str = f"{int(f_start)}-{int((f_idx+1)*8.0)}ns"
        traj_path = traj_tpl.format(time_range=time_str)
        
        if not os.path.exists(traj_path): continue
        
        try: 
            u = mda.Universe(traj_path, format='LAMMPSDUMP', atom_style='id type x y z')
        except: continue
        
        traj_iter = u.trajectory.__iter__()
        
        # Pre-select atom groups
        all_o = u.select_atoms(f"type {TYPE_O}")
        all_h = u.select_atoms(f"type {TYPE_H}")
        
        # ID Mapping (Local Index -> Global ID)
        o_ids = all_o.ids
        h_ids = all_h.ids
        
        for ts in traj_iter:
            # Assuming LAMMPS dump unit is ps (check your trajectory timestep)
            # MDAnalysis reads 'time' in ps. Convert to ns.
            curr_time = f_start + ts.time / 1000.0
            
            if curr_time < start_ns: continue
            if curr_time >= end_ns: break
            
            # Get interface position
            r_int = get_interface_pos(curr_time, f_start, log_tpl.replace("{time_range}", time_str))
            if r_int is None: continue
            
            # Update coordinates
            o_coords = all_o.positions
            h_coords = all_h.positions
            
            # 1. Topology Construction (Water O-H)
            # Re-evaluate H ownership every frame (crucial for protons, usually static for TIP4P/SPC)
            pairs_wh = capped_distance(o_coords, h_coords, max_cutoff=1.2, return_distances=False)
            local_wh = {} # O_local_idx -> [H_local_idx, ...]
            for p in pairs_wh:
                if p[0] not in local_wh: local_wh[p[0]] = []
                local_wh[p[0]].append(p[1])
                
            # 2. Geometric Screening (O-O distance < 3.5)
            # Find all O-O pairs first
            oo_pairs = capped_distance(o_coords, o_coords, max_cutoff=3.5, return_distances=False)
            
            # 3. Identify all existing HBs in current frame
            present_hbs = set() # (d_id, a_id, h_id) - Type tag excluded here
            
            for p in oo_pairs:
                d_idx, a_idx = p[0], p[1]
                if d_idx == a_idx: continue
                
                # Check if Donor has an H pointing to Acceptor
                if d_idx in local_wh:
                    d_pos = o_coords[d_idx]
                    a_pos = o_coords[a_idx]
                    for h_idx in local_wh[d_idx]:
                        h_pos = h_coords[h_idx]
                        if check_single_hb_geom(d_pos, a_pos, h_pos):
                            # Record unique ID tuple
                            hb_key = (o_ids[d_idx], o_ids[a_idx], h_ids[h_idx])
                            present_hbs.add(hb_key)
                            
                            # --- Check if this is a NEW HB in the Shell ---
                            
                            # 1. Check Donor in Shell
                            # If this bond is not already being tracked as Wat_D
                            full_key_d = ('Wat_D',) + hb_key
                            if full_key_d not in active_hbs:
                                d_dist = np.linalg.norm(d_pos)
                                rel = r_int - d_dist
                                if bin_s <= rel <= bin_e:
                                    active_hbs[full_key_d] = curr_time
                                    
                            # 2. Check Acceptor in Shell
                            # If this bond is not already being tracked as Wat_A
                            full_key_a = ('Wat_A',) + hb_key
                            if full_key_a not in active_hbs:
                                a_dist = np.linalg.norm(a_pos)
                                rel = r_int - a_dist
                                if bin_s <= rel <= bin_e:
                                    active_hbs[full_key_a] = curr_time

            # 4. Check for Breakage (Lagrangian Check)
            # Iterate through currently active HBs.
            # If the core key (d, a, h) is not in present_hbs, the bond has broken.
            
            dead_keys = []
            for full_key, start_t in active_hbs.items():
                core_key = full_key[1:] # (d, a, h)
                if core_key not in present_hbs:
                    # Broken! Record duration
                    duration = curr_time - start_t
                    # full_key[0] is 'Wat_D' or 'Wat_A'
                    lifetimes[full_key[0]].append(duration)
                    dead_keys.append(full_key)
            
            for k in dead_keys:
                del active_hbs[k]
                
            if ts.frame % 500 == 0:
                print(f"  Time {curr_time:.3f} ns | Active HBs: {len(active_hbs)}")

    return lifetimes

def main():
    parser = argparse.ArgumentParser(description="Calculate Hydrogen Bond Lifetimes in Shell")
    parser.add_argument("--bin_start", type=float, required=True, help="Start of the shell relative to interface (A)")
    parser.add_argument("--bin_end", type=float, required=True, help="End of the shell relative to interface (A)")
    parser.add_argument("--save_path", default="./", help="Directory to save output files")
    
    # Pure water (sys1) and Saline (sys2)
    parser.add_argument("--sys1_name", required=True, help="Name of System 1")
    parser.add_argument("--sys1_start", type=float, required=True, help="Start time for System 1 (ns)")
    parser.add_argument("--sys1_end", type=float, required=True, help="End time for System 1 (ns)")
    parser.add_argument("--sys1_traj", required=True, help="Trajectory template for System 1")
    parser.add_argument("--sys1_log", required=True, help="Log file template for System 1")
    
    parser.add_argument("--sys2_name", required=True, help="Name of System 2")
    parser.add_argument("--sys2_start", type=float, required=True, help="Start time for System 2 (ns)")
    parser.add_argument("--sys2_end", type=float, required=True, help="End time for System 2 (ns)")
    parser.add_argument("--sys2_traj", required=True, help="Trajectory template for System 2")
    parser.add_argument("--sys2_log", required=True, help="Log file template for System 2")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_path): os.makedirs(args.save_path, exist_ok=True)
    
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
    
    for name, cfg in systems.items():
        print(f"Processing Lifetime for {name}...")
        res = process_lifetime(name, cfg['range'], cfg, args.bin_start, args.bin_end)
        
        outfile = os.path.join(args.save_path, f"{name}_life_stats.txt")
        with open(outfile, 'w') as f:
            f.write(f"BinStart: {args.bin_start}\nBinEnd: {args.bin_end}\n")
            
            # Calculate mean lifetime
            wd_mean = np.mean(res['Wat_D']) if res['Wat_D'] else 0.0
            wa_mean = np.mean(res['Wat_A']) if res['Wat_A'] else 0.0
            
            f.write(f"Water_Donor_Life: {wd_mean:.6f}\n")
            f.write(f"Water_Acceptor_Life: {wa_mean:.6f}\n")
            f.write(f"Sample_Count: {len(res['Wat_D']) + len(res['Wat_A'])}\n")
            
        print(f"Saved {outfile}")

if __name__ == "__main__":
    main()