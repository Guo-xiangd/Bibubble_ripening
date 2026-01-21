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
    """
    idx = int(time_ns - start_ns) + 1
    idx = max(1, min(8, idx))
    path = log_tpl.format(block_idx=idx)
    if not os.path.exists(path): return None
    try:
        with open(path, 'r') as f:
            # Regex updated to match English output. 
            # Input file must contain: "Interface Position: X.XX"
            m = re.search(r"Interface Position:\s*(\d+\.\d+)", f.read())
            return float(m.group(1)) if m else None
    except: return None

def check_single_hb_geom(d_pos, a_pos, h_pos, r_cut_sq=12.25, cos_cut=-0.766):
    """
    Check the geometry of a single hydrogen bond (Optimized).
    
        
    Geometric Criteria:
    1. Distance: r_DA < 3.5 Angstrom (r_cut_sq = 12.25).
    2. Angle: theta > 140 degrees.
       Vectors defined as v1 = H->D, v2 = H->A (vertex at H).
       Ideally, the angle at H is close to 180 degrees.
       cos(140) = -0.766.
       If angle > 140, then cos(theta) < -0.766.
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
    
    # Check angle condition
    if val < cos_cut: return True
    return False

def process_lifetime(sys_name, time_range, cfg, bin_s, bin_e):
    """
    Calculate HB lifetime with tolerance window.
        """
    start_ns, end_ns = time_range
    traj_tpl, log_tpl = cfg['traj'], cfg['log']
    
    # === Tolerance Definition (2.1 ps = 0.0021 ns) ===
    # Allows the bond to break transiently for up to 2.1 ps without being considered "broken".
    T_ALLOW_NS = 0.0021 
    
    # Active HB Record
    # Key: (Type_Flag, d_id, a_id, h_id) -> Value: start_time
    active_hbs = {}
    
    # === Last Seen Record ===
    # Used to handle tolerance logic.
    # Key: (Type_Flag, d_id, a_id, h_id) -> Value: last_seen_time
    last_seen = {}
    
    lifetimes = {'Wat_D': [], 'Wat_A': []}
    
    s_idx = int(start_ns // 8)
    e_idx = int(np.ceil(end_ns / 8)) - 1
    if e_idx < s_idx: e_idx = s_idx
    
    print(f"Analyzing {sys_name} from {start_ns} to {end_ns} ns with 2.1ps tolerance...")

    for f_idx in range(s_idx, e_idx + 1):
        f_start = f_idx * 8.0
        time_str = f"{int(f_start)}-{int((f_idx+1)*8.0)}ns"
        traj_path = traj_tpl.format(time_range=time_str)
        
        if not os.path.exists(traj_path): continue
        try: 
            u = mda.Universe(traj_path, format='LAMMPSDUMP', atom_style='id type x y z')
        except: continue
        
        traj_iter = u.trajectory.__iter__()
        all_o = u.select_atoms(f"type {TYPE_O}")
        all_h = u.select_atoms(f"type {TYPE_H}")
        o_ids = all_o.ids
        h_ids = all_h.ids
        
        for ts in traj_iter:
            # Assuming ts.time is in ps, convert to ns
            curr_time = f_start + ts.time / 1000.0
            
            if curr_time < start_ns: continue
            if curr_time >= end_ns: break
            
            r_int = get_interface_pos(curr_time, f_start, log_tpl.replace("{time_range}", time_str))
            if r_int is None: continue
            
            o_coords = all_o.positions
            h_coords = all_h.positions
            
            # 1. Topology & 2. Neighbor Search
            pairs_wh = capped_distance(o_coords, h_coords, max_cutoff=1.2, return_distances=False)
            local_wh = {} 
            for p in pairs_wh:
                if p[0] not in local_wh: local_wh[p[0]] = []
                local_wh[p[0]].append(p[1])
                
            oo_pairs = capped_distance(o_coords, o_coords, max_cutoff=3.5, return_distances=False)
            
            # 3. Identify all HBs present in the current frame
            # Stores only core key (d, a, h)
            present_hbs = set() 
            
            # Temporarily store Full Keys confirmed alive this frame to update 'last_seen'
            confirmed_alive_keys = set()
            
            for p in oo_pairs:
                d_idx, a_idx = p[0], p[1]
                if d_idx == a_idx: continue
                
                if d_idx in local_wh:
                    d_pos = o_coords[d_idx]
                    a_pos = o_coords[a_idx]
                    for h_idx in local_wh[d_idx]:
                        h_pos = h_coords[h_idx]
                        if check_single_hb_geom(d_pos, a_pos, h_pos):
                            # Geometry satisfied, bond exists
                            hb_key = (o_ids[d_idx], o_ids[a_idx], h_ids[h_idx])
                            present_hbs.add(hb_key)
                            
                            # Construct Full Key and check tracking status
                            
                            # Case 1: Wat_D check
                            full_key_d = ('Wat_D',) + hb_key
                            if full_key_d in active_hbs:
                                confirmed_alive_keys.add(full_key_d)
                            else:
                                # New bond detection
                                d_dist = np.linalg.norm(d_pos)
                                if bin_s <= (r_int - d_dist) <= bin_e:
                                    active_hbs[full_key_d] = curr_time
                                    last_seen[full_key_d] = curr_time # Initialize last_seen
                                    confirmed_alive_keys.add(full_key_d)
                                    
                            # Case 2: Wat_A check
                            full_key_a = ('Wat_A',) + hb_key
                            if full_key_a in active_hbs:
                                confirmed_alive_keys.add(full_key_a)
                            else:
                                # New bond detection
                                a_dist = np.linalg.norm(a_pos)
                                if bin_s <= (r_int - a_dist) <= bin_e:
                                    active_hbs[full_key_a] = curr_time
                                    last_seen[full_key_a] = curr_time # Initialize last_seen
                                    confirmed_alive_keys.add(full_key_a)

            # === Update last_seen ===
            # Update the last observation time for bonds seen in this frame
            for key in confirmed_alive_keys:
                last_seen[key] = curr_time

            # 4. Check for Breakage (With Tolerance Logic)
            dead_keys = []
            
            # Iterate through all tracked bonds
            for full_key, start_t in active_hbs.items():
                # If not seen in this frame
                if full_key not in confirmed_alive_keys:
                    # Get the time it was last seen
                    t_last = last_seen.get(full_key, start_t)
                    
                    # Calculate how long it has been missing
                    time_lost = curr_time - t_last
                    
                    # Only confirm death if missing time exceeds tolerance
                    if time_lost > T_ALLOW_NS:
                        # Actual lifetime = Last Seen Time - Start Time
                        # (Note: Does not include the time spent in the "missing" state)
                        duration = t_last - start_t
                        lifetimes[full_key[0]].append(duration)
                        dead_keys.append(full_key)
                
                # If seen this frame, do nothing (alive), last_seen already updated above.
            
            # Clean up permanently broken bonds
            for k in dead_keys:
                del active_hbs[k]
                del last_seen[k]
                
            if ts.frame % 500 == 0:
                print(f"  Time {curr_time:.3f} ns | Active: {len(active_hbs)}")

    return lifetimes

def main():
    parser = argparse.ArgumentParser(description="Calculate Hydrogen Bond Lifetime with Tolerance")
    parser.add_argument("--bin_start", type=float, required=True, help="Shell start relative to interface")
    parser.add_argument("--bin_end", type=float, required=True, help="Shell end relative to interface")
    parser.add_argument("--save_path", default="./", help="Output directory")
    
    # System 1 (e.g., Pure Water) and System 2 (e.g., Saline)
    parser.add_argument("--sys1_name", required=True, help="Name of System 1")
    parser.add_argument("--sys1_start", type=float, required=True, help="Start time (ns)")
    parser.add_argument("--sys1_end", type=float, required=True, help="End time (ns)")
    parser.add_argument("--sys1_traj", required=True, help="Trajectory template")
    parser.add_argument("--sys1_log", required=True, help="Log template")
    
    parser.add_argument("--sys2_name", required=True, help="Name of System 2")
    parser.add_argument("--sys2_start", type=float, required=True, help="Start time (ns)")
    parser.add_argument("--sys2_end", type=float, required=True, help="End time (ns)")
    parser.add_argument("--sys2_traj", required=True, help="Trajectory template")
    parser.add_argument("--sys2_log", required=True, help="Log template")
    
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