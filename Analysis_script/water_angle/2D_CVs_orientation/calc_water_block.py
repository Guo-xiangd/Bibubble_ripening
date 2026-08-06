#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance
import pickle
import re
from collections import defaultdict

# ==================== Helper Functions ====================

def parse_log_file(log_path):
    """Parse the machine-readable interface position from a log file."""
    pattern = re.compile(r"(?m)^interface_position\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$")
    if not os.path.exists(log_path): return None
    with open(log_path, 'r') as f:
        match = pattern.search(f.read())
        return float(match.group(1)) if match else None

def calculate_angle_vectors(v1, v2):
    """Calculate vector angles (0-180)"""
    # v1, v2 shape: (N, 3)
    norm_v1 = np.linalg.norm(v1, axis=1)
    norm_v2 = np.linalg.norm(v2, axis=1)
    
    mask = (norm_v1 > 0) & (norm_v2 > 0)
    angles = np.full(len(v1), np.nan)
    
    if np.any(mask):
        dot = np.sum(v1[mask] * v2[mask], axis=1)
        cos_angle = dot / (norm_v1[mask] * norm_v2[mask])
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angles[mask] = np.degrees(np.arccos(cos_angle))
    
    return angles

def process_block(u, frames, interface_pos, dist_edges, angle_edges,
                  folder_start_ns, user_start_ns, user_end_ns):
    """Process specific frame list, calculate 2D distribution"""
    
    # Init 2D Histogram Matrix (Rows: Distance, Cols: Angle)
    # shape: (n_dist_bins, n_angle_bins)
    n_dist = len(dist_edges) - 1
    n_angle = len(angle_edges) - 1
    histogram_2d = np.zeros((n_dist, n_angle), dtype=np.float64)
    
    all_oxygens = u.select_atoms("type 2") 
    all_hydrogens = u.select_atoms("type 1")
    dt = 0.001 
    
    for ts in frames:
        current_time = folder_start_ns + (ts * dt)
        if current_time < user_start_ns or current_time >= user_end_ns:
            continue
            
        u.trajectory[ts]
        
        # 1. Spatial coarse filter (Interface +/- 15A)
        o_positions = all_oxygens.positions
        o_dists_origin = np.linalg.norm(o_positions, axis=1)
        mask_near = (o_dists_origin > (interface_pos - 15.0)) & (o_dists_origin < (interface_pos + 15.0))
        target_oxygens = all_oxygens[mask_near] 
        
        if len(target_oxygens) == 0: continue

        # 2. Topology construction (H2O)
        r_o = target_oxygens.positions
        r_h = all_hydrogens.positions
        pairs = capped_distance(r_o, r_h, max_cutoff=1.2, return_distances=False)
        
        if len(pairs) == 0: continue

        o_connectivity = defaultdict(list)
        for o_idx, h_idx in pairs:
            o_connectivity[o_idx].append(h_idx)
            
        valid_o_indices = [] 
        valid_h1_indices = [] 
        valid_h2_indices = [] 
        
        for o_idx, h_list in o_connectivity.items():
            if len(h_list) == 2:
                valid_o_indices.append(o_idx)
                valid_h1_indices.append(h_list[0])
                valid_h2_indices.append(h_list[1])
        
        if not valid_o_indices: continue
        
        # Coordinate Extraction
        # Note: Must calculate two OH bonds separately
        pos_O = r_o[valid_o_indices]
        pos_H1 = r_h[valid_h1_indices]
        pos_H2 = r_h[valid_h2_indices]
        
        # 3. Calculate Vectors
        # Definition: Angle between v_HO (H to O) and v_radial (O to Bubble Center/Origin)
        
        # Vector 1: H -> O
        v_H1_O = pos_O - pos_H1
        v_H2_O = pos_O - pos_H2
        
        # Vector 2: O -> Center (0,0,0) => -pos_O
        v_O_Center = -pos_O
        
        # Stack Data: One water molecule has two bonds, flatten to stats together
        # Stacked v_HO: (2N, 3)
        v_HO_all = np.vstack((v_H1_O, v_H2_O))
        # Stacked v_radial: (2N, 3) - Same radial vector for both bonds
        v_radial_all = np.vstack((v_O_Center, v_O_Center))
        
        # 4. Calculate Angles
        angles_all = calculate_angle_vectors(v_HO_all, v_radial_all)
        
        # 5. Calculate Distance (Relative to Interface)
        # Stack distances as well
        d_to_center = np.linalg.norm(pos_O, axis=1)
        rel_dists = interface_pos - d_to_center # r>0 inside, r<0 outside
        rel_dists_all = np.hstack((rel_dists, rel_dists)) # duplicate for 2 bonds
        
        # 6. Filter NaN and Bin
        valid_mask = ~np.isnan(angles_all)
        final_dists = rel_dists_all[valid_mask]
        final_angles = angles_all[valid_mask]
        
        # Use numpy histogram2d for fast binning
        H, _, _ = np.histogram2d(final_dists, final_angles, bins=[dist_edges, angle_edges])
        
        # Accumulate
        histogram_2d += H
                
    return histogram_2d

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trj_file", required=True)
    parser.add_argument("--geo_file", default=None)
    parser.add_argument("--log_file", required=True)
    parser.add_argument("--output_pkl", required=True)
    parser.add_argument("--block_id", type=int, required=True)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--dist_min_r", type=float, default=-10.0)
    parser.add_argument("--dist_max_r", type=float, default=10.0)
    parser.add_argument("--dist_bin", type=float, default=0.2)
    parser.add_argument("--folder_start_ns", type=float, required=True)
    parser.add_argument("--user_start_ns", type=float, required=True)
    parser.add_argument("--user_end_ns", type=float, required=True)
    
    args = parser.parse_args()
    
    interface_pos = parse_log_file(args.log_file)
    if interface_pos is None: return

    if args.geo_file and os.path.exists(args.geo_file):
        u = mda.Universe(args.geo_file, args.trj_file, atom_style='id type x y z', format='LAMMPSDUMP')
    else:
        u = mda.Universe(args.trj_file, atom_style='id type x y z', format='LAMMPSDUMP', dt=1.0)
        
    total_frames = len(u.trajectory)
    frames_per_block = total_frames // args.num_blocks
    start_frame = args.block_id * frames_per_block
    end_frame = (args.block_id + 1) * frames_per_block
    if args.block_id == args.num_blocks - 1: end_frame = total_frames
    
    current_frames = range(start_frame, end_frame)
    print(f"Processing Block {args.block_id}, Frames {start_frame}-{end_frame}")

    # Define Bins
    dist_edges = np.arange(args.dist_min_r, args.dist_max_r + args.dist_bin, args.dist_bin)
    # Angle Bin: 0 to 180, 1 degree per Bin
    angle_edges = np.linspace(0, 180, 181) 
    
    stats_matrix = process_block(u, current_frames, interface_pos, dist_edges, angle_edges,
                                 args.folder_start_ns, args.user_start_ns, args.user_end_ns)
    
    # Save results (Save edges for merge verification)
    result = {
        'histogram': stats_matrix,
        'dist_edges': dist_edges,
        'angle_edges': angle_edges
    }
    
    with open(args.output_pkl, 'wb') as f:
        pickle.dump(result, f)
    print(f"Saved: {args.output_pkl}")

if __name__ == "__main__":
    main()
