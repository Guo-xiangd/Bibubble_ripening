#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base System: Residence-Coupled Hydrogen Bond (HB) Dynamics.
Includes dynamic OH- ion shielding mechanism.
"""

import os
import re
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('Agg')

# ========================= User Configuration =========================
CONFIG = {
    # Trajectory settings
    "traj_path": "./data/centered_bubble1.lammpstrj", 
    "traj_start_time_ns": 40.0,  
    "dt_ps": 0.1,                
    
    # OH- ion tracking file
    "oh_file": "./data/oh_ions.txt", 
    
    # Log folder template
    "log_template": "./data/logs/{chunk}ns/bubble1_block{block}/log.txt",
    "chunk_size_ns": 4.0,        
    "blocks_per_chunk": 8,       
    
    # Dynamics calculation parameters
    "max_obs_time_ps": 40.0,     
    "t0_spacing_ps": 2.0,        
    "interface_range": 2.5,      
    
    # Output settings
    "out_txt": "base_water_HB_Res_Lifetimes.txt",
    "out_img": "base_water_HB_Res_Lifetimes.png"
}

TYPE_O = '2'
TYPE_H = '1'

# ========================= OH- Ion File Parser =========================

def load_oh_info(filepath):
    """Extract O and H sets for each frame to identify OH- ions."""
    print(f"Parsing OH- ion topology file: {filepath} ...")
    oh_o_ids = {}
    oh_h_ids = {}
    
    if not os.path.exists(filepath):
        print(f"[Error] OH- ion file not found: {filepath}")
        return oh_o_ids, oh_h_ids
        
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip(): 
                continue
            
            parts = line.split()
            n_ions = int(parts[0])
            md_step = int(parts[1])  
            
            o_set = set()
            h_set = set()
            
            for i in range(n_ions):
                base_idx = 2 + i * 8
                if base_idx + 6 >= len(parts): 
                    break 
                
                o_id = int(parts[base_idx])
                if o_id == -1: 
                    continue 
                h_id = int(parts[base_idx + 4])
                
                o_set.add(o_id)
                if h_id > 0:
                    h_set.add(h_id)
                
            oh_o_ids[md_step] = o_set
            oh_h_ids[md_step] = h_set
            
    print(f"[Success] Loaded {len(oh_o_ids)} frames of OH- ion data.")
    return oh_o_ids, oh_h_ids

# ========================= Dynamic Interface Position Reader =========================

class InterfacePositionCache:
    """Reads and caches dynamic interface positions."""
    def __init__(self, template, chunk_size, blocks_per_chunk):
        self.template = template
        self.chunk_size = chunk_size
        self.blocks_per_chunk = blocks_per_chunk
        self.block_duration = chunk_size / blocks_per_chunk
        self.cache = {}
        
    def get_z_int(self, current_time_ns):
        chunk_idx = int(current_time_ns // self.chunk_size)
        chunk_start = int(chunk_idx * self.chunk_size)
        chunk_end = int(chunk_start + self.chunk_size)
        chunk_str = f"{chunk_start}-{chunk_end}"
        
        offset = current_time_ns - chunk_start
        block_idx = int(offset // self.block_duration) + 1
        block_idx = max(1, min(self.blocks_per_chunk, block_idx))
        
        key = (chunk_str, block_idx)
        if key in self.cache:
            return self.cache[key]
            
        log_path = self.template.format(chunk=chunk_str, block=block_idx)
        if not os.path.exists(log_path):
            self.cache[key] = None
            return None
            
        with open(log_path, 'r') as f:
            m = re.search(r"界面位置:\s*([-\d\.]+)", f.read())
            if m:
                z_int = float(m.group(1))
                self.cache[key] = z_int
                return z_int
                
        self.cache[key] = None
        return None

# ========================= Helper Functions =========================

def check_single_hb_geom(d_pos, a_pos, h_pos, r_cut_sq=12.25, cos_cut=-0.766):
    """Check hydrogen bond geometry for a single molecule pair."""
    diff_da = d_pos - a_pos
    dist_sq = np.dot(diff_da, diff_da)
    if dist_sq > r_cut_sq: 
        return False
    
    v1 = d_pos - h_pos
    v2 = a_pos - h_pos
    dot = np.dot(v1, v2)
    norm_sq1 = np.dot(v1, v1)
    norm_sq2 = np.dot(v2, v2)
    
    if norm_sq1 == 0 or norm_sq2 == 0: 
        return False
    return (dot / np.sqrt(norm_sq1 * norm_sq2)) < cos_cut

def find_all_hbs_and_residence(o_atoms, h_atoms, z_int, z_range, exclude_o_ids, exclude_h_ids):
    """Determine HB residence status for water molecules (excluding ions)."""
    o_coords = o_atoms.positions
    h_coords = h_atoms.positions
    o_ids = o_atoms.ids
    h_ids = h_atoms.ids
    
    o_dists = np.linalg.norm(o_coords, axis=1)
    buffer_zone = z_range + 3.5 
    valid_o_mask = np.abs(z_int - o_dists) <= buffer_zone
    valid_o_indices = np.where(valid_o_mask)[0]
    
    if len(valid_o_indices) == 0:
        return {}
        
    valid_o_coords = o_coords[valid_o_indices]
    valid_o_ids = o_ids[valid_o_indices]
    
    if exclude_o_ids:
        keep_mask = np.array([o_id not in exclude_o_ids for o_id in valid_o_ids])
        valid_o_coords = valid_o_coords[keep_mask]
        valid_o_ids = valid_o_ids[keep_mask]
        
    if len(valid_o_ids) == 0:
        return {}
    
    pairs_wh = capped_distance(valid_o_coords, h_coords, max_cutoff=1.2, return_distances=False)
    local_wh = {}
    for p in pairs_wh:
        h_global_idx = p[1]
        if h_ids[h_global_idx] in exclude_h_ids:
            continue
        local_wh.setdefault(p[0], []).append(h_global_idx)
        
    oo_pairs = capped_distance(valid_o_coords, valid_o_coords, max_cutoff=3.5, return_distances=False)
    
    current_hbs = {}
    for p in oo_pairs:
        d_idx_local, a_idx_local = p[0], p[1]
        if d_idx_local == a_idx_local: 
            continue
        
        if d_idx_local in local_wh:
            d_pos = valid_o_coords[d_idx_local]
            a_pos = valid_o_coords[a_idx_local]
            
            for h_idx_global in local_wh[d_idx_local]:
                h_pos = h_coords[h_idx_global]
                
                if check_single_hb_geom(d_pos, a_pos, h_pos):
                    hb_key = (valid_o_ids[d_idx_local], valid_o_ids[a_idx_local], h_ids[h_idx_global])
                    
                    rel_d = z_int - np.linalg.norm(d_pos)
                    rel_a = z_int - np.linalg.norm(a_pos)
                    
                    in_interface = False
                    if (-z_range <= rel_d <= z_range) or (-z_range <= rel_a <= z_range):
                        in_interface = True
                        
                    current_hbs[hb_key] = in_interface
                    
    return current_hbs

# ========================= Core Dynamics Calculation =========================

def main():
    oh_o_dict, oh_h_dict = load_oh_info(CONFIG['oh_file'])
    
    print(f"Loading trajectory: {CONFIG['traj_path']}")
    try:
        u = mda.Universe(CONFIG['traj_path'], format='LAMMPSDUMP', atom_style='id type x y z')
    except Exception as e:
        print(f"Failed to load trajectory: {e}")
        return

    o_atoms = u.select_atoms(f"type {TYPE_O}")
    h_atoms = u.select_atoms(f"type {TYPE_H}")
    
    z_cache = InterfacePositionCache(
        template=CONFIG['log_template'], 
        chunk_size=CONFIG['chunk_size_ns'], 
        blocks_per_chunk=CONFIG['blocks_per_chunk']
    )
    
    frames_per_t0 = int(round(CONFIG['t0_spacing_ps'] / CONFIG['dt_ps']))
    max_frames = int(round(CONFIG['max_obs_time_ps'] / CONFIG['dt_ps']))
    
    c_intermittent_num = np.zeros(max_frames + 1)
    c_continuous_num = np.zeros(max_frames + 1)
    c_den = np.zeros(max_frames + 1)
    
    tracked_origins = []
    total_frames = len(u.trajectory)
    last_valid_z_int = 12.5 
    
    print("Starting TCF forward scan (OH- excluded)...")
    
    for ts in u.trajectory:
        curr_frame = ts.frame
        curr_time_ns = CONFIG['traj_start_time_ns'] + (curr_frame * CONFIG['dt_ps'] / 1000.0)
        
        z_int = z_cache.get_z_int(curr_time_ns)
        if z_int is None: 
            z_int = last_valid_z_int
        else: 
            last_valid_z_int = z_int
            
        exclude_o = oh_o_dict.get(curr_frame, set())
        exclude_h = oh_h_dict.get(curr_frame, set())
        
        current_hbs = find_all_hbs_and_residence(
            o_atoms, h_atoms, z_int, CONFIG['interface_range'], 
            exclude_o_ids=exclude_o, exclude_h_ids=exclude_h
        )
        
        active_origins = []
        for origin in tracked_origins:
            dt_frames = curr_frame - origin['frame']
            if dt_frames <= max_frames:
                active_origins.append(origin)
                
                for hb_key in origin['hbs'].keys():
                    is_present_and_in_residence = (hb_key in current_hbs) and (current_hbs[hb_key] == True)
                    
                    if is_present_and_in_residence:
                        c_intermittent_num[dt_frames] += 1
                    else:
                        origin['hbs'][hb_key] = 0 
                        
                    c_continuous_num[dt_frames] += origin['hbs'][hb_key]
                    c_den[dt_frames] += 1
                    
        tracked_origins = active_origins
        
        if curr_frame % frames_per_t0 == 0 and curr_frame + max_frames < total_frames:
            new_origin = {'frame': curr_frame, 'hbs': {}}
            for hb_key, in_interface in current_hbs.items():
                if in_interface:
                    new_origin['hbs'][hb_key] = 1 
                    c_intermittent_num[0] += 1
                    c_continuous_num[0] += 1
                    c_den[0] += 1
            
            tracked_origins.append(new_origin)
            print(f"  --> [t0 = {curr_time_ns:.3f} ns] Captured pure water HBs: {len(new_origin['hbs'])}")

    # ========================= Post-processing & Output =========================
    print("Scan complete. Exporting data and generating plots...")
    t_axis = np.arange(max_frames + 1) * CONFIG['dt_ps']
    
    valid = c_den > 0
    C_intermittent = np.zeros_like(c_den)
    C_continuous = np.zeros_like(c_den)
    C_intermittent[valid] = c_intermittent_num[valid] / c_den[valid]
    C_continuous[valid] = c_continuous_num[valid] / c_den[valid]
    
    header = (
        "Base System (Water-Water HB) Residence-Coupled Dynamics\n"
        "Time(ps)    C_Interm(t)    C_Contin(t)    Count_Interm(t)    Count_Contin(t)    Total_Samples(t)"
    )
    data_out = np.column_stack((
        t_axis, C_intermittent, C_continuous, 
        c_intermittent_num, c_continuous_num, c_den
    ))
    
    np.savetxt(CONFIG['out_txt'], data_out, header=header, 
               fmt=["%.3f", "%.6f", "%.6f", "%d", "%d", "%d"], delimiter="    ", comments="# ")
    print(f"[✔] Text file saved successfully: {CONFIG['out_txt']}")
    
    set_nature_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_axis, C_intermittent, '-', color='#d62728', lw=2.5, label='Intermittent (Unified)')
    ax.plot(t_axis, C_continuous, '--', color='#1f77b4', lw=2.5, label='Continuous (Unified)')
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel(r'$C_{HB-Res}(t)$')
    ax.set_xlim(0, CONFIG['max_obs_time_ps'])
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right')
    ax.set_title("Base System: Water-Water HB Dynamics")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(CONFIG['out_img'], dpi=300)
    print(f"[✔] Image saved successfully: {CONFIG['out_img']}")

def set_nature_style():
    """Applies a publication-ready plotting style."""
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10,
        'lines.linewidth': 2.0,
        'axes.linewidth': 1.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
    })

if __name__ == "__main__":
    main()