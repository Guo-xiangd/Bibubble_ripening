#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance
from tqdm import tqdm
import warnings

# Configure warning filters
warnings.filterwarnings("ignore", category=UserWarning)  # suppress user warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # suppress deprecation warnings

def setup_output_files(save_path, scheme, step, cutoff, trj_name, findpart, ion_num_upper):
    """Create output files and write their headers"""
    # Set file names
    index_file = os.path.join(save_path, f"{scheme}_ohidx_dump{step}_cut{cutoff}_{trj_name}_{findpart}.txt")
    h_xyz_file = os.path.join(save_path, f"{scheme}_h_xyz_dump{step}_cut{cutoff}_{trj_name}_{findpart}.txt")
    o_xyz_file = os.path.join(save_path, f"{scheme}_o_xyz_dump{step}_cut{cutoff}_{trj_name}_{findpart}.txt")
    
    # Create files and write headers
    with open(index_file, 'w') as f_idx, open(h_xyz_file, 'w') as f_h, open(o_xyz_file, 'w') as f_o:
        # Write the index-file header
        f_idx.write(f"# {'Ions':>6}{'MD_step':>16}")
        f_h.write(f"# {'Ions':>6}{'MD_step':>16}")
        f_o.write(f"# {'Ions':>6}{'MD_step':>16}")
        
        
        # Write headers for the selected scheme
        if scheme == 'h3o':
            for i in range(ion_num_upper):
                f_idx.write(f"{f'O{i+1}':>8}")
                f_h.write(f"{f'O{i+1}':>8}{'X':>16}{'Y':>16}{'Z':>16}")
                f_o.write(f"{f'O{i+1}':>8}")
                for j in range(3):
                    f_idx.write(f"{f'H{j+1}':>8}")
                    f_h.write(f"{f'H{j+1}':>8}{'X':>16}{'Y':>16}{'Z':>16}")
                f_o.write(f"{'X':>16}{'Y':>16}{'Z':>16}")
        else:  # 'oh'
            for i in range(ion_num_upper):
                f_idx.write(f"{f'O{i+1}':>8}{'H':>8}")
                f_h.write(f"{f'O{i+1}':>8}{'X':>16}{'Y':>16}{'Z':>16}{'H':>8}{'X':>16}{'Y':>16}{'Z':>16}")
                f_o.write(f"{f'O{i+1}':>8}{'X':>16}{'Y':>16}{'Z':>16}")
                
        f_idx.write("\n")
        f_h.write("\n")
        f_o.write("\n")        
    
    # Return file objects for appending
    return open(index_file, 'a'), open(h_xyz_file, 'a'), open(o_xyz_file, 'a')

def extract_box_from_timestep(u, frame_idx):
    """
    Extract the current frame box information directly from the trajectory file
    Return format: [Lx, Ly, Lz, 90.0, 90.0, 90.0] (for an orthorhombic box)
    """
    u.trajectory[frame_idx]  # Move to the requested frame
    ts = u.trajectory.ts
    
    # Try to extract box information from the underlying data
    try:
        # Obtain the original cell information
        unitcell = ts._unitcell
        #print(unitcell) unitcell lx still keeps 2L for [-L,L]
        
        dimensions = unitcell
        return dimensions
    except Exception as e:
        # If box information is unavailable, try to infer it from atomic positions
        try:
            print(f"Warning: Unable to obtain box information from the trajectory; attempting to infer it from atomic positions: {str(e)}")
            
            # Obtain all atomic positions
            coords = u.atoms.positions
            
            # Return None if the trajectory is empty
            if len(coords) == 0:
                return None
                
            # Calculate the bounding box
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)
            
            # Calculate dimensions
            Lx = max_coords[0] - min_coords[0]
            Ly = max_coords[1] - min_coords[1]
            Lz = max_coords[2] - min_coords[2]
            
            # Construct standard box information
            dimensions = [Lx, Ly, Lz, 90.0, 90.0, 90.0]
            return dimensions
        except Exception as e2:
            print(f"Unable to obtain box information: {str(e2)}")
            return None

        

def find_ions(scheme, system, cutoff, u, frame_idx):
    """Find ions in the current timestep"""
    use_pbc = (system != 'sphere')
    
    # Extract box information manually - format is [Lx, Ly, Lz, 90, 90, 90]
    dimensions = extract_box_from_timestep(u, frame_idx)
    #print(dimensions)
    
    oxygens = u.select_atoms('type 2')  # O atom
    hydrogens = u.select_atoms('type 1')  # H atom
    
    # Return an empty result if no oxygen or hydrogen atoms are present
    if len(oxygens) == 0 or len(hydrogens) == 0:
        return {-1: []}, {-1: []}, {-1: []}
    
    # Obtain bonding information
    o_h_pairs = []
    
    try:
        # Calculate interatomic distances - using the six-component box dimensions
        pairs = capped_distance(
            oxygens.positions, 
            hydrogens.positions,
            max_cutoff=cutoff, 
            box=dimensions,  # now using the correct format
            return_distances=False
        )
        
        # Process bonded pairs
        for o_idx, h_idx in pairs:
            o_id = oxygens[o_idx].id
            h_id = hydrogens[h_idx].id
            h_pos = hydrogens[h_idx].position
            o_h_pairs.append((o_id, h_id, h_pos))
    
    except Exception as e:
        print(f"Error while calculating distances: {str(e)}")
        return {-1: []}, {-1: []}, {-1: []}
    
    # Group by oxygen atom
    o_h_bonds = {}
    for o_id, h_id, h_pos in o_h_pairs:
        if o_id not in o_h_bonds:
            o_h_bonds[o_id] = []
        o_h_bonds[o_id].append((h_id, h_pos))
    
    # Classify ions according to the selected scheme
    o_dict, h_coord_dict, o_coord_dict = {}, {}, {}
    for o_id, h_list in o_h_bonds.items():
        o_pos = oxygens.select_atoms(f'id {o_id}').positions[0]
        
        if scheme == 'h3o' and len(h_list) == 3:  # hydronium ion
            o_dict[o_id] = [h[0] for h in h_list]
            h_coord_dict[o_id] = [h[1] for h in h_list]
            o_coord_dict[o_id] = o_pos
        elif scheme == 'oh' and len(h_list) == 1:  # hydroxide ion
            o_dict[o_id] = [h[0] for h in h_list]
            h_coord_dict[o_id] = [h[1] for h in h_list]
            o_coord_dict[o_id] = o_pos
    
    # If no ions are found
    if not o_dict:
        o_dict = {-1: []}
        o_coord_dict = {-1: []}
        h_coord_dict = {-1: []}
    
    return o_dict, o_coord_dict, h_coord_dict

def write_placeholder(f_idx, f_h, f_o, scheme):
    """Write placeholder data"""
    if scheme == 'h3o':
        f_idx.write(f"{'-1':>8}{'-1':>8}{'-1':>8}{'-1':>8}")
        f_h.write(f"{'-1':>8}{'0':>16}{'0':>16}{'0':>16}")
        for _ in range(3):  # Three hydrogen atoms
            f_h.write(f"{0:>8}{'0':>16}{'0':>16}{'0':>16}")
    else:  # 'oh'
        f_idx.write(f"{'-1':>8}{'-1':>8}")
        f_h.write(f"{'-1':>8}{'0':>16}{'0':>16}{'0':>16}{0:>8}{'0':>16}{'0':>16}{'0':>16}")
    
    f_o.write(f"{'-1':>8}{'0':>16}{'0':>16}{'0':>16}")

def write_ion_data(f_idx, f_h, f_o, o_id, o_pos, h_ids, h_positions, scheme, dimensions):
    """Write ion data"""
    
    # Calculate translation offsets
    shift_x = -dimensions[0] / 2  # x direction: subtract L
    shift_y = -dimensions[1] / 2  # y direction: subtract 0.5L
    shift_z = -dimensions[2] / 2  # z direction: subtract 0.5L
 
    # Translate oxygen coordinates
    shifted_o_pos = [
        o_pos[0] + shift_x,
        o_pos[1] + shift_y,
        o_pos[2] + shift_z
    ]
    
    # Translate hydrogen coordinates
    shifted_h_positions = [
        [
            pos[0] + shift_x,
            pos[1] + shift_y,
            pos[2] + shift_z
        ] for pos in h_positions
    ]
    
    # Write the index file
    f_idx.write(f"{o_id:>8}")
    if scheme == 'h3o':
        for h_id in h_ids:
            f_idx.write(f"{h_id:>8}")
    else:  # 'oh'
        f_idx.write(f"{h_ids[0]:>8}" if h_ids else "-1")  # Write only the first hydrogen atom ID
    
    # Write coordinate files
    f_h.write(f"{o_id:>8}{shifted_o_pos[0]:>16.8f}{shifted_o_pos[1]:>16.8f}{shifted_o_pos[2]:>16.8f}")
    for i, h_pos in enumerate(shifted_h_positions):
        h_id = h_ids[i] if i < len(h_ids) else -1
        f_h.write(f"{h_id:>8}{h_pos[0]:>16.8f}{h_pos[1]:>16.8f}{h_pos[2]:>16.8f}")
    
    # Write oxygen coordinates
    f_o.write(f"{o_id:>8}{shifted_o_pos[0]:>16.8f}{shifted_o_pos[1]:>16.8f}{shifted_o_pos[2]:>16.8f}")

if __name__ == '__main__':
    import argparse
    
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Ion detection program')
    
    # Add command-line arguments
    parser.add_argument('trj_path', type=str, help='trajectory file path')
    parser.add_argument('save_path', type=str, help='output path')
    parser.add_argument('trj_file', type=str, help='trajectory file name')
    parser.add_argument('--findpart', type=int, default=1, help='partition to process (default: 1)')
    parser.add_argument('--choose_part', type=int, default=2, help='total number of partitions (default: 2)')
    parser.add_argument('--step', type=int, default=1, help='frame stride (default: 1)')
    parser.add_argument('--scheme', type=str, default='h3o', choices=['oh', 'h3o'], help='detection scheme: oh or h3o (default: h3o)')
    parser.add_argument('--system', type=str, default='bulk', choices=['slab', 'sphere', 'bulk'], help='system type (default: bulk)')
    parser.add_argument('--ion_num', type=int, default=40, help='expected ion count (default: 40)')
    parser.add_argument('--cutoff', type=float, default=1.20, help='cutoff distance (default: 1.20)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set parameters
    trj_path = args.trj_path
    save_path = args.save_path
    trj_file = args.trj_file
    findpart = args.findpart
    choose_part = args.choose_part
    step = args.step
    scheme = args.scheme
    system = args.system
    ion_num = args.ion_num
    ion_num_upper = ion_num * 2 + 2
    cutoff = args.cutoff
    
    trj_name = trj_file.split('.')[0]
    
    print(f"Starting ion detection: {trj_file}")
    print(f"Scheme: {scheme}, system: {system}, cutoff distance: {cutoff}Å")
    print(f"Processing partition: {findpart}/{choose_part}")
    
    # Load the trajectory file
    try:
        full_path = os.path.join(trj_path, trj_file)
        u = mda.Universe(full_path, format='LAMMPSDUMP')
        total_frames = len(u.trajectory)
        frames_per_part = total_frames // choose_part
        start_frame = findpart * frames_per_part
        stop_frame = min((findpart + 1) * frames_per_part, total_frames)
        
        print(f"Loaded trajectory contains {total_frames} frames")
        print(f"Frame range: {start_frame} to {stop_frame} ({stop_frame - start_frame} frames total)")
        print(f"Maximum ion count: {ion_num_upper}")
        
        # Prepare output files
        f_idx, f_h, f_o = setup_output_files(save_path, scheme, step, cutoff, trj_name, findpart, ion_num_upper)
        
        # Process trajectory
        frame_count = 0
        with tqdm(total=(stop_frame - start_frame), desc="Frame processing progress") as pbar:
            for frame_idx in range(start_frame, stop_frame, step):
                frame_count += 1
                try:
                    # Move to the current frame
                    u.trajectory[frame_idx]
                    md_step = u.trajectory.frame
                    
                    # Obtain box dimensions
                    dimensions = extract_box_from_timestep(u, frame_idx)
                    
                    # Find ions
                    o_dict, o_coord_dict, h_coord_dict = find_ions(
                        scheme, system, cutoff, u, frame_idx
                    )
                    
                    # Write ion-count information
                    ion_count = 0 if -1 in o_dict else len(o_dict)
                    f_idx.write(f"{ion_count:>8}{md_step:>16}")
                    f_h.write(f"{ion_count:>8}{md_step:>16}")
                    f_o.write(f"{ion_count:>8}{md_step:>16}")
                    
                    # Write ion data
                    if ion_count > 0:
                        for o_id in o_dict:
                            if o_id == -1: 
                                continue
                            write_ion_data(
                                f_idx, f_h, f_o, 
                                o_id, o_coord_dict[o_id], 
                                o_dict[o_id], h_coord_dict[o_id],
                                scheme, dimensions  
                            )
                    
                    # Write placeholders
                    remaining = ion_num_upper - ion_count
                    for _ in range(remaining):
                        write_placeholder(f_idx, f_h, f_o, scheme)
                    
                    # Finish the output row
                    f_idx.write("\n")
                    f_h.write("\n")
                    f_o.write("\n")
                    
                    # Update the progress bar
                    if frame_count % 10 == 0:
                        pbar.update(10)
                
                except Exception as e:
                    print(f"Error processing frame {frame_idx}: {str(e)}")
        
        print("\nProcessing complete!")
        print(f"Processed {frame_count} frames")
        print(f"Index file: {f_idx.name}")
        print(f"Hydrogen coordinate file: {f_h.name}")
        print(f"Oxygen coordinate file: {f_o.name}")
    
    except Exception as e:
        print(f"Error while processing the trajectory: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure files are closed
        if 'f_idx' in locals(): f_idx.close()
        if 'f_h' in locals(): f_h.close()
        if 'f_o' in locals(): f_o.close()
