#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trajectory Analysis Script:
Searches for Na-Oa-Ob-Cl atom quadruplets in LAMMPS trajectory files,
calculates relevant distances and angles, and handles Periodic Boundary Conditions (PBC).

Quadruplet Constraints (Updated):
1. Na - Oa Distance <= R1_NA_OA (r1)
2. Oa - Ob Distance <= R_W_OA_OB (rw)
3. Ob - Cl Distance <= R2_OB_CL (r2)
4. Na - Ob Distance >= R1_NA_OA (r1)
5. Cl - Oa Distance >= R2_OB_CL (r2)
6. Na - Cl Distance between R_ION_MIN and R_ION_MAX (New Constraint)

Dependencies: numpy, scipy
Ensure installed: pip install numpy scipy
"""

import numpy as np
from scipy.spatial import cKDTree
import sys
import os
import logging
import time
from math import sqrt, acos, degrees

# --- Constants ---

ATOM_TYPE_MAP = {
    1: 'H',
    2: 'O',
    3: 'N',
    4: 'Na',
    5: 'Cl'
}
# Target Atom Types
NA_TYPE = 4
O_TYPE = 2
CL_TYPE = 5

# Ion-Ion Distance Constraint (Na-Cl) - New
R_ION_MIN = 5.8
R_ION_MAX = 7.8

# Max quadruplets per frame in output
MAX_QUADRUPLETS_PER_FRAME = 60
# Number of data fields per quadruplet (ID*4 + XYZ*4 + 4*Distances + 2*Angles)
# (ID+XYZ)*4 + 4*Distances + 2*Angles = 22
FIELDS_PER_QUADRUPLET = 22


def setup_logging(log_file):
    """Configure logging to output to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'), # Write to log file
            logging.StreamHandler(sys.stdout)        # Print to console
        ]
    )


def mic_vector(p1, p2, box_dims):
    """
    Calculate the Minimum Image Convention (MIC) vector (p1 - p2) between two points.
    """
    vec = p1 - p2
    # np.round automatically handles signs, shifting vector to [-L/2, L/2]
    vec = vec - box_dims * np.round(vec / box_dims)
    return vec


def calculate_angle(v1, v2):
    """Safely calculate angle between two vectors (Unit: degrees)."""
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
        
    dot_prod = np.dot(v1, v2)
    cosine_angle = np.clip(dot_prod / (norm_v1 * norm_v2), -1.0, 1.0)
    
    return degrees(acos(cosine_angle))


def calculate_metrics(p_na, p_oa, p_ob, p_cl, box_dims):
    """
    Calculate all required distances and angles.
    """
    # 1. Calculate Vectors
    # For Na-Oa-Ob angle
    v_oa_na = mic_vector(p_na, p_oa, box_dims)
    v_oa_ob = mic_vector(p_ob, p_oa, box_dims)
    
    # For Oa-Ob-Cl angle
    v_ob_oa = mic_vector(p_oa, p_ob, box_dims)
    v_ob_cl = mic_vector(p_cl, p_ob, box_dims)
    
    # For Na-Cl distance
    v_na_cl = mic_vector(p_cl, p_na, box_dims)
    
    # For new constraints
    v_na_ob = mic_vector(p_ob, p_na, box_dims)
    v_cl_oa = mic_vector(p_oa, p_cl, box_dims)

    # 2. Calculate Distances
    d_na_oa = np.linalg.norm(v_oa_na)
    d_oa_ob = np.linalg.norm(v_oa_ob)
    d_ob_cl = np.linalg.norm(v_ob_cl)
    d_na_cl = np.linalg.norm(v_na_cl)
    
    # Distances for new constraints
    d_na_ob = np.linalg.norm(v_na_ob)
    d_cl_oa = np.linalg.norm(v_cl_oa)

    # 3. Calculate Angles
    angle_na_oa_ob = calculate_angle(v_oa_na, v_oa_ob)
    angle_oa_ob_cl = calculate_angle(v_ob_oa, v_ob_cl)

    return (d_na_oa, d_oa_ob, d_ob_cl, d_na_cl, 
            d_na_ob, d_cl_oa, 
            angle_na_oa_ob, angle_oa_ob_cl)


def process_frame(atoms_data, box_dims, timestep, r1, rw, r2, f_out):
    """
    Process single frame, search for Na-Oa-Ob-Cl quadruplets.
    r1: Na-Oa cutoff (also used for Na-Ob min distance)
    rw: Oa-Ob cutoff
    r2: Ob-Cl cutoff (also used for Cl-Oa min distance)
    """
    start_time = time.time()

    na_atoms = []
    o_atoms = []
    cl_atoms = []

    # 1. Categorize atoms
    for atom in atoms_data:
        atom_id, atom_type, pos = atom
        if atom_type == NA_TYPE:
            na_atoms.append((atom_id, pos))
        elif atom_type == O_TYPE:
            o_atoms.append((atom_id, pos))
        elif atom_type == CL_TYPE:
            cl_atoms.append((atom_id, pos))

    # Skip if missing required atoms
    if not na_atoms or len(o_atoms) < 2 or not cl_atoms:
        logging.warning(f"  > TIMESTEP {timestep}: Not enough Na, O, or Cl. Skipping frame.")
        # Write filler line
        output_line = [str(timestep)] + ["-1"] * (MAX_QUADRUPLETS_PER_FRAME * FIELDS_PER_QUADRUPLET)
        f_out.write(" ".join(output_line) + "\n")
        return

    # 2. Prepare cKDTree data
    na_ids, na_pos_list = zip(*na_atoms)
    o_ids, o_pos_list = zip(*o_atoms)
    cl_ids, cl_pos_list = zip(*cl_atoms)

    # Convert to Numpy arrays
    na_pos = np.array(na_pos_list)
    o_pos = np.array(o_pos_list)
    cl_pos = np.array(cl_pos_list)
    
    # --- PBC Boundary Correction (Fix potential negative coordinates for cKDTree) ---
    # Ensure all coordinates are in [0, L)
    for dim in range(3):
        # na_pos, o_pos, cl_pos are relative to origin
        o_pos[:, dim] = o_pos[:, dim] - box_dims[dim] * np.floor(o_pos[:, dim] / box_dims[dim])
        cl_pos[:, dim] = cl_pos[:, dim] - box_dims[dim] * np.floor(cl_pos[:, dim] / box_dims[dim])
        na_pos[:, dim] = na_pos[:, dim] - box_dims[dim] * np.floor(na_pos[:, dim] / box_dims[dim])
    
    # --- Correction End ---

    # 3. Build KD-Trees
    try:
        o_tree = cKDTree(o_pos, boxsize=box_dims)
        cl_tree = cKDTree(cl_pos, boxsize=box_dims)
    except ValueError as e:
        logging.error(f"  > TIMESTEP {timestep}: Failed to build cKDTree: {e}")
        logging.error(f"  > Box Dims: {box_dims}")
        # Write -1 line
        output_line = [str(timestep)] + ["-1"] * (MAX_QUADRUPLETS_PER_FRAME * FIELDS_PER_QUADRUPLET)
        f_out.write(" ".join(output_line) + "\n")
        return

    found_quadruplets = []

    # 4. Four-Layer Search Na -> Oa -> Ob -> Cl
    # Iterate all Na
    for i in range(len(na_ids)):
        p_na = na_pos[i]
        id_na = na_ids[i]

        # 4a. Find Oa within r1 of Na
        oa_indices = o_tree.query_ball_point(p_na, r1)
        if not oa_indices: continue

        # Iterate found Oa
        for j in oa_indices:
            p_oa = o_pos[j]
            id_oa = o_ids[j]
            
            # 4b. Find Ob within rw of Oa
            ob_indices = o_tree.query_ball_point(p_oa, rw)
            if not ob_indices: continue
            
            # Iterate found Ob
            for k in ob_indices:
                # Ensure Oa and Ob are different atoms
                if j == k: continue
                
                p_ob = o_pos[k]
                id_ob = o_ids[k]

                # 4c. Find Cl within r2 of Ob
                cl_indices = cl_tree.query_ball_point(p_ob, r2)
                if not cl_indices: continue

                # Iterate found Cl
                for l in cl_indices:
                    # 4d. Found a (Na, Oa, Ob, Cl) quadruplet
                    p_cl = cl_pos[l]
                    id_cl = cl_ids[l]

                    # 4e. Calculate all metrics
                    (d_na_oa, d_oa_ob, d_ob_cl, d_na_cl, 
                     d_na_ob, d_cl_oa, 
                     angle_na_oa_ob, angle_oa_ob_cl) = calculate_metrics(
                         p_na, p_oa, p_ob, p_cl, box_dims
                     )
                    
                    # 4f. Apply new constraints
                    # Constraint 4: Na-Ob Distance >= r1
                    if d_na_ob < r1:
                        continue
                    
                    # Constraint 5: Cl-Oa Distance >= r2
                    if d_cl_oa < r2:
                        continue
                        
                    # Constraint 6: Na-Cl Distance between R_ION_MIN and R_ION_MAX (New)
                    if not (R_ION_MIN <= d_na_cl <= R_ION_MAX):
                        continue
                        
                    # Store Result
                    quadruplet_data = [
                        # Na (p1)
                        f"{id_na:d}", f"{p_na[0]:.6f}", f"{p_na[1]:.6f}", f"{p_na[2]:.6f}",
                        # Oa (p2)
                        f"{id_oa:d}", f"{p_oa[0]:.6f}", f"{p_oa[1]:.6f}", f"{p_oa[2]:.6f}",
                        # Ob (p3)
                        f"{id_ob:d}", f"{p_ob[0]:.6f}", f"{p_ob[1]:.6f}", f"{p_ob[2]:.6f}",
                        # Cl (p4)
                        f"{id_cl:d}", f"{p_cl[0]:.6f}", f"{p_cl[1]:.6f}", f"{p_cl[2]:.6f}",
                        # Distances (Na-Oa, Oa-Ob, Ob-Cl, Na-Cl)
                        f"{d_na_oa:.4f}", f"{d_oa_ob:.4f}", f"{d_ob_cl:.4f}", f"{d_na_cl:.4f}",
                        # Angles (Na-Oa-Ob, Oa-Ob-Cl)
                        f"{angle_na_oa_ob:.3f}", f"{angle_oa_ob_cl:.3f}"
                    ]
                    
                    if len(found_quadruplets) < MAX_QUADRUPLETS_PER_FRAME:
                        found_quadruplets.append(quadruplet_data)
                    else:
                        # Reached max count, break loop
                        break
                
                if len(found_quadruplets) >= MAX_QUADRUPLETS_PER_FRAME: break
            if len(found_quadruplets) >= MAX_QUADRUPLETS_PER_FRAME: break
        if len(found_quadruplets) >= MAX_QUADRUPLETS_PER_FRAME: break

    # 5. Format and Write to File
    output_parts = [str(timestep)]
    num_found = len(found_quadruplets)

    for i in range(MAX_QUADRUPLETS_PER_FRAME):
        if i < num_found:
            output_parts.extend(found_quadruplets[i])
        else:
            # Fill with -1 if fewer than max quadruplets found
            output_parts.extend(["-1"] * FIELDS_PER_QUADRUPLET)

    f_out.write(" ".join(output_parts) + "\n")
    end_time = time.time()
    logging.info(f"  > TIMESTEP {timestep}: Found {num_found} quadruplets. "
                      f"(Time: {end_time - start_time:.2f}s)")


def write_output_header(f_out):
    """Write output file header, including Na-Oa-Ob-Cl quadruplet structure."""
    header_parts = ["TIMESTEP"]
    for i in range(1, MAX_QUADRUPLETS_PER_FRAME + 1):
        # Na
        header_parts.extend([
            f"Na{i}_ID", f"Na{i}_x", f"Na{i}_y", f"Na{i}_z"
        ])
        # Oa
        header_parts.extend([
            f"Oa{i}_ID", f"Oa{i}_x", f"Oa{i}_y", f"Oa{i}_z"
        ])
        # Ob
        header_parts.extend([
            f"Ob{i}_ID", f"Ob{i}_x", f"Ob{i}_y", f"Ob{i}_z"
        ])
        # Cl
        header_parts.extend([
            f"Cl{i}_ID", f"Cl{i}_x", f"Cl{i}_y", f"Cl{i}_z"
        ])
        # Metrics: 4 Distances + 2 Angles
        header_parts.extend([
            f"Na{i}-Oa{i}_d", f"Oa{i}-Ob{i}_d", f"Ob{i}-Cl{i}_d", f"Na{i}-Cl{i}_d",
            f"Na{i}-Oa{i}-Ob{i}_angle", f"Oa{i}-Ob{i}-Cl{i}_angle"
        ])
    f_out.write(" ".join(header_parts) + "\n")


def process_trajectory(traj_file, r1, rw, r2, frame_interval, output_file, log_file):
    """
    Main trajectory processing function.
    """
    setup_logging(log_file)
    logging.info(f"Script started...")
    logging.info(f"Reading trajectory file: {traj_file}")
    logging.info(f"Na-Oa Radius (r1): {r1} Å (Also Na-Ob min dist)")
    logging.info(f"Oa-Ob Radius (rw): {rw} Å")
    logging.info(f"Ob-Cl Radius (r2): {r2} Å (Also Cl-Oa min dist)")
    logging.info(f"Na-Cl Constraint Range: [{R_ION_MIN} Å, {R_ION_MAX} Å]")
    logging.info(f"Frame Interval: {frame_interval}")
    logging.info(f"Output File: {output_file}")
    logging.info(f"Log File: {log_file}")
    
    try:
        # Open with 'w' to clear and write
        with open(traj_file, 'r') as f_in, open(output_file, 'w') as f_out:
            
            # Write header
            write_output_header(f_out)

            frame_count = 0
            processed_count = 0
            in_atom_section = False
            timestep = 0
            num_atoms = 0
            atoms_data = []
            box_dims = np.zeros(3)

            line = f_in.readline()
            while line:
                line_strip = line.strip()

                if line_strip.startswith("ITEM: TIMESTEP"):
                    # Process previous frame if interval met
                    if frame_count > 0 and (frame_count % frame_interval == 0):
                        logging.info(f"Processing Frame {frame_count} (TIMESTEP {timestep})...")
                        process_frame(atoms_data, box_dims, timestep, r1, rw, r2, f_out)
                        processed_count += 1

                    # ... (Read TIMESTEP and NUMBER OF ATOMS) ...
                    try:
                        timestep = int(f_in.readline().strip())
                    except ValueError:
                        logging.error("Failed to read TIMESTEP.")
                        break
                        
                    frame_count += 1
                    # Reset data
                    atoms_data = []
                    in_atom_section = False

                elif line_strip.startswith("ITEM: NUMBER OF ATOMS"):
                    try:
                        num_atoms = int(f_in.readline().strip())
                    except ValueError:
                        logging.error("Failed to read NUMBER OF ATOMS.")
                        break
                        
                elif line_strip.startswith("ITEM: BOX BOUNDS"):
                    try:
                        # Read x, y, z bounds
                        b1 = list(map(float, f_in.readline().split()))
                        b2 = list(map(float, f_in.readline().split()))
                        b3 = list(map(float, f_in.readline().split()))
                        # Calculate box dimensions (Lx, Ly, Lz)
                        box_dims = np.array([b1[1]-b1[0], b2[1]-b2[0], b3[1]-b3[0]])
                    except Exception as e:
                        logging.error(f"Failed to read BOX BOUNDS: {e}")
                        break

                elif line_strip.startswith("ITEM: ATOMS"):
                    in_atom_section = True
                        
                elif in_atom_section:
                    # Read atom data line
                    try:
                        parts = line_strip.split()
                        atom_id = int(parts[0])
                        atom_type = int(parts[1])
                        # Store only needed atoms
                        if atom_type in (NA_TYPE, O_TYPE, CL_TYPE):
                            # parts[2], [3], [4] are x, y, z
                            pos = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
                            atoms_data.append((atom_id, atom_type, pos))
                        
                    except (IndexError, ValueError) as e:
                        logging.warning(f"Failed to parse atom line: '{line_strip}' -> {e}")
                
                # Read next line
                line = f_in.readline()

            # End of loop, process last frame
            if frame_count > 0 and (frame_count % frame_interval == 0) and atoms_data:
                logging.info(f"Processing Last Frame {frame_count} (TIMESTEP {timestep})...")
                process_frame(atoms_data, box_dims, timestep, r1, rw, r2, f_out)
                processed_count += 1
            elif frame_count > 0 and (frame_count % frame_interval != 0):
                # If last frame index is not a multiple of frame_interval, but still needs processing
                logging.info(f"Processing Last Frame {frame_count} (TIMESTEP {timestep})...")
                process_frame(atoms_data, box_dims, timestep, r1, rw, r2, f_out)
                processed_count += 1


            logging.info(f"Trajectory processing complete. Read {frame_count} frames.")
            logging.info(f"Processed {processed_count} frames with interval {frame_interval}.")

    except FileNotFoundError:
        logging.error(f"Error: Trajectory file '{traj_file}' not found.")
        sys.exit(1)
    except ImportError:
        logging.error("Error: Missing dependency (numpy or scipy).")
        logging.error("Please install using 'pip install numpy scipy'.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        logging.exception("Error Details:")
        sys.exit(1)


# =============================================================================
# --- Main Entry Point ---
# =============================================================================
if __name__ == "__main__":
    
    # --- User Configuration ---
    
    # Trajectory filename
    TRAJ_FILE = "data/biNBs/NaCl/002/0-8ns/COM_shift/centered_bubble1.lammpstrj"
    
    # Cutoff Radius r1 (Na-Oa), Unit: Angstrom
    R1_NA_OA = 3.2
    
    # Cutoff Radius rw (Oa-Ob), Unit: Angstrom (Water O-O dist usually 3.0-3.5A)
    R_W_OA_OB = 3.2 
    
    # Cutoff Radius r2 (Ob-Cl), Unit: Angstrom
    R2_OB_CL = 4.2
    
    
    # Frame reading interval (e.g., 1 for every frame, 10 for every 10th frame)
    FRAME_INTERVAL = 1
    
    # Output filename
    OUTPUT_FILE = "na_oa_ob_cl_quadruplets_results_1.dat"
    
    # Log filename
    LOG_FILE = "trajectory_analysis_quadruplet_1.log"
    
    # --- Configuration End ---

    # Check input file
    if not os.path.exists(TRAJ_FILE):
        print(f"Error: Input file '{TRAJ_FILE}' does not exist. Check filename and path.")
        sys.exit(1)

    # Run main program
    process_trajectory(TRAJ_FILE, R1_NA_OA, R_W_OA_OB, R2_OB_CL, FRAME_INTERVAL, OUTPUT_FILE, LOG_FILE)