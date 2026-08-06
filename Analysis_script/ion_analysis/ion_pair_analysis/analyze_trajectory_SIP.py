#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trajectory Analysis Script:
Searches for Na-O-Cl atom triplets in LAMMPS trajectory files,
calculates relevant distances and angles, and handles Periodic Boundary Conditions (PBC).

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

# Atom Type Mapping
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

# --- Constraints ---
# Ion-Ion Distance Constraint (Na-Cl)
D_NA_CL_MIN = 3.8
D_NA_CL_MAX = 5.8
# --------------------

# Max triplets per frame in output file
MAX_TRIPLETS_PER_FRAME = 40
# Number of data fields per triplet (ID*3 + XYZ*3 + 3*Dist + 1*Angle)
FIELDS_PER_TRIPLET = 16


def setup_logging(log_file):
    """Configure logging to output to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'), # Write to log file
            logging.StreamHandler(sys.stdout)       # Print to console
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


def calculate_metrics(p_na, p_o, p_cl, box_dims):
    """
    Calculate all distances and angles given coordinates of Na, O, and Cl.
    """
    # 1. Calculate vectors (using MIC)
    # v_ona = Na - O
    v_ona = mic_vector(p_na, p_o, box_dims)
    # v_ocl = Cl - O
    v_ocl = mic_vector(p_cl, p_o, box_dims)
    # v_nacl = Na - Cl
    v_nacl = mic_vector(p_na, p_cl, box_dims)

    # 2. Calculate distances (Vector norms)
    d_na_o = np.linalg.norm(v_ona)
    d_o_cl = np.linalg.norm(v_ocl)
    d_na_cl = np.linalg.norm(v_nacl)

    # 3. Calculate Na-O-Cl angle
    # Angle based on two vectors originating from O: O->Na (v_ona) and O->Cl (v_ocl)
    v_o_na = v_ona 
    v_o_cl = v_ocl

    # Check for zero vectors (unlikely physically, but safe to check)
    if d_na_o == 0 or d_o_cl == 0:
        angle = 0.0
    else:
        dot_prod = np.dot(v_o_na, v_o_cl)
        # Normalize and clip to avoid floating point errors in arccos
        cosine_angle = np.clip(dot_prod / (d_na_o * d_o_cl), -1.0, 1.0)
        angle = degrees(acos(cosine_angle))

    return d_na_o, d_o_cl, d_na_cl, angle


def process_frame(atoms_data, box_dims, timestep, r1_sq, r2_sq, f_out):
    """
    Process a single frame.
    Note: r1 and r2 are passed as squared values for efficiency check if needed,
    but cKDTree uses raw values.
    """
    start_time = time.time()
    r1 = sqrt(r1_sq)
    r2 = sqrt(r2_sq)

    na_atoms = []
    o_atoms = []
    cl_atoms = []

    # 1. Categorize atoms by type
    for atom in atoms_data:
        atom_id, atom_type, pos = atom
        if atom_type == NA_TYPE:
            na_atoms.append((atom_id, pos))
        elif atom_type == O_TYPE:
            o_atoms.append((atom_id, pos))
        elif atom_type == CL_TYPE:
            cl_atoms.append((atom_id, pos))

    # Skip frame if any atom type is missing
    if not na_atoms or not o_atoms or not cl_atoms:
        logging.warning(f"  > TIMESTEP {timestep}: Missing Na, O, or Cl. Skipping frame.")
        # Write a filler line
        output_line = [str(timestep)] + ["-1"] * (MAX_TRIPLETS_PER_FRAME * FIELDS_PER_TRIPLET)
        f_out.write(" ".join(output_line) + "\n")
        return

    # 2. Prepare data for cKDTree
    # Unzip (id, pos) lists
    na_ids, na_pos_list = zip(*na_atoms)
    o_ids, o_pos_list = zip(*o_atoms)
    cl_ids, cl_pos_list = zip(*cl_atoms)

    # Convert to Numpy arrays
    na_pos = np.array(na_pos_list)
    o_pos = np.array(o_pos_list)
    cl_pos = np.array(cl_pos_list)
    
    # --- PBC Boundary Correction (Fix potential negative coordinates for cKDTree) ---
    # Ensure all coordinates strictly satisfy 0 <= p < L
    # Use floor division to wrap coordinates
    
    for dim in range(3):
        # na_pos, o_pos, cl_pos are already relative to origin
        o_pos[:, dim] = o_pos[:, dim] - box_dims[dim] * np.floor(o_pos[:, dim] / box_dims[dim])
        cl_pos[:, dim] = cl_pos[:, dim] - box_dims[dim] * np.floor(cl_pos[:, dim] / box_dims[dim])
        na_pos[:, dim] = na_pos[:, dim] - box_dims[dim] * np.floor(na_pos[:, dim] / box_dims[dim])
    
    # --- Correction End ---

    # 3. Build KD-Trees
    # boxsize argument informs cKDTree about periodic boundaries
    try:
        o_tree = cKDTree(o_pos, boxsize=box_dims)
        cl_tree = cKDTree(cl_pos, boxsize=box_dims)
    except ValueError as e:
        logging.error(f"  > TIMESTEP {timestep}: Failed to build cKDTree: {e}")
        logging.error(f"  > Box Dims: {box_dims}")
        # Write -1 line
        output_line = [str(timestep)] + ["-1"] * (MAX_TRIPLETS_PER_FRAME * FIELDS_PER_TRIPLET)
        f_out.write(" ".join(output_line) + "\n")
        return

    found_triplets = []

    # 4. Perform Three-Layer Search
    # Iterate over all Na
    for i in range(len(na_ids)):
        p_na = na_pos[i]
        id_na = na_ids[i]

        # 4a. Find all O within r1 of Na (using KDTree)
        o_indices = o_tree.query_ball_point(p_na, r1)
        if not o_indices:
            continue

        # Iterate found O
        for j in o_indices:
            p_o = o_pos[j]
            id_o = o_ids[j]

            # 4b. Find all Cl within r2 of O (using KDTree)
            cl_indices = cl_tree.query_ball_point(p_o, r2)
            if not cl_indices:
                continue

            # Iterate found Cl
            for k in cl_indices:
                # 4c. Found a (Na, O, Cl) triplet
                p_cl = cl_pos[k]
                id_cl = cl_ids[k]

                # Calculate metrics (Distances and Angle)
                d_na_o, d_o_cl, d_na_cl, angle = calculate_metrics(p_na, p_o, p_cl, box_dims)

                # --- New Constraint: Na-Cl Distance ---
                if not (D_NA_CL_MIN <= d_na_cl <= D_NA_CL_MAX):
                    continue
                # --- Constraint Passed ---

                # Store result
                triplet_data = [
                    f"{id_na:d}", f"{p_na[0]:.6f}", f"{p_na[1]:.6f}", f"{p_na[2]:.6f}",
                    f"{id_o:d}", f"{p_o[0]:.6f}", f"{p_o[1]:.6f}", f"{p_o[2]:.6f}",
                    f"{id_cl:d}", f"{p_cl[0]:.6f}", f"{p_cl[1]:.6f}", f"{p_cl[2]:.6f}",
                    f"{d_na_o:.4f}", f"{d_o_cl:.4f}", f"{d_na_cl:.4f}", f"{angle:.3f}"
                ]
                found_triplets.append(triplet_data)

    # 5. Format and write to file
    output_parts = [str(timestep)]
    num_found = len(found_triplets)

    for i in range(MAX_TRIPLETS_PER_FRAME):
        if i < num_found:
            output_parts.extend(found_triplets[i])
        else:
            # Fill with -1 if fewer than 40 triplets found
            output_parts.extend(["-1"] * FIELDS_PER_TRIPLET)

    f_out.write(" ".join(output_parts) + "\n")
    end_time = time.time()
    logging.info(f"  > TIMESTEP {timestep}: Found {num_found} triplets. "
                 f"(Time: {end_time - start_time:.2f}s)")


def write_output_header(f_out):
    """Write data header to output file."""
    header_parts = ["TIMESTEP"]
    for i in range(1, MAX_TRIPLETS_PER_FRAME + 1):
        # Na
        header_parts.extend([
            f"Na{i}_ID", f"Na{i}_x", f"Na{i}_y", f"Na{i}_z"
        ])
        # O
        header_parts.extend([
            f"O{i}_ID", f"O{i}_x", f"O{i}_y", f"O{i}_z"
        ])
        # Cl
        header_parts.extend([
            f"Cl{i}_ID", f"Cl{i}_x", f"Cl{i}_y", f"Cl{i}_z"
        ])
        # Metrics
        header_parts.extend([
            f"Na{i}-O{i}_d", f"O{i}-Cl{i}_d", f"Na{i}-Cl{i}_d", f"Na{i}-O{i}-Cl{i}_angle"
        ])
    f_out.write(" ".join(header_parts) + "\n")


def process_trajectory(traj_file, r1, r2, frame_interval, output_file, log_file):
    """
    Main trajectory processing function.
    """
    setup_logging(log_file)
    logging.info(f"Script started...")
    logging.info(f"Reading trajectory file: {traj_file}")
    logging.info(f"Na-O Radius (r1): {r1} Å")
    logging.info(f"O-Cl Radius (r2): {r2} Å")
    logging.info(f"Na-Cl Constraint Range: [{D_NA_CL_MIN}, {D_NA_CL_MAX}] Å")
    logging.info(f"Frame Interval: {frame_interval}")
    logging.info(f"Output File: {output_file}")
    logging.info(f"Log File: {log_file}")
    
    # Pre-calculate squared distances
    r1_sq = r1 * r1
    r2_sq = r2 * r2

    try:
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
                    # Start of a frame
                    # Process previous frame if exists
                    if frame_count > 0 and (frame_count % frame_interval == 0):
                        logging.info(f"Processing Frame {frame_count} (TIMESTEP {timestep})...")
                        process_frame(atoms_data, box_dims, timestep, r1_sq, r2_sq, f_out)
                        processed_count += 1

                    # Read new timestep
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
                        atoms_data = [] 
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
                    # Read atom data lines
                    try:
                        parts = line_strip.split()
                        atom_id = int(parts[0])
                        atom_type = int(parts[1])
                        # Store only needed atom types
                        if atom_type in (NA_TYPE, O_TYPE, CL_TYPE):
                            pos = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
                            atoms_data.append((atom_id, atom_type, pos))
                        
                    except (IndexError, ValueError) as e:
                        logging.warning(f"Failed to parse atom line: '{line_strip}' -> {e}")
                
                # Read next line
                line = f_in.readline()

            # End of file, process last frame
            if frame_count > 0 and (frame_count % frame_interval == 0):
                logging.info(f"Processing Last Frame {frame_count} (TIMESTEP {timestep})...")
                process_frame(atoms_data, box_dims, timestep, r1_sq, r2_sq, f_out)
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
    
    # Cutoff Radius r1 (Na-O), Unit: Angstrom
    R1_NA_O = 3.2
    
    # Cutoff Radius r2 (O-Cl), Unit: Angstrom
    R2_O_CL = 4.2
    
    # Frame reading interval (e.g., 1 for every frame, 100 for every 100th frame)
    FRAME_INTERVAL = 1
    
    # Output filename
    OUTPUT_FILE = "na_o_cl_triplets_results_1.dat"
    
    # Log filename
    LOG_FILE = "trajectory_analysis_1.log"
    
    # --- Configuration End ---

    # Check if input file exists
    if not os.path.exists(TRAJ_FILE):
        print(f"Error: Input file '{TRAJ_FILE}' does not exist. Check filename and path.")
        sys.exit(1)

    # Run main program
    process_trajectory(TRAJ_FILE, R1_NA_O, R2_O_CL, FRAME_INTERVAL, OUTPUT_FILE, LOG_FILE)