import numpy as np
import argparse
import os
import sys
from pathlib import Path

def read_com_file(filename):
    """Read Center of Mass (COM) file, returns dict: {frame: [x, y, z]}"""
    try:
        com_dict = {}
        with open(filename, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.split()
                if not parts: 
                    continue
                frame = int(parts[0])
                x, y, z = map(float, parts[1:4])
                com_dict[frame] = [x, y, z]
        return com_dict
    except FileNotFoundError:
        print(f"Error: File not found {filename}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file {filename}: {str(e)}")
        sys.exit(1)

def shift_atoms_position(atoms, shift, box_bounds):
    """Shift atom positions and handle Periodic Boundary Conditions (PBC)"""
    shifted_atoms = []
    Lx = box_bounds[0][1] - box_bounds[0][0]
    Ly = box_bounds[1][1] - box_bounds[1][0]
    Lz = box_bounds[2][1] - box_bounds[2][0]
    
    for atom in atoms:
        x = atom[2] + shift[0]
        y = atom[3] + shift[1]
        z = atom[4] + shift[2]
        
        # Handle PBC
        x = (x - box_bounds[0][0]) % Lx + box_bounds[0][0]
        y = (y - box_bounds[1][0]) % Ly + box_bounds[1][0]
        z = (z - box_bounds[2][0]) % Lz + box_bounds[2][0]
        
        shifted_atoms.append([atom[0], atom[1], x, y, z])
    return shifted_atoms

def format_coord(value):
    """Format coordinate: width 10, 6 decimal places"""
    return f"{value:10.6f}"

def recenter_traj(input_filename, output_filename):
    """Recenter trajectory: Modify box bounds and atom coordinates"""
    try:
        with open(input_filename, 'r') as infile, \
             open(output_filename, 'w') as outfile:
            
            frame_count = 0
            while True:
                # Read frame header
                timestep_line = infile.readline().strip()
                if not timestep_line:  # End of file
                    break
                step_line = infile.readline().strip()
                num_atoms_line = infile.readline().strip()
                if not num_atoms_line: break
                num_atoms = int(infile.readline().strip())
                box_line = infile.readline().strip()
                
                # Read box bounds
                box_bounds = []
                for _ in range(3):
                    lo, hi = map(float, infile.readline().split())
                    box_bounds.append([lo, hi])
                
                atoms_line = infile.readline().strip()
                
                # Read atom info
                atoms = []
                for _ in range(num_atoms):
                    atom_line = infile.readline().split()
                    if not atom_line: break
                    atom_id = int(atom_line[0])
                    atom_type = int(atom_line[1])
                    x, y, z = map(float, atom_line[2:5])
                    atoms.append([atom_id, atom_type, x, y, z])
                
                # Get box dimensions
                Lx = box_bounds[0][1] - box_bounds[0][0]
                Ly = box_bounds[1][1] - box_bounds[1][0]
                Lz = box_bounds[2][1] - box_bounds[2][0]
                
                # Calculate shift (move center of original box to 0,0,0)
                shift_x = -(box_bounds[0][0] + Lx/2)
                shift_y = -(box_bounds[1][0] + Ly/2)
                shift_z = -(box_bounds[2][0] + Lz/2)
                
                # Calculate new box bounds (from -0.5L to 0.5L)
                new_box_bounds = [
                    [-Lx/2, Lx/2],            # x: -L to L (Note: code logic in original seems to imply specific scaling, kept as is)
                    [-Ly/2, Ly/2],            # y
                    [-Lz/2, Lz/2]             # z
                ]
                
                # Shift all atoms
                recentered_atoms = []
                for atom in atoms:
                    x = atom[2] + shift_x
                    y = atom[3] + shift_y
                    z = atom[4] + shift_z
                    recentered_atoms.append([
                        atom[0], 
                        atom[1], 
                        x, 
                        y, 
                        z
                    ])
                
                # Write to output file
                outfile.write(f"{timestep_line}\n")
                outfile.write(f"{step_line}\n")
                outfile.write(f"{num_atoms_line}\n")
                outfile.write(f"{num_atoms}\n")
                outfile.write(f"{box_line}\n")
                for bounds in new_box_bounds:
                    lo_str = format_coord(bounds[0])
                    hi_str = format_coord(bounds[1])
                    outfile.write(f"{lo_str} {hi_str}\n")
                
                outfile.write(f"{atoms_line}\n")
                for atom in recentered_atoms:
                    id_str = f"{atom[0]:>6}"       
                    type_str = f"{atom[1]:>3}"      
                    x_str = format_coord(atom[2])
                    y_str = format_coord(atom[3])
                    z_str = format_coord(atom[4])
                    outfile.write(f"{id_str} {type_str} {x_str} {y_str} {z_str}\n")
                
                frame_count += 1
                print(f"Recentered {frame_count} frames...", end='\r')
    except FileNotFoundError:
        print(f"Error: File not found {input_filename}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)

def process_trajectory(input_traj, com1_dict, com2_dict, output_traj1, output_traj2):
    """Process trajectory and generate two shifted trajectories"""
    try:
        # Create temp files
        output_dir = os.path.dirname(output_traj1)
        temp_traj1 = os.path.join(output_dir, "temp_shifted_bubble1.lammpstrj")
        temp_traj2 = os.path.join(output_dir, "temp_shifted_bubble2.lammpstrj")

        with open(input_traj, 'r') as infile, \
             open(temp_traj1, 'w') as outfile1, \
             open(temp_traj2, 'w') as outfile2:
            
            frame_count = 0
            while True:
                # Read header
                timestep = infile.readline().strip()
                if not timestep: 
                    break
                step = infile.readline().strip()
                num_atoms_line = infile.readline().strip()
                if not num_atoms_line: break
                num_atoms = int(infile.readline().strip())
                box_line = infile.readline().strip()
                box_bounds = []
                for _ in range(3):
                    line = infile.readline()
                    if not line: 
                        break
                    lo, hi = map(float, line.split())
                    box_bounds.append([lo, hi])
                
                atoms_line = infile.readline().strip()
                # Read atoms
                atoms = []
                for _ in range(num_atoms):
                    atom_line = infile.readline().split()
                    if not atom_line: break
                    atom_id = int(atom_line[0])
                    atom_type = int(atom_line[1])
                    x, y, z = map(float, atom_line[2:5])
                    atoms.append([atom_id, atom_type, x, y, z])
                
                frame_count += 1
                # Get COM for current frame
                com1 = com1_dict.get(frame_count)
                com2 = com2_dict.get(frame_count)
                if com1 is None or com2 is None:
                    raise ValueError(f"Missing COM data for frame {frame_count}")
                
                Ly = box_bounds[1][1] - box_bounds[1][0]
                Lz = box_bounds[2][1] - box_bounds[2][0]
                
                # Calculate target position (Based on dimensions)
                target1 = [1.0 * Ly, 0.5 * Ly, 0.5 * Lz] 
                target2 = [1.0 * Ly, 0.5 * Ly, 0.5 * Lz] 
                
                # Calculate shift vector
                shift1 = [target1[0] - com1[0], target1[1] - com1[1], target1[2] - com1[2]]
                shift2 = [target2[0] - com2[0], target2[1] - com2[1], target2[2] - com2[2]]
                
                # Shift atoms
                shifted_atoms1 = shift_atoms_position(atoms, shift1, box_bounds)
                shifted_atoms2 = shift_atoms_position(atoms, shift2, box_bounds)
                
                # Write Trajectory 1
                outfile1.write(f"{timestep}\n")
                outfile1.write(f"{step}\n")
                outfile1.write(f"{num_atoms_line}\n")
                outfile1.write(f"{num_atoms}\n")
                outfile1.write(f"{box_line}\n")
                for bounds in box_bounds:
                    lo_str = format_coord(bounds[0])
                    hi_str = format_coord(bounds[1])
                    outfile1.write(f"{lo_str} {hi_str}\n")
                
                outfile1.write(f"{atoms_line}\n")
                for atom in shifted_atoms1:
                    id_str = f"{atom[0]:>6}"       
                    type_str = f"{atom[1]:>3}"      
                    x_str = format_coord(atom[2])
                    y_str = format_coord(atom[3])
                    z_str = format_coord(atom[4])
                    outfile1.write(f"{id_str} {type_str} {x_str} {y_str} {z_str}\n")
                
                # Write Trajectory 2
                outfile2.write(f"{timestep}\n")
                outfile2.write(f"{step}\n")
                outfile2.write(f"{num_atoms_line}\n")
                outfile2.write(f"{num_atoms}\n")
                outfile2.write(f"{box_line}\n")
                for bounds in box_bounds:
                    lo_str = format_coord(bounds[0])
                    hi_str = format_coord(bounds[1])
                    outfile2.write(f"{lo_str} {hi_str}\n")
                
                outfile2.write(f"{atoms_line}\n")
                for atom in shifted_atoms2:
                    id_str = f"{atom[0]:>6}"       
                    type_str = f"{atom[1]:>3}"      
                    x_str = format_coord(atom[2])
                    y_str = format_coord(atom[3])
                    z_str = format_coord(atom[4])
                    outfile2.write(f"{id_str} {type_str} {x_str} {y_str} {z_str}\n")
                
                print(f"Shifted {frame_count} frames...", end='\r')
            print() 

        # Recenter after processing
        recenter_traj(temp_traj1, output_traj1)
        recenter_traj(temp_traj2, output_traj2)

        # Clean temp files
        try:
            os.remove(temp_traj1)
            os.remove(temp_traj2)
        except OSError:
            pass 

    except FileNotFoundError as e:
        print(f"Error: File not found {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing trajectory: {str(e)}")
        sys.exit(1)

def main(args):
    # Check input files
    for input_file in [args.input_traj, args.com1_file, args.com2_file]:
        if not os.path.exists(input_file):
            print(f"Error: Input file does not exist: {input_file}")
            sys.exit(1)
    
    # Ensure output dir exists
    for output_file in [args.output_traj1, args.output_traj2]:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except OSError as e:
                print(f"Error: Cannot create directory {output_dir}: {str(e)}")
                sys.exit(1)
    
    # Read COM data
    print("Reading COM data...")
    com1_dict = read_com_file(args.com1_file)
    com2_dict = read_com_file(args.com2_file)
    
    print(f"Read {len(com1_dict)} frames for Bubble 1 COM")
    print(f"Read {len(com2_dict)} frames for Bubble 2 COM")
    
    print("\nStarting process: Shift bubble COM and recenter...")
    process_trajectory(
        args.input_traj,
        com1_dict,
        com2_dict,
        args.output_traj1,
        args.output_traj2
    )
    
    print("\nProcess complete! Trajectories saved to:")
    print(f"Bubble 1: {args.output_traj1}")
    print(f"Bubble 2: {args.output_traj2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process trajectory and shift COM')
    parser.add_argument('--input-traj', required=True, help='Input trajectory file path')
    parser.add_argument('--com1-file', required=True, help='Bubble 1 COM file path')
    parser.add_argument('--com2-file', required=True, help='Bubble 2 COM file path')
    parser.add_argument('--output-traj1', required=True, help='Bubble 1 output trajectory path')
    parser.add_argument('--output-traj2', required=True, help='Bubble 2 output trajectory path')
    
    args = parser.parse_args()
    main(args)