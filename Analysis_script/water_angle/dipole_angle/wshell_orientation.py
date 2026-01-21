#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.lib.distances import calc_angles
import warnings
import os
warnings.filterwarnings('ignore')

def create_universe_from_lammpstrj(filename):
    """Create MDAnalysis Universe from LAMMPS trajectory"""
    # Create Universe, specify atom style
    u = mda.Universe(filename, format='LAMMPSDUMP', 
                    atom_style='id type x y z',
                    dt=1.0)  # timestep 1.0
    
    # Output System Info
    print("\nSystem Info:")
    print(f"- Total atoms: {len(u.atoms)}")
    print(f"- Hydrogen atoms: {len(u.select_atoms('type 1'))}")
    print(f"- Oxygen atoms: {len(u.select_atoms('type 2'))}")
    print(f"- Box dimensions: {u.dimensions[:3]}")
    
    return u

def find_water_molecules_in_shell(universe, frame_idx, r_min, r_max, debug=False):
    """Find water molecules within specified shell range"""
    # Switch frame
    universe.trajectory[frame_idx]
    
    # Select O and H
    oxygen_atoms = universe.select_atoms('type 2')
    hydrogen_atoms = universe.select_atoms('type 1')
    
    # Get positions
    o_positions = oxygen_atoms.positions
    h_positions = hydrogen_atoms.positions 
    
    # Calculate distance from origin (O atoms)
    o_distances = np.linalg.norm(o_positions, axis=1)
    
    # Filter O atoms in shell
    shell_mask = (o_distances >= r_min) & (o_distances <= r_max)
    shell_oxygen = oxygen_atoms[shell_mask]
    shell_o_positions = o_positions[shell_mask]
    
    if debug:
        print(f"\nDebug info for Frame {frame_idx + 1}:")
        print(f"- Total O atoms: {len(oxygen_atoms)}")
        print(f"- O atoms in shell: {len(shell_oxygen)}")
        print(f"- Total H atoms: {len(hydrogen_atoms)}")
    
    if len(shell_oxygen) == 0:
        return [], {}
    
    water_angles = []
    water_details = {
        'total_oxygen': len(oxygen_atoms),
        'shell_oxygen': len(shell_oxygen),
        'total_hydrogen': len(hydrogen_atoms),
        'water_molecules': []
    }
    
    # Find two nearest H atoms for each O in shell
    for o_idx, o_pos in enumerate(shell_o_positions):
        # Distances to all H atoms
        h_distances = np.linalg.norm(h_positions - o_pos, axis=1)
        
        # Find 2 nearest H indices
        nearest_h_indices = np.argsort(h_distances)[:2]
        h1_pos = h_positions[nearest_h_indices[0]]
        h2_pos = h_positions[nearest_h_indices[1]]
        
        h1_dist = h_distances[nearest_h_indices[0]]
        h2_dist = h_distances[nearest_h_indices[1]]
        
        # Check O-H bond length
        if h1_dist < 1.2 and h2_dist < 1.2:
            # Calculate H-H midpoint
            h_midpoint = (h1_pos + h2_pos) / 2
            
            # Calculate vector
            water_vector = h_midpoint - o_pos
            bubble_vector = o_pos  # Bubble center at origin (0,0,0)
            
            # Calculate Angle
            cos_angle = np.dot(water_vector, bubble_vector)
            cos_angle /= (np.linalg.norm(water_vector) * np.linalg.norm(bubble_vector))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            
            water_angles.append(angle)
            
            # Record details
            water_details['water_molecules'].append({
                'o_id': shell_oxygen[o_idx].id,
                'h1_id': hydrogen_atoms[nearest_h_indices[0]].id,
                'h2_id': hydrogen_atoms[nearest_h_indices[1]].id,
                'o_distance': o_distances[shell_mask][o_idx],
                'h1_bond_length': h1_dist,
                'h2_bond_length': h2_dist,
                'angle': angle
            })
    
    if debug and water_details['water_molecules']:
        print("Water Molecule Example:")
        example = water_details['water_molecules'][0]
        print(f"- O ID: {example['o_id']}")
        print(f"- H IDs: {example['h1_id']}, {example['h2_id']}")
        print(f"- Bond Lengths: {example['h1_bond_length']:.3f}Å, {example['h2_bond_length']:.3f}Å")
        print(f"- Distance to center: {example['o_distance']:.3f}Å")
        print(f"- Orientation Angle: {example['angle']:.2f}°")
    
    return water_angles, water_details

def analyze_water_orientation(traj_file, r_min, r_max, output_dir=".", debug=True):
    """Analyze water orientation distribution"""
    print(f"\nStart analysis: {traj_file}")
    print(f"Range: {r_min} Å - {r_max} Å")
    print(f"Output Dir: {output_dir}")
    
    # Create output dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Create Universe
    try:
        universe = create_universe_from_lammpstrj(traj_file)
    except Exception as e:
        print(f"Error: Cannot read trajectory - {str(e)}")
        return
    
    n_frames = len(universe.trajectory)
    print(f"\nTrajectory contains {n_frames} frames")
    
    # Collect angles
    all_angles = []
    total_waters = 0
    frame_statistics = []
    
    # Analyze frame by frame
    for ts in universe.trajectory:
        frame_idx = ts.frame
        if frame_idx % 10 == 0:
            print(f"\nProcessing Frame {frame_idx + 1}/{n_frames}...")
        
        # Analyze current frame
        frame_angles, frame_details = find_water_molecules_in_shell(
            universe, frame_idx, r_min, r_max, debug=(debug and frame_idx % 10 == 0)
        )
        
        all_angles.extend(frame_angles)
        total_waters += len(frame_angles)
        
        # Record stats
        frame_statistics.append({
            'frame': frame_idx + 1,
            'total_oxygen': frame_details.get('total_oxygen', 0),
            'shell_oxygen': frame_details.get('shell_oxygen', 0),
            'water_count': len(frame_angles)
        })
        
        if frame_idx % 10 == 0:
            print(f"  Found {len(frame_angles)} target water molecules")
    
    if not all_angles:
        print("\nWarning: No water molecules found in specified shell!")
        return
    
    print("\nAnalysis Complete!")
    print(f"Statistics:")
    print(f"- Total Frames: {n_frames}")
    print(f"- Total Water Molecules: {total_waters}")
    print(f"- Avg Waters per Frame: {total_waters/n_frames:.1f}")
    
    # Calculate Distribution
    print("\nCalculating Angle Distribution...")
    hist, bins = np.histogram(all_angles, bins=180, range=(0, 180), density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Calculate Stats
    angle_mean = np.mean(all_angles)
    angle_std = np.std(all_angles)
    angle_median = np.median(all_angles)
    print(f"Angle Stats:")
    print(f"- Mean: {angle_mean:.2f}°")
    print(f"- Std Dev: {angle_std:.2f}°")
    print(f"- Median: {angle_median:.2f}°")
    
    # Save Data
    output_prefix = os.path.join(output_dir, f'orientation_distribution_{r_min}_{r_max}')
    
    # Save Distribution Data
    output_txt = f'{output_prefix}.txt'
    data = np.column_stack((bin_centers, hist))
    header = (f"# Water Orientation Angle Distribution\n"
             f"# Range: {r_min}-{r_max} Å\n"
             f"# Total Frames: {n_frames}\n"
             f"# Total Waters: {total_waters}\n"
             f"# Mean Angle: {angle_mean:.2f}°\n"
             f"# Std Dev: {angle_std:.2f}°\n"
             f"# Median: {angle_median:.2f}°\n"
             f"# Angle(degrees) Probability_Density")
    np.savetxt(output_txt, data, header=header, comments='')
    
    # Save Frame Stats
    stats_txt = f'{output_prefix}_frame_stats.txt'
    with open(stats_txt, 'w') as f:
        f.write("# Frame Statistics\n")
        f.write("Frame TotalOxygen ShellOxygen WaterCount\n")
        for stat in frame_statistics:
            f.write(f"{stat['frame']} {stat['total_oxygen']} "
                   f"{stat['shell_oxygen']} {stat['water_count']}\n")
    
    # Plot
    print("\nGenerating Plots...")
    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    plt.plot(bin_centers, hist, 'b-', linewidth=2)
    plt.xlabel('Orientation Angle')
    plt.ylabel('Probability Density')
    plt.title(f'Orientation Distribution (r = {r_min}-{r_max} Å)')
    plt.grid(True)
    
    # Plot count change
    plt.subplot(212)
    frame_nums = [stat['frame'] for stat in frame_statistics]
    water_counts = [stat['water_count'] for stat in frame_statistics]
    plt.plot(frame_nums, water_counts, 'g-')
    plt.xlabel('Frame Number')
    plt.ylabel('Water Molecule Count')
    plt.title('Water Molecule Count Change')
    plt.grid(True)
    
    plt.tight_layout()
    output_png = f'{output_prefix}.png'
    plt.savefig(output_png, dpi=300)
    plt.close()
    
    print(f"\nOutput files:")
    print(f"- Distribution Data: {output_txt}")
    print(f"- Frame Statistics: {stats_txt}")
    print(f"- Plot: {output_png}")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) not in [4, 5]:
        print("Usage: python wshell_orientation.py <traj_file> <min_r> <max_r> [output_dir]")
        print("Example: python wshell_orientation.py traj.lammpstrj 5 15 ./output")
        sys.exit(1)
    
    traj_file = sys.argv[1]
    r_min = float(sys.argv[2])
    r_max = float(sys.argv[3])
    output_dir = sys.argv[4] if len(sys.argv) == 5 else "."
    
    analyze_water_orientation(traj_file, r_min, r_max, output_dir)