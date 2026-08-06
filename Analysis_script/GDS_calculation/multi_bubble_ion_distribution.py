#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import MDAnalysis as mda

# ==================== Command Line Argument Parsing ====================

parser = argparse.ArgumentParser(description="Ion distribution analysis for double bubble systems over multiple time blocks")
parser.add_argument("--save_path", default="./results", help="Path to save results")
parser.add_argument("--oh_files", nargs=2, help="Paths to hydroxide ion files for the two bubbles")
parser.add_argument("--na_files", nargs=2, help="Paths to sodium ion files for the two bubbles")
parser.add_argument("--trj_files", nargs=2, help="Paths to trajectory files for the two bubbles")
parser.add_argument("--geo_files", nargs=2, help="Paths to geometry files for the two bubbles")
parser.add_argument("--delta_r", type=float, default=0.1, help="Bin width for radial distribution calculation")
parser.add_argument("--time_blocks", type=int, default=8, help="Number of time blocks")
parser.add_argument("--z_low", type=float, default=0.0, help="Lower limit for Z axis")
parser.add_argument("--z_high", type=float, default=30.0, help="Upper limit for Z axis")
parser.add_argument("--skip_first_frames", type=int, default=0, help="Number of initial frames to skip")
parser.add_argument("--skip_last_frames", type=int, default=0, help="Number of final frames to ignore")

args = parser.parse_args()

# Create directory for saving results
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

# ==================== Global Font Settings ====================

SMALL_SIZE = 18
MEDIUM_SIZE = 18
BIG_SIZE = 20
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE-4)
plt.rc('figure', titlesize=BIG_SIZE)

# ==================== Physical Constants ====================
c_mass  = 1.6605390666e-27  # kg
h_mass  = 1.008  * c_mass * 1e3  # g
o_mass  = 15.999 * c_mass * 1e3  # g
n_mass  = 14.007 * c_mass * 1e3  # g
na_mass = 22.990 * c_mass * 1e3  # g
cl_mass = 35.450 * c_mass * 1e3  # g

# ==================== Utility Functions ====================

def get_mid_density(coord_list, density_list):
    """
    Calculate the interface position corresponding to half the density.
    """
    # Find the maximum position index corresponding to approximately 80% of z_high, 
    # to avoid unstable densities at box edges.
    max_idx = next((i for i, val in enumerate(coord_list) if val > args.z_high * 0.8), len(coord_list)-1)
    
    # Calculate start and end indices based on the valid region
    start = int(max_idx * 0.825)
    end = int(max_idx * 0.995)
    
    # Ensure indices are within valid range
    start = max(0, min(start, len(density_list)-2))
    end = max(start+1, min(end, len(density_list)-1))
    
    print(f"Calculating mid_density using coordinate range: {coord_list[start]:.2f} - {coord_list[end]:.2f} A")
    
    mid_density = np.mean(density_list[start:end]) / 2
    for i in range(len(density_list)-1):
        if density_list[i] < mid_density and density_list[i+1] > mid_density:
            mid_interface = (mid_density - density_list[i]) / (density_list[i+1] - density_list[i]) * (coord_list[i+1] - coord_list[i]) + coord_list[i]
            return mid_interface, mid_density
    return None, mid_density

def get_ion_list(txt):
    """
    Extract ion list from hydroxide or sodium ion files.
    """
    total_list = []
    for i in range(len(txt)):
        for j in range(int(txt[i][0])):
            if len(txt[i]) > 4 and txt[i][0] > 0:  # Ensure enough columns exist
                if len(txt[i]) >= 6+4*j:  # Hydroxide ion format (ID, x, y, z, ID, x, y, z)
                    total_list.append([txt[i][3+4*j], txt[i][4+4*j], txt[i][5+4*j]])
                elif len(txt[i]) >= 3+4*j:  # Sodium ion format (ID, x, y, z)
                    total_list.append([txt[i][2+4*j], txt[i][3+4*j], txt[i][4+4*j]])
    return total_list

def get_distance(xyz1, xyz2):
    """
    Calculate distance between two points.
    """
    distance = 0
    for i in range(3):
        distance += np.square((xyz1[i]-xyz2[i]))
    return np.sqrt(distance)

def get_center(list1):
    """
    Calculate distribution of ion distances to the center.
    """
    center = [0.0, 0.0, 0.0]
    distribution = []
    for list_i in list1:
        dist_list = [get_distance(i, center) for i in list_i]
        distribution.append(dist_list)
    return distribution

def calculate_density(u, frames, delta_r, box_center=[0,0,0]):
    """
    Calculate density distribution.
    """
    # Calculate box length
    box_length = min([u.trajectory[i].dimensions[0] for i in frames])
    
    # Use z_high as upper limit for density calculation, but not exceeding half box size
    max_radius = min(args.z_high, box_length/2)
    bin_edges = np.linspace(0, max_radius, int(max_radius/delta_r)+1)
    bin_centers = bin_edges[:-1] + (bin_edges[1]-bin_edges[0])/2
    
    # Calculate volumes
    v_list = [4/3*np.pi*(bin_edges[i+1]**3 - bin_edges[i]**3)*1e-24 for i in range(len(bin_centers))]
    
    # Initialize count matrices
    o_counts = np.zeros(len(bin_centers))
    h_counts = np.zeros(len(bin_centers))
    n_counts = np.zeros(len(bin_centers))
    na_counts = np.zeros(len(bin_centers))
    cl_counts = np.zeros(len(bin_centers))
    
    # Calculate distance of each atom to center and count
    for frame in frames:
        u.trajectory[frame]
        for atom in u.atoms:
            d_cent_atom = get_distance(atom.position, box_center)
            bin_idx = np.searchsorted(bin_edges[1:], d_cent_atom)
            if bin_idx < len(bin_centers):
                if atom.type == '1':
                    h_counts[bin_idx] += 1
                elif atom.type == '2':
                    o_counts[bin_idx] += 1
                elif atom.type == '3':
                    n_counts[bin_idx] += 1
                elif atom.type == '4':
                    na_counts[bin_idx] += 1
                else:
                    cl_counts[bin_idx] += 1
    
    # Calculate average density
    h_density = h_counts * h_mass / (len(frames) * np.array(v_list))
    o_density = o_counts * o_mass / (len(frames) * np.array(v_list))
    n_density = n_counts * n_mass / (len(frames) * np.array(v_list))
    na_density = na_counts * na_mass / (len(frames) * np.array(v_list))
    cl_density = cl_counts * cl_mass / (len(frames) * np.array(v_list))
    
    # Calculate total density and water density
    total_density = h_density + o_density + n_density + na_density + cl_density
    water_density = h_density + o_density
    
    return bin_centers, total_density, water_density, n_density

def draw_density(radial_d, total_density, water_density, n_density, interface, z_low, z_high, savepath, time_section):
    """
    Plot density distribution.
    """
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    ax.hlines(0.5, -0., 30, colors="black", linestyles='dashed')
    for k in interface:
        ax.vlines(k, -0.1, 1.1, colors="grey", linestyles='dashed')

    ax.plot(radial_d, total_density, color='#2BBBD8', lw=2, label='Total Density')
    ax.plot(radial_d, water_density, color='#1A936F', lw=2, label='Water Density')
    ax.plot(radial_d, n_density, color='#E76F51', lw=2, label='Nitrogen Density')
    
    ax.set_xlim(z_low, z_high)
    ax.legend(fontsize=10)
    ax.set_xlabel(r"$D_\mathrm{radial}$"+r"$ \ \rm (\AA)$")
    ax.set_ylabel(r"$\rho$ (g/" + '$\mathregular{cm^3)}$')
    ax.set_ylim(-0.1, 1.1)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.04))
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    plt.savefig(os.path.join(savepath, f"density_distribution_bubble{time_section}.png"), dpi=300, bbox_inches='tight')

    output_file = os.path.join(savepath, f"density_distribution_bubble{time_section}.txt")
    with open(output_file, 'w') as f:
        f.write("# Distance(A)\tTotal_Density(g/cm^3)\tWater_Density(g/cm^3)\tN_Density(g/cm^3)\n")
        for d, tot, wat, n in zip(radial_d, total_density, water_density, n_density):
            f.write(f"{d:.5f}\t{tot:.5f}\t{wat:.5f}\t{n:.5f}\n")

def draw_ion_distribution(d_list1, d_list2, z_low, z_high, delta_r, interface, savepath, time_section):
    """
    Plot ion distribution.
    """
    bins = np.linspace(z_low, z_high, int((z_high-z_low)/delta_r)+1)
    bin_centers = bins[:-1] + (bins[1]-bins[0])/2
    v_list = []
    for i in bins[1:]:
        v = 4/3*np.pi*(np.power(i,3)-np.power(i-delta_r,3))*1e-3 #nm^3
        v_list.append(v)

    y1, y2 = [], []
    for i in range(len(d_list2)):
        _y1, _ = np.histogram(d_list1[i], bins, density=True)
        _y2, _ = np.histogram(d_list2[i], bins, density=True)
        y1.append(_y1)
        y2.append(_y2)

    avg_oh = y1[0]
    avg_na = y2[0]
    sum1 = sum(avg_oh)
    sum2 = sum(avg_na)

    Y1, Y2 = [],[]
    for i in range(len(avg_oh)):
        Y1.append(avg_oh[i]/v_list[i]/sum1*100)
        Y2.append(avg_na[i]/v_list[i]/sum2*100)

    # Calculate distance relative to the interface (interface position is 0)
    d_interface_array = interface[0] - bin_centers

    output_file = os.path.join(savepath, f"ion_radical_distribution_bubble{time_section}.txt")
    with open(output_file, 'w') as f:
        f.write("# Distance_to_Interface(A)\tNa+(%/nm^3)\tOH-(%/nm^3)\n")
        for i in range(len(bin_centers)):
            f.write(f"{d_interface_array[i]:.5f}\t{Y2[i]:.5f}\t{Y1[i]:.5f}\n")

    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    ax.vlines(0, -0.6, 12.6, colors="grey", linestyles='dashed')
    l1 = ax.plot(d_interface_array, Y2, lw=2, color='#2BBBD8', label='Na⁺')
    ax.set_xlabel(r"$D_\mathrm{interface}$" + r" $\ \rm (\AA)$")
    ax.set_ylabel(r"$\mathrm{Na^+} \ (\%/\mathrm{nm^3})$")
    ax.set_xlim(-20, 10)
    ax.set_ylim(-0.1, 2)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax2 = ax.twinx()
    l2 = ax2.plot(d_interface_array, Y1, lw=2, color='#d9534f', label='OH⁻')
    ax2.set_ylabel(r"$\mathrm{OH^-} \ (\%/\mathrm{nm^3})$")
    ax2.set_ylim(-0.1, 2)
    ax2.yaxis.set_major_locator(MultipleLocator(1))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.5))
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0, fontsize=8)
    plt.savefig(os.path.join(savepath, f"ion_radical_distribution_bubble{time_section}.png"), dpi=300, bbox_inches='tight')

# ==================== Main Program ====================

def main():
    # Analyze each bubble
    for bubble_idx in range(2):
        bubble_name = f"bubble{bubble_idx+1}"
        print(f"Processing {bubble_name}")
        
        # Create bubble-specific save directory
        bubble_savepath = os.path.join(args.save_path, bubble_name)
        if not os.path.exists(bubble_savepath):
            os.makedirs(bubble_savepath)
        
        # Load LAMMPS trajectory file
        geo_file = args.geo_files[bubble_idx]
        trj_file = args.trj_files[bubble_idx]
        u = mda.Universe(geo_file, trj_file, atom_style='id type x y z', format='LAMMPSDUMP')
        
        # Determine total frames and analysis frame range
        original_total_frames = len(u.trajectory)
        analysis_start_frame = args.skip_first_frames
        analysis_end_frame = original_total_frames - args.skip_last_frames
        
        total_analysis_frames = analysis_end_frame - analysis_start_frame
        
        if total_analysis_frames <= 0 or analysis_start_frame >= analysis_end_frame:
            print(f"{bubble_name} does not have enough frames for analysis (skip {args.skip_first_frames}, ignore {args.skip_last_frames}, total {original_total_frames}). Skipping this bubble.")
            continue # Skip to next bubble
            
        print(f"{bubble_name}: Original total frames {original_total_frames}, Analysis frame range {analysis_start_frame}-{analysis_end_frame} (Total {total_analysis_frames} frames)")

        if total_analysis_frames < args.time_blocks:
            print(f"Warning: Number of analyzed frames ({total_analysis_frames}) is less than time blocks ({args.time_blocks}).")
            # Can choose to adjust time_blocks or raise error, proceeding here but may result in empty blocks
        
        frames_per_block = total_analysis_frames // args.time_blocks
        if frames_per_block == 0 and total_analysis_frames > 0:
             frames_per_block = 1 # Ensure at least 1 frame
             print(f"Warning: Frames per block is less than 1. Please check time_blocks and skip/ignore settings.")
        
        # Process each time block
        for block_idx in range(args.time_blocks):
            # Calculate relative block indices (relative to the analyzed subset)
            block_start_rel = block_idx * frames_per_block
            block_end_rel = (block_idx + 1) * frames_per_block
            
            if block_idx == args.time_blocks - 1:
                block_end_rel = total_analysis_frames  # Ensure last block contains all remaining frames
            
            # Map back to absolute frame indices
            block_start_abs = analysis_start_frame + block_start_rel
            block_end_abs = analysis_start_frame + block_end_rel
                
            print(f"Processing time block {block_idx+1}/{args.time_blocks}, Frame range: {block_start_abs}-{block_end_abs}")
            time_section = f"{bubble_name}_block{block_idx+1}"
            
            # Select frames for current time block (absolute indices)
            current_frames = list(range(block_start_abs, block_end_abs))
            
            # Calculate density distribution
            bin_centers, total_density, water_density, n_density = calculate_density(
                u, current_frames, args.delta_r
            )
            
            # Calculate interface position
            interface_pos, mid_density = get_mid_density(bin_centers, water_density)
            interface = [interface_pos]
            # Emit stable, machine-readable English key-value records for downstream tools.
            print(f"bubble={bubble_name}")
            print(f"block_index={block_idx + 1}")
            print(f"interface_position={interface_pos:.10g}")
            print(f"mid_density={mid_density:.10g}")
            
            # Plot density distribution
            draw_density(
                bin_centers, total_density, water_density, n_density, 
                interface, args.z_low, args.z_high, bubble_savepath, time_section
            )
            
            # Process ion distribution
            # Read ion files for corresponding time period
            oh_file = args.oh_files[bubble_idx]
            na_file = args.na_files[bubble_idx]
            
            oh_data = np.loadtxt(oh_file)
            na_data = np.loadtxt(na_file)
            
            # Use original_total_frames to calculate mapping relationship
            frames_per_oh_line = original_total_frames / len(oh_data) if len(oh_data) > 0 else 1
            frames_per_na_line = original_total_frames / len(na_data) if len(na_data) > 0 else 1
            
            # Use absolute block_start_abs and block_end_abs for slicing
            oh_start_line = int(block_start_abs / frames_per_oh_line)
            oh_end_line = int(block_end_abs / frames_per_oh_line)
            na_start_line = int(block_start_abs / frames_per_na_line)
            na_end_line = int(block_end_abs / frames_per_na_line)
          
            oh_block_data = oh_data[oh_start_line:oh_end_line]
            na_block_data = na_data[na_start_line:na_end_line]
            
            # Extract ion lists
            oh_list = get_ion_list(oh_block_data)
            na_list = get_ion_list(na_block_data)
            
            # Calculate ion distribution
            oh_distribution = get_center([oh_list])
            na_distribution = get_center([na_list])
            
            # Plot ion distribution
            draw_ion_distribution(
                oh_distribution, na_distribution, args.z_low, args.z_high,
                args.delta_r, interface, bubble_savepath, time_section
            )

if __name__ == "__main__":
    main()
