import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend to Agg, must be done before importing pyplot
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import os
import copy
import argparse

class UnionFind:
    """Union-Find implementation for fast clustering."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0]*n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        xroot = self.find(x)
        yroot = self.find(y)
        if xroot == yroot: return
        if self.rank[xroot] < self.rank[yroot]:
            self.parent[xroot] = yroot
        else:
            self.parent[yroot] = xroot
            if self.rank[xroot] == self.rank[yroot]:
                self.rank[xroot] += 1

def periodic_distance(a, b, box):
    """Calculate minimum image distance under periodic boundary conditions."""
    a = np.array(a)
    b = np.array(b)
    delta = a - b
    for i in range(3):
        length = box[i][1] - box[i][0]
        if length > 0:  # Ensure no division by zero
            delta[i] -= round(delta[i]/length) * length
    return np.linalg.norm(delta)

def periodic_centroid(coords, box):
    """Calculate centroid coordinates under periodic boundary conditions."""
    if len(coords) == 0:
        return np.zeros(3)
    
    coords = np.array(coords)
    box_lengths = np.array([dim[1] - dim[0] for dim in box])
    
    # 1. Convert to fractional coordinates
    fractions = np.zeros_like(coords)
    for d in range(3):
        fractions[:, d] = (coords[:, d] - box[d][0]) / box_lengths[d]
    
    # 2. Calculate average fractional coordinates
    avg_frac = np.zeros(3)
    for d in range(3):
        angles = 2 * np.pi * fractions[:, d]
        x_avg = np.mean(np.cos(angles))
        y_avg = np.mean(np.sin(angles))
        avg_frac[d] = np.arctan2(y_avg, x_avg) / (2 * np.pi)
        if avg_frac[d] < 0:
            avg_frac[d] += 1
    
    # 3. Convert back to Cartesian coordinates
    centroid = np.zeros(3)
    for d in range(3):
        centroid[d] = box[d][0] + avg_frac[d] * box_lengths[d]
    
    return centroid

def cluster_analysis(frame_coords, box, cutoff):
    """Perform cluster analysis."""
    frame_coords = np.array(frame_coords)
    n = len(frame_coords)
    uf = UnionFind(n)
    
    # Build adjacency matrix
    for i in range(n):
        for j in range(i+1, n):
            if periodic_distance(frame_coords[i], frame_coords[j], box) <= cutoff:
                uf.union(i, j)
    
    # Collect clustering results
    clusters = defaultdict(list)
    for atom in range(n):
        clusters[uf.find(atom)].append(atom)
    return sorted(clusters.values(), key=len, reverse=True)

def write_snapshot(frame_idx, timestep, coords, box, cluster_labels):
    """Write trajectory snapshot to file."""
    if not os.path.exists('adjusted_snapshots'):
        os.makedirs('adjusted_snapshots')
    filename = f"adjusted_snapshots/warning_{frame_idx:04d}.data"
    with open(filename, 'w') as f:
        # Write frame header info
        f.write("ITEM: TIMESTEP\n")
        f.write(f"{timestep}\n")
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(f"{len(coords)}\n")
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        for dim in box:
            f.write(f"{dim[0]:.11f} {dim[1]:.11f}\n")
        
        # Write atom info
        f.write("ITEM: ATOMS id Cluster type x y z\n")
        for j in range(len(coords)):
            atom_id = j + 1
            cluster_id = int(cluster_labels[j])
            atom_type = 3  # Fixed as nitrogen atom type
            x, y, z = coords[j]
            f.write(f"{atom_id} {cluster_id} {atom_type} {x:.11f} {y:.11f} {z:.11f}\n")

def track_bubbles(frames, cutoff, timesteps, initial_merged=False, merged_bubble_id=None):
    """
    Main bubble tracking algorithm using historical averaging to detect anomalies.
    
    Args:
        frames: Frame data
        cutoff: Cutoff distance
        timesteps: Timestep data
        initial_merged: Whether initially merged, specified by command line argument
        merged_bubble_id: ID of the bubble being merged (1 or 2), valid only if initial_merged is True
    """
    history = []  # For tracking bubble identity
    bubble1_sizes = []  # Store bubble 1 sizes
    bubble2_sizes = []  # Store bubble 2 sizes
    bubble1_coms = []   # Store bubble 1 COM coordinates
    bubble2_coms = []   # Store bubble 2 COM coordinates
    all_cluster_labels = []  # Store cluster labels for each frame
    adjusted_frames = []  # Store indices of adjusted frames
    normal_history = []  # Store bubble sizes of normal frames
    adjusted_info = []  # Record adjustment info (is_adjusted, new_cutoff)
    merged = initial_merged  # Initialize merge status from args
    consecutive_abnormal = 0  # Counter for consecutive abnormal frames
    merged_start_frame = 0 if initial_merged else None  # Set start frame to 0 if initially merged
    
    # If initially merged, use the provided merged_bubble_id
    current_merged_bubble_id = merged_bubble_id if initial_merged else None
    
    # Counters for tracking small bubbles
    small_bubble1_count = 0  # Consecutive frames where bubble 1 < 10
    small_bubble2_count = 0  # Consecutive frames where bubble 2 < 10

    # Open files to write COM coordinates
    with open('COM1_1k_new.txt', 'w') as f_com1, open('COM2_1k_new.txt', 'w') as f_com2:
        f_com1.write("# num_frame COMx COMy COMz\n")
        f_com2.write("# num_frame COMx COMy COMz\n")

        for frame_idx, ((coords, box), timestep) in enumerate(zip(frames, timesteps)):
            # Initialize data for current frame
            bubble1_size = 0
            bubble2_size = 0
            com1 = np.zeros(3)
            com2 = np.zeros(3)
            frame_labels = np.full(len(coords), 3) # Initialize as 3 (other clusters)
            current_cutoff = cutoff
            adjusted = False
            adjusted_cutoff = cutoff

            # Simplified logic if bubbles are already merged
            if merged:
                # Only process the largest bubble
                clusters = cluster_analysis(coords, box, cutoff)

                # If clusters exist, mark the largest one
                if clusters and clusters[0]:
                    largest_cluster = set(clusters[0])

                    # Mark largest cluster as bubble 1 or 2 based on merged_bubble_id
                    if merged_bubble_id == 1:
                        # If bubble 1 was merged, the largest cluster should be bubble 2
                        for atom_idx in largest_cluster:
                            frame_labels[atom_idx] = 2
                        bubble2_size = len(largest_cluster)
                        bubble1_size = 0  # Bubble 1 is merged
                        com2 = periodic_centroid([coords[i] for i in largest_cluster], box)
                        com1 = np.zeros(3)  # Bubble 1 merged, COM is 0
                    else:
                        # If bubble 2 was merged, the largest cluster is bubble 1
                        for atom_idx in largest_cluster:
                            frame_labels[atom_idx] = 1
                        bubble1_size = len(largest_cluster)
                        bubble2_size = 0  # Bubble 2 is merged
                        com1 = periodic_centroid([coords[i] for i in largest_cluster], box)
                        com2 = np.zeros(3)  # Bubble 2 merged, COM is 0

                    # Update history
                    if merged_bubble_id == 1:
                        history = [set(), largest_cluster]  # Bubble 1 merged, max cluster is Bubble 2
                    else:
                        history = [largest_cluster, set()]  # Bubble 2 merged, max cluster is Bubble 1
                else:
                    # No clusters found
                    history = [set(), set()]
                    bubble1_size = 0
                    bubble2_size = 0
                    com1 = np.zeros(3)
                    com2 = np.zeros(3)

                # Store results
                bubble1_sizes.append(bubble1_size)
                bubble2_sizes.append(bubble2_size)
                bubble1_coms.append(com1)
                bubble2_coms.append(com2)
                all_cluster_labels.append(frame_labels)
                adjusted_info.append((False, cutoff))

            else: # Normal logic when bubbles are not merged
                is_normal_frame = True  # Assume current frame is normal

                # Cluster using current cutoff
                clusters = cluster_analysis(coords, box, current_cutoff)

                # Get top two clusters (sorted by size descending)
                top_clusters = []
                if clusters and clusters[0]:
                    top_clusters.append(set(clusters[0]))
                if len(clusters) > 1 and clusters[1]:
                    top_clusters.append(set(clusters[1]))

                current_clusters = top_clusters

                # ===== Initial Identity Matching =====
                if not history:  # First frame initialization
                    if current_clusters and current_clusters[0]:
                        history = copy.deepcopy(current_clusters)
                        # Mark bubble 1
                        for atom_idx in current_clusters[0]:
                            frame_labels[atom_idx] = 1

                        bubble1_size = len(current_clusters[0])

                        # If bubble 2 exists, mark it
                        if len(current_clusters) > 1 and current_clusters[1]:
                            for atom_idx in current_clusters[1]:
                                frame_labels[atom_idx] = 2

                            bubble2_size = len(current_clusters[1])
                            normal_history.append((bubble1_size, bubble2_size))
                        else:
                            bubble2_size = 0
                            normal_history.append((bubble1_size, 0))

                    # Calculate centroids
                    com1 = periodic_centroid([coords[i] for i in current_clusters[0]], box) if current_clusters and current_clusters[0] else np.zeros(3)
                    com2 = periodic_centroid([coords[i] for i in current_clusters[1]], box) if len(current_clusters) > 1 and current_clusters[1] else np.zeros(3)

                    # Store results
                    all_cluster_labels.append(frame_labels)
                    bubble1_sizes.append(bubble1_size)
                    bubble2_sizes.append(bubble2_size)
                    bubble1_coms.append(com1)
                    bubble2_coms.append(com2)
                    adjusted_info.append((adjusted, adjusted_cutoff))
                    # Skip subsequent regular processing for this frame, continue to next
                    f_com1.write(f"{frame_idx+1} {com1[0]:.6f} {com1[1]:.6f} {com1[2]:.6f}\n")
                    f_com2.write(f"{frame_idx+1} {com2[0]:.6f} {com2[1]:.6f} {com2[2]:.6f}\n")
                    continue 

                # Calculate overlap matrix
                prev1, prev2 = history
                overlap_matrix = []
                if current_clusters:
                    overlap_matrix.append([
                        len(current_clusters[0] & prev1) if current_clusters else 0,
                        len(current_clusters[0] & prev2) if current_clusters else 0
                    ])
                    if len(current_clusters) > 1:
                        overlap_matrix.append([
                            len(current_clusters[1] & prev1),
                            len(current_clusters[1] & prev2)
                        ])
                    else:
                        overlap_matrix.append([0, 0])
                else:
                    overlap_matrix = [[0,0], [0,0]]

                # Determine optimal matching
                option1 = overlap_matrix[0][0] + (overlap_matrix[1][1] if len(overlap_matrix)>1 else 0)
                option2 = overlap_matrix[0][1] + (overlap_matrix[1][0] if len(overlap_matrix)>1 else 0)

                if option1 >= option2:  # Maintain match
                    bubble1 = current_clusters[0] if current_clusters and len(current_clusters) > 0 else set()
                    bubble2 = current_clusters[1] if len(current_clusters) > 1 else set()
                else:  # Swap match
                    bubble1 = current_clusters[1] if len(current_clusters) > 1 else set()
                    bubble2 = current_clusters[0] if current_clusters and len(current_clusters) > 0 else set()

                bubble1_size = len(bubble1) if bubble1 is not None else 0
                bubble2_size = len(bubble2) if bubble2 is not None else 0

                # ===== Anomaly Detection (Using History Average) =====
                window_size = 5  # Use previous 5 frames as reference
                valid_history = []

                # Collect valid history frames (skipping abnormal ones)
                start_index = max(0, len(normal_history) - window_size)
                for i in range(start_index, len(normal_history)):
                    valid_history.append(normal_history[i])

                if valid_history:  # If history is available
                    hist_bubble1 = np.mean([size[0] for size in valid_history])
                    hist_bubble2 = np.mean([size[1] for size in valid_history]) if any(size[1] > 0 for size in valid_history) else 0

                    # Check if bubbles are abnormal (change > 80)
                    delta1 = abs(bubble1_size - hist_bubble1) if bubble1_size > 0 else 0
                    delta2 = abs(bubble2_size - hist_bubble2) if bubble2_size > 0 else 0

                    if delta1 > 80 or delta2 > 80:
                        is_normal_frame = False
                        print(f"Frame {frame_idx+1} anomaly detected - Bubble 1 delta: {delta1:.1f}, Bubble 2 delta: {delta2:.1f}")

                # Update small bubble counters
                if bubble1_size < 10:
                    small_bubble1_count += 1
                else:
                    small_bubble1_count = 0

                if bubble2_size < 10:
                    small_bubble2_count += 1
                else:
                    small_bubble2_count = 0

                # ===== Handle Anomalous Frames or Small Bubble Merging =====
                if not is_normal_frame:
                    # Increment consecutive abnormal counter
                    consecutive_abnormal += 1

                    # Check for 10 consecutive abnormal frames or 5 consecutive small bubble frames
                    if consecutive_abnormal >= 10 or small_bubble1_count >= 5 or small_bubble2_count >= 5:
                        print(f"Frame {frame_idx+1} bubble merge detected:")
                        if consecutive_abnormal >= 10:
                            print(f"  - {consecutive_abnormal} consecutive abnormal frames")
                        if small_bubble1_count >= 5:
                            print(f"  - Bubble 1 has been less than 10 atoms for {small_bubble1_count} frames")
                            merged = True
                            merged_start_frame = frame_idx

                    original_cutoff = current_cutoff
                    adjusted = True
                    adjusted_cutoff = original_cutoff
                    original_clusters = copy.deepcopy(current_clusters)

                    # Variables to record best adjustment
                    best_adjustment = None
                    min_total_delta = float('inf')

                    # Try reducing cutoff distance (0.01 steps)
                    for adj_cut in np.arange(cutoff, 4.5, -0.01):
                        clusters_adj = cluster_analysis(coords, box, adj_cut)

                        # Get adjusted clusters
                        top_clusters_adj = []
                        if clusters_adj:
                            top_clusters_adj.append(set(clusters_adj[0]))
                        if len(clusters_adj) > 1:
                            top_clusters_adj.append(set(clusters_adj[1]))

                        current_adj = top_clusters_adj

                        # Calculate overlap matrix (using previous frame identity)
                        overlap_matrix_adj = []
                        if current_adj:
                            overlap_matrix_adj.append([
                                len(current_adj[0] & prev1) if current_adj else 0,
                                len(current_adj[0] & prev2) if current_adj else 0
                            ])
                            if len(current_adj) > 1:
                                overlap_matrix_adj.append([
                                    len(current_adj[1] & prev1),
                                    len(current_adj[1] & prev2)
                                ])
                            else:
                                overlap_matrix_adj.append([0, 0])
                        else:
                            overlap_matrix_adj = [[0,0], [0,0]]

                        # Determine optimal matching
                        option1_adj = overlap_matrix_adj[0][0] + (overlap_matrix_adj[1][1] if len(overlap_matrix_adj)>1 else 0)
                        option2_adj = overlap_matrix_adj[0][1] + (overlap_matrix_adj[1][0] if len(overlap_matrix_adj)>1 else 0)

                        if option1_adj >= option2_adj:  # Maintain
                            bubble1_adj = current_adj[0] if current_adj and len(current_adj) > 0 else set()
                            bubble2_adj = current_adj[1] if len(current_adj) > 1 else set()
                        else:  # Swap
                            bubble1_adj = current_adj[1] if len(current_adj) > 1 else set()
                            bubble2_adj = current_adj[0] if current_adj and len(current_adj) > 0 else set()

                        bubble1_size_adj = len(bubble1_adj) if bubble1_adj is not None else 0
                        bubble2_size_adj = len(bubble2_adj) if bubble2_adj is not None else 0

                        # Check if adjustment meets condition (both bubbles delta < 20)
                        delta1_adj = abs(bubble1_size_adj - hist_bubble1) if bubble1_size_adj > 0 else 0
                        delta2_adj = abs(bubble2_size_adj - hist_bubble2) if bubble2_size_adj > 0 and hist_bubble2 > 0 else 0
                        total_delta = delta1_adj + delta2_adj

                        # Record current result
                        current_result = {
                            'cutoff': adj_cut,
                            'bubble1_size': bubble1_size_adj,
                            'bubble2_size': bubble2_size_adj,
                            'bubble1_cluster': bubble1_adj,
                            'bubble2_cluster': bubble2_adj,
                            'clusters': current_adj,
                            'delta1': delta1_adj,
                            'delta2': delta2_adj,
                            'total_delta': total_delta
                        }

                        # Check if this is a better result
                        if total_delta < min_total_delta:
                            best_adjustment = current_result
                            min_total_delta = total_delta

                        if delta1_adj <= 20 and delta2_adj <= 20:
                            adjusted = True
                            bubble1_size = bubble1_size_adj
                            bubble2_size = bubble2_size_adj
                            bubble1 = bubble1_adj
                            bubble2 = bubble2_adj
                            current_clusters = current_adj
                            adjusted_cutoff = adj_cut
                            print(f"  Used cutoff {adjusted_cutoff:.2f}, Bubble 1 delta: {delta1_adj:.1f}, Bubble 2 delta: {delta2_adj:.1f}")
                            break
                    else: # loop ended without break, no adjustment found with delta <= 20
                        # Use the best found adjustment
                        if best_adjustment is not None:
                            bubble1_size = best_adjustment['bubble1_size']
                            bubble2_size = best_adjustment['bubble2_size']
                            bubble1 = best_adjustment['bubble1_cluster']
                            bubble2 = best_adjustment['bubble2_cluster']
                            current_clusters = best_adjustment['clusters']
                            adjusted_cutoff = best_adjustment['cutoff']
                            print(f"  Used best cutoff {adjusted_cutoff:.3f}, Total delta: {min_total_delta:.1f}")
                            print(f"  Bubble 1 delta: {best_adjustment['delta1']:.1f}, Bubble 2 delta: {best_adjustment['delta2']:.1f}")
                        else:
                            # Revert to original clustering
                            print(f"  No valid adjustment found, using original cutoff {original_cutoff:.1f}")
                            adjusted_cutoff = original_cutoff
                            bubble1_size = len(bubble1) if bubble1 is not None else 0
                            bubble2_size = len(bubble2) if bubble2 is not None else 0

                    # Save snapshot
                    write_snapshot(frame_idx, timestep, coords, box, frame_labels)
                    adjusted_frames.append(frame_idx)
                else:
                    # Reset consecutive abnormal counter for normal frames
                    consecutive_abnormal = 0

                # ===== Update Cluster Labels =====
                if bubble1 is not None:
                    for atom_idx in bubble1:
                        frame_labels[atom_idx] = 1

                if bubble2 is not None:
                    for atom_idx in bubble2:
                        frame_labels[atom_idx] = 2

                bubble1_sizes.append(bubble1_size)
                bubble2_sizes.append(bubble2_size)

                if is_normal_frame:
                    # Only add normal frames to history
                    normal_history.append((bubble1_size, bubble2_size))

                # Update history for next frame
                if bubble1 is not None and bubble2 is not None:
                    history = [bubble1, bubble2]
                elif bubble1 is not None:
                    history = [bubble1, set()]
                else:
                    history = [set(), set()]

                # ===== Calculate Centroids (Executed whether merged or not) =====
                if bubble1 is not None:
                    bubble1_atoms = [coords[i] for i in bubble1]
                    # If bubble 1 is merged, do not calculate its centroid
                    if merged and merged_bubble_id == 1:
                        com1 = np.zeros(3)
                    else:
                        com1 = periodic_centroid(bubble1_atoms, box) if bubble1_atoms else np.zeros(3)
                else:
                    com1 = np.zeros(3)

                if bubble2 is not None:
                    bubble2_atoms = [coords[i] for i in bubble2]
                    # If bubble 2 is merged, do not calculate its centroid
                    if merged and merged_bubble_id == 2:
                        com2 = np.zeros(3)
                    else:
                        com2 = periodic_centroid(bubble2_atoms, box) if bubble2_atoms else np.zeros(3)
                else:
                    com2 = np.zeros(3)

            # Write COM data (regardless of merge status or adjustment)
            f_com1.write(f"{frame_idx+1} {com1[0]:.6f} {com1[1]:.6f} {com1[2]:.6f}\n")
            f_com2.write(f"{frame_idx+1} {com2[0]:.6f} {com2[1]:.6f} {com2[2]:.6f}\n")

            all_cluster_labels.append(frame_labels) 
            bubble1_coms.append(com1)
            bubble2_coms.append(com2)
            adjusted_info.append((adjusted, adjusted_cutoff))

    return bubble1_sizes, bubble2_sizes, bubble1_coms, bubble2_coms, all_cluster_labels, adjusted_frames, adjusted_info, merged, merged_start_frame

def calculate_bubble_distances(bubble1_coms, bubble2_coms, boxes, merged_start_frame=None):
    """Calculate distance between two bubbles for each frame (considering PBC)."""
    distances = []
    for idx, ((com1, com2), box) in enumerate(zip(zip(bubble1_coms, bubble2_coms), boxes)):
        # Check if in merged state (after merged_start_frame)
        if merged_start_frame is not None and idx >= merged_start_frame:
            # Do not calculate distance after merge
            distances.append(np.nan)
            continue
            
        # Check if both bubbles exist and are non-zero
        bubble1_exists = not np.all(com1 == 0) and not np.isnan(com1).any()
        bubble2_exists = not np.all(com2 == 0) and not np.isnan(com2).any()
        
        # If both bubbles exist, calculate distance
        if bubble1_exists and bubble2_exists:
            dist = periodic_distance(com1, com2, box)
        else:
            dist = np.nan  # Set to NaN if a bubble is missing
            
        distances.append(dist)
    return distances

def read_and_filter_traj(filepath, target_type=3):
    """Read LAMMPS trajectory file and filter for specific atom type."""
    frames = []  # Store processed frame data
    frame_boxes = []  # Store box info for each frame
    frame_timesteps = []  # Store timestep for each frame
    box = []
    
    with open(filepath, 'r') as f_in:
        timestep = None
        num_atoms = None
        frame_coords = []
        
        while True:
            line = f_in.readline()
            if not line: 
                # End of file, process last frame
                if frame_coords:
                    frames.append((frame_coords, box))
                    frame_boxes.append(box)
                    frame_timesteps.append(timestep)
                break
                
            if line.startswith('ITEM: TIMESTEP'):
                # Process previous frame (if exists)
                if frame_coords:
                    frames.append((frame_coords, box))
                    frame_boxes.append(box)
                    frame_timesteps.append(timestep)
                
                # Reset current frame variables
                frame_coords = []
                box = []
                
                # Read timestep
                timestep = f_in.readline().strip()
                
            elif line.startswith('ITEM: NUMBER OF ATOMS'):
                # Read total atoms
                num_atoms = int(f_in.readline().strip())
                
            elif line.startswith('ITEM: BOX'):
                # Read box info
                for _ in range(3):
                    parts = f_in.readline().split()
                    if len(parts) >= 2:
                        lo = float(parts[0])
                        hi = float(parts[1])
                        box.append((lo, hi))
                    else:
                        box.append((0.0, 1.0))  # Default box size
                
            elif line.startswith('ITEM: ATOMS'):
                # Start reading atom data
                parts = line.split()
                
                # Parse column indices
                columns = parts[2:]  # Remove 'ITEM: ATOMS'
                
                # Determine indices for important columns
                id_idx = columns.index('id') if 'id' in columns else 0
                type_idx = columns.index('type') if 'type' in columns else 1
                x_idx = columns.index('x') if 'x' in columns else 2
                y_idx = columns.index('y') if 'y' in columns else 3
                z_idx = columns.index('z') if 'z' in columns else 4
                
                # Read all atoms in current frame
                for _ in range(num_atoms):
                    parts = f_in.readline().split()
                    if len(parts) < max(id_idx, type_idx, x_idx, y_idx, z_idx) + 1:
                        continue
                    
                    try:
                        atom_type = int(parts[type_idx])
                        if atom_type == target_type:
                            x = float(parts[x_idx])
                            y = float(parts[y_idx])
                            z = float(parts[z_idx])
                            frame_coords.append([x, y, z])
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Atom data parsing error - {str(e)}", file=sys.stderr)
                        continue
    
    if frames:
        print(f"Extraction complete. Processed {len(frames)} frames, average {sum(len(f[0]) for f in frames)/len(frames):.1f} nitrogen atoms per frame")
    else:
        print("Warning: No frames extracted", file=sys.stderr)
    return frame_timesteps, frames, frame_boxes

def write_clustered_traj(f_out, timesteps, frames, cluster_labels):
    """Write trajectory file with cluster information."""
    try:
        for i, (timestep, (coords, box), labels) in enumerate(zip(timesteps, frames, cluster_labels)):
            # Write frame header info
            f_out.write("ITEM: TIMESTEP\n")
            f_out.write(f"{timestep}\n")
            f_out.write("ITEM: NUMBER OF ATOMS\n")
            f_out.write(f"{len(coords)}\n")
            f_out.write("ITEM: BOX BOUNDS pp pp pp\n")
            for dim in box:
                f_out.write(f"{dim[0]:.11f} {dim[1]:.11f}\n")
            
            # Write atom info
            f_out.write("ITEM: ATOMS id Cluster type x y z\n")
            for j in range(len(coords)):
                atom_id = j + 1
                cluster_id = int(labels[j])
                atom_type = 3  # Fixed as nitrogen atom type
                x, y, z = coords[j]
                f_out.write(f"{atom_id} {cluster_id} {atom_type} {x:.11f} {y:.11f} {z:.11f}\n")
            f_out.flush()  # Ensure data is written
    except Exception as e:
        print(f"Error writing trajectory file: {str(e)}", file=sys.stderr)
        raise

# Main Program
if __name__ == "__main__":
    try:
        # Argument parsing
        parser = argparse.ArgumentParser(description='Bubble Clustering and Analysis Script')
        parser.add_argument('INITIAL_CUTOFF', type=float, help='Initial cutoff distance')
        parser.add_argument('TRAJ_FILE', type=str, help='Raw trajectory file path')
        parser.add_argument('merged', type=int, choices=[0,1], help='Whether bubbles are merged: 0 for No, 1 for Yes')
        parser.add_argument('t', type=float, help='Initial time of trajectory')
        parser.add_argument('save_dir', type=str, help='Output directory path')
        parser.add_argument('--merged_bubble', type=int, choices=[1,2], 
                          help='Specify which bubble was merged (1 or 2), valid only when merged=1', default=2)
        args = parser.parse_args()

        INITIAL_CUTOFF = args.INITIAL_CUTOFF
        TRAJ_FILE = args.TRAJ_FILE
        initial_merged = bool(args.merged)
        t = args.t
        save_dir = args.save_dir
        merged_bubble_id = args.merged_bubble if initial_merged else None

        # Create output directory
        os.makedirs(save_dir, exist_ok=True)

        # Output file paths
        com1_path = os.path.join(save_dir, 'COM1_1k_new.txt')
        com2_path = os.path.join(save_dir, 'COM2_1k_new.txt')
        traj_out_path = os.path.join(save_dir, 'bubble_onlyN2_1k_new.lammpstrj')
        dist_path = os.path.join(save_dir, 'bubble_distance_1k_new.txt')
        size_path = os.path.join(save_dir, 'bubble_sizes_1k_new.txt')
        fig_path = os.path.join(save_dir, 'bubble_evolution_and_distance_1k_new.png')
        adjusted_snapshots_dir = os.path.join(save_dir, 'adjusted_snapshots')

        print("Processing trajectory file, extracting nitrogen atoms...")
        timesteps, frames, frame_boxes = read_and_filter_traj(TRAJ_FILE, target_type=3)
        if not frames:
            print("Error: No valid frame data found", file=sys.stderr)
            sys.exit(1)
        frame_data = [(coords, box) for coords, box in frames]
        boxes = [box for _, box in frames]

        print("Performing bubble analysis (with history averaging anomaly detection)...")
        results = track_bubbles(frame_data, INITIAL_CUTOFF, timesteps, initial_merged, merged_bubble_id)
        bubble1_sizes, bubble2_sizes, bubble1_coms, bubble2_coms, cluster_labels, adjusted_frames, adjusted_info, merged_flag, merged_start_frame = results

        if initial_merged:
            print(f"Command line argument specified bubbles are merged. Calculating only Bubble {3-merged_bubble_id} info from Frame 1 (Bubble {merged_bubble_id} is merged).")
        elif merged_flag:
            print(f"Bubbles merged! From Frame {merged_start_frame+1} onwards, calculating single bubble info only.")

        print("Writing trajectory file with cluster info...")
        with open(traj_out_path, 'w') as f_out:
            write_clustered_traj(f_out, timesteps, frames, cluster_labels)

        print("Calculating bubble distances...")
        bubble_distances = calculate_bubble_distances(bubble1_coms, bubble2_coms, boxes, merged_start_frame)

        print("Writing bubble distance data...")
        with open(dist_path, 'w') as f_dist:
            f_dist.write("# frame distance(Angstrom)\n")
            for i, dist in enumerate(bubble_distances):
                f_dist.write(f"{i+1} {dist:.6f}\n")
                f_dist.flush()  # Ensure data is written

        print("Outputting bubble size evolution info...")
        with open(size_path, 'w') as f_size:
            f_size.write("# MD time(ns)   Bubble1     Bubble2\n")
            for i, (size1, size2) in enumerate(zip(bubble1_sizes, bubble2_sizes)):
                time_ns = t + i * 0.001  # Corrected time calculation, adding initial time t
                adjusted = False
                adjusted_cutoff = INITIAL_CUTOFF
                if i < len(adjusted_info):
                    adjusted, adjusted_cutoff = adjusted_info[i]
                if i in adjusted_frames:
                    if adjusted:
                        f_size.write(f"{time_ns:.3f}\t{size1}\t{size2}\t# adjusted_cutoff={adjusted_cutoff:.2f}\n")
                    else:
                        f_size.write(f"{time_ns:.3f}\t{size1}\t{size2}\t# (adjustment attempted but not applied)\n")
                elif merged_flag and merged_start_frame is not None and i >= merged_start_frame:
                    f_size.write(f"{time_ns:.3f}\t{size1}\t{size2}\t# merged\n")
                else:
                    f_size.write(f"{time_ns:.3f}\t{size1}\t{size2}\n")
                f_size.flush()  # Ensure data is written

        print("Generating visualization...")
        plt.figure(figsize=(14, 8))
        frame_numbers = list(range(1, len(bubble1_sizes) + 1))
        MD_time = [t + frame * 0.001 for frame in frame_numbers]  # Corrected time calculation
        
        fig, ax1 = plt.subplots(figsize=(14, 8))
        ax1.plot(MD_time, bubble1_sizes, 'b-', linewidth=2, label='Bubble 1 Size')
        ax1.plot(MD_time, bubble2_sizes, 'r-', linewidth=2, label='Bubble 2 Size')
        ax1.set_xlabel('MD Time (ns)', fontsize=14)
        ax1.set_ylabel('Bubble Size (Atoms)', fontsize=14)
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        ax2 = ax1.twinx()
        ax2.plot(MD_time, bubble_distances, 'g-', linewidth=2, label='Bubble Distance')
        ax2.set_ylabel('Bubble Distance (Å)', fontsize=14)
        ax2.tick_params(axis='y', labelcolor='green')
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', ncol=3, fontsize=12)
        
        plt.title('Nitrogen Bubble Evolution and Distance', fontsize=16, pad=20)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()  # Explicitly close figure
        
        print("Analysis complete!")
        print(f"Generated COM files: {com1_path}, {com2_path}")
        print(f"Generated Nitrogen trajectory file: {traj_out_path}")
        print(f"Generated Bubble distance file: {dist_path}")
        print(f"Generated Bubble size file: {size_path}")
        print(f"Generated Evolution and Distance plot: {fig_path}")
        
        if adjusted_frames:
            print(f"Detected {len(adjusted_frames)} frames requiring cutoff adjustment, saved to {adjusted_snapshots_dir}/warning_*.data")
        else:
            print("No frames requiring cutoff adjustment detected.")
        if merged_flag and merged_start_frame is not None:
            print(f"From Frame {merged_start_frame+1} onwards, bubbles are merged. Only single bubble info is calculated.")

    except Exception as e:
        print(f"Error during execution: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)