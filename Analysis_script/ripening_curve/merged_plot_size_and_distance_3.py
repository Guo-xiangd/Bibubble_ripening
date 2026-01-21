import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import csv

# === Configuration Section ===

bubble_size_dir = "../"
bubble_dist_dir = "../"
is_merged = True  # Whether to merge bubble data
time_interval = 0.001
output_csv = "merged_2.csv"

# === Global Variables: Record Merge Frame Index ===
merge_frame_index = None

# === Helper Functions ===

def read_size_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            bubble1 = int(parts[1])
            bubble2 = int(parts[2])
            has_adjusted = '# adjusted_cutoff' in line
            data.append((bubble1, bubble2, has_adjusted))
    return data

def read_distance_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            distance = float(parts[1])
            data.append(distance)
    return data
    
def extract_start_time(path):
    match = re.search(r'(\d+)-(\d+)', path)
    if match:
        start = int(match.group(1))
        return start
    else:
        return float('inf')

def swap_bubbles(size_data):
    return [(b2, b1, adj) for (b1, b2, adj) in size_data]

def merge_files():
    size_files = glob.glob('../**/bubble_sizes_1k.txt', recursive=True)
    dist_files = glob.glob('../**/bubble_distance_1k.txt', recursive=True)

    size_files = sorted(size_files, key=extract_start_time)
    dist_files = sorted(dist_files, key=extract_start_time)

    print("File order:")
    for f in size_files:
        print(f)

    size_segments = [read_size_file(f) for f in size_files]
    dist_segments = [read_distance_file(f) for f in dist_files]

    merged_size_segments = [size_segments[0]]
    merged_dist_segments = [dist_segments[0]]

    for i in range(1, len(size_segments)):
        prev = merged_size_segments[-1]
        curr = size_segments[i]

        N = 5
        prev_b1 = np.mean([b1 for b1, b2, _ in prev[-N:]])
        prev_b2 = np.mean([b2 for b1, b2, _ in prev[-N:]])
        curr_b1 = np.mean([b1 for b1, b2, _ in curr[:N]])
        curr_b2 = np.mean([b2 for b1, b2, _ in curr[:N]])
        print("End of segment {0}: B1 avg {1}, B2 avg {2}".format(i, prev_b1, prev_b2))
        print("Start of segment {0}: B1 avg {1}, B2 avg {2}".format(i + 1, curr_b1, curr_b2))

        normal_diff = abs(prev_b1 - curr_b1) + abs(prev_b2 - curr_b2)
        swapped_diff = abs(prev_b1 - curr_b2) + abs(prev_b2 - curr_b1)

        threshold = 10
        if swapped_diff + threshold < normal_diff:
            print(f"⚠️  Bubble identity swapped at segment {i}, auto-corrected.")
            curr = swap_bubbles(curr)
        else:
            print(f"✅ Identity consistent at segment {i}, no swap needed.")

        merged_size_segments.append(curr)
        merged_dist_segments.append(dist_segments[i])

    merged_data = []
    for size_data, dist_data in zip(merged_size_segments, merged_dist_segments):
        if len(size_data) != len(dist_data):
            raise ValueError("File line counts do not match!")

        for i in range(len(size_data)):
            b1, b2, adjusted = size_data[i]
            dist = dist_data[i]
            merged_data.append((b1, b2, dist, adjusted))

    return merged_data

def is_valid_frame(b1, b2, last_b1, last_b2, delta_threshold=15):
    delta1 = abs(b1 - last_b1)
    delta2 = abs(b2 - last_b2)
    return delta1 <= delta_threshold and delta2 <= delta_threshold
    
def handle_no_merge(data):
    cleaned_data = []
    last_valid = None

    for i, row in enumerate(data):
        b1, b2, dist, adjusted = row
        if not adjusted:
            cleaned_data.append((b1, b2, dist))
            last_valid = (b1, b2)
        else:
            if last_valid is None:
                continue
            if is_valid_frame(b1, b2, last_valid[0], last_valid[1]):
                cleaned_data.append((b1, b2, dist))
                last_valid = (b1, b2)
            else:
                continue
    return cleaned_data

def handle_merge(data):
    global merge_frame_index
    cleaned_data = []
    last_valid = None

    for i, (b1, b2, dist, adjusted) in enumerate(data):
        if not adjusted:
            cleaned_data.append((b1, b2, dist))
            last_valid = (b1, b2)
        else:
            if last_valid is None:
                continue  # Skip if no baseline
            if is_valid_frame(b1, b2, last_valid[0], last_valid[1]):
                cleaned_data.append((b1, b2, dist))
                last_valid = (b1, b2)
            else:
                # Remove abnormal frame
                print(f"⚠️  Abnormal frame removed: Frame {i}, B1={b1}, B2={b2}")
                continue

    # Merge frame identification logic:
    b1_sizes = [row[0] for row in cleaned_data]
    b2_sizes = [row[1] for row in cleaned_data]

    def find_merge_frame(sizes):
        for i in range(len(sizes) - 4):
            window = sizes[i:i + 5]
            if all(s < 15 for s in window):
                return i
        return None

    idx1 = find_merge_frame(b1_sizes)
    idx2 = find_merge_frame(b2_sizes)

    if idx1 is None and idx2 is None:
        print("No merge frame found! Keeping all data.")
        merge_frame_index = None
        return cleaned_data

    merge_frame_index = min([x for x in [idx1, idx2] if x is not None])
    merge_time = merge_frame_index * time_interval
    print(f"Merge frame detected at: {merge_time:.3f} ns")
    return cleaned_data

def write_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Time(ns)', 'Bubble1', 'Bubble2', 'Distance'])
        for idx, row in enumerate(data):
            time = idx * time_interval
            b1, b2, dist = row
            writer.writerow([f"{time:.3f}", b1, b2, dist])
    print(f"Merged data exported to {filename}")

def plot_data(data):
    times = np.arange(0, len(data) * time_interval, time_interval)
    b1 = [row[0] for row in data]
    b2 = [row[1] for row in data]
    dist = [row[2] for row in data]

    # Stop plotting distance after merge frame
    if merge_frame_index is not None:
        for i in range(merge_frame_index + 5, len(dist)):
            dist[i] = np.nan
            
        # Determine which bubble is smaller
        if b1[merge_frame_index] < b2[merge_frame_index]:
            smaller_bubble = 1  # b1 is smaller
        else:
            smaller_bubble = 2  # b2 is smaller
            
        # Set small bubble size to 0 after merge
        for i in range(merge_frame_index + 5, len(b1)):
            if smaller_bubble == 1:
                b1[i] = 0
            else:
                b2[i] = 0

    # Set global font size
    plt.rcParams.update({'font.size': 32})
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()

    ax1.plot(times, b1, color='red', label='Bubble1 size', linewidth=2)
    ax1.plot(times, b2, color='blue', label='Bubble2 size', linewidth=2)
    ax1.set_xlabel('Time (ns)', fontsize=32)
    ax1.set_ylabel('Bubble Size', fontsize=32)
    ax1.set_ylim(0, 400)
    
    # Set x-axis ticks: major at 8, minor at 2
    ax1.xaxis.set_major_locator(plt.MultipleLocator(8))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(2))
    
    # Set y-axis ticks: major at 80, minor at 20
    ax1.yaxis.set_major_locator(plt.MultipleLocator(80))
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(20))
    
    # Adjust tick label size
    ax1.tick_params(axis='both', which='major', labelsize=32)

    ax2.plot(times, dist, color='green', label='Distance', linewidth=2)
    ax2.set_ylabel('Distance (Å)', fontsize=32)
    ax2.set_ylim(0, 70)
    
    # Adjust tick label size for second y-axis
    ax2.tick_params(axis='y', which='major', labelsize=32)

    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("Bubble_size_and_distance_2.png", dpi=300)

# === Main Entry Point ===

if __name__ == "__main__":
    merged_data = merge_files()

    if is_merged:
        processed_data = handle_merge(merged_data)
    else:
        processed_data = handle_no_merge(merged_data)

    write_csv(processed_data, output_csv)
    plot_data(processed_data)