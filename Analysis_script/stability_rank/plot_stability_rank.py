import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import os
import numpy as np
import matplotlib as mpl

# --- Apply "Nature" Style ---

def set_nature_style():
    """Apply 'Nature' style Matplotlib settings"""
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12,
        'lines.linewidth': 2.0,
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True, 
        'ytick.right': True, 
        'axes.grid': False, 
        'axes.unicode_minus': False 
    })

def read_simulation_data(file_path):
    """Read simulation data, return time and N2 percentage"""
    time, percentage = [], []
    with open(file_path, 'r') as f:
        for line in f:
            cols = line.split() 
            if len(cols) == 7:
                time.append(float(cols[0]))
                percentage.append(int(cols[-1])/120 * 100) # Convert to percentage
    return time, percentage

# Define System Configuration (Added 100NaOH as 5th system)
SYSTEM_CONFIG = {
    '100NaOH': { # Added high concentration NaOH
        'paths': [
            'data/100naoh/022_nep_v2/001/accumulated/30ns/N2Cluster_5.5.txt',
            'data/100naoh/022_nep_v2/002/accumulated/30ns/N2Cluster_5.5.txt'
        ],
        'color': '#9467bd', # Purple
        'style': 'solid'
    },
    '20NaOH': {
        'paths': [
            'data/20naoh/022_nep_v2/001/accumulated/30ns/N2Cluster_5.5.txt',
            'data/20naoh/022_nep_v2/002/accumulated/30ns/N2Cluster_5.5.txt'
        ],
        'color': '#1f77b4',
        'style': 'solid'
    },
    '20NaCl': {
        'paths': [
            'data/20nacl/022_nep_v2/iter000/9000/001/accumulated/30ns/N2Cluster_5.5.txt',
            'data/20nacl/022_nep_v2/iter000/9000/002/accumulated/30ns/N2Cluster_5.5.txt',
        ],
        'color': '#2ca02c',
        'style': 'solid'
    },
    'purewater': {
        'paths': [
            'data/purewater/iter000/nep_022_v2-9000/001/accumulated/30ns/N2Cluster_5.5.txt',
            'data/purewater/iter000/nep_022_v2-9000/002/accumulated/30ns/N2Cluster_5.5.txt',
        ],
        'color': '#d62728',
        'style': 'solid'
    },
    '20HCl': {
        'paths': [
            'data/20hcl/022_nep/022_nep_v2-9000/001/accumulated/30ns/N2Cluster_5.5.txt',
            'data/20hcl/022_nep/022_nep_v2-9000/002/accumulated/30ns/N2Cluster_5.5.txt'
        ],
        'color': '#ff7f0e',
        'style': 'solid'
    }
}

def analyze_and_plot_rank_distribution(avg_data, system_order, time_points):
    """Rank distribution stats and visualization (Extended to 5 ranks)"""
    # Initialize rank counters
    rank_counts = {sys: [0]*5 for sys in system_order} 
    
    # Iterate through time points
    for t_idx in range(len(time_points)):
        current_values = {sys: avg_data[sys][t_idx] for sys in system_order}
        sorted_systems = sorted(system_order, key=lambda x: current_values[x], reverse=True)
        
        # Record top 5 ranks
        for rank, sys in enumerate(sorted_systems[:5]):
            rank_counts[sys][rank] += 1
    
    # Convert to proportions
    proportions = {}
    for sys in system_order:
        total = sum(rank_counts[sys])
        proportions[sys] = [count/total for count in rank_counts[sys]]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(9, 3)) 
    
    # Plot settings
    bar_width = 0.17 
    rank_colors = ['#9467bd', '#1f77b4', '#2ca02c', '#d62728', '#ff7f0e'] 
    system_pos = np.arange(len(system_order))
    
    # Custom legend labels
    legend_labels = [f'Rank{rank+1}' for rank in range(5)]
    
    # Draw bars
    for rank in range(5): 
        offset = bar_width * (rank - 2) 
        values = [proportions[sys][rank] for sys in system_order]
        ax.bar(
            system_pos + offset,
            values,
            bar_width,
            color=rank_colors[rank],
            label=legend_labels[rank]
        )
    
    # Axis settings
    ax.set_xticks(system_pos)
    ax.set_xticklabels([sys for sys in system_order])
    ax.set_ylabel('Time Proportion', labelpad=12)
    ax.set_ylim(0, 1)
    
    # Legend settings
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles, 
        labels=labels,
        title="Stability:", 
        ncol=5, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 0.98), 
        frameon=True, 
        handletextpad=0.5 
    )
    
    plt.tight_layout()
    plt.savefig('Rank_Distribution.png', bbox_inches='tight', dpi=1000)
    
def plot_rank_probability(avg_data, system_order, time_points, target_rank):
    """Plot cumulative probability for a specific Rank (Supports Rank 5)"""
    # Validation
    if target_rank < 1 or target_rank > 5:
        raise ValueError("Target Rank must be between 1-5")
    
    # Initialize cumulative counters
    cumulative_counts = {sys: 0 for sys in system_order}
    probabilities = {sys: [] for sys in system_order}
    time_indices = np.arange(len(time_points))
    
    # Iterate time points
    for t_idx in time_indices:
        current_values = {sys: avg_data[sys][t_idx] for sys in system_order}
        sorted_systems = sorted(system_order, key=lambda x: current_values[x], reverse=True)
        
        # Update counter for target rank
        if target_rank-1 < len(sorted_systems):
            current_system = sorted_systems[target_rank-1]
            cumulative_counts[current_system] += 1
            
        # Calculate probability
        for sys in system_order:
            prob = cumulative_counts[sys] / (t_idx + 1)
            probabilities[sys].append(prob)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('white')
    
    # Plot curves
    for sys in system_order:
        ax.plot(
            time_points,
            probabilities[sys],
            color=SYSTEM_CONFIG[sys]['color'],
            linestyle=SYSTEM_CONFIG[sys]['style'],
            linewidth=2.5, 
            label=sys
        )
    
    # Elements
    ax.set_xlabel("Simulation Time (ns)")
    ax.set_ylabel(f'Cumulative Rank {target_rank} Probability')
    ax.legend(ncol=2, frameon=True, loc='lower right')
    
    # Axis
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    
    # Grid
    ax.grid(True, which='major', linestyle='--', alpha=0.7)
    
    # Save
    plt.tight_layout()
    plt.savefig(f'Rank{target_rank}_Probability.png', bbox_inches='tight', dpi=300)

# Main Execution Flow
if __name__ == "__main__":
    # --- Apply Nature Style ---
    set_nature_style()
    
    # Init data storage
    avg_percentages = {}
    system_order = []
    base_time = None
    
    # Create Line Plot
    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    
    # Process each system
    for sys_name, config in SYSTEM_CONFIG.items():
        system_order.append(sys_name)
        all_percent = []
        time_validation = None
        
        # Process multiple samples
        for path in config['paths']:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            
            time, percent = read_simulation_data(path)
            
            # Time validation
            if base_time is None:
                base_time = time
            if time_validation is None:
                time_validation = time
            else:
                if time != time_validation:
                    raise ValueError(f"Time sequence mismatch: {sys_name}")
            
            all_percent.append(percent)
        
        # Average
        avg_percent = [sum(col)/len(col) for col in zip(*all_percent)]
        avg_percentages[sys_name] = avg_percent
        
        # Plot
        ax.plot(
            base_time, avg_percent,
            color=config['color'],
            linestyle=config['style'],
            linewidth=2.5,
            label=sys_name
        )
    
    # Style Line Plot
    ax.legend(ncol=3, frameon=True, loc='upper right') 
    ax.set_xlabel("Simulation Time (ns)")
    ax.set_ylabel('$\mathregular{N_2}$ in Bubble (%)')
    
    # Axis Locators
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(25))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    
    # Limits
    ax.set_xlim(0, 30)
    ax.set_ylim(30, 100)
    
    # Save Line Plot
    plt.tight_layout()
    plt.savefig('MultiSystem_N2_Comparison.png', bbox_inches='tight')
    
    # Rank Distribution
    analyze_and_plot_rank_distribution(avg_percentages, system_order, base_time)
    
    # Rank Probability (1-5)
    for rank in range(1, 6):
        plot_rank_probability(avg_percentages, system_order, base_time, rank)