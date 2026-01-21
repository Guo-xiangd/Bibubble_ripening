import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- 1. Define file names and parameters ---
FILE_NAMES = [
    '../../moving_mode_filter_and_merge/cluster1_evolution_mode61_merge200_1.log',
    '../../moving_mode_filter_and_merge/cluster2_evolution_mode61_merge200_2.log',
    '../../moving_mode_filter_and_merge/cluster3_evolution_mode61_merge200_3.log'
]
OUTPUT_FILE_P = 'Transition_Probability_Matrix_P.txt'
OUTPUT_FILE_C = 'Transition_Count_Matrix_C.txt'
N_STATES = 3  # States: 1 (Bubble 1), 2 (Bubble 2), 3 (Free)
LAG_TIME = 400  # Lag time tau, based on your smoothing, we use 1 frame as default step

# --- 2. Data loading and stitching function ---

def load_and_stitch_data(file_list: list) -> pd.DataFrame:
    """
    Load all state sequence files and stitch horizontally by molecule ID.
    Returns a DataFrame where rows are time steps, columns are molecule IDs, and values are cluster IDs (1, 2, or 3).
    """
    all_data = []
    
    # Determine which columns to read: skip time column (0)
    use_cols = lambda x: x > 0
    
    print(f"--- 1. Loading and stitching {len(file_list)} sequence files ---")
    for fname in file_list:
        if not os.path.exists(fname):
            print(f"Warning: File {fname} does not exist, skipping.")
            continue
        
        # Use pandas to read data, skip time column (0th column)
        # header=0 reads the header line as column names
        # skiprows=1 skips the first comment line (#time...)
        df = pd.read_csv(fname, sep='\t', header=0, index_col=0)
        
        # State data should be integers
        df = df.apply(pd.to_numeric, downcast='integer')
        
        all_data.append(df)
        print(f"Loaded {fname}: {df.shape[0]} frames, {df.shape[1]} molecules.")

    if not all_data:
        raise FileNotFoundError("No sequence files loaded successfully. Please check filenames and paths.")
        
    # Concatenate all data horizontally by column (molecule)
    # Ensure all files have the same number of rows (frames)
    full_state_matrix = pd.concat(all_data, axis=1)
    
    print(f"Stitching complete. Total shape: {full_state_matrix.shape[0]} frames x {full_state_matrix.shape[1]} molecules.")
    return full_state_matrix.values # Return underlying numpy array for efficient calculation

# --- 3. Core calculation function: Transition Matrix ---

def compute_transition_matrix(states_array: np.ndarray, lag_time: int, n_states: int):
    """
    Compute transition count matrix C and transition probability matrix P.
    states_array: (n_frames, n_molecules) numpy state array.
    """
    print("--- 2. Start calculating transition count matrix C ---")
    
    # Initialize count matrix C
    count_matrix = np.zeros((n_states, n_states), dtype=int)
    
    n_frames, n_mols = states_array.shape
    
    # Iterate over each molecule
    for mol_idx in range(n_mols):
        mol_states = states_array[:, mol_idx]
        
        # Iterate over all time steps for this molecule (t -> t + lag_time)
        for t in range(n_frames - lag_time):
            # Get current state (i) and future state (j)
            state_i = mol_states[t]
            state_j = mol_states[t + lag_time]
            
            # State values (1, 2, 3) correspond to matrix indices (0, 1, 2)
            i_idx = state_i - 1
            j_idx = state_j - 1
            
            # Check if state is within valid range
            if 0 <= i_idx < n_states and 0 <= j_idx < n_states:
                count_matrix[i_idx, j_idx] += 1
                
    # Ensure row sums of count matrix C are non-zero to avoid division by zero
    row_sums = count_matrix.sum(axis=1, keepdims=True)
    
    # Compute transition probability matrix P
    # Use np.divide to safely handle division by zero: rows with zero sum (state never visited) will be zero.
    probability_matrix = np.divide(count_matrix, row_sums, 
                                   out=np.zeros_like(count_matrix, dtype=float), 
                                   where=row_sums != 0)
    
    print("--- 3. Transition matrix P calculation complete ---")
    return count_matrix, probability_matrix
    
def plot_heatmap(Q, savefig=f'P_heatmap.png'):
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(Q, cmap='Blues', aspect='auto')
    # Annotate values
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{Q[i, j]:.3f}',
                    ha='center', va='center', color='white' if Q[i, j]>0.5 else 'black')
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(['1', '2', '3'])
    ax.set_yticklabels(['1', '2', '3'])
    ax.set_xlabel('The next state')
    ax.set_ylabel('The current state')
    ax.set_title(f'Transition Probability Matrix P')
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(savefig, dpi=300)
    plt.close()
    print(f'Heatmap saved to {savefig}')

# --- 4. Main execution flow ---
def main():
    try:
        # 1. Load and stitch data
        full_states = load_and_stitch_data(FILE_NAMES)
        
        # 2. Calculate matrices
        count_matrix, probability_matrix = compute_transition_matrix(
            full_states, LAG_TIME, N_STATES
        )
        
        # 3. Format and save results
        state_labels = ['Bubble 1 (State 1)', 'Bubble 2 (State 2)', 'Free (State 3)']
        
        # Save count matrix C
        df_c = pd.DataFrame(count_matrix, index=state_labels, columns=state_labels)
        df_c.to_csv(OUTPUT_FILE_C, sep='\t', header=True, index=True, float_format='%d')
        print(f"\nTransition Count Matrix C saved to: {OUTPUT_FILE_C}")
        print("\nTransition Count Matrix C (Counts):\n", df_c)
        
        # Save probability matrix P
        df_p = pd.DataFrame(probability_matrix, index=state_labels, columns=state_labels)
        df_p.to_csv(OUTPUT_FILE_P, sep='\t', header=True, index=True, float_format='%.6f')
        print(f"\nTransition Probability Matrix P saved to: {OUTPUT_FILE_P}")
        print("\nTransition Probability Matrix P (Probabilities):\n", df_p)
        
        # Verify row normalization property of P matrix
        row_sums = df_p.sum(axis=1)
        print("\n--- Verify P matrix row sums ---")
        print(row_sums)
        
        plot_heatmap(probability_matrix, savefig=f'P_heatmap.png')
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unknown error occurred: {e}")

if __name__ == '__main__':
    main()