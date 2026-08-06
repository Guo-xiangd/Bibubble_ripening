# Bibubble_ripening Repository Overview

This repository collects the data sets, models, molecular dynamics (MD) inputs, and analysis tools used for the study of **Ostwald ripening** of nitrogen nanobubble pairs in various electrolyte solutions. It provides the project data and inputs needed to prepare NEP fine-tuning, run the supplied GPUMD systems, and analyze ion-modulated nanobubble ripening dynamics. Some external model files and software must be obtained separately as described below.

## Model Resources

The NEP89 foundation model used in this work is uploaded to [Zenodo](https://zenodo.org/uploads/20290846). The `NEP_training/nep.in` file expects `nep89.txt` and `nep89.restart`; these files are not included in this repository and must be downloaded before fine-tuning. The fine-tuned model distributed with this repository is `NEP_training/nep.txt`.

## Software Environment

The following software packages and versions were used in this study:

* **VASP** (v5.4.4)
* **deepmd-kit** (v2.1.5)
* **dpgen** (v0.11.0)
* **GPUMD** *(Note: The version of GPUMD used in this work is an unpublished version, which is uploaded to [Zenodo](https://zenodo.org/uploads/20290846))*
* **PLUMED** (v2.8.2)

The analysis scripts also use Python packages such as NumPy, SciPy, pandas, Matplotlib, MDAnalysis, and tqdm. Exact Python package versions are not currently recorded in this repository, so users should verify compatibility with their local environment.

## Repository Structure

The directories in this repository are organized to cover data preparation, NEP model training, molecular dynamics simulations, enhanced sampling, and post-processing analyses:

- **Analysis_script/** – Python scripts used to post-process molecular dynamics trajectories and characterize nanobubble structures, interfacial properties, ion distributions, and molecular transport. The directory includes Gibbs dividing surface calculations (`multi_bubble_ion_distribution.py`), hydrogen-bond analyses (`*_interfacial_w_life_based_on_TCF.py`, `interface_area_*_HB_calculation_vs_nocounterion.py`, and `calc_*_hb.py`), radial distribution function calculations (`RDF_normal.py`), and molecular-cluster identification for nanobubbles (`BiBNs_2.py`). It also contains scripts for ion distribution, coordination, and orientation analyses (`Many_Ions_Find.py`, `calculate_*_orientation.py`, `analyze_trajectory_*.py`, and `ion_pair_distri.py`), radial molecular diffusion calculations (`compute_block_msd_moving5A.py` and `opt_and_plot_moving5A.py`), nanobubble ripening analyses (`merged_plot_size_and_distance_3.py` and `plot_bubble_distance_and_bothsize_evolution.py`), interfacial water orientation analyses (`wshell_orientation.py` and `calc_water_block.py`), trajectory translation and re-centering (`shift_2.py`), single-nanobubble stability analysis (`plot_stability_rank.py`), and Markov-state analysis of gas-molecule migration (`moving_mode_filter_and_short_segment_merge.py`, `msm_calculation.py`, and `n2_migration_analysis_ID.py`).

- **Molecular_dynamics/** – Input files for running molecular dynamics simulations with GPUMD. The `double_bubble/run.in` and `single_bubble/run.in` files are shared templates for the system-specific subdirectories that contain `model.xyz`. Enhanced-sampling systems have their own `Enhanced_sampling/*/run.in` and `plumed.dat` files.

- **NEP_training/** – Input files used to train the neuroevolution potential model. The principal training parameters, neural-network architecture, descriptor settings, and optimization options are specified in `nep.in`.

- **dataset/** – Atomic configurations used for NEP model training, validation, and testing. All structures, atomic species, energies, forces, and virial information are stored in extended XYZ-format files (`*.xyz`).

## Getting Started

### 1. Obtain the external software and model resources

1. Install the software required for the workflow you intend to run. GPUMD is required for the supplied MD inputs, PLUMED is required for enhanced sampling, and the Python packages listed above are required for post-processing.
2. Download the study-specific GPUMD version and the NEP89 foundation-model files from the Zenodo record linked above when those resources are required.
3. For NEP fine-tuning, place `nep89.txt` and `nep89.restart` in `NEP_training/`, next to `nep.in`.

### 2. Prepare or inspect the NEP data

The extended XYZ files in `dataset/` are the source configurations. Before training, assemble the desired training, validation, and test sets according to the intended experiment. The repository does not currently prescribe a single split or automatically combine the individual system files.

Run NEP fine-tuning from `NEP_training/` after the foundation-model files and the selected training data have been prepared. Consult the documentation for the downloaded GPUMD/NEP version for the exact command and expected training-data file names, because this study used a specific GPUMD build.

### 3. Run a standard GPUMD simulation

The single- and double-bubble folders use one shared `run.in` template per simulation class. Run GPUMD from the selected system directory—the directory containing `model.xyz`—and make the corresponding parent template available there as `run.in` by copying or linking it.

For example, from a shell:

```bash
cd Molecular_dynamics/double_bubble/40NaCl
cp ../run.in run.in
gpumd
```

The potential entry in the supplied template is `../../../NEP_training/nep.txt`, which resolves correctly when GPUMD is launched from a system directory such as the one above. Apply the same procedure to a subdirectory of `single_bubble/`.

### 4. Run an enhanced-sampling simulation

Enhanced-sampling systems are self-contained at the input-file level: each system directory contains `model.xyz`, `run.in`, and `plumed.dat`.

```bash
cd Molecular_dynamics/Enhanced_sampling/Pure_water
gpumd
```

The enhanced-sampling `run.in` files use `../../../NEP_training/nep.txt` and load the local `plumed.dat`. GPUMD writes its normal trajectory and restart outputs, while PLUMED writes `HILLS` and `COLVAR` in the working directory.

### 5. Run post-processing analyses

Analysis scripts are grouped by purpose under `Analysis_script/`. Paths that were previously tied to the original compute environment are now repository-relative and begin with `data/`. Create the corresponding local data layout or update the script configuration/arguments for your trajectory and output locations.

Scripts that use `argparse` expose their accepted inputs through `--help`, for example:

```bash
python Analysis_script/cluster_analysis/BiBNs_2.py --help
python Analysis_script/GDS_calculation/multi_bubble_ion_distribution.py --help
python Analysis_script/radial_diffusion/compute_block_msd_moving5A.py --help
```

Several plotting and workflow-specific scripts still define relative input lists near the beginning or end of the file. Review those values before execution. Typical processing stages are trajectory re-centering, bubble or ion identification, structural/distribution analysis, and final plotting or Markov-state analysis; only the stages needed for a particular observable must be run.

Interface-position logs use a machine-readable English key-value record:

```text
interface_position=12.592
```

`Analysis_script/GDS_calculation/multi_bubble_ion_distribution.py` emits this record together with `bubble`, `block_index`, and `mid_density`. Downstream scripts parse only the `interface_position` key. When processing historical logs, regenerate or normalize them to this format before running the current analysis scripts.

All commands above assume the repository directory structure is preserved. Output files are generally written to the current working directory or to the output path supplied on the command line.

## Related Publication

Ion-Modulated Ostwald Ripening Dynamics of Nitrogen Nanobubble Pairs. [J. Am. Chem. Soc.](https://pubs.acs.org/doi/10.1021/jacs.6c08865)
