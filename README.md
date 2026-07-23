# Bibubble_ripening Repository Overview

This repository collects the data sets, models, molecular dynamics (MD) inputs, and analysis tools used for the study of **Ostwald ripening** of the nitrogen nanobubble pair in various electrolyte solutions. The material here enables researchers to reproduce the fine-tuned neuroevolution potential (NEP) model, carry out the GPUMD simulation described in the linked publication, and further explore ion-modulated nanobubble ripening dynamics.

## Model Resources

The NEP89 foundation model used in this work is uploaded to [Zenodo](https://zenodo.org/uploads/20290846).

## Software Environment

The following software packages and versions were used in this study:

* **VASP** (v5.4.4)
* **deepmd-kit** (v2.1.5)
* **dpgen** (v0.11.0)
* **GPUMD** *(Note: The version of GPUMD used in this work is an unpublished version, which is uploaded to [Zenodo](https://zenodo.org/uploads/20290846))*
* **PLUMED** (v2.8.2)

## Repository Structure

The directories in this repository are organized to cover data preparation, NEP model training, molecular dynamics simulations, enhanced sampling, and post-processing analyses:

- **Analysis_script/** – Python scripts used to post-process molecular dynamics trajectories and characterize nanobubble structures, interfacial properties, ion distributions, and molecular transport. The directory includes Gibbs dividing surface calculations (`multi_bubble_ion_distribution.py`), hydrogen-bond analyses (`*_interfacial_w_life_based_on_TCF.py`, `interface_area_*_HB_calculation_vs_nocounterion.py`, and `calc_*_hb.py`), radial distribution function calculations (`RDF_normal.py`), and molecular-cluster identification for nanobubbles (`BiBNs_2.py`). It also contains scripts for ion distribution, coordination, and orientation analyses (`Many_Ions_Find.py`, `calculate_*_orientation.py`, `analyze_trajectory_*.py`, and `ion_pair_distri.py`), radial molecular diffusion calculations (`compute_block_msd_moving5A.py` and `opt_and_plot_moving5A.py`), nanobubble ripening analyses (`merged_plot_size_and_distance_3.py` and `plot_bubble_distance_and_bothsize_evolution.py`), interfacial water orientation analyses (`wshell_orientation.py` and `calc_water_block.py`), trajectory translation and re-centering (`shift_2.py`), single-nanobubble stability analysis (`plot_stability_rank.py`), and Markov-state analysis of gas-molecule migration (`moving_mode_filter_and_short_segment_merge.py`, `msm_calculation.py`, and `n2_migration_analysis_ID.py`).

- **Molecular_dynamics/** – Input files for running molecular dynamics simulations with GPUMD. Standard GPUMD simulations are controlled by system-specific `*/run.in` files, while metadynamics-based enhanced-sampling simulations are defined by `plumed.dat` together with the corresponding `Enhanced_sampling/*/run.in` input files.

- **NEP_training/** – Input files used to train the neuroevolution potential model. The principal training parameters, neural-network architecture, descriptor settings, and optimization options are specified in `nep.in`.

- **dataset/** – Atomic configurations used for NEP model training, validation, and testing. All structures, atomic species, energies, forces, and virial information are stored in extended XYZ-format files (`*.xyz`).

Use these descriptions as a starting point to locate the files relevant to your workflow—whether you are preparing the training data set, retraining the NEP model, performing GPUMD or enhanced-sampling simulations, or analyzing nanobubble ripening trajectories.

## Related Publication

Ion-Modulated Ostwald Ripening Dynamics of Nitrogen Nanobubble Pairs. [J. Am. Chem. Soc.](https://pubs.acs.org/doi/10.1021/jacs.6c08865?utm_source=SendGrid_ealert&utm_medium=ealert&utm_campaign=ASAP_jacsat_v0_i0)
