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

```text
├── datasets/          # Data sets used for training/testing
├── models/            # Fine-tuned NEP models
├── md_inputs/         # Molecular dynamics (MD) inputs for GPUMD
├── analysis_tools/    # Scripts and tools for data analysis
└── README.md          # Repository overview
