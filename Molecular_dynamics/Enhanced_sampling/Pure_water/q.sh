#!/bin/bash
#SBATCH -J es_pw_NB
#SBATCH -p gnall
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -o stdout.%j
#SBATCH -e stderr.%j

cd $SLURM_SUBMIT_DIR

module load CUDA

/WORK/xuxf_work/app/gpumd/20250318/GPUMD-master/src/gpumd
