#!/bin/bash
#SBATCH --job-name=simsopt_auglag
#SBATCH --time=20:00:00  # 20 Hours in HH:MM:SS format

#SBATCH --mem=48G  # 48GB memory
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-30
#SBATCH --output=simsopt_%A_%a.out  # %A = job ID, %a = array index
#SBATCH --error=simsopt_%A_%a.err

# Submit an array of 30 jobs (1-30 inclusive). Each job runs
# a different order value for the coil optimization problem.

# Set threading environment variables
export OMP_NUM_THREADS=1  # number of threads for OpenMP
export MKL_NUM_THREADS=1  # number of threads for Intel MKL


# Activate conda environment
source ~/.bashrc
conda activate AJT

# Add simsopt_gil to Python path
export PYTHONPATH="/scratch/projects/kaptanoglulab/JS/simsopt_gil/src:$PYTHONPATH"

# Run the Python script with order parameter from SLURM array index
# $SLURM_ARRAY_TASK_ID will be 1-30, which we pass as the order parameter
srun python auglag_w7x_Gaussian.py $SLURM_ARRAY_TASK_ID  