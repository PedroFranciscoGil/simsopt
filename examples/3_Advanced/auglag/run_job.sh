#!/bin/bash
#SBATCH --job-name=simsopt_auglag
#SBATCH --time=08:20:00  # 500 minutes in HH:MM:SS format
#SBATCH --nodes=1
#SBATCH --mem=48G  # 48GB memory
#SBATCH --ntasks-per-node=32
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
source ~/.conda/etc/profile.d/conda.sh
conda activate AJT

# Run the Python script with order parameter from SLURM array index
# $SLURM_ARRAY_TASK_ID will be 1-30, which we pass as the order parameter
srun python auglag_qa_reactorscale_Gaussian.py $SLURM_ARRAY_TASK_ID  