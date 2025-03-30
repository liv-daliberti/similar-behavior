#!/bin/bash
#
#SBATCH --job-name=dm_control_finger-spin # Name of the job
#SBATCH -N 1                              # Request 1 node
#SBATCH --ntasks-per-node=1               # 1 task per node
#SBATCH --gres=gpu:1                      # Request 1 GPU
#SBATCH -t 128:00:00                        # Walltime (adjust as needed)
#SBATCH --mem=8GB                         # Memory per node (adjust as needed)
#SBATCH --mail-type=FAIL                  # Email notification on failure
#SBATCH --mail-user=od2961@cs.princeton.edu  # Replace with your actual CS email

# Load necessary modules (if required, e.g., CUDA or any other dependencies)
# module load cuda/11.0  # Adjust module name/version as needed

# Run your script
srun TD3/dm-td3-comparison-5.sh
