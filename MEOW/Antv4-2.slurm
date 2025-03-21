#!/bin/bash
#SBATCH --job-name=meow_ant-v4   # Job name
#SBATCH --nodes=1                # Node count
#SBATCH --ntasks=1               # Total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # CPU cores per task
#SBATCH --mem=16G                # Memory per CPU core
#SBATCH --gres=gpu:1             # Number of GPUs per node
#SBATCH --time=128:00:00          # Time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # Send email when the job starts
#SBATCH --mail-type=end          # Send email when the job ends
#SBATCH --mail-user=od2961@princeton.edu  # Your email for notifications

# This script runs MEOW experiments on 1 GPU sequentially,
# running seed 1 across all specified α values.

# Load necessary modules and activate the Conda environment
module purge
module load anaconda3/2024.6
conda activate similar-behavior
export WANDB_MODE=offline

# Only one GPU available: assign GPU 0 for all experiments.
export CUDA_VISIBLE_DEVICES=0

# Sweep parameters for Ant-v4: run seed 1 across all α values.
SEEDS=(2)
ALPHAS=(0.2 0.175 0.15 0.125 0.1 0.075 0.05 0.025 0)

# Loop over each seed and alpha value sequentially.
for SEED in "${SEEDS[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
        # Construct run name, e.g.: meow_ant-v4-seed-1-alpha-0.2
        RUN_NAME="meow_ant-v4-seed-${SEED}-alpha-${ALPHA}"
        echo "Launching experiment ${RUN_NAME}"
        
        # Run the experiment.
        python cleanrl/cleanrl/meow_continuous_action.py \
            --seed ${SEED} \
            --total_timesteps 2500000 \
            --learning_starts 5000 \
            --alpha ${ALPHA} \
            --no-autotune \
            --env_id Ant-v4 \
            --exp_name ant-v4 \
            --track \
            --torch_deterministic \
            --cuda \
            --wandb_project_name ant-v4 \
            --no-capture_video \
            --num_envs 1 \
            --buffer_size 1000000 \
            --gamma 0.99 \
            --tau 0.005 \
            --batch_size 256 \
            --q_lr 1e-3 \
            --policy_frequency 2 \
            --target_network_frequency 1 \
            --noise_clip 0.5 \
            --grad_clip 30 \
            --sigma_max -0.3 \
            --sigma_min -5.0 \
            --no-deterministic_action
    done
done
