#!/bin/bash
# This script runs SAC experiments on 1 GPU,
# with 2 experiments running concurrently.

# Load necessary modules and activate the Conda environment
module purge
module load anaconda3/2024.6
conda activate similar-behavior
export WANDB_MODE=offline

# Define the alphas and seeds to sweep
alphas=(0.5)
seeds=(1 2 5)

counter=0

# Loop over each alpha value
for alpha in "${alphas[@]}"; do
    # Loop over each specified seed for the given alpha
    for seed in "${seeds[@]}"; do
        # Construct a unique run name
        RUN_NAME="sac_walker2d-v4-seed-${seed}-alpha-${alpha}"
        echo "Launching experiment ${RUN_NAME} on GPU 0"
        
        # Launch the experiment in the background using GPU 0
        env CUDA_VISIBLE_DEVICES=0 python cleanrl/cleanrl/sac_continuous_action.py \
            --seed ${seed} \
            --total_timesteps 5000000 \
            --learning_starts 5000 \
            --alpha ${alpha} \
            --no-autotune \
            --env_id Walker2d-v4 \
            --exp_name walker2d-v4 \
            --run_name ${RUN_NAME} \
            --track \
            --torch_deterministic \
            --cuda \
            --wandb_project_name walker2d-v4 \
            --no-capture_video \
            --num_envs 1 \
            --buffer_size 1000000 \
            --gamma 0.99 \
            --tau 0.005 \
            --batch_size 256 \
            --policy_lr 3e-4 \
            --q_lr 3e-4 \
            --policy_frequency 2 \
            --target_network_frequency 1 &
        
        ((counter++))
        # Limit concurrent experiments to 2 at a time
        if (( counter % 2 == 0 )); then
            wait
        fi
    done
done

# Wait for any remaining background processes to finish
wait
