#!/bin/bash
# This script runs SAC experiments on 1 GPU (GPU 0),
# launching 2 experiments concurrently on that GPU.

# Load necessary modules and activate the Conda environment
module purge
module load anaconda3/2024.6
conda activate similar-behavior

export WANDB_MODE=offline

# Sweep parameters: Only seed 5 is used and a range of alpha values
SEEDS=(3)
ALPHAS=(0.6 0.5)

counter=0

# Loop over the single seed and each alpha value
for SEED in "${SEEDS[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
        # Construct run name e.g.: meow_walker2d-v4-seed-5-alpha-Y
        RUN_NAME="meow_walker2d-v4-seed-${SEED}-alpha-${ALPHA}"
        echo "Launching experiment ${RUN_NAME}"
        
        # Since we're using 1 GPU, set GPU_ID to 0
        GPU_ID=0
        echo "Using GPU ${GPU_ID}"
        
        # Launch the experiment in the background using GPU 0
        CUDA_VISIBLE_DEVICES=${GPU_ID} python cleanrl/cleanrl/meow_continuous_action.py \
            --seed ${SEED} \
            --total_timesteps 5000000 \
            --learning_starts 5000 \
            --alpha ${ALPHA} \
            --no-autotune \
            --env_id Walker2d-v4 \
            --exp_name walker2d-v4 \
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
            --q_lr 1e-3 \
            --policy_frequency 2 \
            --target_network_frequency 1 \
            --noise_clip 0.5 \
            --grad_clip 30 \
            --sigma_max -0.3 \
            --sigma_min -5.0 \
            --no-deterministic_action  &
        
        ((counter++))
        
        # After launching 2 experiments concurrently, wait for them to finish
        if (( counter % 2 == 0 )); then
            wait
        fi
    done
done

# Wait for any remaining background processes to finish.
wait
