#!/bin/bash
# This script runs SAC experiments on 4 GPUs,
# launching 2 experiments concurrently on each GPU.

# Load necessary modules and activate the Conda environment
module purge
module load anaconda3/2024.6
conda activate similar-behavior

export WANDB_MODE=offline

# Sweep parameters
SEEDS=(1 2 3 4 5)
ALPHAS=(0.2 0.3 0.4 0.5 0.6 0 0.1)

counter=0

# Loop over each seed and alpha value
for SEED in "${SEEDS[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
        # Construct run name e.g.: meow_halfcheetah-v4-seed-X-alpha-Y
        RUN_NAME="meow_halfcheetah-v4-seed-${SEED}-alpha-${ALPHA}"
        echo "Launching experiment ${RUN_NAME}"
        
        # Determine GPU id based on counter.
        # Two experiments per GPU are launched in a round-robin fashion across 4 GPUs.
        GPU_ID=$(( (counter / 2) % 4 ))
        echo "Using GPU ${GPU_ID}"
        
        # Launch the experiment in the background using the assigned GPU.
        CUDA_VISIBLE_DEVICES=${GPU_ID} python cleanrl/cleanrl/meow_continuous_action.py \
            --seed ${SEED} \
            --total_timesteps 5000000 \
            --learning_starts 5000 \
            --alpha ${ALPHA} \
            --no-autotune \
            --env_id HalfCheetah-v4 \
            --exp_name halfcheetah-v4 \
            --track \
            --torch_deterministic \
            --cuda \
            --wandb_project_name halfcheetah-v4 \
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
        
        # Wait after launching 8 experiments concurrently (2 per GPU across 4 GPUs)
        if (( counter % 8 == 0 )); then
            wait
        fi
    done
done

# Wait for any remaining background processes to finish.
wait
