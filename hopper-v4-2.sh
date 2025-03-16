#!/bin/bash
# This script runs SAC experiments locally on 1 GPU,
# with only 3 experiments running concurrently.

# Load necessary modules and activate the Conda environment
module purge
module load anaconda3/2024.6
conda activate similar-behavior

# Sweep parameters
SEEDS=(6 7 8 9 10)
ALPHAS=(0 0.1 0.2 0.3 0.4 0.5 0.6)

counter=0

# Loop over each seed and alpha value
for SEED in "${SEEDS[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
        # Construct run name: sac_hopper-v4-seed-X-alpha-Y
        RUN_NAME="sac_hopper-v4-seed-${SEED}-alpha-${ALPHA}"
        echo "Launching experiment ${RUN_NAME}"

        # Set GPU explicitly (only GPU 0 available)
        export CUDA_VISIBLE_DEVICES=0

        # Run the experiment in the background
        python cleanrl/cleanrl/sac_continuous_action.py \
            --seed ${SEED} \
            --total_timesteps 2500000 \
            --learning_starts 5000 \
            --alpha ${ALPHA} \
            --no-autotune \
            --env_id Hopper-v4 \
            --exp_name hopper-v4 \
            --run_name ${RUN_NAME} \
            --track \
            --torch_deterministic \
            --cuda \
            --wandb_project_name hopper-v4 \
            --no-capture_video \
            --num_envs 1 \
            --buffer_size 1000000 \
            --gamma 0.99 \
            --tau 0.005 \
            --batch_size 256 \
            --policy_lr 0.0003 \
            --q_lr 0.001 \
            --policy_frequency 2 \
            --target_network_frequency 1 &

        ((counter++))

        # Launch in batches of 3 concurrently (single GPU)
        if (( counter % 3 == 0 )); then
            wait
        fi
    done
done

# Wait for any remaining background processes to finish.
wait
