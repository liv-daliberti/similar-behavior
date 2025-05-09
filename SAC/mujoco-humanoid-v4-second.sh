#!/bin/bash
# This script runs SAC experiments locally on 1 GPU,
# launching 2 experiments concurrently on GPU 0.

# Load necessary modules and activate the Conda environment
module purge
module load anaconda3/2024.6
conda activate similar-behavior

# Sweep parameters
SEEDS=(3)
ALPHAS=(0.5 0.4)
export WANDB_MODE=offline

# Set the GPU (only GPU 0 is available)
export CUDA_VISIBLE_DEVICES=0

counter=0

# Loop over each seed and alpha value
for SEED in "${SEEDS[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
        # Construct run name: sac_humanoid-v4-seed-X-alpha-Y
        RUN_NAME="sac_humanoid-v4-seed-${SEED}-alpha-${ALPHA}"
        echo "Launching experiment ${RUN_NAME}"

        # Run the experiment in the background.
        python cleanrl/cleanrl/sac_continuous_action.py \
            --seed ${SEED} \
            --total_timesteps 12000000 \
            --learning_starts 5000 \
            --alpha ${ALPHA} \
            --no-autotune \
            --env_id Humanoid-v4 \
            --exp_name humanoid-v4 \
            --run_name ${RUN_NAME} \
            --track \
            --torch_deterministic \
            --cuda \
            --wandb_project_name humanoid-v4 \
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

        # After launching 2 experiments, wait for them to finish.
        if (( counter % 2 == 0 )); then
            wait
        fi
    done
done

# Wait for any remaining background processes to finish.
wait
