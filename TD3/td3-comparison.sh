#!/bin/bash
# Script to run TD3 experiments locally on 1 GPU with 3 concurrent runs.

# Load necessary modules and activate Conda environment
module purge
module load anaconda3/2024.6
conda activate similar-behavior
export WANDB_MODE=offline

# Define parameters
SEEDS=(1 2 3 4 5)
ENVS=("Humanoid-v4")

counter=0
export CUDA_VISIBLE_DEVICES=0

# Loop over seeds and environments
for SEED in "${SEEDS[@]}"; do
    for ENV in "${ENVS[@]}"; do

        RUN_NAME="td3_${ENV}_seed_${SEED}"
        echo "Launching experiment ${RUN_NAME}"

        python cleanrl/cleanrl/td3_continuous_action.py \
            --seed ${SEED} \
            --total_timesteps 12000000 \
            --learning_starts 5000 \
            --env_id ${ENV} \
            --exp_name ${ENV} \
            --track \
            --torch_deterministic \
            --cuda \
            --wandb_project_name td3-comparison \
            --num_envs 1 \
            --buffer_size 1000000 \
            --gamma 0.99 \
            --tau 0.005 \
            --batch_size 256 \
            --learning_rate 3e-4 \
            --policy_noise 0.2 \
            --exploration_noise 0.1 \
            --policy_frequency 2 \
            --noise_clip 0.5 \
            --no-capture_video &

        ((counter++))

        # Limit to 3 concurrent runs
        if (( counter % 2 == 0 )); then
            wait
        fi

    done
done

# Wait for remaining processes
wait