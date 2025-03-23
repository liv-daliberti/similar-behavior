#!/bin/bash
# This script runs SAC experiments locally on 1 GPU,
# with 2 experiments running concurrently.

# Load necessary modules and activate the Conda environment
module purge
module load anaconda3/2024.6
conda activate similar-behavior

export WANDB_MODE=offline

# Define parameters
SEEDS=(1 2 3 4 5)
ENVS=("Hopper-v4" "Ant-v4" "HalfCheetah-v4" "Walker2d-v4")
ALPHA=1

counter=0

# Loop over each environment.
for ENV in "${ENVS[@]}"; do
    # Loop over each seed.
    for SEED in "${SEEDS[@]}"; do
        # Since we have 1 GPU, assign it always.
        gpu_id=0
        
        # Construct the run name.
        RUN_NAME="sac-${ENV}-seed-${SEED}-alpha-${ALPHA}"
        echo "Launching experiment ${RUN_NAME} on GPU ${gpu_id}"

        # Launch the experiment with the assigned GPU.
        CUDA_VISIBLE_DEVICES=${gpu_id} python cleanrl/cleanrl/sac_continuous_action-liv-autotune.py \
            --seed ${SEED} \
            --total_timesteps 5000000 \
            --learning_starts 5000 \
            --autotune \
            --alpha ${ALPHA} \
            --env_id ${ENV} \
            --exp_name ${ENV} \
            --run_name ${RUN_NAME} \
            --track \
            --torch_deterministic \
            --cuda \
            --wandb_project_name autotune \
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

        # Once 2 experiments are launched, wait for the batch to finish.
        if (( counter % 2 == 0 )); then
            wait
        fi
    done
done

# Wait for any remaining background processes to finish.
wait
