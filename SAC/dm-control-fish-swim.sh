#!/bin/bash
# This script runs SAC experiments locally on 1 GPU,
# with one experiment running at a time.

# Load necessary modules and activate the Conda environment
module purge
module load anaconda3/2024.6
conda activate similar-behavior-dm

# Sweep parameters
ALPHAS=(0.1 0.2 0.3 0.4 0.5 0.6 0)
SEEDS=(1 2)

# Always use GPU 0
export CUDA_VISIBLE_DEVICES=0

# Loop over each alpha then seed
for ALPHA in "${ALPHAS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        # Construct run name: sac_fish-swim-alpha-Y-seed-X
        RUN_NAME="sac_fish-swim-alpha-${ALPHA}-seed-${SEED}"
        echo "Launching experiment ${RUN_NAME}"

        # Run the experiment and wait for it to complete before starting the next
        python cleanrl/cleanrl/sac_continuous_action_2.py \
            --seed ${SEED} \
            --total_timesteps 2500000 \
            --learning_starts 5000 \
            --alpha ${ALPHA} \
            --no-autotune \
            --env_id "dm_control/fish-swim-v0" \
            --exp_name dmcontrol-fish-swim-v0 \
            --wandb_project_name dmcontrol_fish \
            --run_name ${RUN_NAME} \
            --track \
            --torch_deterministic \
            --cuda \
            --no-capture_video \
            --num_envs 1 \
            --buffer_size 1000000 \
            --gamma 0.99 \
            --tau 0.005 \
            --batch_size 256 \
            --policy_lr 3e-4 \
            --q_lr 3e-4 \
            --policy_frequency 2 \
            --target_network_frequency 1

        # No background processes: each experiment runs sequentially.
    done
done
