#!/bin/bash
# This script runs SAC Hopper Meow experiments on 1 GPU,
# launching only the specified experiments concurrently (2 at a time).

# Load necessary modules and activate the Conda environment
module purge
module load anaconda3/2024.6
conda activate similar-behavior

# Only one GPU available: assign GPU 0 for all experiments.
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline

# Define the alphas to run in a fixed order.
ALPHAS_LIST=(0.3 0.4 0.5 0.6)

# For each alpha, run only seed 1.
declare -A ALPHA_SEEDS
for alpha in "${ALPHAS_LIST[@]}"; do
    ALPHA_SEEDS[$alpha]="5"
done

counter=0

# Loop over each alpha and its corresponding seed (only seed 1).
for ALPHA in "${ALPHAS_LIST[@]}"; do
    SEEDS=${ALPHA_SEEDS[$ALPHA]}
    for SEED in $SEEDS; do
        # Construct run name: sac_hopper-v4-seed-X-alpha-Y
        RUN_NAME="sac_hopper-v4-seed-${SEED}-alpha-${ALPHA}"
        echo "Launching experiment ${RUN_NAME}"

        # Run the experiment in the background.
        python cleanrl/cleanrl/meow_continuous_action.py \
            --seed ${SEED} \
            --total_timesteps 2500000 \
            --learning_starts 5000 \
            --alpha ${ALPHA} \
            --no-autotune \
            --env_id Hopper-v4 \
            --exp_name hopper-v4 \
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
            --q_lr 1e-3 \
            --policy_frequency 2 \
            --target_network_frequency 1 \
            --noise_clip 0.5 \
            --grad_clip 30 \
            --sigma_max -0.3 \
            --sigma_min -5.0 \
            --no-deterministic_action  &

        ((counter++))

        # Wait after launching 2 experiments concurrently.
        if (( counter % 2 == 0 )); then
            wait
        fi
    done
done

# Wait for any remaining background processes to finish.
wait
