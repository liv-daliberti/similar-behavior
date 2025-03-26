#!/bin/bash
# This script runs Meow SAC experiments on Half Cheetah with 1 GPU,
# with 2 experiments running concurrently.

# Load necessary modules and activate the Conda environment
module purge
module load anaconda3/2024.6
conda activate similar-behavior

# Only one GPU available: assign GPU 0 for all experiments.
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline

# Define the experiments as "alpha:seeds"
# Seeds are space separated if more than one.
declare -a experiments=(
  "0:2 3"
  "0.1:2 3"
  "0.2:3"
  "0.3:2 3"
  "0.4:2 3"
  "0.5:2 3"
  "0.6:2 3"
)

counter=0

# Loop over each experiment configuration
for exp in "${experiments[@]}"; do
    # Split the configuration into alpha and its corresponding seeds.
    alpha=$(echo $exp | cut -d':' -f1)
    seeds=$(echo $exp | cut -d':' -f2)
    
    # Loop over each specified seed for the given alpha.
    for seed in $seeds; do
        # Construct run name e.g.: meow_halfcheetah-v4-seed-X-alpha-Y
        RUN_NAME="meow_halfcheetah-v4-seed-${seed}-alpha-${alpha}"
        echo "Launching experiment ${RUN_NAME}"
        
        # Run the experiment in the background.
        python cleanrl/cleanrl/meow_continuous_action.py \
            --seed ${seed} \
            --total_timesteps 2500000 \
            --learning_starts 5000 \
            --alpha ${alpha} \
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
        
        # Wait after launching 2 experiments concurrently.
        if (( counter % 2 == 0 )); then
            wait
        fi
    done
done

# Wait for any remaining background processes to finish.
wait
