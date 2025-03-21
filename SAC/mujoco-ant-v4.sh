#!/bin/bash
# This script runs SAC experiments locally on 4 GPUs,
# with 8 experiments running concurrently (2 per GPU).

# Load necessary modules and activate the Conda environment
module purge
module load anaconda3/2024.6
conda activate similar-behavior

export WANDB_MODE=offline

# Define the alpha values.
ALPHAS=(0.2 0.175 0.15 0.125 0.1 0.075 0.05 0.025 0)

counter=0

# Loop over each alpha value.
for ALPHA in "${ALPHAS[@]}"; do
    # For α = 0.05, 0.025, 0 use seeds 4 and 5; otherwise use seeds 3, 4, 5.
    if [[ "$ALPHA" == "0.05" || "$ALPHA" == "0.025" || "$ALPHA" == "0" ]]; then
         SEEDS=(4 5)
    else
         SEEDS=(3 4 5)
    fi
    # Loop over the appropriate seeds.
    for SEED in "${SEEDS[@]}"; do
        # Assign a GPU based on the counter.
        # There are 8 slots (4 GPUs x 2 experiments each), so:
        # gpu_id cycles as: 0,0,1,1,2,2,3,3,...
        gpu_id=$(( (counter % 8) / 2 ))
        
        # Construct the run name.
        RUN_NAME="sac_ant-v4-seed-${SEED}-alpha-${ALPHA}"
        echo "Launching experiment ${RUN_NAME} on GPU ${gpu_id}"

        # Launch the experiment with the assigned GPU.
        CUDA_VISIBLE_DEVICES=${gpu_id} python cleanrl/cleanrl/sac_continuous_action.py \
            --seed ${SEED} \
            --total_timesteps 2500000 \
            --learning_starts 5000 \
            --alpha ${ALPHA} \
            --no-autotune \
            --env_id Ant-v4 \
            --exp_name ant-v4 \
            --run_name ${RUN_NAME} \
            --track \
            --torch_deterministic \
            --cuda \
            --wandb_project_name ant-v4 \
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

        # Once 8 experiments are launched, wait for the batch to finish.
        if (( counter % 8 == 0 )); then
            wait
        fi
    done
done

# Wait for any remaining background processes to finish.
wait
