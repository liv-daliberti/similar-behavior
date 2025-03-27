#!/bin/bash
# This script runs SAC experiments on 1 GPU,
# launching experiments in the background in batches of 2.

# Load necessary modules and activate the Conda environment
source ~/.bashrc
conda activate similar-behavior-dm
export WANDB_MODE=offline
pip install -r cleanrl/requirements/requirements.txt
pip install -r cleanrl/requirements/requirements-dm_control.txt
pip install --upgrade typing_extensions
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Define parameters
ALPHAS=(1)
SEEDS=(1 2 3 4 5)
# DM-Control environments to run:
ENVS=("dm_control/finger-spin-v0" "dm_control/cheetah-run-v0" "dm_control/fish-swim-v0")

counter=0

# Loop over each DM-Control environment.
for ENV in "${ENVS[@]}"; do
    # Set experiment name and run prefix based on environment.
    if [[ ${ENV} == "dm_control/finger-spin-v0" ]]; then
        RUN_PREFIX="finger-spin"
        EXP_NAME="dmcontrol-finger-spin-v0"
    elif [[ ${ENV} == "dm_control/cheetah-run-v0" ]]; then
        RUN_PREFIX="quadruped-run"
        EXP_NAME="dmcontrol-quadruped-run-v0"
    elif [[ ${ENV} == "dm_control/fish-swim-v0" ]]; then
        RUN_PREFIX="fish-swim"
        EXP_NAME="dmcontrol-fish-swim-v0"
    fi

    # Loop over each alpha and seed.
    for ALPHA in "${ALPHAS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            # Always use GPU 0.
            gpu_id=0

            # Construct the run name.
            RUN_NAME="sac_${RUN_PREFIX}-alpha-${ALPHA}-seed-${SEED}"
            echo "Launching experiment ${RUN_NAME} on GPU ${gpu_id}"

            # Launch the experiment in background.
            CUDA_VISIBLE_DEVICES=${gpu_id} python cleanrl/cleanrl/sac_continuous_action_2.py \
                --seed ${SEED} \
                --total_timesteps 10000000 \
                --learning_starts 5000 \
                --autotune \
                --alpha ${ALPHA} \
                --env_id ${ENV} \
                --exp_name ${EXP_NAME} \
                --run_name ${RUN_NAME} \
                --track \
                --torch_deterministic \
                --cuda \
                --wandb_project_name dmcontrol \
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

            # Launch at most 2 experiments concurrently.
            if (( counter % 2 == 0 )); then
                wait
            fi
        done
    done
done

# Wait for any remaining background processes to finish.
wait
