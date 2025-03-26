#!/bin/bash
# This script runs SAC experiments locally on 1 GPU,
# with one experiment running at a time.

# Load necessary modules and activate the Conda environment
source ~/.bashrc
conda activate similar-behavior-dm
export WANDB_MODE=offline
pip install -r cleanrl/requirements/requirements.txt
pip install -r cleanrl/requirements/requirements-dm_control.txt
pip install --upgrade typing_extensions
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Sweep parameters
ALPHAS=(0 0.05 0.1 0.15 0.2 0.25)
SEEDS=(1 2 3 4 5)

# Always use GPU 0
export CUDA_VISIBLE_DEVICES=0

# Loop over each alpha then seed
for ALPHA in "${ALPHAS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        # Construct run name: sac_quadruped-run-alpha-Y-seed-X
        RUN_NAME="sac_quadruped-run-alpha-${ALPHA}-seed-${SEED}"
        echo "Launching experiment ${RUN_NAME}"

        # Run the experiment and wait for it to complete before starting the next
        python cleanrl/cleanrl/sac_continuous_action_2.py \
            --seed ${SEED} \
            --total_timesteps 10000000 \
            --learning_starts 5000 \
            --alpha ${ALPHA} \
            --no-autotune \
            --env_id "dm_control/cheetah-run-v0" \
            --exp_name dmcontrol-quadruped-run-v0 \
            --wandb_project_name dmcontrol_quadruped \
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

        # Each experiment runs sequentially.
    done
done
