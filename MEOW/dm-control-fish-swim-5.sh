#!/bin/bash
source ~/.bashrc
conda activate similar-behavior-dm
export WANDB_MODE=offline
pip install -r cleanrl/requirements/requirements.txt
pip install -r cleanrl/requirements/requirements-dm_control.txt
pip install --upgrade typing_extensions
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Sweep parameters
SEEDS=(5)
ALPHAS=(0.2 0.3 0.4 0.5 0.6 0 0.1)

counter=0

# Loop over each seed and alpha value
for SEED in "${SEEDS[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
        # Construct run name e.g.: meow_dm_control_fish-swim-v0-seed-X-alpha-Y
        RUN_NAME="meow_dm_control_fish-swim-v0-seed-${SEED}-alpha-${ALPHA}"
        echo "Launching experiment ${RUN_NAME}"
        
        # Use a single GPU (GPU 0)
        CUDA_VISIBLE_DEVICES=0 python cleanrl/cleanrl/meow_continuous_action_2.py \
            --seed ${SEED} \
            --total_timesteps 10000000 \
            --learning_starts 5000 \
            --alpha ${ALPHA} \
            --no-autotune \
            --env_id dm_control/fish-swim-v0 \
            --exp_name fish-swim-v0 \
            --track \
            --torch_deterministic \
            --cuda \
            --wandb_project_name dmcontrol_fish \
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
        
        # Wait after launching two experiments concurrently
        if (( counter % 2 == 0 )); then
            wait
        fi
    done
done

# Wait for any remaining background processes to finish.
wait
