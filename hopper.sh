#!/usr/bin/env bash
export WANDB_PROJECT=autotune

export MUJOCO_PY_MUJOCO_PATH="$PWD/.mujoco/mujoco210"
export MUJOCO_PY_MJKEY_PATH="$PWD/.mujoco/mjkey.txt"
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:$LD_LIBRARY_PATH"

# your 4 GPUs
GPUS=(0 1 2 3)
N_GPUS=${#GPUS[@]}

for seed in {1..5}; do
  # pick GPU in round-robin
  gpu=${GPUS[$(( (seed-1) % N_GPUS ))]}
  echo "Launching seed $seed on GPU $gpu…"
  CUDA_VISIBLE_DEVICES=$gpu python /scratch/gpfs/od2961/similarity-paper/similar-behavior/cleanrl/cleanrl/adaptive_sac.py \
    --env_id Hopper-v4 \
    --total_timesteps 2500000 \
    --seed $seed \
    --track \
    --wandb-project-name autotune \
    &
done

# wait for all background jobs
wait
echo "All background Hopper-v4 runs complete."