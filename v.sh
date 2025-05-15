#!/usr/bin/env bash
export WANDB_PROJECT=autotune

export MUJOCO_PY_MUJOCO_PATH="$PWD/.mujoco/mujoco210"
export MUJOCO_PY_MJKEY_PATH="$PWD/.mujoco/mjkey.txt"
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:$LD_LIBRARY_PATH"


# run seed 3 on GPU 1
CUDA_VISIBLE_DEVICES=0 python /scratch/gpfs/od2961/similarity-paper/similar-behavior/cleanrl/cleanrl/adaptive_sac.py \
  --env_id Humanoid-v4 \
  --total_timesteps 5500000 \
  --seed 5 \
  --track \
  --wandb-project-name autotune \
  --resume wandb/run-20250512_213646-7978r6ec/files/files/checkpoint_2200000.pth \
  &
pids+=($!)  # record its PID

 run seed 4 on GPU 2
CUDA_VISIBLE_DEVICES=1 python /scratch/gpfs/od2961/similarity-paper/similar-behavior/cleanrl/cleanrl/adaptive_sac.py \
  --env_id Humanoid-v4 \
  --total_timesteps 5500000 \
  --seed 4 \
  --track \
  --wandb-project-name autotune \
  --resume  wandb/run-20250512_213646-prrs8r1a/files/files/checkpoint_2200000.pth \
  &
pids+=($!)

# run seed 5 on GPU 3
CUDA_VISIBLE_DEVICES=2 python /scratch/gpfs/od2961/similarity-paper/similar-behavior/cleanrl/cleanrl/adaptive_sac.py \
  --env_id Humanoid-v4 \
  --total_timesteps 5500000 \
  --seed 3 \
  --track \
  --wandb-project-name autotune \
  --resume  wandb/run-20250512_233519-7errfnnx/files/files/checkpoint_1950000.pth \
  &
pids+=($!)

# wait for ALL jobs to finish
wait

# (the EXIT trap will fire here, sending SIGUSR1 to each $pid)
