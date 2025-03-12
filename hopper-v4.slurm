#!/bin/bash
#SBATCH --job-name=sac_hopper          # Job name
#SBATCH --nodes=8                      # Request 8 nodes
#SBATCH --ntasks-per-node=1            # One task per node
#SBATCH --cpus-per-task=1              # 1 CPU-core per task
#SBATCH --mem=16G                      # Memory per node
#SBATCH --gres=gpu:1                   # 1 GPU per node
#SBATCH --time=64:00:00               # Time limit (HH:MM:SS)
#SBATCH --mail-type=begin              # Email when job starts
#SBATCH --mail-type=end                # Email when job ends
#SBATCH --mail-user=od2961@princeton.edu  # Your email

# Load necessary modules and activate the Conda environment
module purge
module load anaconda3/2024.6
conda activate similar-behavior

# Sweep parameters
SEEDS=(1 2 3 4 5 6 7 8 9 10)
ALPHAS=(0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5)

counter=0

# Loop over each seed and alpha value
for SEED in "${SEEDS[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
        # Construct run name: sac_hopper-v4-seed-X-alpha-Y
        RUN_NAME="sac_hopper-v4-seed-${SEED}-alpha-${ALPHA}"
        echo "Launching experiment ${RUN_NAME}"

        # Launch one experiment on a single node (with 1 GPU) using srun.
        srun --nodes=1 --ntasks=1 --gres=gpu:1 --exclusive \
            python cleanrl/cleanrl/sac_continuous_action.py \
                --seed ${SEED} \
                --total_timesteps 3000000 \
                --learning_starts 50000 \
                --alpha ${ALPHA} \
                --autotune False \
                --env_id Hopper-v4 \
                --exp_name hopper-v4 \
                --run_name ${RUN_NAME} \
                --track True \
                --torch_deterministic True \
                --cuda True \
                --wandb_project_name cleanRL \
                --wandb_entity "" \
                --capture_video False \
                --num_envs 1 \
                --buffer_size 1000000 \
                --gamma 0.99 \
                --tau 0.005 \
                --batch_size 256 \
                --policy_lr 0.0003 \
                --q_lr 0.001 \
                --policy_frequency 2 \
                --target_network_frequency 1 &
                
        ((counter++))

        # Run in batches of 8 concurrently (one per node)
        if (( counter % 8 == 0 )); then
            wait
        fi
    done
done

# Wait for any remaining processes to finish
wait
