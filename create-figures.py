#!/usr/bin/env python3
"""
Inference script for running one video-recorded episode per checkpoint/model,
saving each video to the "vidoes" folder and producing a metadata file plus a
static image that visualizes the walking trajectory.

Requirements:
 - Gymnasium environment that supports video recording.
 - A function to load actors from checkpoints.
 - Matplotlib for plotting the trajectory.
"""

import os
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def load_actor(checkpoint_path, env, device, is_meow=False, alpha=None, sigma_max=None, sigma_min=None):
    """
    Loads the actor from a checkpoint.
    This simplified version assumes the checkpoint contains an "actor_state_dict" or "policy_state_dict".
    Depending on the type of algorithm (e.g., MEOW) you might need to adjust the state dict keys.
    
    For non-meow algorithms, we assume an Actor similar to your provided code.
    For meow algorithms, we prepend "flow_policy." to the keys.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "actor_state_dict" in checkpoint:
        state = checkpoint["actor_state_dict"]
    elif "policy_state_dict" in checkpoint:
        state = checkpoint["policy_state_dict"]
    else:
        raise KeyError(f"No recognized actor keys in checkpoint: {checkpoint_path}")

    if is_meow: 
        # Adjust keys for FlowActor
        state = {"flow_policy." + k: v for k, v in state.items()}
        # Here you’d construct your FlowActor, e.g.:
        actor = FlowActor(env, alpha=alpha, sigma_max=sigma_max, sigma_min=sigma_min, device=device)
    else:
        actor = Actor(env)  # Assuming Actor is imported or defined
    actor.load_state_dict(state)
    actor.to(device)
    actor.eval()
    return actor

def extract_position(obs):
    """
    Extract the position from the observation.
    Modify this function to extract the (x, y) position from your environment's observation.
    For example, if obs is an array and the first two entries are the x and y coordinates:
    """
    # Here we assume the first two values are positions; adjust as needed.
    return np.array(obs[:2])

def save_trajectory_plot(trajectory, save_path):
    """
    Creates and saves a plot representing the walking trajectory.
    """
    trajectory = np.array(trajectory)
    plt.figure()
    plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', linewidth=2)
    plt.title("Walking Trajectory")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Trajectory image saved to {save_path}")

def write_metadata(metadata_file, model_name, checkpoint_name, autotune_flag):
    """
    Writes metadata about the model/checkpoint to a file.
    """
    with open(metadata_file, "w") as f:
        f.write("model_name: {}\n".format(model_name))
        f.write("checkpoint: {}\n".format(checkpoint_name))
        f.write("label: {}\n".format(autotune_flag))
    logger.info(f"Metadata written to {metadata_file}")

# --- Main Inference Loop ---
def run_inference_for_checkpoint(checkpoint_path, env_id, device, output_dir, is_meow=False, alpha=None, sigma_max=None, sigma_min=None):
    # Create output directories
    video_dir = Path(output_dir) / "vidoes"  # as per requested name
    video_dir.mkdir(parents=True, exist_ok=True)
    traj_img_dir = Path(output_dir) / "trajectories"
    traj_img_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = Path(output_dir) / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Determine model and checkpoint metadata based on filename.
    checkpoint_basename = os.path.basename(checkpoint_path)
    # For example, if the file name includes 'liv-autotune', set label accordingly.
    if "liv-autotune" in checkpoint_basename:
        autotune_flag = "sac-liv-autotune"
    else:
        autotune_flag = "sac"
    
    # You may also extract model names or run names from the directory structure.
    model_name = Path(checkpoint_path).parent.name

    # Prepare the Gym environment with video recording.
    # The RecordVideo wrapper automatically records one episode if the `episode_trigger`
    # returns True. Here we ensure we record the first (and only) episode.
    env = gym.make(env_id)
    env = gym.wrappers.RecordVideo(
        env, 
        video_dir=str(video_dir), 
        name_prefix=f"{model_name}_{checkpoint_basename}",
        episode_trigger=lambda episode_id: episode_id == 0  # record only the first episode
    )
    # Seed the environment if necessary
    env.action_space.seed(0)

    # Load the actor model.
    actor = load_actor(checkpoint_path, env, device, is_meow=is_meow, alpha=alpha, sigma_max=sigma_max, sigma_min=sigma_min)

    done = False
    obs, info = env.reset()
    trajectory = []  # List to store (x, y) positions

    while not done:
        # Record position (modify extraction based on your env's observation)
        pos = extract_position(obs)
        trajectory.append(pos)

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        # Get action from the actor. If your actor expects no-grad calls, wrap in torch.no_grad()
        with torch.no_grad():
            # For instance, for a standard actor get_action returns (action, log_prob, mean)
            action, _, _ = actor.get_action(obs_tensor, deterministic=True)
        # Convert action to numpy array
        a_np = action.cpu().numpy().squeeze(0)
        obs, reward, terminated, truncated, info = env.step(a_np)
        done = terminated or truncated

    # Close the environment to ensure the video is saved.
    env.close()

    # Save the trajectory plot
    traj_img_path = traj_img_dir / f"{model_name}_{checkpoint_basename}_traj.png"
    save_trajectory_plot(trajectory, str(traj_img_path))

    # Write metadata file
    metadata_file = metadata_dir / f"{model_name}_{checkpoint_basename}_metadata.txt"
    write_metadata(metadata_file, model_name, checkpoint_basename, autotune_flag)

# --- Example Usage ---
if __name__ == "__main__":
    # Define parameters here or load via a configuration system (like Hydra/omegaconf)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_id = "Humanoid-v2"  # Replace with your actual environment ID
    output_dir = "./inference_results"  # Change as needed

    # List your checkpoints (for example, list of file paths)
    checkpoint_list = [
        "/path/to/model1/checkpoint_5M.pth",
        "/path/to/model2/liv-autotune_checkpoint_5M.pth",
        # add further checkpoints as needed
    ]

    # If you're using the FlowActor (for meow) you may need to set these parameters:
    is_meow = False  # Set to True if this checkpoint belongs to a meow algorithm
    alpha = 0.2      # Example value, adjust as required
    sigma_max = 1.0  # Example value
    sigma_min = 0.1  # Example value

    for checkpoint_path in checkpoint_list:
        logger.info(f"Processing checkpoint: {checkpoint_path}")
        run_inference_for_checkpoint(
            checkpoint_path=checkpoint_path,
            env_id=env_id,
            device=device,
            output_dir=output_dir,
            is_meow=is_meow,
            alpha=alpha,
            sigma_max=sigma_max,
            sigma_min=sigma_min
        )
