#!/usr/bin/env python3
"""
This script gathers valid run checkpoints from a specified directory,
loads actor networks from checkpoints at a specified step, and performs
pairwise KL divergence evaluations between actors that have the same
alpha value, different seeds, and that were trained with the same algorithm.
The evaluation runs on GPU if available. Results are saved to a CSV file whose name
is based on the environment ID.

This script uses Hydra to load configuration from a YAML file (e.g. "configs/hopper-v4.yaml").
"""

import os
import re
import json
import torch
import glob
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random

# ------- Global Constants -----------------------------------------------------

LOG_STD_MAX = 2
LOG_STD_MIN = -5


# ------- Actor Definition -----------------------------------------------------
class Actor(nn.Module):
    """
    Actor neural network for continuous action spaces.

    Args:
        env (gym.Env): An environment instance to infer observation and action space shapes.
    """
    def __init__(self, env):
        super().__init__()
        obs_dim = np.array(env.single_observation_space.shape).prod()
        action_dim = np.prod(env.single_action_space.shape)
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        # Action rescaling based on environment's action space.
        self.register_buffer(
            "action_scale",
            torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input observation tensor.
        
        Returns:
            tuple: mean (Tensor) and log_std (Tensor) for the action distribution.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        # Scale log_std to be within [LOG_STD_MIN, LOG_STD_MAX].
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        """
        Returns an action given the observation x, along with the log probabilities.

        Args:
            x (Tensor): Input observation tensor.
        
        Returns:
            tuple: (action, log_prob, mean, std, log_prob_test)
        """
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick.
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing action bounds.
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob_test = log_prob
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean, std, log_prob_test


# ------- Helper Functions -----------------------------------------------------

def episode_trigger(episode_id: int) -> bool:
    """
    Determines whether to record an episode based on its ID.

    Args:
        episode_id (int): Current episode ID.
    
    Returns:
        bool: True if the episode should be recorded (every 100 episodes).
    """
    return episode_id % 100 == 0


def make_env(env_id: str, seed: int, idx: int, capture_video: bool, run_name: str) -> callable:
    """
    Creates a thunk function to instantiate the environment.

    Args:
        env_id (str): Environment ID.
        seed (int): Seed for the environment.
        idx (int): Index for environment instances.
        capture_video (bool): Flag to enable video capture.
        run_name (str): Run name (used for video logging).

    Returns:
        callable: A function that returns an instantiated environment.
    """
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger)
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk


def _evaluate_agent(env, n_eval_episodes: int, actor_1: Actor, actor_2: Actor, seed: int = 0):
    """
    Evaluates two actor networks over a specified number of episodes.

    Args:
        env: A vectorized gym environment.
        n_eval_episodes (int): Number of evaluation episodes.
        actor_1 (Actor): The first actor network.
        actor_2 (Actor): The second actor network.
        seed (int, optional): Seed for evaluation. Defaults to 0.
    
    Returns:
        tuple: (mean_reward, std_reward, mean_kl_div, std_kl_div)
    """
    episode_rewards = []
    KL_divergence = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for episode in range(n_eval_episodes):
        obs, _ = env.reset(seed=seed)
        infos = {}
        total_rewards_ep = 0
        total_divergence = 0
        steps = 0

        while "final_info" not in infos:
            obs_tensor = torch.Tensor(obs).to(device)
            actions, _, mean1, std1, _ = actor_1.get_action(obs_tensor)
            _, _, mean2, std2, _ = actor_2.get_action(obs_tensor)

            # Compute KL divergence between two Gaussian distributions.
            normal1 = torch.distributions.Normal(mean1, std1)
            normal2 = torch.distributions.Normal(mean2, std2)
            divergence = torch.distributions.kl.kl_divergence(normal1, normal2)
            total_divergence += divergence

            actions_np = actions.detach().cpu().numpy()
            next_obs, rewards, terminations, truncations, infos = env.step(actions_np)
            total_rewards_ep += rewards
            steps += 1
            obs = next_obs

        episode_rewards.append(total_rewards_ep)
        KL_divergence.append(total_divergence.detach().cpu().numpy() / steps)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_divergence = np.mean(KL_divergence)
    std_divergence = np.std(KL_divergence)

    return mean_reward, std_reward, mean_divergence, std_divergence


def gather_valid_runs(root_dir: str, target_env: str, checkpoint_step: int,
                      training_seeds: list, algorithms: list):
    """
    Walks the `root_dir` recursively to gather valid run checkpoints.

    This function:
      - Searches for subdirectories containing 'files/config.yaml'.
      - Extracts configuration values from the YAML.
      - Validates that:
            * env_id equals the desired target environment.
            * seed is in the provided training_seeds list.
            * The run was trained with one of the specified algorithms (i.e., one of the provided substrings is in code_path).
      - Searches within the run directory (assumed structure: run_dir/files/files/)
        for a checkpoint file matching:
            "{target_env}__{target_env.lower()}__<seed>__<alpha>__*_step{checkpoint_step}.pth"
      - Returns a list of dictionaries for valid runs, including the algorithm.
    
    Args:
        root_dir (str): Root directory to search.
        target_env (str): Target environment ID (e.g., "Hopper-v4").
        checkpoint_step (int): Checkpoint step to use (e.g., 2450000).
        training_seeds (list): List of training seeds to consider.
        algorithms (list): List of algorithm substrings to filter for.
    
    Returns:
        list: A list of dictionaries for valid runs.
    """
    valid_runs = []
    for subdir, dirs, files in os.walk(root_dir):
        if "config.yaml" in files:
            full_config_path = os.path.join(subdir, "config.yaml")
            with open(full_config_path, "r") as f:
                config_data = yaml.safe_load(f)

            # Extract configuration values.
            env_val = (
                (config_data.get("env_id", {}) or {}).get("value", None)
                if isinstance(config_data.get("env_id"), dict)
                else config_data.get("env_id")
            )
            seed_val = (
                (config_data.get("seed", {}) or {}).get("value", None)
                if isinstance(config_data.get("seed"), dict)
                else config_data.get("seed")
            )
            alpha_val = (
                (config_data.get("alpha", {}) or {}).get("value", None)
                if isinstance(config_data.get("alpha"), dict)
                else config_data.get("alpha")
            )

            # Extract algorithm info from the _wandb key.
            algorithm_val = None
            if "_wandb" in config_data and isinstance(config_data["_wandb"], dict):
                wandb_value = config_data["_wandb"].get("value", {})
                algorithm_val = wandb_value.get("code_path", None)

            print("Loaded config from:", full_config_path)
            print("env:", env_val, "seed:", seed_val, "alpha:", alpha_val, "algorithm:", algorithm_val)

            if env_val == target_env and seed_val in training_seeds:
                # Filter runs by algorithm: require that one of the provided substrings is in algorithm_val.
                if algorithm_val is None or not any(alg in algorithm_val for alg in algorithms):
                    continue

                # Assume config.yaml is in a "files" folder inside the run directory.
                run_dir = os.path.dirname(subdir)
                # Expected checkpoint location: run_dir/files/files/
                checkpoint_dir = os.path.join(run_dir, "files", "files")
                pattern = os.path.join(
                    checkpoint_dir,
                    f"{target_env}__{target_env.lower()}__{seed_val}__{alpha_val}__*_step{checkpoint_step}.pth"
                )
                matching_files = glob.glob(pattern)
                if matching_files:
                    actor_path = matching_files[0]
                    valid_runs.append({
                        "alpha": alpha_val,
                        "seed": seed_val,
                        "path": run_dir,
                        "actor_path": actor_path,
                        "algorithm": algorithm_val  # Save the algorithm
                    })
    return valid_runs


@hydra.main(config_path="configs", config_name="hopper-v4")
def main(cfg: DictConfig):
    """
    Main function to compare KL divergence between pairs of actor networks
    from valid runs and output the results to a CSV file.
    
    Args:
        cfg (DictConfig): Hydra configuration.
    """
    # Convert the root_dir to an absolute path.
    root_dir = to_absolute_path(cfg.root_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Configuration:\n", OmegaConf.to_yaml(cfg))

    # Collect valid runs.
    runs = gather_valid_runs(root_dir, cfg.env, cfg.checkpoint_step, cfg.training_seeds, cfg.algorithms)
    print("Found valid runs:", runs)

    envs = gym.vector.SyncVectorEnv(
        [make_env(cfg.env, cfg.seed, 0, cfg.capture_video, cfg.run_name)]
    )

    # Write CSV to a fixed (absolute) folder.
    results_dir = to_absolute_path(cfg.results_dir) if "results_dir" in cfg else to_absolute_path("results")
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f"{cfg.env}-KL_divergence_final.csv")
    with open(output_file, "w") as f:
        f.write("actor_0_alpha,actor_0_seed,actor_1_alpha,actor_1_seed,algorithm,n_eval_episodes,mean_reward,mean_KL\n")

    # Compare valid runs pairwise.
    for i in range(len(runs)):
        alpha_1 = runs[i]["alpha"]
        seed_1 = runs[i]["seed"]
        checkpoint_path_1 = runs[i]["actor_path"]
        algorithm_1 = runs[i]["algorithm"]

        actor_1 = Actor(envs)
        checkpoint_1 = torch.load(checkpoint_path_1, map_location=device)
        actor_state_dict_1 = checkpoint_1["actor_state_dict"]
        if actor_state_dict_1["action_scale"].dim() == 1:
            actor_state_dict_1["action_scale"] = actor_state_dict_1["action_scale"].unsqueeze(0)
            actor_state_dict_1["action_bias"] = actor_state_dict_1["action_bias"].unsqueeze(0)
        actor_1.load_state_dict(actor_state_dict_1)
        actor_1.to(device)

        for j in range(len(runs)):
            if j == i:
                continue

            alpha_2 = runs[j]["alpha"]
            seed_2 = runs[j]["seed"]
            checkpoint_path_2 = runs[j]["actor_path"]
            algorithm_2 = runs[j]["algorithm"]

            # Only evaluate if alpha is the same, seeds differ, and both agents were trained with the same algorithm.
            if alpha_1 == alpha_2 and seed_1 != seed_2 and algorithm_1 == algorithm_2:
                actor_2 = Actor(envs)
                checkpoint_2 = torch.load(checkpoint_path_2, map_location=device)
                actor_state_dict_2 = checkpoint_2["actor_state_dict"]
                if actor_state_dict_2["action_scale"].dim() == 1:
                    actor_state_dict_2["action_scale"] = actor_state_dict_2["action_scale"].unsqueeze(0)
                    actor_state_dict_2["action_bias"] = actor_state_dict_2["action_bias"].unsqueeze(0)
                actor_2.load_state_dict(actor_state_dict_2)
                actor_2.to(device)

                mean_reward, _, mean_KL, _ = _evaluate_agent(
                    envs,
                    n_eval_episodes=cfg.n_eval_episodes,
                    actor_1=actor_1,
                    actor_2=actor_2,
                    seed=cfg.seed,
                )

                print(alpha_1, seed_1, alpha_2, seed_2, algorithm_1,
                      cfg.n_eval_episodes, mean_reward, mean_KL)
                with open(output_file, "a") as f:
                    f.write(
                        f"{alpha_1},{seed_1},{alpha_2},{seed_2},{algorithm_1},{cfg.n_eval_episodes},{mean_reward},{mean_KL}\n"
                    )


if __name__ == "__main__":
    main()
