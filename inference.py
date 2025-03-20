#!/usr/bin/env python3
"""
This script gathers valid run checkpoints from a specified directory,
loads actor and Q networks (qf1 and qf2) from checkpoints at a specified step,
and performs pairwise evaluations between runs that have the same alpha value and algorithm.
For each pair, it computes:
 - KL divergence between the action distributions (naive for MEOW)
 - The L∞ norm (infinite norm) between the Q networks
 - The average Frobenius norm difference between the Jacobians
Evaluations are only performed if at least two valid runs exist for a given (alpha, algorithm) group.
"""

import os
import glob
import yaml
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import defaultdict

# ----------------------------------------------------------------------
# If your local FlowPolicy code lives in similar-behavior/cleanrl/cleanrl/nf,
# import it here. E.g.:
try:
    from cleanrl.nf.nets import MLP
    from cleanrl.nf.transforms import Preprocessing
    from cleanrl.nf.distributions import ConditionalDiagLinearGaussian
    from cleanrl.nf.flows import MaskedCondAffineFlow, CondScaling
    # Or, if your local path is different, adjust accordingly
except ImportError:
    # fallback or relative import if needed
    pass

# If you keep FlowPolicy in the same script or a local module:
try:
    from .meow_script import FlowPolicy  # Adjust if needed
except ImportError:
    # define a fallback or do nothing if you already have it above
    pass

# ------- Global Constants for SAC Actor --------------------------------------
LOG_STD_MAX = 2
LOG_STD_MIN = -5

# ------- Actor Definition (For SAC/TD3) --------------------------------------
class Actor(nn.Module):
    """
    Actor neural network for continuous action spaces (SAC-style).
    """
    def __init__(self, env):
        super().__init__()
        obs_dim = int(np.prod(env.single_observation_space.shape))
        action_dim = int(np.prod(env.single_action_space.shape))
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        # Action rescaling based on environment's action space.
        self.register_buffer(
            "action_scale",
            torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick.
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        # "mean" in the environment's scale
        scaled_mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, scaled_mean


# ------- FlowActor Wrapper (For MEOW) ----------------------------------------
class FlowActor(nn.Module):
    """
    A thin wrapper around MEOW's FlowPolicy so that the rest of the inference
    script (which expects an .forward() returning (mean, log_std), and a
    .get_action(...) returning (action, log_prob, mean)) does not crash.
    
    **Warning**: This is a naive approach. We produce a single 'deterministic sample'
    as the 'mean' and set 'log_std' = constant, so the script can run the same logic
    (like KL-divergence with Normal(...) ) without failing. The numbers won't truly
    represent the real flow distribution.
    """
    def __init__(self, env, alpha, sigma_max, sigma_min, device):
        super().__init__()
        self.env = env
        obs_dim = int(np.prod(env.single_observation_space.shape))
        action_dim = int(np.prod(env.single_action_space.shape))

        # Create the actual FlowPolicy from your MEOW script
        # If FlowPolicy requires `action_sizes`, `state_sizes`, do:
        self.flow_policy = FlowPolicy(
            alpha=alpha,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            action_sizes=action_dim,
            state_sizes=obs_dim,
            device=device
        ).to(device)

        # We choose some constant log_std for the naive approach:
        self._fake_log_std_val = -1.0  # or e.g. -2.0
        self.device = device

    def forward(self, x):
        """
        This is used by _evaluate_agent to get (mean, log_std).
        We'll define 'mean' = a deterministic sample from the flow,
        and 'log_std' = a constant.
        """
        x = x.to(self.device)
        with torch.no_grad():
            # We can do a 'deterministic' sample
            # i.e. sample(..., deterministic=True)
            # That yields the 'mean' from the prior + flow transforms (somewhat).
            a, _ = self.flow_policy.sample(num_samples=x.shape[0], obs=x, deterministic=True)

        mean = a
        log_std = torch.full_like(mean, self._fake_log_std_val)  # shape = [batch, action_dim]
        return mean, log_std

    def get_action(self, x):
        """
        Called at runtime to get the actual action used for stepping the env,
        along with a log_prob, and a 'mean' for logging. We'll do a random sample
        from the flow for the action, the 'mean' from the deterministic sample,
        and log_prob from the flow distribution.
        """
        x = x.to(self.device)
        with torch.no_grad():
            # random sample
            action, log_q = self.flow_policy.sample(num_samples=x.shape[0], obs=x, deterministic=False)
            # naive "mean" via a deterministic sample
            det_mean, fake_log_std = self.forward(x)
        log_prob = log_q.unsqueeze(-1)  # make it shape (batch, 1)
        return action, log_prob, det_mean


# ------- SoftQNetwork Definition (SAC style) -------------------------------
class SoftQNetwork(nn.Module):
    """
    A simple Q network (SAC style).
    """
    def __init__(self, env):
        super().__init__()
        obs_dim = int(np.prod(env.single_observation_space.shape))
        action_dim = int(np.prod(env.single_action_space.shape))
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

# ------- Jacobian Computation (for the "Actor") -----------------------------
def single_sample_jacobian(actor, state):
    """
    Manually compute d(mean(action)) / d(state) for a single sample.
    For the FlowActor, this will be a derivative w.r.t. the 'det sample.'
    """
    state = state.clone().requires_grad_(True)
    mean, _ = actor(state)  # shape: [1, act_dim]
    act_dim = mean.shape[1]

    jac_rows = []
    for a in range(act_dim):
        actor.eval()
        actor.zero_grad()
        if state.grad is not None:
            state.grad.zero_()
        mean[0, a].backward(retain_graph=True)
        jac_rows.append(state.grad[0].clone())
    return torch.stack(jac_rows, dim=0)  # [act_dim, obs_dim]

def compute_avg_jacobian_difference(env, n_eval_episodes, actorA, actorB, device, seed):
    """
    Average Frobenius norm of difference between the Jacobians of actorA and actorB.
    For MEOW (FlowActor), this is naive: we treat the deterministic sample as 'mean.'
    """
    frob_differences = []
    valid_count = 0

    for episode in range(n_eval_episodes):
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            s = torch.from_numpy(obs).float().to(device)
            if len(s.shape) == 1:
                s = s.unsqueeze(0)
            Ja = single_sample_jacobian(actorA, s)
            Jb = single_sample_jacobian(actorB, s)
            frob_diff = (Ja - Jb).norm(p='fro')
            if not torch.isnan(frob_diff):
                valid_count += 1
                frob_differences.append(frob_diff.item())
            # Step env with actorA's action
            actions, _, _ = actorA.get_action(s)
            actions = actions.squeeze(0).detach().cpu().numpy()
            next_obs, rewards, terminated, truncated, infos = env.step(actions)
            if terminated or truncated:
                done = True
            obs = next_obs
    if valid_count == 0:
        return float('nan')
    else:
        return sum(frob_differences) / valid_count


# ------- Helper / Evaluate / Gather Functions -------------------------------
def episode_trigger(episode_id: int) -> bool:
    return episode_id % 100 == 0

def make_env(env_id: str, seed: int, idx: int, capture_video: bool, run_name: str) -> callable:
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

# ------- Evaluate Actors (KL & Reward) ----------------------------------
def _evaluate_agent(env, n_eval_episodes: int, actor_1: nn.Module, actor_2: nn.Module, seed: int = 0):
    """
    Evaluates two actors by running episodes in the environment.
    Computes the mean reward (actor_1’s actions) 
    and the mean KL divergence between the action distributions.
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
            actions1, _, mean1 = actor_1.get_action(obs_tensor)
            _, _, mean2 = actor_2.get_action(obs_tensor)
            # standard dev from forward pass
            _, log_std1 = actor_1(obs_tensor)
            _, log_std2 = actor_2(obs_tensor)
            std1 = log_std1.exp()
            std2 = log_std2.exp()
            normal1 = torch.distributions.Normal(mean1, std1)
            normal2 = torch.distributions.Normal(mean2, std2)
            divergence = torch.distributions.kl.kl_divergence(normal1, normal2)
            total_divergence += divergence
            actions_np = actions1.detach().cpu().numpy()
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

# ------- Evaluate Q-output difference -----------------------------------
def _evaluate_q_output_difference(
    env,
    n_eval_episodes: int,
    actor_1: nn.Module,
    qf1_i: nn.Module,
    qf2_i: nn.Module,
    qf1_j: nn.Module,
    qf2_j: nn.Module,
    device,
    seed: int = 0,
) -> float:
    """
    Runs `n_eval_episodes` in `env` using actor_1's actions to step the environment.
    At each step, we compute:
       Q_i = min(qf1_i(s, a), qf2_i(s, a))
       Q_j = min(qf1_j(s, a), qf2_j(s, a))
    and measure abs(Q_i - Q_j).
    
    Returns the average absolute difference across all steps in all episodes.
    """
    total_q_diff = 0.0
    total_steps = 0

    # Make sure Qs are in eval mode
    qf1_i.eval()
    qf2_i.eval()
    qf1_j.eval()
    qf2_j.eval()

    for episode in range(n_eval_episodes):
        obs, _ = env.reset(seed=seed)
        done = False
        infos = {}
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                # actor_1 picks the environment action
                action_1, _, _ = actor_1.get_action(obs_tensor)
                q1_i_val = qf1_i(obs_tensor, action_1)
                q2_i_val = qf2_i(obs_tensor, action_1)
                q_i = torch.min(q1_i_val, q2_i_val)

                q1_j_val = qf1_j(obs_tensor, action_1)
                q2_j_val = qf2_j(obs_tensor, action_1)
                q_j = torch.min(q1_j_val, q2_j_val)

                diff = (q_i - q_j).abs().item()
                total_q_diff += diff
                total_steps += 1

            # Step environment with that action
            next_obs, rewards, terminations, truncations, infos = env.step(action_1.squeeze(0).cpu().numpy())
            obs = next_obs
            done = (terminations or truncations)
    
    if total_steps == 0:
        return float("nan")
    return total_q_diff / total_steps

# ------- Helper Functions ----------------------------------------------------
def episode_trigger(episode_id: int) -> bool:
    return episode_id % 100 == 0

def make_env(env_id: str, seed: int, idx: int, capture_video: bool, run_name: str) -> callable:
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

def gather_valid_runs(root_dir: str, target_env: str, checkpoint_step: int,
                      training_seeds: list, algorithms: list):
    valid_runs = []
    for subdir, dirs, files in os.walk(root_dir):
        if "config.yaml" in files:
            full_config_path = os.path.join(subdir, "config.yaml")
            with open(full_config_path, "r") as f:
                config_data = yaml.safe_load(f)
 
            env_val = config_data.get("env_id", {}) or config_data.get("env_id")
            if isinstance(env_val, dict):
                env_val = env_val.get("value", None)
 
            seed_val = config_data.get("seed", {}) or config_data.get("seed")
            if isinstance(seed_val, dict):
                seed_val = seed_val.get("value", None)
 
            alpha_val = config_data.get("alpha", {}) or config_data.get("alpha")
            if isinstance(alpha_val, dict):
                alpha_val = alpha_val.get("value", None)
 
            algorithm_val = None
            if "_wandb" in config_data and isinstance(config_data["_wandb"], dict):
                wandb_value = config_data["_wandb"].get("value", {})
                algorithm_val = wandb_value.get("code_path", None)
 
            print("Loaded config from:", full_config_path)
            print("env:", env_val, "seed:", seed_val, "alpha:", alpha_val, "algorithm:", algorithm_val)
 
            if env_val == target_env and seed_val in training_seeds:
                if algorithm_val is None or not any(alg in algorithm_val for alg in algorithms):
                    continue
 
                run_dir = os.path.dirname(subdir)
                checkpoint_dir = os.path.join(run_dir, "files", "files")
                if alpha_val is None:
                    pattern = os.path.join(
                        checkpoint_dir,
                        f"{target_env}__*__{seed_val}__*_step{checkpoint_step}.pth"
                    )
                else:
                    pattern = os.path.join(
                        checkpoint_dir,
                        f"{target_env}__*__{seed_val}__{alpha_val}__*_step{checkpoint_step}.pth"
                    )
                matching_files = glob.glob(pattern)
                if not matching_files and alpha_val is not None:
                    pattern = os.path.join(
                        checkpoint_dir,
                        f"{target_env}__*__{seed_val}__*_step{checkpoint_step}.pth"
                    )
                    matching_files = glob.glob(pattern)
                if matching_files:
                    actor_path = matching_files[0]
                    valid_runs.append({
                        "env": env_val,
                        "alpha": alpha_val,
                        "seed": seed_val,
                        "path": run_dir,
                        "actor_path": actor_path,
                        "algorithm": algorithm_val
                    })
    return valid_runs

# ------- Hydra Main Entry ----------------------------------------------------
@hydra.main(config_path="configs", config_name="inference")
def main(cfg: DictConfig):
    root_dir = to_absolute_path(cfg.root_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Configuration:\n", OmegaConf.to_yaml(cfg))

    for env in cfg.envs:
        print(f"\nProcessing environment: {env}")
        checkpoint_step = cfg.checkpoint_steps.get(env)
        if checkpoint_step is None:
            raise ValueError(f"checkpoint_step for {env} not found in the config!")
        print(f"Using checkpoint step {checkpoint_step} for {env}")

        runs = gather_valid_runs(root_dir, env, checkpoint_step, cfg.training_seeds, cfg.algorithms)
        print(f"Found valid runs for {env}:", runs)
        if not runs:
            print(f"No valid runs found for {env}; skipping.")
            continue

        grouped = defaultdict(list)
        for run in runs:
            key = (run["alpha"], run["algorithm"])
            grouped[key].append(run)

        results_dir = to_absolute_path(cfg.results_dir) if "results_dir" in cfg else to_absolute_path("results")
        os.makedirs(results_dir, exist_ok=True)
        output_file = os.path.join(results_dir, f"{env}-KL_divergence_final.csv")
        with open(output_file, "w") as f:
            # columns
            f.write("env,actor_0_alpha,actor_0_seed,actor_1_alpha,actor_1_seed,algorithm,n_eval_episodes,mean_reward,mean_KL,q_output_diff,jacobian_diff\n")

        envs_inst = gym.vector.SyncVectorEnv(
            [make_env(env, cfg.seed, 0, cfg.capture_video, cfg.run_name)]
        )

        for key, group in grouped.items():
            alpha_val, algorithm_val = key
            if len(group) < 2:
                print(f"Skipping env {env}, alpha {alpha_val}, algorithm {algorithm_val} because only {len(group)} run(s) available")
                continue
            for i in range(len(group)):
                run_i = group[i]
                actor_path_i = run_i["actor_path"]
                seed_i = run_i["seed"]
 
                # Load checkpoint i
                checkpoint_i = torch.load(actor_path_i, map_location=device)

                # Build the standard Actor (SAC or TD3) or FlowActor (if needed)
                actor_i = Actor(envs_inst)  # Or FlowActor if you are handling MEOW
                # Decide which key to use
                if "actor_state_dict" in checkpoint_i:
                    state_i = checkpoint_i["actor_state_dict"]
                elif "policy_state_dict" in checkpoint_i:
                    state_i = checkpoint_i["policy_state_dict"]
                else:
                    raise KeyError(f"No recognized actor keys in checkpoint: {actor_path_i}")
                
                # If this is TD3, do key renames + zero out logstd
                if run_i["algorithm"].endswith("td3_continuous_action.py"):
                    if "fc_mu.weight" in state_i:
                        state_i["fc_mean.weight"] = state_i.pop("fc_mu.weight")
                        state_i["fc_mean.bias"] = state_i.pop("fc_mu.bias")
                    fc_logstd_weight_shape = actor_i.fc_logstd.weight.shape
                    fc_logstd_bias_shape = actor_i.fc_logstd.bias.shape
                    state_i["fc_logstd.weight"] = torch.zeros(fc_logstd_weight_shape)
                    state_i["fc_logstd.bias"] = torch.zeros(fc_logstd_bias_shape)

                # Fix shape if needed
                if "action_scale" in state_i and state_i["action_scale"].ndim == 2:
                    state_i["action_scale"] = state_i["action_scale"].squeeze(0)
                if "action_bias" in state_i and state_i["action_bias"].ndim == 2:
                    state_i["action_bias"] = state_i["action_bias"].squeeze(0)

                actor_i.load_state_dict(state_i)
                actor_i.to(device)

                for j in range(i + 1, len(group)):
                    run_j = group[j]
                    actor_path_j = run_j["actor_path"]
                    seed_j = run_j["seed"]
 
                    checkpoint_j = torch.load(actor_path_j, map_location=device)

                    # build actor_j
                    actor_j = Actor(envs_inst)
                    if "actor_state_dict" in checkpoint_j:
                        state_j = checkpoint_j["actor_state_dict"]
                    elif "policy_state_dict" in checkpoint_j:
                        state_j = checkpoint_j["policy_state_dict"]
                    else:
                        raise KeyError(f"No recognized actor keys in checkpoint: {actor_path_j}")
 
                    if run_j["algorithm"].endswith("td3_continuous_action.py"):
                        if "fc_mu.weight" in state_j:
                            state_j["fc_mean.weight"] = state_j.pop("fc_mu.weight")
                            state_j["fc_mean.bias"] = state_j.pop("fc_mu.bias")
                        fc_logstd_weight_shape = actor_j.fc_logstd.weight.shape
                        fc_logstd_bias_shape = actor_j.fc_logstd.bias.shape
                        state_j["fc_logstd.weight"] = torch.zeros(fc_logstd_weight_shape)
                        state_j["fc_logstd.bias"] = torch.zeros(fc_logstd_bias_shape)
 
                    if "action_scale" in state_j and state_j["action_scale"].ndim == 2:
                        state_j["action_scale"] = state_j["action_scale"].squeeze(0)
                    if "action_bias" in state_j and state_j["action_bias"].ndim == 2:
                        state_j["action_bias"] = state_j["action_bias"].squeeze(0)
 
                    actor_j.load_state_dict(state_j)
                    actor_j.to(device)

                    # Evaluate for KL & reward (using _evaluate_agent)
                    mean_reward, _, mean_KL, _ = _evaluate_agent(
                        envs_inst,
                        n_eval_episodes=cfg.n_eval_episodes,
                        actor_1=actor_i,
                        actor_2=actor_j,
                        seed=cfg.seed
                    )

                    # Build Q networks if available
                    if all(k in checkpoint_i for k in ["qf1_state_dict", "qf2_state_dict"]) and \
                       all(k in checkpoint_j for k in ["qf1_state_dict", "qf2_state_dict"]):

                        qf1_i = SoftQNetwork(envs_inst).to(device)
                        qf2_i = SoftQNetwork(envs_inst).to(device)
                        qf1_i.load_state_dict(checkpoint_i["qf1_state_dict"])
                        qf2_i.load_state_dict(checkpoint_i["qf2_state_dict"])

                        qf1_j = SoftQNetwork(envs_inst).to(device)
                        qf2_j = SoftQNetwork(envs_inst).to(device)
                        qf1_j.load_state_dict(checkpoint_j["qf1_state_dict"])
                        qf2_j.load_state_dict(checkpoint_j["qf2_state_dict"])

                        # Evaluate Q-output difference
                        # create a fresh single-env for Q difference if you like:
                        single_env_for_q = make_env(env, cfg.seed, 999, cfg.capture_video, cfg.run_name)()
                        q_norm_diff = _evaluate_q_output_difference(
                            env=single_env_for_q,
                            n_eval_episodes=cfg.n_eval_episodes,
                            actor_1=actor_i,      # actor_i drives the env
                            qf1_i=qf1_i,
                            qf2_i=qf2_i,
                            qf1_j=qf1_j,
                            qf2_j=qf2_j,
                            device=device,
                            seed=cfg.seed
                        )
                        single_env_for_q.close()
                    else:
                        q_norm_diff = float('nan')

                    # Jacobian difference
                    single_env_for_jac = make_env(env, cfg.seed, 998, cfg.capture_video, cfg.run_name)()
                    jacobian_diff = compute_avg_jacobian_difference(
                        env=single_env_for_jac,
                        n_eval_episodes=cfg.n_eval_episodes,
                        actorA=actor_i,
                        actorB=actor_j,
                        device=device,
                        seed=cfg.seed
                    )
                    single_env_for_jac.close()

                    print(env, alpha_val, seed_i, alpha_val, seed_j, algorithm_val,
                          cfg.n_eval_episodes, mean_reward, mean_KL, q_norm_diff, jacobian_diff)
                    with open(output_file, "a") as f:
                        f.write(f"{env},{alpha_val},{seed_i},{alpha_val},{seed_j},"
                                f"{algorithm_val},{cfg.n_eval_episodes},{mean_reward},"
                                f"{mean_KL},{q_norm_diff},{jacobian_diff}\n")

        print(f"Results for {env} saved to {output_file}")


if __name__ == "__main__":
    main()