#!/usr/bin/env python3
"""
This script gathers valid run checkpoints from a specified directory,
loads actor and Q networks (qf1 and qf2) from checkpoints at a specified step,
and performs pairwise evaluations between runs that have the same alpha value and algorithm.
For each pair, it computes:
 - KL divergence between the action distributions (naive for MEOW)
 - The L∞ norm (infinite norm) between the Q networks
 - The average Frobenius norm difference between the Jacobians

Additionally, for each pair the script evaluates actor_i (the “in charge” actor)
over n_eval_episodes to obtain its reward distribution and then computes:
 - The median reward and its bootstrap 95% confidence interval,
 - The interquartile mean (IQM) reward and its bootstrap 95% confidence interval, and
 - A score distribution via key quantiles (10th, 25th, 50th, 75th, and 90th).

These metrics are appended to the CSV output.
"""

import os
import glob
import yaml
import torch
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import defaultdict
import itertools

# Configure logging with moderate verbosity.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Define constants for SAC.
LOG_STD_MAX = 2
LOG_STD_MIN = -5

# ----------------------------------------------------------------------
# Import FlowPolicy dependencies
try:
    from cleanrl.cleanrl.nf.nets import MLP
    from cleanrl.cleanrl.nf.transforms import Preprocessing
    from cleanrl.cleanrl.nf.distributions import ConditionalDiagLinearGaussian
    from cleanrl.cleanrl.nf.flows import MaskedCondAffineFlow, CondScaling
except ImportError:
    from nf.nets import MLP
    from nf.transforms import Preprocessing
    from nf.distributions import ConditionalDiagLinearGaussian
    from nf.flows import MaskedCondAffineFlow, CondScaling

# ----------------------- Flow Initialization --------------------------
def init_Flow(sigma_max, sigma_min, action_sizes, state_sizes):
    init_parameter = "zero"
    init_parameter_flow = "orthogonal"
    dropout_rate_flow = 0.1
    dropout_rate_scale = 0.0
    layer_norm_flow = True
    layer_norm_scale = False
    hidden_layers = 2
    flow_layers = 2
    hidden_sizes = 64

    prior_list = [state_sizes] + [hidden_sizes] * hidden_layers + [action_sizes]
    loc = None
    log_scale = MLP(prior_list, init=init_parameter)
    q0 = ConditionalDiagLinearGaussian(action_sizes, loc, log_scale, SIGMA_MIN=sigma_min, SIGMA_MAX=sigma_max)

    flows = []
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(action_sizes)])
    for _ in range(flow_layers):
        layers_list = [action_sizes + state_sizes] + [hidden_sizes] * hidden_layers + [action_sizes]
        t1 = MLP(layers_list, init=init_parameter_flow, dropout_rate=dropout_rate_flow, layernorm=layer_norm_flow)
        t2 = MLP(layers_list, init=init_parameter_flow, dropout_rate=dropout_rate_flow, layernorm=layer_norm_flow)
        flows.extend([
            MaskedCondAffineFlow(b, t1, None),
            MaskedCondAffineFlow(1 - b, t2, None)
        ])
    scale_list = [state_sizes] + [256] * hidden_layers + [1]
    learnable_scale_1 = MLP(scale_list, init=init_parameter, dropout_rate=dropout_rate_scale, layernorm=layer_norm_scale)
    learnable_scale_2 = MLP(scale_list, init=init_parameter, dropout_rate=dropout_rate_scale, layernorm=layer_norm_scale)
    flows.append(CondScaling(learnable_scale_1, learnable_scale_2))
    flows.append(Preprocessing())
    return flows, q0

# ----------------------- FlowPolicy Class -------------------------------
class FlowPolicy(nn.Module):
    def __init__(self, alpha, sigma_max, sigma_min, action_sizes, state_sizes, device):
        super().__init__()
        self.device = device
        self.alpha = alpha
        flows, q0 = init_Flow(sigma_max, sigma_min, action_sizes, state_sizes)
        self.flows = nn.ModuleList(flows).to(self.device)
        self.prior = q0.to(self.device)

    def forward(self, obs, act):
        log_q = torch.zeros(act.shape[0], device=act.device)
        z = act
        for flow in self.flows:
            z, log_det = flow.forward(z, context=obs)
            log_q -= log_det
        return z, log_q

    @torch.jit.export
    def inverse(self, obs, act):
        log_q = torch.zeros(act.shape[0], device=act.device)
        z = act
        for flow in self.flows[::-1]:
            z, log_det = flow.inverse(z, context=obs)
            log_q += log_det
        return z, log_q

    @torch.jit.ignore
    def sample(self, num_samples, obs, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if deterministic:
            eps = torch.randn((num_samples,) + self.prior.shape, dtype=obs.dtype, device=obs.device)
            act, _ = self.prior.get_mean_std(eps, context=obs)
            log_q = self.prior.log_prob(act, context=obs)
        else:
            act, log_q = self.prior.sample(num_samples=num_samples, context=obs)
        a, log_det = self.forward(obs=obs, act=act)
        log_q -= log_det
        return a, log_q

    @torch.jit.export
    def log_prob(self, obs, act):
        z, log_q = self.inverse(obs=obs, act=act)
        log_q += self.prior.log_prob(z, context=obs)
        return log_q

    @torch.jit.export
    def get_qv(self, obs, act):
        q = torch.zeros((act.shape[0]), device=act.device)
        v = torch.zeros((act.shape[0]), device=act.device)
        z = act
        for flow in self.flows[::-1]:
            if hasattr(flow, "get_qv"):
                z, q_, v_ = flow.get_qv(z, context=obs)
                if q_.numel() > 0 and v_.numel() > 0:
                    q += q_
                    v += v_
        q_, v_ = self.prior.get_qv(z, context=obs)
        q += q_
        v += v_
        q = q * self.alpha
        v = v * self.alpha
        return q[:, None], v[:, None]

    @torch.jit.export
    def get_v(self, obs):
        act = torch.zeros((obs.shape[0], self.prior.shape[0]), device=self.device)
        v = torch.zeros((act.shape[0]), device=act.device)
        z = act
        for flow in self.flows[::-1]:
            z, _, v_ = flow.get_qv(z, context=obs)
            v += v_
        _, v_ = self.prior.get_qv(z, context=obs)
        v += v_
        return (v * self.alpha)[:, None]

# ----------------------- Actor Classes ----------------------------------
class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = int(np.prod(env.single_observation_space.shape))
        action_dim = int(np.prod(env.single_action_space.shape))
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
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
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x, deterministic=False):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = (normal.log_prob(x_t) - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)).sum(1, keepdim=True)
        scaled_mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, scaled_mean

class FlowActor(nn.Module):
    """
    A wrapper around FlowPolicy so that the inference script (which expects .forward() to return (mean, log_std)
    and .get_action() to return (action, log_prob, mean)) continues to work.
    """
    def __init__(self, env, alpha, sigma_max, sigma_min, device):
        super().__init__()
        self.env = env
        obs_dim = int(np.prod(env.single_observation_space.shape))
        action_dim = int(np.prod(env.single_action_space.shape))
        self.flow_policy = FlowPolicy(alpha, sigma_max, sigma_min, action_dim, obs_dim, device).to(device)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        with torch.no_grad():
            a, _ = self.flow_policy.sample(num_samples=x.shape[0], obs=x, deterministic=True)
        mean = a
        log_prob = self.flow_policy.log_prob(x, a)
        d = mean.shape[1]
        const = 0.5 * np.log(2 * np.pi)
        pseudo_log_std = ((-log_prob) - d * const) / d
        pseudo_log_std = pseudo_log_std.unsqueeze(1).expand_as(mean)
        return mean, pseudo_log_std

    def forward_for_grad(self, x):
        x = x.to(self.device)
        a, _ = self.flow_policy.sample(num_samples=x.shape[0], obs=x, deterministic=True)
        mean = a
        log_prob = self.flow_policy.log_prob(x, a)
        d = mean.shape[1]
        const = 0.5 * np.log(2 * np.pi)
        pseudo_log_std = ((-log_prob) - d * const) / d
        return mean, pseudo_log_std.unsqueeze(1).expand_as(mean)

    def get_action(self, x, deterministic=False):
        x = x.to(self.device)
        with torch.no_grad():
            action, log_q = self.flow_policy.sample(num_samples=x.shape[0], obs=x, deterministic=deterministic)
            det_mean, _ = self.forward(x)
        return action, log_q.unsqueeze(-1), det_mean

# ----------------------- SoftQNetwork -----------------------------------
class SoftQNetwork(nn.Module):
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
        return self.fc3(x)

# ----------------------- Jacobian Computation ---------------------------
def single_sample_jacobian(actor, state):
    """
    Compute d(mean(action)) / d(state) for a single sample.
    state: [1, obs_dim]
    returns: [act_dim, obs_dim]
    """
    state = state.clone().requires_grad_(True)
    actor.eval()
    if hasattr(actor, "forward_for_grad"):
        mean, _ = actor.forward_for_grad(state)
    else:
        mean, _ = actor(state)
    act_dim = mean.shape[1]
    jac_rows = []
    for a in range(act_dim):
        actor.zero_grad()
        if state.grad is not None:
            state.grad.zero_()
        mean[0, a].backward(retain_graph=True)
        jac_rows.append(state.grad[0].clone())
    return torch.stack(jac_rows, dim=0)

def compute_avg_jacobian_similarity_loop(env, n_eval_episodes, actorA, actorB, device, seed):
    """
    Computes the average Frobenius norm of the difference between the Jacobians of actorA and actorB.
    This function mimics the KL evaluation loop by using actorA's decisions.
    """
    frob_differences = []
    for episode in range(n_eval_episodes):
        logger.info(f"Jacobian Eval: Starting episode {episode}")
        obs, _ = env.reset(seed=seed)
        total_frob = 0.0
        steps = 0
        infos = {}
        while "final_info" not in infos:
            steps += 1
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            if obs_tensor.ndim == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            # Compute Jacobians.
            Ja = single_sample_jacobian(actorA, obs_tensor)
            Jb = single_sample_jacobian(actorB, obs_tensor)
            total_frob += (Ja - Jb).norm(p='fro').item()
            # Get next action from actorA.
            a1, _, _ = actorA.get_action(obs_tensor, deterministic=False)
            actions_np = a1.detach().cpu().numpy().squeeze(0)  # Remove extra batch dimension.
            next_obs, rewards, terminated, truncated, infos = env.step(actions_np)
            if terminated or truncated:
                break
            obs = next_obs
        avg_frob = total_frob / steps if steps > 0 else float('nan')
        frob_differences.append(avg_frob)
        logger.info(f"Jacobian Eval: Episode {episode} completed in {steps} steps, avg frob diff: {avg_frob}")
    overall_avg = sum(frob_differences) / len(frob_differences) if frob_differences else float('nan')
    logger.info(f"Jacobian Eval: Final average frob diff over {n_eval_episodes} episodes: {overall_avg}")
    return overall_avg

# ----------------------- Evaluation Functions ---------------------------
def _evaluate_agent(env, n_eval_episodes, actor_1, actor_2, seed=0):
    episode_rewards = []
    KL_divergence = []
    device = next(actor_1.parameters()).device
    for episode in range(n_eval_episodes):
        logger.info(f"Agent Eval: Starting episode {episode}")
        obs, _ = env.reset(seed=seed)
        total_rewards_ep, total_divergence, steps = 0, 0, 0
        infos = {}
        while "final_info" not in infos:
            steps += 1
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            a1, _, mean1 = actor_1.get_action(obs_tensor, deterministic=False)
            _, _, mean2 = actor_2.get_action(obs_tensor, deterministic=False)
            _, log_std1 = actor_1(obs_tensor)
            _, log_std2 = actor_2(obs_tensor)
            std1, std2 = log_std1.exp(), log_std2.exp()
            normal1 = torch.distributions.Normal(mean1, std1)
            normal2 = torch.distributions.Normal(mean2, std2)
            divergence = torch.distributions.kl.kl_divergence(normal1, normal2)
            total_divergence += divergence
            actions_np = a1.detach().cpu().numpy()
            next_obs, rewards, _, _, infos = env.step(actions_np)
            total_rewards_ep += rewards
            steps += 1
            obs = next_obs
        episode_rewards.append(total_rewards_ep)
        avg_divergence = (total_divergence.detach().cpu().numpy() / steps)
        KL_divergence.append(avg_divergence)
        logger.info(f"Agent Eval: Episode {episode} reward: {total_rewards_ep}, avg KL divergence: {avg_divergence}")
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_KL = np.mean(KL_divergence)
    std_KL = np.std(KL_divergence)
    logger.info(f"Agent Eval: Final results: mean_reward={mean_reward}, mean_KL={mean_KL}")
    return mean_reward, std_reward, mean_KL, std_KL

def _evaluate_agent_meow(env, n_eval_episodes, actor_1, actor_2, seed=0, num_samples=10):
    episode_rewards = []
    KL_divergence = []
    device = next(actor_1.parameters()).device
    actor_1.eval()
    actor_2.eval()
    with torch.no_grad():
        for episode in range(n_eval_episodes):
            logger.info(f"MEOW Eval: Starting episode {episode}")
            obs, _ = env.reset(seed=seed)
            total_rewards_ep, total_divergence, steps = 0.0, 0.0, 0
            infos = {}
            while "final_info" not in infos:
                steps += 1
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                context = obs_tensor.unsqueeze(0).repeat(num_samples, 1) if obs_tensor.ndim == 1 else obs_tensor.repeat(num_samples, 1)
                a_samples, _ = actor_1.flow_policy.sample(num_samples=num_samples, obs=context, deterministic=False)
                log_pi1 = actor_1.flow_policy.log_prob(context, a_samples)
                log_pi2 = actor_2.flow_policy.log_prob(context, a_samples)
                divergence = (log_pi1 - log_pi2).mean().item()
                total_divergence += divergence
                a1, _, _ = actor_1.get_action(obs_tensor, deterministic=False)
                actions_np = a1.detach().cpu().numpy()
                next_obs, rewards, terminated, truncated, infos = env.step(actions_np)
                total_rewards_ep += rewards
                if terminated or truncated:
                    break
                obs = next_obs
            episode_rewards.append(total_rewards_ep)
            avg_divergence = total_divergence / steps if steps > 0 else float("nan")
            KL_divergence.append(avg_divergence)
            logger.info(f"MEOW Eval: Episode {episode} reward: {total_rewards_ep}, avg KL divergence: {avg_divergence}")
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_KL = np.mean(KL_divergence)
    std_KL = np.std(KL_divergence)
    logger.info(f"MEOW Eval: Final results: mean_reward={mean_reward}, mean_KL={mean_KL}")
    return mean_reward, std_reward, mean_KL, std_KL

def get_algo_priority(algo):
    if "meow_continuous_action.py" in algo:
        return 0
    elif "td3_continuous_action.py" in algo:
        return 1
    elif "sac_continuous_action.py" in algo:
        return 2
    else:
        return 3

# ----------------------- Helper Functions -------------------------------
# Environment creation functions.
def make_env_sac(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk

def make_env_meow(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env_inst = gym.make(env_id, render_mode="rgb_array")
            env_inst = gym.wrappers.RecordVideo(env_inst, f"videos/{run_name}")
        else:
            env_inst = gym.make(env_id)
        env_inst = gym.wrappers.RecordEpisodeStatistics(env_inst)
        env_inst.action_space.seed(seed)
        env_inst = gym.wrappers.RescaleAction(env_inst, min_action=-1.0, max_action=1.0)
        return env_inst
    return thunk

def gather_valid_runs(root_dir, target_env, checkpoint_step, training_seeds, algorithms):
    valid_runs = []
    for subdir, dirs, files in os.walk(root_dir):
        if "config.yaml" in files:
            full_config_path = os.path.join(subdir, "config.yaml")
            with open(full_config_path, "r") as f:
                config_data = yaml.safe_load(f)
            env_val = config_data.get("env_id", {}) or config_data.get("env_id")
            if isinstance(env_val, dict):
                env_val = env_val.get("value")
            seed_val = config_data.get("seed", {}) or config_data.get("seed")
            if isinstance(seed_val, dict):
                seed_val = seed_val.get("value")
            alpha_val = config_data.get("alpha", {}) or config_data.get("alpha")
            if isinstance(alpha_val, dict):
                alpha_val = alpha_val.get("value")
            algorithm_val = None
            if "_wandb" in config_data and isinstance(config_data["_wandb"], dict):
                algorithm_val = config_data["_wandb"].get("value", {}).get("code_path")
            print("Loaded config from:", full_config_path)
            print("env:", env_val, "seed:", seed_val, "alpha:", alpha_val, "algorithm:", algorithm_val)
            if env_val == target_env and seed_val in training_seeds:
                if algorithm_val is None or not any(alg in algorithm_val for alg in algorithms):
                    continue
                run_dir = os.path.dirname(subdir)
                checkpoint_dir = os.path.join(run_dir, "files", "files")
                if alpha_val is None:
                    pattern = os.path.join(checkpoint_dir, f"{target_env}__*__{seed_val}__*_step{checkpoint_step}.pth")
                else:
                    pattern = os.path.join(checkpoint_dir, f"{target_env}__*__{seed_val}__{alpha_val}__*_step{checkpoint_step}.pth")
                matching_files = glob.glob(pattern)
                if not matching_files and alpha_val is not None:
                    pattern = os.path.join(checkpoint_dir, f"{target_env}__*__{seed_val}__*_step{checkpoint_step}.pth")
                    matching_files = glob.glob(pattern)
                if matching_files:
                    valid_runs.append({
                        "env": env_val,
                        "alpha": alpha_val,
                        "seed": seed_val,
                        "path": run_dir,
                        "actor_path": matching_files[0],
                        "algorithm": algorithm_val
                    })
    return valid_runs

def load_actor(run, env_inst, device, cfg):
    """Helper to load and return the actor from a checkpoint."""
    checkpoint = torch.load(run["actor_path"], map_location=device)
    if "actor_state_dict" in checkpoint:
        state = checkpoint["actor_state_dict"]
    elif "policy_state_dict" in checkpoint:
        state = checkpoint["policy_state_dict"]
    else:
        raise KeyError(f"No recognized actor keys in checkpoint: {run['actor_path']}")
    if "meow_continuous_action.py" in run["algorithm"]:
        from copy import deepcopy
        state = {"flow_policy." + k: v for k, v in state.items()}
        actor = FlowActor(env_inst, alpha=run["alpha"], sigma_max=cfg.sigma_max,
                          sigma_min=cfg.sigma_min, device=device)
    else:
        actor = Actor(env_inst)
        if run["algorithm"].endswith("td3_continuous_action.py"):
            if "fc_mu.weight" in state:
                state["fc_mean.weight"] = state.pop("fc_mu.weight")
                state["fc_mean.bias"] = state.pop("fc_mu.bias")
            fc_logstd_weight_shape = actor.fc_logstd.weight.shape
            fc_logstd_bias_shape = actor.fc_logstd.bias.shape
            state["fc_logstd.weight"] = torch.zeros(fc_logstd_weight_shape)
            state["fc_logstd.bias"] = torch.zeros(fc_logstd_bias_shape)
    actor.load_state_dict(state)
    actor.to(device)
    return actor

def get_algo_priority(algo):
    """
    Returns a priority number for the given algorithm string.
    Lower numbers mean higher priority.
    """
    if "meow_continuous_action.py" in algo:
        return 0
    elif "td3_continuous_action.py" in algo:
        return 1
    elif "sac_continuous_action.py" in algo:
        return 2
    else:
        return 3

# ----------------------- New Helper Functions for Reward Metrics -----------------------
def compute_IQM(scores):
    """Compute the interquartile mean (IQM) of a list of scores."""
    scores_sorted = np.sort(scores)
    n = len(scores_sorted)
    trim = int(0.25 * n)
    if n - 2 * trim > 0:
        return np.mean(scores_sorted[trim:n - trim])
    else:
        return np.mean(scores_sorted)

def bootstrap_CI(scores, stat_func, n_bootstrap=1000, alpha=0.05):
    """Compute a bootstrap confidence interval for a given statistic (e.g. np.median or compute_IQM)."""
    boot_stats = []
    scores = np.array(scores)
    n = len(scores)
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=n, replace=True)
        boot_stats.append(stat_func(sample))
    lower = np.percentile(boot_stats, 100 * (alpha / 2))
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return lower, upper

def evaluate_actor(actor, env, n_eval_episodes, seed):
    """
    Run a single actor on the environment for n_eval_episodes and return a list of total rewards per episode.
    """
    rewards_list = []
    for episode in range(n_eval_episodes):
        obs, _ = env.reset(seed=seed)
        total_reward = 0.0
        done = False
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=actor.device).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = actor.get_action(obs_tensor, deterministic=False)
            next_obs, reward, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
            total_reward += reward
            done = terminated or truncated
            obs = next_obs
        rewards_list.append(total_reward)
    return rewards_list

def compute_score_distribution(scores):
    """
    Compute the score distribution for a list of scores, returning key quantiles.
    """
    scores = np.array(scores)
    distribution = {
        '10th': np.percentile(scores, 10),
        '25th': np.percentile(scores, 25),
        '50th': np.percentile(scores, 50),
        '75th': np.percentile(scores, 75),
        '90th': np.percentile(scores, 90)
    }
    return distribution

# ----------------------- Main Function -------------------------------
@hydra.main(config_path="configs", config_name="inference")
def main(cfg: DictConfig):
    root_dir = to_absolute_path(cfg.root_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))
    
    for env in cfg.envs:
        logger.info(f"Processing environment: {env}")
        checkpoint_step = cfg.checkpoint_steps.get(env)
        if checkpoint_step is None:
            raise ValueError(f"checkpoint_step for {env} not found in the config!")
        logger.info(f"Using checkpoint step {checkpoint_step} for {env}")
        
        runs = gather_valid_runs(root_dir, env, checkpoint_step, cfg.training_seeds, cfg.algorithms)
        logger.info(f"Found valid runs for {env}: {runs}")
        if not runs:
            logger.info(f"No valid runs found for {env}; skipping.")
            continue
        
        grouped = defaultdict(list)
        for run in runs:
            grouped[(run["alpha"], run["algorithm"])].append(run)
        
        results_dir = to_absolute_path(cfg.results_dir) if "results_dir" in cfg else to_absolute_path("results")
        os.makedirs(results_dir, exist_ok=True)
        output_file = os.path.join(results_dir, f"{env}-KL_divergence_final.csv")
        with open(output_file, "w") as f:
            f.write("env,actor_0_alpha,actor_0_seed,actor_1_alpha,actor_1_seed,algorithm,"
                    "n_eval_episodes,mean_reward,mean_KL,q_output_diff,jacobian_diff,"
                    "median_reward,median_CI_lower,median_CI_upper,IQM_reward,IQM_CI_lower,IQM_CI_upper,"
                    "quantile_10,quantile_25,quantile_50,quantile_75,quantile_90\n")
        
        if any("meow_continuous_action.py" in run["algorithm"] for run in runs):
            envs_inst = gym.vector.SyncVectorEnv([make_env_meow(env, cfg.seed, 0, cfg.capture_video, cfg.run_name)])
        else:
            envs_inst = gym.vector.SyncVectorEnv([make_env_sac(env, cfg.seed, 0, cfg.capture_video, cfg.run_name)])
        
        for key in sorted(grouped.keys(), key=lambda k: get_algo_priority(k[1])):
            run_group = grouped[key]
            if len(run_group) < 2:
                continue
            for run_i, run_j in itertools.combinations(run_group, 2):
                if run_i["seed"] == run_j["seed"]:
                    continue
                
                actor_i = load_actor(run_i, envs_inst, device, cfg)
                actor_j = load_actor(run_j, envs_inst, device, cfg)
                
                if "meow_continuous_action.py" in run_i["algorithm"]:
                    mean_reward, std_reward, mean_KL, std_KL = _evaluate_agent_meow(
                        env=envs_inst, n_eval_episodes=cfg.n_eval_episodes,
                        actor_1=actor_i, actor_2=actor_j, seed=cfg.seed, num_samples=10
                    )
                else:
                    mean_reward, std_reward, mean_KL, std_KL = _evaluate_agent(
                        env=envs_inst, n_eval_episodes=cfg.n_eval_episodes,
                        actor_1=actor_i, actor_2=actor_j, seed=cfg.seed
                    )
                
                if "meow_continuous_action.py" in run_i["algorithm"]:
                    single_env_for_q = make_env_meow(env, cfg.seed, 999, cfg.capture_video, cfg.run_name)()
                    q_norm_diff = _evaluate_q_output_difference_meow(
                        env=single_env_for_q, n_eval_episodes=cfg.n_eval_episodes,
                        actor_1=actor_i, actor_2=actor_j, device=device, seed=cfg.seed
                    )
                    single_env_for_q.close()
                elif all(k in torch.load(run_i["actor_path"], map_location=device) for k in ["qf1_state_dict", "qf2_state_dict"]) and \
                     all(k in torch.load(run_j["actor_path"], map_location=device) for k in ["qf1_state_dict", "qf2_state_dict"]):
                    checkpoint_i = torch.load(run_i["actor_path"], map_location=device)
                    checkpoint_j = torch.load(run_j["actor_path"], map_location=device)
                    qf1_i = SoftQNetwork(envs_inst).to(device)
                    qf2_i = SoftQNetwork(envs_inst).to(device)
                    qf1_i.load_state_dict(checkpoint_i["qf1_state_dict"])
                    qf2_i.load_state_dict(checkpoint_i["qf2_state_dict"])
                    qf1_j = SoftQNetwork(envs_inst).to(device)
                    qf2_j = SoftQNetwork(envs_inst).to(device)
                    qf1_j.load_state_dict(checkpoint_j["qf1_state_dict"])
                    qf2_j.load_state_dict(checkpoint_j["qf2_state_dict"])
                    single_env_for_q = make_env_sac(env, cfg.seed, 999, cfg.capture_video, cfg.run_name)()
                    q_norm_diff = _evaluate_q_output_difference(
                        env=single_env_for_q, n_eval_episodes=cfg.n_eval_episodes,
                        actor_1=actor_i, qf1_i=qf1_i, qf2_i=qf2_i,
                        qf1_j=qf1_j, qf2_j=qf2_j, device=device, seed=cfg.seed
                    )
                    single_env_for_q.close()
                else:
                    q_norm_diff = float('nan')
                
                single_env_for_jac = make_env_sac(env, cfg.seed, 998, cfg.capture_video, cfg.run_name)()
                jacobian_diff = compute_avg_jacobian_similarity_loop(
                    env=single_env_for_jac, n_eval_episodes=cfg.n_eval_episodes,
                    actorA=actor_i, actorB=actor_j, device=device, seed=cfg.seed
                )
                single_env_for_jac.close()
                
                # New metrics for actor_i based on reward evaluation
                rewards_i = evaluate_actor(actor_i, envs_inst, cfg.n_eval_episodes, cfg.seed)
                median_reward_i = np.median(rewards_i)
                IQM_reward_i = compute_IQM(rewards_i)
                median_CI_lower, median_CI_upper = bootstrap_CI(rewards_i, np.median)
                IQM_CI_lower, IQM_CI_upper = bootstrap_CI(rewards_i, compute_IQM)
                score_distribution = compute_score_distribution(rewards_i)
                quantile_10 = score_distribution['10th']
                quantile_25 = score_distribution['25th']
                quantile_50 = score_distribution['50th']
                quantile_75 = score_distribution['75th']
                quantile_90 = score_distribution['90th']
                
                logger.info(f"{env}, {run_i['alpha']}, {run_i['seed']}, {run_j['alpha']}, {run_j['seed']}, "
                            f"{run_i['algorithm']}, {cfg.n_eval_episodes}, {mean_reward}, {mean_KL}, {q_norm_diff}, {jacobian_diff}, "
                            f"{median_reward_i}, {median_CI_lower}, {median_CI_upper}, {IQM_reward_i}, {IQM_CI_lower}, {IQM_CI_upper}, "
                            f"{quantile_10}, {quantile_25}, {quantile_50}, {quantile_75}, {quantile_90}")
                with open(output_file, "a") as f:
                    f.write(f"{env},{run_i['alpha']},{run_i['seed']},{run_j['alpha']},{run_j['seed']},"
                            f"{run_i['algorithm']},{cfg.n_eval_episodes},{mean_reward},"
                            f"{mean_KL},{q_norm_diff},{jacobian_diff},"
                            f"{median_reward_i},{median_CI_lower},{median_CI_upper},"
                            f"{IQM_reward_i},{IQM_CI_lower},{IQM_CI_upper},"
                            f"{quantile_10},{quantile_25},{quantile_50},{quantile_75},{quantile_90}\n")
        
        logger.info(f"Results for {env} saved to {output_file}")
    
if __name__ == "__main__":
    main()
