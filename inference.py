#!/usr/bin/env python3
"""
This script gathers valid run checkpoints from a specified directory,
loads actor and Q networks (qf1 and qf2) from checkpoints at a specified step,
and performs pairwise evaluations between runs that have the same alpha value and algorithm.

For each pair, it now logs **per-episode** results (no averaging). 
We report:
 - Episode reward
 - KL divergence (naive for MEOW)
 - The L∞ norm (max) between the Q networks for that episode
 - The Frobenius norm difference between the Jacobians for that episode

Additionally, we include columns for the exact checkpoint paths used (checkpoint_i, checkpoint_j).
No median, IQM, or bootstrap CIs are computed here.

Changes in this version:
 - We do **not** use a Gym vector environment anymore for evaluation. Instead, we create a single
   normal environment each time, so the action shape is always (action_dim,).
 - For MEOW runs, we wrap the environment with `RescaleAction`.
"""

import os
import glob
import yaml
import torch
import hydra
import logging
import csv
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import defaultdict
import itertools

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LOG_STD_MAX = 2
LOG_STD_MIN = -5

# ----------------------------------------------------------------------
# FlowPolicy dependencies (try local import if needed)
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
    q0 = ConditionalDiagLinearGaussian(
        action_sizes, loc, log_scale,
        SIGMA_MIN=sigma_min, SIGMA_MAX=sigma_max
    )

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
    learnable_scale_1 = MLP(scale_list, init=init_parameter, dropout_rate=0.0, layernorm=False)
    learnable_scale_2 = MLP(scale_list, init=init_parameter, dropout_rate=0.0, layernorm=False)
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

    def inverse(self, obs, act):
        log_q = torch.zeros(act.shape[0], device=act.device)
        z = act
        for flow in self.flows[::-1]:
            z, log_det = flow.inverse(z, context=obs)
            log_q += log_det
        return z, log_q

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

    def log_prob(self, obs, act):
        z, log_q = self.inverse(obs=obs, act=act)
        log_q += self.prior.log_prob(z, context=obs)
        return log_q

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
def _get_env_spaces(env):
    if hasattr(env, "single_observation_space"):
        obs_space = env.single_observation_space
        act_space = env.single_action_space
    else:
        obs_space = env.observation_space
        act_space = env.action_space
    return obs_space, act_space

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_space, act_space = _get_env_spaces(env)
        obs_dim = int(np.prod(obs_space.shape))
        action_dim = int(np.prod(act_space.shape))

        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)

        action_scale = (act_space.high - act_space.low) / 2.0
        action_bias = (act_space.high + act_space.low) / 2.0

        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor(action_bias, dtype=torch.float32))

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
        if deterministic:
            x_t = mean
        else:
            x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = (
            normal.log_prob(x_t)
            - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        ).sum(1, keepdim=True)
        scaled_mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, scaled_mean

class FlowActor(nn.Module):
    def __init__(self, env, alpha, sigma_max, sigma_min, device):
        super().__init__()
        obs_space, act_space = _get_env_spaces(env)
        obs_dim = int(np.prod(obs_space.shape))
        action_dim = int(np.prod(act_space.shape))

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
            action, log_q = self.flow_policy.sample(
                num_samples=x.shape[0], obs=x, deterministic=deterministic
            )
            det_mean, _ = self.forward(x)
        return action, log_q.unsqueeze(-1), det_mean

# ----------------------- SoftQNetwork -----------------------------------
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_space, act_space = _get_env_spaces(env)
        obs_dim = int(np.prod(obs_space.shape))
        action_dim = int(np.prod(act_space.shape))

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

def compute_jacobian_diff_episode(env, actorA, actorB, device, seed=0):
    obs, _ = env.reset(seed=seed)
    done = False
    total_frob = 0.0
    steps = 0
    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        Ja = single_sample_jacobian(actorA, obs_tensor)
        Jb = single_sample_jacobian(actorB, obs_tensor)
        total_frob += (Ja - Jb).norm(p='fro').item()
        steps += 1

        with torch.no_grad():
            a1, _, _ = actorA.get_action(obs_tensor, deterministic=False)
        a1_np = a1.cpu().numpy().squeeze(0)

        next_obs, _, terminated, truncated, _ = env.step(a1_np)
        obs = next_obs
        done = terminated or truncated
    return total_frob / max(1, steps)

# ------------- Q Output Diff, single episode versions -------------------
def evaluate_q_output_diff_episode(env, actor_1, qf1_i, qf2_i, qf1_j, qf2_j, device, seed=0):
    obs, _ = env.reset(seed=seed)
    done = False
    max_q_diff_episode = 0.0
    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_1, _, _ = actor_1.get_action(obs_tensor, deterministic=False)
            q_i = torch.max(qf1_i(obs_tensor, action_1), qf2_i(obs_tensor, action_1))
            q_j = torch.max(qf1_j(obs_tensor, action_1), qf2_j(obs_tensor, action_1))
            current_diff = (q_i - q_j).abs().item()
            if current_diff > max_q_diff_episode:
                max_q_diff_episode = current_diff

        a1_np = action_1.cpu().numpy().squeeze(0)
        next_obs, _, terminated, truncated, _ = env.step(a1_np)
        obs = next_obs
        done = terminated or truncated
    return max_q_diff_episode

def evaluate_q_output_diff_episode_meow(env, actor_1, actor_2, device, seed=0):
    obs, _ = env.reset(seed=seed)
    done = False
    max_q_diff_episode = 0.0
    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            a1, _, _ = actor_1.get_action(obs_tensor, deterministic=False)
            q1, _ = actor_1.flow_policy.get_qv(obs_tensor, a1)
            q2, _ = actor_2.flow_policy.get_qv(obs_tensor, a1)
            current_diff = (q1 - q2).abs().max().item()
            if current_diff > max_q_diff_episode:
                max_q_diff_episode = current_diff

        a1_np = a1.cpu().numpy().squeeze(0)
        next_obs, _, terminated, truncated, _ = env.step(a1_np)
        obs = next_obs
        done = terminated or truncated
    return max_q_diff_episode

# -------------------- KL + Reward, single-episode -----------------------
def evaluate_agent_episode(env, actor_1, actor_2, device, seed=0):
    obs, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    kl_sum = 0.0
    steps = 0
    while not done:
        steps += 1
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            a1, _, mean1 = actor_1.get_action(obs_tensor, deterministic=False)
            _, _, mean2 = actor_2.get_action(obs_tensor, deterministic=False)
            mean_tmp1, log_std1 = actor_1(obs_tensor)
            mean_tmp2, log_std2 = actor_2(obs_tensor)
            std1 = log_std1.exp()
            std2 = log_std2.exp()
            dist1 = torch.distributions.Normal(mean1, std1)
            dist2 = torch.distributions.Normal(mean2, std2)
            kl_tensor = torch.distributions.kl.kl_divergence(dist1, dist2).mean()
            # Clamp negative KL values to zero
            kl = kl_tensor.clamp(min=0).item()
            kl_sum += kl

            a1_np = a1.cpu().numpy().squeeze(0)
        next_obs, reward, terminated, truncated, _ = env.step(a1_np)
        total_reward += reward
        obs = next_obs
        done = terminated or truncated

    avg_kl = kl_sum / max(1, steps)
    return total_reward, avg_kl

def evaluate_agent_episode_meow(env, actor_1, actor_2, device, seed=0, num_samples=10):
    obs, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    kl_sum = 0.0
    steps = 0
    while not done:
        steps += 1
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            context = obs_tensor.repeat(num_samples, 1)
            a_samples, log_pi1 = actor_1.flow_policy.sample(
                num_samples=num_samples, obs=context, deterministic=False
            )
            log_pi2 = actor_2.flow_policy.log_prob(context, a_samples)
            kl_tensor = (log_pi1 - log_pi2).mean()
            step_kl = kl_tensor.clamp(min=0).item()
            kl_sum += step_kl

            a1, _, _ = actor_1.get_action(obs_tensor, deterministic=False)
            a1_np = a1.cpu().numpy().squeeze(0)

        next_obs, reward, terminated, truncated, _ = env.step(a1_np)
        total_reward += reward
        obs = next_obs
        done = terminated or truncated

    avg_kl = kl_sum / max(1, steps)
    return total_reward, avg_kl

# -------------------- Env Creation Helpers -----------------------------
def get_algo_priority(algo):
    if "meow_continuous_action.py" in algo:
        return 0
    elif "td3_continuous_action.py" in algo:
        return 1
    elif "sac_continuous_action.py" in algo:
        return 2
    else:
        return 3

def make_env_sac(env_id, seed, idx, capture_video, run_name):
    env_ = gym.make(env_id)
    env_ = gym.wrappers.RecordEpisodeStatistics(env_)
    if capture_video and idx == 0:
        env_ = gym.wrappers.RecordVideo(env_, f"videos/{run_name}", episode_trigger=lambda x: True)
    env_.action_space.seed(seed)
    return env_

def make_env_meow(env_id, seed, idx, capture_video, run_name):
    env_ = gym.make(env_id)
    env_ = gym.wrappers.RecordEpisodeStatistics(env_)
    if capture_video and idx == 0:
        env_ = gym.wrappers.RecordVideo(env_, f"videos/{run_name}", episode_trigger=lambda x: True)
    env_.action_space.seed(seed)
    env_ = gym.wrappers.RescaleAction(env_, -1.0, 1.0)
    return env_

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

            if env_val == target_env and seed_val in training_seeds:
                if algorithm_val is None or not any(alg in algorithm_val for alg in algorithms):
                    continue

                run_dir = os.path.dirname(subdir)
                checkpoint_dir = os.path.join(run_dir, "files")
                env_for_filename = target_env.split("/")[-1]

                if alpha_val is not None:
                    pattern = os.path.join(
                        checkpoint_dir, '**',
                        f"{env_for_filename}__*__{seed_val}__{alpha_val}__*_step{checkpoint_step}.pth"
                    )
                    matching_files = glob.glob(pattern, recursive=True)
                    if not matching_files:
                        pattern = os.path.join(
                            checkpoint_dir, '**',
                            f"{env_for_filename}__*__{seed_val}__*_step{checkpoint_step}.pth"
                        )
                        matching_files = glob.glob(pattern, recursive=True)
                else:
                    pattern = os.path.join(
                        checkpoint_dir, '**',
                        f"{env_for_filename}__*__{seed_val}__*_step{checkpoint_step}.pth"
                    )
                    matching_files = glob.glob(pattern, recursive=True)

                if matching_files:
                    valid_runs.append({
                        "env": env_val,
                        "alpha": alpha_val,
                        "seed": seed_val,
                        "path": run_dir,
                        "actor_path": matching_files[0],
                        "algorithm": algorithm_val,
                    })
    print(valid_runs)
    return valid_runs

def load_actor(run, device, cfg):
    checkpoint = torch.load(run["actor_path"], map_location=device)
    if "actor_state_dict" in checkpoint:
        state = checkpoint["actor_state_dict"]
    elif "policy_state_dict" in checkpoint:
        state = checkpoint["policy_state_dict"]
    else:
        raise KeyError(f"No recognized actor keys in checkpoint: {run['actor_path']}")

    if "meow_continuous_action.py" in run["algorithm"]:
        env_for_actor = make_env_meow(run["env"], cfg.seed, 9999, cfg.capture_video, cfg.run_name)
    else:
        env_for_actor = make_env_sac(run["env"], cfg.seed, 9999, cfg.capture_video, cfg.run_name)

    if "meow_continuous_action.py" in run["algorithm"]:
        state = {"flow_policy." + k: v for k, v in state.items()}
        actor = FlowActor(
            env_for_actor,
            alpha=run["alpha"],
            sigma_max=cfg.sigma_max,
            sigma_min=cfg.sigma_min,
            device=device
        )
    else:
        actor = Actor(env_for_actor)
        if run["algorithm"].endswith("td3_continuous_action.py"):
            if "fc_mu.weight" in state:
                state["fc_mean.weight"] = state.pop("fc_mu.weight")
                state["fc_mean.bias"] = state.pop("fc_mu.bias")
            fc_logstd_w_shape = actor.fc_logstd.weight.shape
            fc_logstd_b_shape = actor.fc_logstd.bias.shape
            state["fc_logstd.weight"] = torch.zeros(fc_logstd_w_shape)
            state["fc_logstd.bias"] = torch.zeros(fc_logstd_b_shape)

    actor.load_state_dict(state)
    actor.to(device)
    env_for_actor.close()
    return actor

def create_single_env_for_eval(run, seed, idx, capture_video, run_name):
    if "meow_continuous_action.py" in run["algorithm"]:
        return make_env_meow(run["env"], seed, idx, capture_video, run_name)
    else:
        return make_env_sac(run["env"], seed, idx, capture_video, run_name)

def create_single_env_for_q(run, env_id, seed, capture_video, run_name, idx=999):
    if "meow_continuous_action.py" in run["algorithm"]:
        return make_env_meow(env_id, seed, idx, capture_video, run_name)
    else:
        return make_env_sac(env_id, seed, idx, capture_video, run_name)

def create_single_env_for_jac(run, env_id, seed, capture_video, run_name, idx=998):
    if "meow_continuous_action.py" in run["algorithm"]:
        return make_env_meow(env_id, seed, idx, capture_video, run_name)
    else:
        return make_env_sac(env_id, seed, idx, capture_video, run_name)

# ----------------------------------------------------------------------
#                    Main evaluation + incremental CSV
# ----------------------------------------------------------------------

HEADER = (
    "env,algorithm,"
    "actor_i_alpha,actor_i_seed,checkpoint_i,"
    "actor_j_alpha,actor_j_seed,checkpoint_j,"
    "episode_idx,episode_reward,episode_kl,episode_q_infnorm,episode_jacobian_diff\n"
)


def _load_existing_keys(csv_path):
    """Return a set of tuple‑keys for rows that are already present."""
    keys = set()
    if not os.path.isfile(csv_path):
        return keys
    with open(csv_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            keys.add(
                (
                    row["algorithm"],
                    row["actor_i_alpha"], row["actor_i_seed"], row["checkpoint_i"],
                    row["actor_j_alpha"], row["actor_j_seed"], row["checkpoint_j"],
                    row["episode_idx"],
                )
            )
    return keys


def _maybe_write_header(csv_path):
    if not os.path.isfile(csv_path):
        with open(csv_path, "w", newline="") as fh:
            fh.write(HEADER)


@hydra.main(config_path="configs", config_name="inference")
def main(cfg: DictConfig):
    root_dir = to_absolute_path(cfg.root_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    results_dir = to_absolute_path(cfg.get("results_dir", "results"))
    os.makedirs(results_dir, exist_ok=True)

    for env in cfg.envs:
        logger.info(f"Processing environment: {env}")
        checkpoint_step = cfg.checkpoint_steps.get(env)
        if checkpoint_step is None:
            raise ValueError(f"checkpoint_step for {env} not found in the config!")
        logger.info(f"Using checkpoint step {checkpoint_step} for {env}")

        # discover runs
        runs = gather_valid_runs(
            root_dir,
            env,
            checkpoint_step,
            cfg.training_seeds,
            cfg.algorithms,
        )
        if not runs:
            logger.info(f"No valid runs found for {env}; skipping.")
            continue

        # Group by (alpha, algorithm)
        grouped = defaultdict(list)
        for run in runs:
            grouped[(run["alpha"], run["algorithm"])].append(run)

        # Prepare CSV bookkeeping
        output_file = os.path.join(results_dir, f"{env}-pairwise_per_episode.csv")
        _maybe_write_header(output_file)
        existing_keys = _load_existing_keys(output_file)

        # --------------------------------------------------------------
        #                 Pairwise comparisons (unchanged)
        # --------------------------------------------------------------
        for key in sorted(grouped.keys(), key=lambda k: get_algo_priority(k[1])):
            run_group = grouped[key]
            if len(run_group) < 2:
                continue

            for run_i, run_j in itertools.permutations(run_group, 2):
                if run_i["seed"] == run_j["seed"]:
                    continue

                # pair description key (incl episode later)
                base_key = (
                    run_i["algorithm"],
                    str(run_i["alpha"]), str(run_i["seed"]), run_i["actor_path"],
                    str(run_j["alpha"]), str(run_j["seed"]), run_j["actor_path"],
                )

                # --- load actors & (optionally) Q‑nets (original code, unmodified) ---
                actor_i = load_actor(run_i, device, cfg)
                actor_j = load_actor(run_j, device, cfg)
                actor_i.eval(); actor_j.eval()

                have_q_networks = False
                qf1_i = qf2_i = qf1_j = qf2_j = None
                checkpoint_i = torch.load(run_i["actor_path"], map_location=device)
                checkpoint_j = torch.load(run_j["actor_path"], map_location=device)
                if all(k in checkpoint_i for k in ["qf1_state_dict", "qf2_state_dict"]) and \
                   all(k in checkpoint_j for k in ["qf1_state_dict", "qf2_state_dict"]):
                    have_q_networks = True
                    env_for_q = create_single_env_for_q(run_i, env, cfg.seed, cfg.capture_video, cfg.run_name, idx=999)
                    qf1_i = SoftQNetwork(env_for_q).to(device)
                    qf2_i = SoftQNetwork(env_for_q).to(device)
                    qf1_i.load_state_dict(checkpoint_i["qf1_state_dict"])
                    qf2_i.load_state_dict(checkpoint_i["qf2_state_dict"])
                    qf1_j = SoftQNetwork(env_for_q).to(device)
                    qf2_j = SoftQNetwork(env_for_q).to(device)
                    qf1_j.load_state_dict(checkpoint_j["qf1_state_dict"])
                    qf2_j.load_state_dict(checkpoint_j["qf2_state_dict"])
                    env_for_q.close()

                # ------------------------------------------------------
                #                   Episode loop
                # ------------------------------------------------------
                for episode_idx in range(cfg.n_eval_episodes):
                    row_key = base_key + (str(episode_idx),)
                    if row_key in existing_keys:
                        continue  # already logged ➜ skip computation

                    eval_env = create_single_env_for_eval(run_i, cfg.seed + episode_idx, episode_idx, cfg.capture_video, cfg.run_name)

                    if "meow_continuous_action.py" in run_i["algorithm"]:
                        ep_reward, ep_kl = evaluate_agent_episode_meow(
                            env=eval_env,
                            actor_1=actor_i,
                            actor_2=actor_j,
                            device=device,
                            seed=cfg.seed + episode_idx,
                            num_samples=25,
                        )
                    else:
                        ep_reward, ep_kl = evaluate_agent_episode(
                            env=eval_env,
                            actor_1=actor_i,
                            actor_2=actor_j,
                            device=device,
                            seed=cfg.seed + episode_idx,
                        )
                    eval_env.close()

                    # Q‑diff --------------------------------------------------
                    if "meow_continuous_action.py" in run_i["algorithm"]:
                        q_env = create_single_env_for_q(run_i, env, cfg.seed + episode_idx, cfg.capture_video, cfg.run_name, idx=993)
                        ep_qdiff = evaluate_q_output_diff_episode_meow(
                            q_env,
                            actor_1=actor_i,
                            actor_2=actor_j,
                            device=device,
                            seed=cfg.seed + episode_idx,
                        )
                        q_env.close()
                    else:
                        if have_q_networks:
                            q_env = create_single_env_for_q(run_i, env, cfg.seed + episode_idx, cfg.capture_video, cfg.run_name, idx=994)
                            ep_qdiff = evaluate_q_output_diff_episode(
                                q_env,
                                actor_1=actor_i,
                                qf1_i=qf1_i,
                                qf2_i=qf2_i,
                                qf1_j=qf1_j,
                                qf2_j=qf2_j,
                                device=device,
                                seed=cfg.seed + episode_idx,
                            )
                            q_env.close()
                        else:
                            ep_qdiff = float("nan")

                    # Jacobian diff -----------------------------------------
                    jac_env = create_single_env_for_jac(run_i, env, cfg.seed + episode_idx, cfg.capture_video, cfg.run_name, idx=995)
                    ep_jacdiff = compute_jacobian_diff_episode(
                        jac_env,
                        actorA=actor_i,
                        actorB=actor_j,
                        device=device,
                        seed=cfg.seed + episode_idx,
                    )
                    jac_env.close()

                    # ------------------- write row -------------------------
                    with open(output_file, "a", newline="") as fh:
                        fh.write(
                            f"{env},{run_i['algorithm']},{run_i['alpha']},{run_i['seed']},{run_i['actor_path']},"
                            f"{run_j['alpha']},{run_j['seed']},{run_j['actor_path']},"
                            f"{episode_idx},{ep_reward},{ep_kl},{ep_qdiff},{ep_jacdiff}\n"
                        )
                    existing_keys.add(row_key)  # ensure dedupe within run

        logger.info(f"Results for {env} written/updated at {output_file}")


if __name__ == "__main__" and os.path.basename(__file__) == "inference.py":
    main()
