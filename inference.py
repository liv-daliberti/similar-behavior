#!/usr/bin/env python3
"""
Parallel CPU-Only Evaluation Script
- Uses all available CPUs
- Avoids GPU usage
- Parallelizes per-episode run comparisons
- TD3 is run deterministically; all other policies are stochastic.
- Always loads both Q-nets out of the same actor checkpoint.
"""
import os
import glob
import yaml
import torch
import hydra
import logging
import gymnasium as gym
import csv
import multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import itertools
from tqdm import tqdm

# ─── Logging & single-thread PyTorch ───────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

LOG_STD_MAX = 2
LOG_STD_MIN = -5

# ─── FlowPolicy dependencies ────────────────────────────────────────────
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

# ─── Flow Initialization ────────────────────────────────────────────────
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


# ─── Actor & FlowActor ─────────────────────────────────────────────────
def _get_env_spaces(env):
    if hasattr(env, "single_observation_space"):
        return env.single_observation_space, env.single_action_space
    return env.observation_space, env.action_space

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_s, act_s = _get_env_spaces(env)
        od = int(np.prod(obs_s.shape))
        ad = int(np.prod(act_s.shape))
        self.fc1      = nn.Linear(od, 256)
        self.fc2      = nn.Linear(256, 256)
        self.fc_mean  = nn.Linear(256, ad)
        self.fc_logstd= nn.Linear(256, ad)
        scale = (act_s.high-act_s.low)/2
        bias  = (act_s.high+act_s.low)/2
        self.register_buffer("action_scale", torch.tensor(scale, dtype=torch.float32))
        self.register_buffer("action_bias",  torch.tensor(bias,  dtype=torch.float32))

    def forward(self, x):
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x))
        m = self.fc_mean(x)
        ls= torch.tanh(self.fc_logstd(x))
        ls= LOG_STD_MIN + 0.5*(LOG_STD_MAX-LOG_STD_MIN)*(ls+1)
        return m, ls

    def get_action(self, x, deterministic=False):
        m, ls = self(x); std = ls.exp()
        dist = torch.distributions.Normal(m, std)
        xt   = m if deterministic else dist.rsample()
        yt   = torch.tanh(xt)
        a    = yt*self.action_scale + self.action_bias
        logp = (dist.log_prob(xt)
                - torch.log(self.action_scale*(1-yt.pow(2))+1e-6))\
               .sum(1, keepdim=True)
        return a, logp, m

class FlowActor(nn.Module):
    def __init__(self, env, alpha, sigma_max, sigma_min, device):
        super().__init__()
        obs_s, act_s = _get_env_spaces(env)
        od = int(np.prod(obs_s.shape)); ad = int(np.prod(act_s.shape))
        self.flow_policy = FlowPolicy(alpha, sigma_max, sigma_min, ad, od, device).to(device)
        self.device      = device

    def forward_for_grad(self, x):
        x = x.to(self.device)
        a, _ = self.flow_policy.sample(num_samples=x.shape[0], obs=x, deterministic=True)
        lp   = self.flow_policy.log_prob(x, a)
        d    = a.shape[1]; c = 0.5*np.log(2*np.pi)
        pseudo = ((-lp) - d*c)/d
        return a, pseudo.unsqueeze(1).expand_as(a)

    def forward(self, x):
        with torch.no_grad():
            return self.forward_for_grad(x)

    def get_action(self, x, deterministic=False):
        x = x.to(self.device)
        with torch.no_grad():
            a, log_q = self.flow_policy.sample(num_samples=x.shape[0], obs=x, deterministic=deterministic)
        return a, log_q.unsqueeze(-1), a

# ─── Critic network ─────────────────────────────────────────────────────
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_s, act_s = _get_env_spaces(env)
        od = int(np.prod(obs_s.shape)); ad = int(np.prod(act_s.shape))
        self.fc1 = nn.Linear(od+ad, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1)
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x))
        return self.fc3(x)

# ─── Jacobian utilities ─────────────────────────────────────────────────
def single_sample_jacobian(actor, state):
    state = state.clone().requires_grad_(True)
    actor.eval()
    mean, _ = (actor.forward_for_grad(state)
               if hasattr(actor, "forward_for_grad")
               else actor(state))
    rows = []
    for i in range(mean.shape[1]):
        actor.zero_grad()
        if state.grad is not None:
            state.grad.zero_()
        mean[0,i].backward(retain_graph=True)
        rows.append(state.grad[0].clone())
    return torch.stack(rows, dim=0)

def compute_jacobian_diff_episode(env, actorA, actorB, device, seed=0, deterministic=False):
    obs, _ = env.reset(seed=seed)
    done = False; total=0.0; steps=0
    while not done:
        st = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        Ja = single_sample_jacobian(actorA, st)
        Jb = single_sample_jacobian(actorB, st)
        total += (Ja-Jb).norm(p='fro').item()
        steps+=1
        with torch.no_grad():
            a1, _, _ = actorA.get_action(st, deterministic=deterministic)
        obs, _, t1, t2, _ = env.step(a1.detach().cpu().numpy().squeeze(0))
        done = t1 or t2
    return total/ max(1, steps)

# ─── Reward + KL per episode ────────────────────────────────────────────
def evaluate_agent_episode(env, a1, a2, device, seed=0, deterministic=False):
    obs, _ = env.reset(seed=seed)
    done=False; tot_r=0.0; tot_kl=0.0; steps=0
    while not done:
        st = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            act1,_,_ = a1.get_action(st, deterministic=deterministic)
            m1,ls1   = a1(st); m2,ls2 = a2(st)
            std1, std2 = ls1.exp(), ls2.exp()
            d1 = torch.distributions.Normal(m1,std1)
            d2 = torch.distributions.Normal(m2,std2)
            kl = torch.distributions.kl.kl_divergence(d1,d2).mean().clamp(min=0).item()
            tot_kl += kl
        obs, r, t1, t2, _ = env.step(act1.detach().cpu().numpy().squeeze(0))
        tot_r += r; steps+=1; done = t1 or t2
    return tot_r, tot_kl/max(1,steps)

def evaluate_agent_episode_meow(env, a1, a2, device, seed=0, num_samples=10):
    obs,_ = env.reset(seed=seed)
    done=False; tot_r=0.0; tot_kl=0.0; steps=0
    while not done:
        st = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            ctx = st.repeat(num_samples,1)
            acts, lp1 = a1.flow_policy.sample(num_samples, ctx, deterministic=False)
            lp2 = a2.flow_policy.log_prob(ctx, acts)
            kl = (lp1-lp2).mean().clamp(min=0).item()
            tot_kl += kl
            act1,_,_ = a1.get_action(st, deterministic=False)
        obs, r, t1, t2, _ = env.step(act1.cpu().numpy().squeeze(0))
        tot_r += r; steps+=1; done=t1 or t2
    return tot_r, tot_kl/max(1,steps)

# ─── Q-difference (mean per step) ───────────────────────────────────────
def evaluate_q_output_diff_episode(env, actor_1,
                                   qf1_i, qf2_i, qf1_j, qf2_j,
                                   device, seed=0, deterministic=False):
    obs,_ = env.reset(seed=seed)
    done=False; total=0.0; steps=0
    while not done:
        st = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            a1,_,_ = actor_1.get_action(st, deterministic=deterministic)
            qi = torch.max(qf1_i(st,a1), qf2_i(st,a1))
            qj = torch.max(qf1_j(st,a1), qf2_j(st,a1))
            total += (qi-qj).abs().item()
        obs, _, t1, t2, _ = env.step(a1.detach().cpu().numpy().squeeze(0))
        done = t1 or t2; steps+=1
    return total/max(1,steps)

# ─── Environment factories ─────────────────────────────────────────────
def make_env_sac(env_id, seed, idx, capture_video, run_name):
    e = gym.make(env_id)
    e = gym.wrappers.RecordEpisodeStatistics(e)
    if capture_video and idx==0:
        e = gym.wrappers.RecordVideo(e, f"videos/{run_name}", episode_trigger=lambda _:True)
    e.action_space.seed(seed)
    return e

def make_env_meow(env_id, seed, idx, capture_video, run_name):
    e = gym.make(env_id)
    e = gym.wrappers.RecordEpisodeStatistics(e)
    if capture_video and idx==0:
        e = gym.wrappers.RecordVideo(e, f"videos/{run_name}", episode_trigger=lambda _:True)
    e.action_space.seed(seed)
    return gym.wrappers.RescaleAction(e, -1.0, 1.0)

def create_single_env_for_eval(run, seed, idx, capture_video, run_name):
    return (make_env_meow if "meow_continuous_action.py" in run["algorithm"]
            else make_env_sac)(run["env"], seed, idx, capture_video, run_name)

def create_single_env_for_q(run, env_id, seed, capture_video, run_name, idx=999):
    return (make_env_meow if "meow_continuous_action.py" in run["algorithm"]
            else make_env_sac)(env_id, seed, idx, capture_video, run_name)

def create_single_env_for_jac(run, env_id, seed, capture_video, run_name, idx=998):
    if "meow_continuous_action.py" in run["algorithm"]:
        return make_env_meow(env_id, seed, idx, capture_video, run_name)
    else:
        return make_env_sac(env_id, seed, idx, capture_video, run_name)
    
    

# ─── Run discovery & actor loading (unchanged) ─────────────────────────
def gather_valid_runs(root_dir, target_env, checkpoint_step, training_seeds, algorithms):
    """
    Find runs in root_dir whose config.yaml matches target_env/seed/algorithms
    and which have exactly one checkpoint file for step=checkpoint_step
    containing actor + qf1 + qf2 state_dicts.
    """
    valid_runs = []

    for run_name in os.listdir(root_dir):
        run_root = os.path.join(root_dir, run_name)
        if not os.path.isdir(run_root):
            continue

        # 1) locate config.yaml
        config_path = None
        for cand in (run_root,
                     os.path.join(run_root, "files"),
                     os.path.join(run_root, "files", "files")):
            p = os.path.join(cand, "config.yaml")
            if os.path.isfile(p):
                config_path = p
                break
        if config_path is None:
            continue

        # 2) load config
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        # 3) extract env, seed, alpha
        env_val = cfg.get("env_id", cfg.get("env_id", None))
        if isinstance(env_val, dict):
            env_val = env_val.get("value")
        seed_val = cfg.get("seed", cfg.get("seed", None))
        if isinstance(seed_val, dict):
            seed_val = seed_val.get("value")
        alpha_val = cfg.get("alpha", cfg.get("alpha", None))
        if isinstance(alpha_val, dict):
            alpha_val = alpha_val.get("value")

        # 4) filter by env & seed
        if env_val != target_env or seed_val not in training_seeds:
            continue

        # 5) filter by algorithm
        wb = cfg.get("_wandb", {}).get("value", {}) or {}
        algo_val = wb.get("code_path") or cfg.get("algorithm") or cfg.get("algo")
        if not algo_val or not any(sub in algo_val for sub in algorithms):
            continue

        # 6) find the single checkpoint file
        files_dir = os.path.dirname(config_path)
        cand = os.path.join(files_dir, "files")
        if os.path.isdir(cand):
            files_dir = cand

        patterns = [
            os.path.join(files_dir, f"*_{checkpoint_step}.pth"),
            os.path.join(files_dir, f"*step{checkpoint_step}.pth"),
        ]
        matches = []
        for pat in patterns:
            matches.extend(glob.glob(pat))
        matches = list(set(matches))
        if len(matches) != 1:
            logger.warning(
                f"Run {run_root} has {len(matches)} matches for step {checkpoint_step}, skipping."
            )
            continue

        ckpt_file = matches[0]
        valid_runs.append({
            "env":         env_val,
            "alpha":       alpha_val,
            "seed":        seed_val,
            "path":        run_root,
            # point all three at the same checkpoint
            "actor_path":  ckpt_file,
            "qf1_path":    ckpt_file,
            "qf2_path":    ckpt_file,
            "algorithm":   algo_val,
        })

    logger.info(f"Found {len(valid_runs)} valid runs for {target_env} @ step {checkpoint_step}")
    return valid_runs


def load_actor(run, device, cfg):
    """
    Load just the actor from run["actor_path"].
    Assumes that checkpoint contains an "actor_state_dict" or equivalent.
    """
    ckpt = torch.load(run["actor_path"], map_location=device)

    # extract actor weights
    if "actor_state_dict" in ckpt:
        state = ckpt["actor_state_dict"]
    elif "policy_state_dict" in ckpt:
        state = ckpt["policy_state_dict"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        # fallback: assume all top‐level tensors belong to actor
        state = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}

    # build a dummy env to get shapes
    if "meow_continuous_action.py" in run["algorithm"]:
        env_for_actor = make_env_meow(run["env"], cfg.seed, 0, cfg.capture_video, cfg.run_name)
    else:
        env_for_actor = make_env_sac(run["env"], cfg.seed, 0, cfg.capture_video, cfg.run_name)

    # instantiate the correct class
    if "meow_continuous_action.py" in run["algorithm"]:
        state = {f"flow_policy.{k}": v for k, v in state.items()}
        actor = FlowActor(
            env_for_actor,
            alpha=run["alpha"],
            sigma_max=cfg.sigma_max,
            sigma_min=cfg.sigma_min,
            device=device
        )
    else:
        actor = Actor(env_for_actor)
        # if it was a TD3 checkpoint you may need to rename fc_mu → fc_mean
        if run["algorithm"].endswith("td3_continuous_action.py"):
            if "fc_mu.weight" in state:
                state["fc_mean.weight"] = state.pop("fc_mu.weight")
                state["fc_mean.bias"]   = state.pop("fc_mu.bias")
        # ensure logstd fields exist
        state.setdefault("fc_logstd.weight", torch.zeros_like(actor.fc_logstd.weight))
        state.setdefault("fc_logstd.bias",   torch.zeros_like(actor.fc_logstd.bias))

    actor.load_state_dict(state, strict=False)
    actor.to(device)
    env_for_actor.close()
    return actor


def get_algo_priority(algo):
    """
    Sort order: meow -> td3 -> sac -> others
    """
    if "meow_continuous_action.py" in algo:
        return 0
    if "td3_continuous_action.py" in algo:
        return 1
    if "sac_continuous_action.py" in algo:
        return 2
    return 3

HEADER = (
    "env,algorithm,"
    "actor_i_alpha,actor_i_seed,checkpoint_i,"
    "actor_j_alpha,actor_j_seed,checkpoint_j,"
    "episode_idx,episode_reward,episode_kl,episode_q_infnorm,episode_jacobian_diff\n"
)

def _maybe_write_header(path):
    if not os.path.isfile(path):
        with open(path,"w") as f: f.write(HEADER)

def _load_existing_keys(path):
    keys=set()
    if os.path.isfile(path):
        with open(path,"r") as f:
            for r in csv.DictReader(f):
                keys.add((
                    r["algorithm"],
                    r["actor_i_alpha"], r["actor_i_seed"], r["checkpoint_i"],
                    r["actor_j_alpha"], r["actor_j_seed"], r["checkpoint_j"],
                    r["episode_idx"],
                ))
    return keys

def evaluate_pair_episode(args):
    run_i, run_j, env_id, cfg_dict, episode_idx, base_key, output_file = args
    cfg = OmegaConf.create(cfg_dict)
    device = torch.device("cpu")

    # 1. Load actors
    actor_i = load_actor(run_i, device, cfg)
    actor_j = load_actor(run_j, device, cfg)
    actor_i.eval()
    actor_j.eval()

    # 2. Reward & KL
    eval_env = create_single_env_for_eval(
        run_i, cfg.seed + episode_idx, episode_idx, cfg.capture_video, cfg.run_name
    )
    if "meow_continuous_action.py" in run_i["algorithm"]:
        ep_reward, ep_kl = evaluate_agent_episode_meow(
            eval_env, actor_i, actor_j, device, cfg.seed + episode_idx
        )
    else:
        ep_reward, ep_kl = evaluate_agent_episode(
            eval_env, actor_i, actor_j, device, cfg.seed + episode_idx
        )
    eval_env.close()

    # 3. Q-difference
    if run_i["qf1_path"] and run_i["qf2_path"]:
        # Load separate Q-networks from checkpoint
        qf1_i = SoftQNetwork(eval_env).to(device)
        qf2_i = SoftQNetwork(eval_env).to(device)
        qf1_j = SoftQNetwork(eval_env).to(device)
        qf2_j = SoftQNetwork(eval_env).to(device)

        ckpt = torch.load(run_i["qf1_path"], map_location=device)
        qf1_i.load_state_dict(ckpt.get("qf1_state_dict", ckpt), strict=False)
        ckpt = torch.load(run_i["qf2_path"], map_location=device)
        qf2_i.load_state_dict(ckpt.get("qf2_state_dict", ckpt), strict=False)
        ckpt = torch.load(run_j["qf1_path"], map_location=device)
        qf1_j.load_state_dict(ckpt.get("qf1_state_dict", ckpt), strict=False)
        ckpt = torch.load(run_j["qf2_path"], map_location=device)
        qf2_j.load_state_dict(ckpt.get("qf2_state_dict", ckpt), strict=False)

        q_env = create_single_env_for_q(
            run_i, env_id, cfg.seed + episode_idx, cfg.capture_video, cfg.run_name
        )
        ep_qdiff = evaluate_q_output_diff_episode(
            q_env, actor_i, qf1_i, qf2_i, qf1_j, qf2_j,
            device, cfg.seed + episode_idx
        )
        q_env.close()
    else:
        # Use MEOW-style Q evaluation via flow_policy.get_qv
        q_env = create_single_env_for_q(
            run_i, env_id, cfg.seed + episode_idx, cfg.capture_video, cfg.run_name
        )
        ep_qdiff = evaluate_q_output_diff_episode_meow(
            q_env, actor_i, actor_j, device, cfg.seed + episode_idx
        )
        q_env.close()

    # 4. Jacobian diff
    jac_env = create_single_env_for_jac(
        run_i, env_id, cfg.seed + episode_idx, cfg.capture_video, cfg.run_name
    )
    ep_jacdiff = compute_jacobian_diff_episode(
        jac_env, actor_i, actor_j, device, cfg.seed + episode_idx
    )
    jac_env.close()

    # 5. Write result
    with open(output_file, "a", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            env_id,
            run_i["algorithm"],
            run_i["alpha"],
            run_i["seed"],
            run_i["actor_path"],
            run_j["alpha"],
            run_j["seed"],
            run_j["actor_path"],
            episode_idx,
            ep_reward,
            ep_kl,
            ep_qdiff,
            ep_jacdiff
        ])

    return base_key + (str(episode_idx),)

@hydra.main(config_path="configs", config_name="inference", version_base="1.1")
def main(cfg: DictConfig):
    root = to_absolute_path(cfg.root_dir)
    out_dir = to_absolute_path(cfg.get("results_dir","results"))
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cpu")
    logger.info(f"Using device: {device}")
    logger.info("Config:\n" + OmegaConf.to_yaml(cfg))

    for env in cfg.envs:
        step = cfg.checkpoint_steps[env]
        runs = gather_valid_runs(root, env, step, cfg.training_seeds, cfg.algorithms)
        grp  = defaultdict(list)
        for r in runs:
            grp[(r["alpha"],r["algorithm"])].append(r)

        out_f = os.path.join(out_dir, f"{env}-pairwise_per_episode.csv")
        _maybe_write_header(out_f)
        seen = _load_existing_keys(out_f)

        jobs = []
        for (α,algo),g in sorted(grp.items(), key=lambda x:get_algo_priority(x[0][1])):
            if len(g)<2: continue
            for i,j in itertools.permutations(g,2):
                if i["seed"]==j["seed"]: continue
                key = (i["algorithm"],str(i["alpha"]),str(i["seed"]),i["actor_path"],
                       str(j["alpha"]),str(j["seed"]),j["actor_path"])
                for epi in range(cfg.n_eval_episodes):
                    rk = key+(str(epi),)
                    if rk in seen: continue
                    jobs.append((i,j,env,OmegaConf.to_container(cfg,resolve=True),
                                 epi, key, out_f))

        logger.info(f"Launching {len(jobs)} jobs for {env}")
        with mp.Pool(mp.cpu_count()) as pool:
            for res in tqdm(pool.imap_unordered(evaluate_pair_episode,jobs), total=len(jobs)):
                if res: seen.add(res)

if __name__=="__main__":
    main()
