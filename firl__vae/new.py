"""
new.py — TraIRL implementation aligned with Algorithm 1 in the paper.

Key differences from TraIRL.py / irl_samples.py:
1. Trajectory Buffer per source task: each iteration's learner trajectories are
   added to a per-task buffer; reward/VAE/critic updates sample uniformly from
   the full buffer (not just the current iteration).
2. Independent SAC agent per source task (paper: πξ1, ..., πξn).
3. VAE reconstruction uses expert trajectories only.
4. Hyperparameters aligned with paper Table 5:
   - reward net: [16, 16], lr=3e-4
   - VAE encoder: [32,32,32]+Tanh, decoder: [64,64,64], lr=3e-4
   - disc: [32, 32], update steps=10
   - reward update steps=10, VAE update steps=10
"""

import sys
import os
import time
import json
import datetime
import dateutil.tz
import collections

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ruamel.yaml import YAML

import envs  # noqa: F401
from utils import system, collect, logger
from common.sac import ReplayBuffer, SAC

from firl.models.reward import MLPReward
from firl.models.discrim import SMMIRLCritic as Critic

from firl__vae.irl_samples import (
    set_requires_grad,
    vae_loss,
    reward_covariance_loss,
    parse_source_envs,
    load_source_expert_data,
    maybe_resample_expert_trajs,
    evaluate_multi_source,
)


# ---------------------------------------------------------------------------
# VAE with paper-aligned architecture: encoder [32,32,32]+Tanh, decoder [64,64,64]
# ---------------------------------------------------------------------------

class TaskDecoder(nn.Module):
    def __init__(self, latent_dim, state_dim, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, state_dim),
        )

    def forward(self, z):
        return self.net(z)


class MultiHeadVAE(nn.Module):
    """Shared encoder + task-specific decoders. Paper arch: enc=[32,32,32]+Tanh, dec=[64,64,64]."""

    def __init__(self, state_dim, latent_dim, num_tasks, enc_hidden=32, dec_hidden=64, device="cpu"):
        super().__init__()
        self.device = device
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, enc_hidden), nn.Tanh(),
            nn.Linear(enc_hidden, enc_hidden), nn.Tanh(),
            nn.Linear(enc_hidden, enc_hidden), nn.Tanh(),
        )
        self.fc_mu = nn.Linear(enc_hidden, latent_dim)
        self.fc_logvar = nn.Linear(enc_hidden, latent_dim)
        self.decoders = nn.ModuleList(
            [TaskDecoder(latent_dim, state_dim, dec_hidden) for _ in range(num_tasks)]
        )
        self.to(device)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x, task_id, sample=True):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if sample else mu
        return self.decoders[task_id](z), mu, logvar, z

    def encode_mean(self, x):
        mu, _ = self.encode(x)
        return mu

    def get_z(self, x):
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            return self.encode_mean(x)


# ---------------------------------------------------------------------------
# Per-task trajectory buffer (stores flat state arrays)
# ---------------------------------------------------------------------------

class TrajectoryBuffer:
    """Fixed-size FIFO buffer storing learner trajectory arrays (shape: [T, state_dim])."""

    def __init__(self, max_trajs=200):
        self.max_trajs = max_trajs
        self._buf = collections.deque(maxlen=max_trajs)

    def add(self, trajs: np.ndarray):
        """trajs: np.ndarray of shape [n_trajs, horizon, state_dim]"""
        for i in range(trajs.shape[0]):
            self._buf.append(trajs[i])  # each element: [horizon, state_dim]

    def sample_flat(self, n_trajs=None) -> np.ndarray:
        """Return flat states sampled uniformly from buffer. Shape: [N, state_dim]"""
        buf = list(self._buf)
        if n_trajs is not None and n_trajs < len(buf):
            idx = np.random.choice(len(buf), n_trajs, replace=False)
            buf = [buf[i] for i in idx]
        return np.concatenate(buf, axis=0)  # [N*horizon, state_dim]

    def sample_trajs(self, n_trajs=None) -> np.ndarray:
        """Return trajectory array sampled uniformly. Shape: [n, horizon, state_dim]"""
        buf = list(self._buf)
        if n_trajs is not None and n_trajs < len(buf):
            idx = np.random.choice(len(buf), n_trajs, replace=False)
            buf = [buf[i] for i in idx]
        return np.stack(buf, axis=0)

    def __len__(self):
        return len(self._buf)


# ---------------------------------------------------------------------------
# Helper: build one SAC agent per source task
# ---------------------------------------------------------------------------

def build_sac_agent(env_name, replay_buffer, config, state_indices, seed, device):
    env_fn = lambda: gym.make(env_name)
    return SAC(
        env_fn,
        replay_buffer,
        steps_per_epoch=config["env"]["T"],
        update_after=config["env"]["T"] * config["sac"]["random_explore_episodes"],
        max_ep_len=config["env"]["T"],
        seed=seed,
        start_steps=config["env"]["T"] * config["sac"]["random_explore_episodes"],
        reward_state_indices=state_indices,
        device=device,
        **config["sac"],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    yaml = YAML()
    v = yaml.load(open(sys.argv[1]))

    source_env_names = parse_source_envs(v)
    state_indices = v["env"]["state_indices"]
    seed = v["seed"]
    num_expert_trajs = v["irl"]["expert_episodes"]

    device = torch.device(
        f"cuda:{v['cuda']}" if torch.cuda.is_available() and v["cuda"] >= 0 else "cpu"
    )
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(seed)
    pid = os.getpid()

    trairl_cfg = v.get("trairl", {})
    latent_dim = trairl_cfg.get("latent_dim", 16)
    beta_kl = trairl_cfg.get("beta_kl", 0.1)
    lambda_vae = trairl_cfg.get("lambda_vae", 1.0)
    lambda_wgan = trairl_cfg.get("lambda_wgan", 1.0)
    lambda_f = trairl_cfg.get("lambda_f", 1.0)
    # Paper Table 5: update steps = 10 for reward, VAE, disc
    reward_update_steps = trairl_cfg.get("reward_update_steps", 10)
    vae_update_steps = trairl_cfg.get("vae_update_steps", 10)
    critic_update_steps = trairl_cfg.get("critic_update_steps", 10)
    buffer_max_trajs = trairl_cfg.get("buffer_max_trajs", 200)
    reward_anchor_lambda = v["reward"].get("anchor_lambda", 0.0)

    v["obj"] = "emd"
    assert v["IS"] is False

    env_tag = f"multi_{len(source_env_names)}src"
    exp_id = f"new_logs/{env_tag}/exp-{num_expert_trajs}/emd-trairl-new/{seed}"
    os.makedirs(exp_id, exist_ok=True)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    log_folder = exp_id + "/" + now.strftime("%Y_%m_%d_%H_%M_%S")
    logger.configure(dir=log_folder)
    print(f"Logging to directory: {log_folder}")
    os.system(f"cp {sys.argv[0]} {log_folder}")
    os.system(f"cp {sys.argv[1]} {log_folder}/variant_{pid}.yml")
    with open(os.path.join(logger.get_dir(), "variant.json"), "w") as f:
        json.dump(v, f, indent=2, sort_keys=True)
    os.makedirs(os.path.join(log_folder, "model"), exist_ok=True)

    # Build envs
    source_specs = []
    state_size, action_size = None, None
    for task_id, env_name in enumerate(source_env_names):
        train_env = gym.make(env_name)
        eval_env = gym.make(env_name)
        s = train_env.observation_space.shape[0]
        a = train_env.action_space.shape[0]
        if state_size is None:
            state_size, action_size = s, a
        assert s == state_size and a == action_size
        source_specs.append({"task_id": task_id, "env_name": env_name,
                              "train_env": train_env, "eval_env": eval_env})

    if state_indices == "all":
        state_indices = list(range(state_size))
    obs_dim = len(state_indices)

    # Expert data
    source_expert_trajs = load_source_expert_data(source_env_names, num_expert_trajs, state_indices, seed)
    expert_samples_eval = np.concatenate(
        [t.reshape(-1, obs_dim) for t in source_expert_trajs], axis=0
    )

    # Models — paper-aligned architectures
    vae = MultiHeadVAE(
        state_dim=obs_dim,
        latent_dim=latent_dim,
        num_tasks=len(source_env_names),
        enc_hidden=trairl_cfg.get("enc_hidden", 32),
        dec_hidden=trairl_cfg.get("dec_hidden", 64),
        device=device,
    )
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=trairl_cfg.get("vae_lr", 3e-4))

    # reward net: paper Table 5 uses [16,16]; set via config reward.hidden_sizes
    reward_func = MLPReward(latent_dim, **v["reward"], device=device).to(device)
    reward_optimizer = torch.optim.Adam(
        reward_func.parameters(),
        lr=v["reward"]["lr"],
        weight_decay=v["reward"]["weight_decay"],
        betas=(v["reward"]["momentum"], 0.999),
    )

    # disc: paper Table 5 uses [32,32]; set via config critic.hid_dim
    critic = Critic(latent_dim, **v["critic"], device=device)

    # Per-task trajectory buffers (Algorithm 1 lines 6-7)
    traj_buffers = [TrajectoryBuffer(max_trajs=buffer_max_trajs) for _ in source_env_names]
    # Pre-fill buffers with expert trajectories (Algorithm 1 line 2)
    for task_id, expert_trajs in enumerate(source_expert_trajs):
        traj_buffers[task_id].add(expert_trajs)

    # Independent SAC agent per source task (paper: πξ1, ..., πξn)
    sac_agents = []
    for spec in source_specs:
        rb = ReplayBuffer(state_size, action_size, device=device, size=v["sac"]["buffer_size"])
        agent = build_sac_agent(spec["env_name"], rb, v, state_indices, seed, device)
        sac_agents.append(agent)

    def make_reward_fn(agent_idx):
        def get_latent_reward(obs):
            with torch.no_grad():
                if not torch.is_tensor(obs):
                    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
                else:
                    obs = obs.to(device)
                if obs.dim() == 1:
                    obs = obs.unsqueeze(0)
                z = vae.get_z(obs)
                return reward_func(z).cpu().numpy().flatten()
        return get_latent_reward

    for i, agent in enumerate(sac_agents):
        agent.reward_function = make_reward_fn(i)

    max_real_return_det, max_real_return_sto = -np.inf, -np.inf
    n_itrs = v["irl"]["n_itrs"]
    print(f">>> [new.py] Sources: {source_env_names}")
    print(f">>> [new.py] latent_dim={latent_dim}, reward_steps={reward_update_steps}, "
          f"vae_steps={vae_update_steps}, critic_steps={critic_update_steps}")

    for itr in range(n_itrs):
        # Algorithm 1 lines 4-6: collect trajectories from each source task
        source_samples = []
        for task_id, (spec, agent) in enumerate(zip(source_specs, sac_agents)):
            agent.env = spec["train_env"]
            agent.test_env = spec["eval_env"]
            agent.test_fn = agent.test_agent_ori_env
            agent.learn_mujoco(print_out=True)
            samples = collect.collect_trajectories_policy_single(
                spec["train_env"], agent,
                n=v["irl"]["training_trajs"],
                state_indices=state_indices,
            )
            source_samples.append(samples)
            # Add to buffer (Algorithm 1 line 6)
            traj_buffers[task_id].add(samples[0])  # samples[0]: [n_trajs, horizon, obs_dim]

        start = time.time()

        # Algorithm 1 line 7: uniformly sample from buffer for updates
        # Critic update (WGAN-GP, paper: 10 steps)
        for task_id in range(len(source_specs)):
            expert_flat = source_expert_trajs[task_id].reshape(-1, obs_dim)
            # Sample from buffer (includes expert + all past learner trajs)
            agent_flat = traj_buffers[task_id].sample_flat()
            with torch.no_grad():
                expert_z = vae.encode_mean(
                    torch.as_tensor(expert_flat, dtype=torch.float32, device=device)
                ).cpu().numpy()
                agent_z = vae.encode_mean(
                    torch.as_tensor(agent_flat, dtype=torch.float32, device=device)
                ).cpu().numpy()
            critic.learn(expert_z, agent_z, iter=critic_update_steps)

        set_requires_grad(critic.model, False)

        # VAE update (paper: 10 steps)
        set_requires_grad(vae, True)
        vae_loss_value, wgan_loss_value = 0.0, 0.0
        for _ in range(vae_update_steps):
            vae_optimizer.zero_grad()
            total_vae_loss = torch.tensor(0.0, device=device)
            total_wgan_loss = torch.tensor(0.0, device=device)

            for task_id in range(len(source_specs)):
                expert_flat_t = torch.as_tensor(
                    source_expert_trajs[task_id].reshape(-1, obs_dim),
                    dtype=torch.float32, device=device,
                )
                # WGAN branch: sample from buffer (historical learner states)
                agent_flat_np = traj_buffers[task_id].sample_flat()
                agent_flat_t = torch.as_tensor(agent_flat_np, dtype=torch.float32, device=device)

                # VAE reconstruction: expert only (paper consistent)
                recon_x, mu, logvar, _ = vae(expert_flat_t, task_id, sample=True)
                src_vae_loss, _, _ = vae_loss(recon_x, expert_flat_t, mu, logvar, beta_kl)
                total_vae_loss = total_vae_loss + src_vae_loss

                expert_z_flat = vae.encode_mean(expert_flat_t)
                agent_z_flat = vae.encode_mean(agent_flat_t)
                total_wgan_loss = total_wgan_loss + (
                    critic.model(agent_z_flat).mean() - critic.model(expert_z_flat).mean()
                )

            total_vae_loss = total_vae_loss / len(source_specs)
            total_wgan_loss = total_wgan_loss / len(source_specs)
            (lambda_vae * total_vae_loss + lambda_wgan * total_wgan_loss).backward()
            vae_optimizer.step()
            vae_loss_value = total_vae_loss.item()
            wgan_loss_value = total_wgan_loss.item()

        # Reward update (paper: 10 steps)
        set_requires_grad(vae, False)
        reward_loss_value = 0.0
        for _ in range(reward_update_steps):
            reward_optimizer.zero_grad()
            total_reward_loss = torch.tensor(0.0, device=device)

            for task_id in range(len(source_specs)):
                expert_trajs_np = source_expert_trajs[task_id]  # [n, T, obs_dim]
                # Sample learner trajectories from buffer
                agent_trajs_np = traj_buffers[task_id].sample_trajs(
                    n_trajs=v["irl"]["training_trajs"]
                )  # [n, T, obs_dim]

                with torch.no_grad():
                    expert_flat_t = torch.as_tensor(
                        expert_trajs_np.reshape(-1, obs_dim), dtype=torch.float32, device=device
                    )
                    agent_flat_t = torch.as_tensor(
                        agent_trajs_np.reshape(-1, obs_dim), dtype=torch.float32, device=device
                    )
                    expert_z_flat = vae.encode_mean(expert_flat_t)
                    agent_z_flat = vae.encode_mean(agent_flat_t)

                horizon = expert_trajs_np.shape[1]
                expert_z_trajs = expert_z_flat.view(expert_trajs_np.shape[0], horizon, latent_dim)
                agent_z_trajs = agent_z_flat.view(agent_trajs_np.shape[0], horizon, latent_dim)
                expert_z_batch = maybe_resample_expert_trajs(expert_z_trajs, v["irl"]["resample_episodes"])
                total_reward_loss = total_reward_loss + reward_covariance_loss(
                    agent_z_trajs, expert_z_batch, critic.model, reward_func
                )

            total_reward_loss = total_reward_loss / len(source_specs)
            reward_objective = lambda_f * total_reward_loss
            if reward_anchor_lambda > 0:
                anchor_loss = sum(
                    (p ** 2).sum() for p in reward_func.parameters()
                )
                reward_objective = reward_objective + reward_anchor_lambda * anchor_loss
            reward_objective.backward()
            torch.nn.utils.clip_grad_norm_(reward_func.parameters(), max_norm=1.0)
            reward_optimizer.step()
            reward_loss_value = total_reward_loss.item()

        set_requires_grad(vae, True)
        set_requires_grad(critic.model, True)
        print(f"train critic/joint {time.time() - start:.0f}s", flush=True)

        # Use first agent for evaluation (consistent with original code)
        real_return_det, real_return_sto = evaluate_multi_source(
            itr, source_specs, source_samples, expert_samples_eval, sac_agents[0], v
        )

        if real_return_det > max_real_return_det and real_return_sto > max_real_return_sto:
            max_real_return_det, max_real_return_sto = real_return_det, real_return_sto
            save_dir = os.path.join(
                logger.get_dir(), "model",
                f"best_itr{itr}_det{max_real_return_det:.0f}_sto{max_real_return_sto:.0f}",
            )
            os.makedirs(save_dir, exist_ok=True)
            torch.save(reward_func.state_dict(), os.path.join(save_dir, "reward_func.pkl"))
            torch.save(vae.state_dict(), os.path.join(save_dir, "vae.pkl"))
            torch.save(reward_func.state_dict(), os.path.join(logger.get_dir(), "model", "best_reward.pkl"))
            torch.save(vae.state_dict(), os.path.join(logger.get_dir(), "model", "best_vae.pkl"))
            print(f">>> [new.py Save] Saved best models to: {save_dir}")

        logger.record_tabular("VAE Loss", round(vae_loss_value, 4))
        logger.record_tabular("WGAN Loss", round(wgan_loss_value, 4))
        logger.record_tabular("Reward Loss", round(reward_loss_value, 4))
        logger.record_tabular("Buffer Size", sum(len(b) for b in traj_buffers))
        if v["sac"]["automatic_alpha_tuning"]:
            logger.record_tabular("alpha", sac_agents[0].alpha.item())
        logger.dump_tabular()
