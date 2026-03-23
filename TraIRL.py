"""
TraIRL training script with expert-only VAE reconstruction.

Compared with `firl__vae/irl_samples.py`, this script keeps the same overall
pipeline but enforces:
  - VAE reconstruction loss uses expert trajectories only.
  - Learner trajectories are used only in WGAN and reward learning branches.
"""

import sys
import os
import time
import json
import datetime
import dateutil.tz

import gym
import numpy as np
import torch
from ruamel.yaml import YAML

import envs  # noqa: F401
from utils import system, collect, logger
from common.sac import ReplayBuffer

from firl_roer.models.reward import MLPReward
from firl_roer.models.discrim import SMMIRLCritic as Critic

from firl__vae.irl_samples import (
    MultiHeadVAE,
    parse_source_envs,
    load_source_expert_data,
    build_sac_agent,
    maybe_resample_expert_trajs,
    evaluate_multi_source,
    set_requires_grad,
    vae_loss,
    critic_gradient_penalty,
    reward_covariance_loss,
)


if __name__ == "__main__":
    yaml = YAML()
    v = yaml.load(open(sys.argv[1]))

    source_env_names = parse_source_envs(v)
    primary_env_name = source_env_names[0]
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
    vae_hidden_size = trairl_cfg.get("hidden_size", 256)
    beta_kl = trairl_cfg.get("beta_kl", 0.1)
    lambda_vae = trairl_cfg.get("lambda_vae", 1.0)
    lambda_wgan = trairl_cfg.get("lambda_wgan", 1.0)
    lambda_f = trairl_cfg.get("lambda_f", 1.0)
    joint_updates = trairl_cfg.get("joint_updates", 1)
    critic_updates = trairl_cfg.get("critic_updates", v["critic"]["iter"])
    vae_lr = trairl_cfg.get("vae_lr", 3e-4)
    reward_anchor_lambda = v["reward"].get("anchor_lambda", 0.0)
    vae_expert_only = trairl_cfg.get("vae_expert_only", True)

    print(">>> [TraIRL Setup] Force setting obj to 'emd' (W1 Distance)")
    v["obj"] = "emd"
    assert v["IS"] is False

    env_tag = f"multi_{len(source_env_names)}src"
    exp_id = f"logs/{env_tag}/exp-{num_expert_trajs}/{v['obj']}-trairl/{seed}"
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

    source_specs = []
    state_size, action_size = None, None
    for task_id, env_name in enumerate(source_env_names):
        train_env = gym.make(env_name)
        eval_env = gym.make(env_name)
        current_state_size = train_env.observation_space.shape[0]
        current_action_size = train_env.action_space.shape[0]
        if state_size is None:
            state_size = current_state_size
            action_size = current_action_size
        assert current_state_size == state_size, "All source envs must share the same state dim."
        assert current_action_size == action_size, "All source envs must share the same action dim."
        source_specs.append(
            {
                "task_id": task_id,
                "env_name": env_name,
                "train_env": train_env,
                "eval_env": eval_env,
            }
        )

    if state_indices == "all":
        state_indices = list(range(state_size))

    source_expert_trajs = load_source_expert_data(source_env_names, num_expert_trajs, state_indices, seed)
    expert_samples_eval = np.concatenate(
        [expert_trajs.reshape(-1, len(state_indices)) for expert_trajs in source_expert_trajs], axis=0
    )

    vae = MultiHeadVAE(
        state_dim=len(state_indices),
        latent_dim=latent_dim,
        num_tasks=len(source_env_names),
        hidden_size=vae_hidden_size,
        device=device,
    )
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=vae_lr)

    reward_func = MLPReward(latent_dim, **v["reward"], device=device).to(device)
    reward_optimizer = torch.optim.Adam(
        reward_func.parameters(),
        lr=v["reward"]["lr"],
        weight_decay=v["reward"]["weight_decay"],
        betas=(v["reward"]["momentum"], 0.999),
    )

    critic = Critic(latent_dim, **v["critic"], device=device)
    critic_batch_size = min(v["critic"]["batch_size"], expert_samples_eval.shape[0])

    replay_buffer = ReplayBuffer(
        state_size,
        action_size,
        device=device,
        size=v["sac"]["buffer_size"],
    )
    sac_agent = build_sac_agent(primary_env_name, replay_buffer, v, state_indices, seed, device)

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

    sac_agent.reward_function = get_latent_reward

    max_real_return_det, max_real_return_sto = -np.inf, -np.inf
    n_itrs = v["irl"]["n_itrs"]
    print(f">>> [TraIRL Setup] Sources: {source_env_names}")
    print(
        f">>> [TraIRL Setup] latent_dim={latent_dim}, lambda_vae={lambda_vae}, "
        f"lambda_wgan={lambda_wgan}, lambda_f={lambda_f}, joint_updates={joint_updates}, "
        f"vae_expert_only={vae_expert_only}"
    )

    for itr in range(n_itrs):
        if itr > 0 and v["sac"]["reinitialize"]:
            replay_buffer = ReplayBuffer(
                state_size,
                action_size,
                device=device,
                size=v["sac"]["buffer_size"],
            )
            sac_agent = build_sac_agent(primary_env_name, replay_buffer, v, state_indices, seed, device)
            sac_agent.reward_function = get_latent_reward

        source_samples = []
        for spec in source_specs:
            sac_agent.env = spec["train_env"]
            sac_agent.test_env = spec["eval_env"]
            sac_agent.test_fn = sac_agent.test_agent_ori_env
            sac_agent.learn_mujoco(print_out=(spec["task_id"] == 0))
            samples = collect.collect_trajectories_policy_single(
                spec["train_env"],
                sac_agent,
                n=v["irl"]["training_trajs"],
                state_indices=state_indices,
            )
            source_samples.append(samples)

        start = time.time()
        for _ in range(critic_updates):
            critic.optimizer.zero_grad()
            critic_total_loss = torch.tensor(0.0, device=device)
            for task_id in range(len(source_specs)):
                expert_flat = torch.as_tensor(
                    source_expert_trajs[task_id].reshape(-1, len(state_indices)),
                    dtype=torch.float32,
                    device=device,
                )
                agent_flat = torch.as_tensor(
                    source_samples[task_id][0].reshape(-1, len(state_indices)),
                    dtype=torch.float32,
                    device=device,
                )
                batch_size = min(critic_batch_size, expert_flat.shape[0], agent_flat.shape[0])
                expert_idx = torch.randint(expert_flat.shape[0], (batch_size,), device=device)
                agent_idx = torch.randint(agent_flat.shape[0], (batch_size,), device=device)
                with torch.no_grad():
                    expert_z = vae.encode_mean(expert_flat[expert_idx])
                    agent_z = vae.encode_mean(agent_flat[agent_idx])
                main_loss = critic.model(agent_z).mean() - critic.model(expert_z).mean()
                gp = critic.lam * critic_gradient_penalty(critic.model, expert_z, agent_z)
                critic_total_loss = critic_total_loss + main_loss + gp
            critic_total_loss = critic_total_loss / len(source_specs)
            critic_total_loss.backward()
            critic.optimizer.step()

        set_requires_grad(critic.model, False)
        joint_total_loss = torch.tensor(0.0, device=device)
        vae_loss_value, wgan_loss_value, reward_loss_value = 0.0, 0.0, 0.0

        for _ in range(joint_updates):
            # Stage A: VAE + WGAN for abstraction
            set_requires_grad(vae, True)
            vae_optimizer.zero_grad()
            total_vae_loss = torch.tensor(0.0, device=device)
            total_wgan_loss = torch.tensor(0.0, device=device)

            for task_id in range(len(source_specs)):
                expert_trajs_t = torch.as_tensor(source_expert_trajs[task_id], dtype=torch.float32, device=device)
                agent_trajs_t = torch.as_tensor(source_samples[task_id][0], dtype=torch.float32, device=device)
                expert_flat_t = expert_trajs_t.reshape(-1, len(state_indices))
                agent_flat_t = agent_trajs_t.reshape(-1, len(state_indices))

                # Paper-consistent variant: reconstruction from expert trajectories only.
                if vae_expert_only:
                    vae_input = expert_flat_t
                else:
                    vae_input = torch.cat([expert_flat_t, agent_flat_t], dim=0)
                recon_x, mu, logvar, _ = vae(vae_input, task_id, sample=True)
                source_vae_loss, _, _ = vae_loss(recon_x, vae_input, mu, logvar, beta_kl)
                total_vae_loss = total_vae_loss + source_vae_loss

                expert_z_flat = vae.encode_mean(expert_flat_t)
                agent_z_flat = vae.encode_mean(agent_flat_t)
                total_wgan_loss = total_wgan_loss + (
                    critic.model(agent_z_flat).mean() - critic.model(expert_z_flat).mean()
                )

            total_vae_loss = total_vae_loss / len(source_specs)
            total_wgan_loss = total_wgan_loss / len(source_specs)
            vae_wgan_loss = lambda_vae * total_vae_loss + lambda_wgan * total_wgan_loss
            vae_wgan_loss.backward()
            vae_optimizer.step()

            # Stage B: freeze VAE and update reward only.
            set_requires_grad(vae, False)
            reward_optimizer.zero_grad()
            old_reward_params = {name: p.detach().clone() for name, p in reward_func.named_parameters()}
            total_reward_loss = torch.tensor(0.0, device=device)

            for task_id in range(len(source_specs)):
                expert_trajs_t = torch.as_tensor(source_expert_trajs[task_id], dtype=torch.float32, device=device)
                agent_trajs_t = torch.as_tensor(source_samples[task_id][0], dtype=torch.float32, device=device)
                expert_flat_t = expert_trajs_t.reshape(-1, len(state_indices))
                agent_flat_t = agent_trajs_t.reshape(-1, len(state_indices))

                with torch.no_grad():
                    expert_z_flat = vae.encode_mean(expert_flat_t)
                    agent_z_flat = vae.encode_mean(agent_flat_t)

                expert_z_trajs = expert_z_flat.view(expert_trajs_t.shape[0], expert_trajs_t.shape[1], latent_dim)
                agent_z_trajs = agent_z_flat.view(agent_trajs_t.shape[0], agent_trajs_t.shape[1], latent_dim)
                expert_z_batch = maybe_resample_expert_trajs(expert_z_trajs, v["irl"]["resample_episodes"])
                total_reward_loss = total_reward_loss + reward_covariance_loss(
                    agent_z_trajs, expert_z_batch, critic.model, reward_func
                )

            total_reward_loss = total_reward_loss / len(source_specs)
            reward_objective = lambda_f * total_reward_loss
            if reward_anchor_lambda > 0:
                anchor_loss = sum(
                    ((param - old_reward_params[name]) ** 2).sum()
                    for name, param in reward_func.named_parameters()
                )
                reward_objective = reward_objective + reward_anchor_lambda * anchor_loss
            reward_objective.backward()
            reward_optimizer.step()

            joint_total_loss = vae_wgan_loss.detach() + reward_objective.detach()
            vae_loss_value = total_vae_loss.item()
            wgan_loss_value = total_wgan_loss.item()
            reward_loss_value = total_reward_loss.item()

        set_requires_grad(vae, True)
        set_requires_grad(critic.model, True)
        print(f"train critic/joint {time.time() - start:.0f}s", flush=True)

        real_return_det, real_return_sto = evaluate_multi_source(
            itr, source_specs, source_samples, expert_samples_eval, sac_agent, v
        )

        if real_return_det > max_real_return_det and real_return_sto > max_real_return_sto:
            max_real_return_det, max_real_return_sto = real_return_det, real_return_sto
            save_dir = os.path.join(
                logger.get_dir(), "model", f"best_itr{itr}_det{max_real_return_det:.0f}_sto{max_real_return_sto:.0f}"
            )
            os.makedirs(save_dir, exist_ok=True)
            torch.save(reward_func.state_dict(), os.path.join(save_dir, "reward_func.pkl"))
            torch.save(vae.state_dict(), os.path.join(save_dir, "vae.pkl"))
            torch.save(sac_agent.ac.state_dict(), os.path.join(save_dir, "policy.pkl"))
            torch.save(reward_func.state_dict(), os.path.join(logger.get_dir(), "model", "best_reward.pkl"))
            torch.save(vae.state_dict(), os.path.join(logger.get_dir(), "model", "best_vae.pkl"))
            torch.save(sac_agent.ac.state_dict(), os.path.join(logger.get_dir(), "model", "best_policy.pkl"))
            print(f">>> [TraIRL Save] Saved best models to: {save_dir}")

        logger.record_tabular("Joint Total Loss", round(joint_total_loss.item(), 4))
        logger.record_tabular("VAE Loss", round(vae_loss_value, 4))
        logger.record_tabular("WGAN Loss", round(wgan_loss_value, 4))
        logger.record_tabular("Reward Loss", round(reward_loss_value, 4))
        if v["sac"]["automatic_alpha_tuning"]:
            logger.record_tabular("alpha", sac_agent.alpha.item())
        logger.dump_tabular()
