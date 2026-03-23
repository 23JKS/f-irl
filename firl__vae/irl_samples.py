'''
TraIRL-style f-IRL variant with:
1. multi-source task training
2. multi-head VAE (shared encoder + task-specific decoders)
3. joint losses where encoder is updated by VAE, WGAN, and reward losses
'''
import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from ruamel.yaml import YAML

from firl.models.reward import MLPReward
from firl.models.discrim import SMMIRLCritic as Critic

import envs
from utils import system, collect, logger, eval
from common.sac import ReplayBuffer, SAC

import datetime
import dateutil.tz
import json


class TaskDecoder(nn.Module):
    def __init__(self, latent_dim, state_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_dim),
        )

    def forward(self, z):
        return self.net(z)


class MultiHeadVAE(nn.Module):
    def __init__(self, state_dim, latent_dim, num_tasks, hidden_size=256, device='cpu'):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.num_tasks = num_tasks

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)
        self.decoders = nn.ModuleList(
            [TaskDecoder(latent_dim, state_dim, hidden_size) for _ in range(num_tasks)]
        )
        self.to(device)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, task_id, sample=True):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if sample else mu
        recon_x = self.decoders[task_id](z)
        return recon_x, mu, logvar, z

    def encode_mean(self, x):
        mu, _ = self.encode(x)
        return mu

    def get_z(self, x):
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            return self.encode_mean(x)


def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad = flag


def vae_loss(recon_x, x, mu, logvar, beta_kl):
    recon = F.mse_loss(recon_x, x, reduction='mean')
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return recon + beta_kl * kl, recon, kl


def critic_gradient_penalty(critic_model, expert_z, agent_z):
    batch_size = expert_z.shape[0]
    eps = torch.rand((batch_size, 1), device=expert_z.device)
    interp = eps * expert_z + (1 - eps) * agent_z
    interp = interp.detach()
    interp.requires_grad_(True)
    critic_out = critic_model(interp).sum()
    gradients = torch.autograd.grad(
        outputs=critic_out,
        inputs=interp,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def reward_covariance_loss(agent_z_trajs, expert_z_trajs, critic_model, reward_func):
    z_trajs = torch.cat([agent_z_trajs, expert_z_trajs], dim=0)
    num_trajs, horizon, latent_dim = z_trajs.shape
    z_flat = z_trajs.reshape(-1, latent_dim)
    critic_terms = (-critic_model(z_flat)).view(num_trajs, horizon).sum(1)
    reward_terms = reward_func.r(z_flat).view(num_trajs, horizon).sum(1)
    cov = (critic_terms * reward_terms).mean() - critic_terms.mean() * reward_terms.mean()
    return cov / horizon


def parse_source_envs(config):
    env_cfg = config.get('env', {})
    source_envs = config.get('source_envs', env_cfg.get('source_envs'))
    if source_envs is None:
        env_name = env_cfg.get('env_name')
        source_envs = env_name if isinstance(env_name, list) else [env_name]
    return source_envs


def load_source_expert_data(source_env_names, num_expert_trajs, state_indices, seed):
    expert_data = []
    for env_name in source_env_names:
        preferred_path = f'expert_data/states/{env_name}_{seed}_det.pt'
        fallback_path = f'expert_data/states/{env_name}.pt'
        path = preferred_path if os.path.exists(preferred_path) else fallback_path
        trajs = torch.load(path).numpy()[:, :, state_indices]
        trajs = trajs[:num_expert_trajs, :, :]
        expert_data.append(trajs)
        print(f"Loaded expert data from {path}: {trajs.shape}")
    return expert_data


def build_sac_agent(env_name, replay_buffer, config, state_indices, seed, device):
    env_fn = lambda: gym.make(env_name)
    return SAC(
        env_fn,
        replay_buffer,
        steps_per_epoch=config['env']['T'],
        update_after=config['env']['T'] * config['sac']['random_explore_episodes'],
        max_ep_len=config['env']['T'],
        seed=seed,
        start_steps=config['env']['T'] * config['sac']['random_explore_episodes'],
        reward_state_indices=state_indices,
        device=device,
        **config['sac'],
    )


def maybe_resample_expert_trajs(expert_z_trajs, resample_episodes):
    if resample_episodes <= 0:
        return expert_z_trajs
    replace = resample_episodes > expert_z_trajs.shape[0]
    idx = np.random.choice(expert_z_trajs.shape[0], resample_episodes, replace=replace)
    return expert_z_trajs[idx]


def evaluate_multi_source(itr, source_specs, source_samples, expert_samples_eval, sac_agent, config):
    env_steps = (itr + 1) * len(source_specs) * config['sac']['epochs'] * config['env']['T']
    agent_eval_samples = np.concatenate(
        [sample[0].reshape(-1, sample[0].shape[2]) for sample in source_samples],
        axis=0,
    )
    metrics = eval.KL_summary(expert_samples_eval, agent_eval_samples, env_steps, "Running")

    det_returns = []
    sto_returns = []
    for spec in source_specs:
        det_return = eval.evaluate_real_return(
            sac_agent.get_action,
            spec['eval_env'],
            config['irl']['eval_episodes'],
            config['env']['T'],
            True,
        )
        sto_return = eval.evaluate_real_return(
            sac_agent.get_action,
            spec['eval_env'],
            config['irl']['eval_episodes'],
            config['env']['T'],
            False,
        )
        det_returns.append(det_return)
        sto_returns.append(sto_return)

    real_return_det = float(np.mean(det_returns))
    real_return_sto = float(np.mean(sto_returns))

    logger.record_tabular("Iteration", itr)
    logger.record_tabular("Env Steps", env_steps)
    logger.record_tabular("Real Det Return", round(real_return_det, 2))
    logger.record_tabular("Real Sto Return", round(real_return_sto, 2))
    for spec, det_return, sto_return in zip(source_specs, det_returns, sto_returns):
        env_name = spec['env_name']
        logger.record_tabular(f"{env_name} Det Return", round(det_return, 2))
        logger.record_tabular(f"{env_name} Sto Return", round(sto_return, 2))
    for k, val in metrics.items():
        logger.record_tabular(k, val)
    return real_return_det, real_return_sto


if __name__ == "__main__":
    yaml = YAML()
    v = yaml.load(open(sys.argv[1]))

    source_env_names = parse_source_envs(v)
    primary_env_name = source_env_names[0]
    state_indices = v['env']['state_indices']
    seed = v['seed']
    num_expert_trajs = v['irl']['expert_episodes']

    device = torch.device(f"cuda:{v['cuda']}" if torch.cuda.is_available() and v['cuda'] >= 0 else "cpu")
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(seed)
    pid = os.getpid()

    trairl_cfg = v.get('trairl', {})
    latent_dim = trairl_cfg.get('latent_dim', 16)
    vae_hidden_size = trairl_cfg.get('hidden_size', 256)
    beta_kl = trairl_cfg.get('beta_kl', 0.1)
    lambda_vae = trairl_cfg.get('lambda_vae', 1.0)
    lambda_wgan = trairl_cfg.get('lambda_wgan', 1.0)
    lambda_f = trairl_cfg.get('lambda_f', 1.0)
    joint_updates = trairl_cfg.get('joint_updates', 1)
    critic_updates = v['critic']['iter']
    vae_lr = trairl_cfg.get('vae_lr', 3e-4)
    reward_anchor_lambda = v['reward'].get('anchor_lambda', 0.0)

    print(">>> [TraIRL Setup] Force setting obj to 'emd' (W1 Distance)")
    v['obj'] = 'emd'
    assert v['IS'] is False

    env_tag =  f"multi_{len(source_env_names)}src"
    exp_id = f"logs/{env_tag}/exp-{num_expert_trajs}/{v['obj']}-trairl/{seed}"
    if not os.path.exists(exp_id):
        os.makedirs(exp_id)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    log_folder = exp_id + '/' + now.strftime('%Y_%m_%d_%H_%M_%S')
    logger.configure(dir=log_folder)
    print(f"Logging to directory: {log_folder}")
    os.system(f'cp {sys.argv[0]} {log_folder}')
    os.system(f'cp {sys.argv[1]} {log_folder}/variant_{pid}.yml')
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(v, f, indent=2, sort_keys=True)
    os.makedirs(os.path.join(log_folder, 'model'))

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
        source_specs.append({
            'task_id': task_id,
            'env_name': env_name,
            'train_env': train_env,
            'eval_env': eval_env,
        })

    if state_indices == 'all':
        state_indices = list(range(state_size))

    source_expert_trajs = load_source_expert_data(source_env_names, num_expert_trajs, state_indices, seed)
    expert_samples_eval = np.concatenate(
        [expert_trajs.reshape(-1, len(state_indices)) for expert_trajs in source_expert_trajs],
        axis=0,
    )

    vae = MultiHeadVAE(
        state_dim=len(state_indices),
        latent_dim=latent_dim,
        num_tasks=len(source_env_names),
        hidden_size=vae_hidden_size,
        device=device,
    )
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=vae_lr)

    reward_func = MLPReward(latent_dim, **v['reward'], device=device).to(device)
    reward_optimizer = torch.optim.Adam(
        reward_func.parameters(),
        lr=v['reward']['lr'],
        weight_decay=v['reward']['weight_decay'],
        betas=(v['reward']['momentum'], 0.999),
    )

    critic = Critic(latent_dim, **v['critic'], device=device)
    critic_batch_size = min(v['critic']['batch_size'], expert_samples_eval.shape[0])

    replay_buffer = ReplayBuffer(
        state_size,
        action_size,
        device=device,
        size=v['sac']['buffer_size'],
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
    n_itrs = v['irl']['n_itrs']

    print(f">>> [TraIRL Setup] Sources: {source_env_names}")
    print(
        f">>> [TraIRL Setup] latent_dim={latent_dim}, lambda_vae={lambda_vae}, "
        f"lambda_wgan={lambda_wgan}, lambda_f={lambda_f}, joint_updates={joint_updates}"
    )

    for itr in range(n_itrs):
        if itr > 0 and v['sac']['reinitialize']:
            print("Reinitializing SAC agent and replay buffer")
            replay_buffer = ReplayBuffer(
                state_size,
                action_size,
                device=device,
                size=v['sac']['buffer_size'],
            )
            sac_agent = build_sac_agent(primary_env_name, replay_buffer, v, state_indices, seed, device)
            sac_agent.reward_function = get_latent_reward

        source_samples = []
        for spec in source_specs:
            sac_agent.env = spec['train_env']
            sac_agent.test_env = spec['eval_env']
            sac_agent.test_fn = sac_agent.test_agent_ori_env
            sac_agent.learn_mujoco(print_out=True)
            samples = collect.collect_trajectories_policy_single(
                spec['train_env'],
                sac_agent,
                n=v['irl']['training_trajs'],
                state_indices=state_indices,
            )
            source_samples.append(samples)

        start = time.time()
        for task_id in range(len(source_specs)):
            expert_flat = source_expert_trajs[task_id].reshape(-1, len(state_indices))
            agent_flat = source_samples[task_id][0].reshape(-1, len(state_indices))
            with torch.no_grad():
                expert_z = vae.encode_mean(
                    torch.as_tensor(expert_flat, dtype=torch.float32, device=device)
                ).cpu().numpy()
                agent_z = vae.encode_mean(
                    torch.as_tensor(agent_flat, dtype=torch.float32, device=device)
                ).cpu().numpy()
            critic.learn(expert_z, agent_z, iter=critic_updates)

        set_requires_grad(critic.model, False)
        joint_total_loss = torch.tensor(0.0, device=device)
        vae_loss_value = 0.0
        wgan_loss_value = 0.0
        reward_loss_value = 0.0

        for _ in range(joint_updates):
            # Stage A: update abstraction (encoder/decoders) with VAE + WGAN losses.
            set_requires_grad(vae, True)
            vae_optimizer.zero_grad()
            total_vae_loss = torch.tensor(0.0, device=device)
            total_wgan_loss = torch.tensor(0.0, device=device)

            for task_id, spec in enumerate(source_specs):
                expert_trajs_t = torch.as_tensor(
                    source_expert_trajs[task_id],
                    dtype=torch.float32,
                    device=device,
                )
                agent_trajs_t = torch.as_tensor(
                    source_samples[task_id][0],
                    dtype=torch.float32,
                    device=device,
                )
                expert_flat_t = expert_trajs_t.reshape(-1, len(state_indices))
                agent_flat_t = agent_trajs_t.reshape(-1, len(state_indices))

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

            # Stage B: freeze encoder/decoders and optimize reward only (paper Eq.5 assumption).
            set_requires_grad(vae, False)
            reward_optimizer.zero_grad()
            old_reward_params = {
                name: param.detach().clone() for name, param in reward_func.named_parameters()
            }
            total_reward_loss = torch.tensor(0.0, device=device)

            for task_id, spec in enumerate(source_specs):
                expert_trajs_t = torch.as_tensor(
                    source_expert_trajs[task_id],
                    dtype=torch.float32,
                    device=device,
                )
                agent_trajs_t = torch.as_tensor(
                    source_samples[task_id][0],
                    dtype=torch.float32,
                    device=device,
                )
                expert_flat_t = expert_trajs_t.reshape(-1, len(state_indices))
                agent_flat_t = agent_trajs_t.reshape(-1, len(state_indices))

                with torch.no_grad():
                    expert_z_flat = vae.encode_mean(expert_flat_t)
                    agent_z_flat = vae.encode_mean(agent_flat_t)

                expert_z_trajs = expert_z_flat.view(expert_trajs_t.shape[0], expert_trajs_t.shape[1], latent_dim)
                agent_z_trajs = agent_z_flat.view(agent_trajs_t.shape[0], agent_trajs_t.shape[1], latent_dim)
                expert_z_batch = maybe_resample_expert_trajs(
                    expert_z_trajs, v['irl']['resample_episodes']
                )
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
            torch.nn.utils.clip_grad_norm_(reward_func.parameters(), max_norm=1.0)
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
                logger.get_dir(),
                "model",
                f"best_itr{itr}_det{max_real_return_det:.0f}_sto{max_real_return_sto:.0f}",
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
        if v['sac']['automatic_alpha_tuning']:
            logger.record_tabular("alpha", sac_agent.alpha.item())

        # if itr % v['irl'].get('save_interval', 100) == 0:
        #     ckpt_dir = os.path.join(logger.get_dir(), "model", f"checkpoint_itr{itr}")
        #     os.makedirs(ckpt_dir, exist_ok=True)
        #     torch.save(vae.state_dict(), os.path.join(ckpt_dir, "vae.pkl"))
        #     torch.save(reward_func.state_dict(), os.path.join(ckpt_dir, "reward_func.pkl"))
        #     torch.save(sac_agent.ac.state_dict(), os.path.join(ckpt_dir, "policy.pkl"))

        logger.dump_tabular()