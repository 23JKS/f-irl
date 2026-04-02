"""
改进版 AIRL (Adversarial Inverse Reinforcement Learning) + SAC
参考官方 imitation 库的最佳实践，但保持代码独立
- Action-conditioned AIRL（标准做法）
- 修复 expert action 索引 off-by-one bug
- 增加 reward clipping 和数值稳定性
- 增加 discriminator accuracy 监控
- 支持 state-only reward（r(s) + γV(s') - V(s)）
"""

import sys, os, time
import numpy as np
import torch
import gym
from ruamel.yaml import YAML
import envs

from common.sac import SAC, ReplayBuffer
from baselines.discrim import ResNetAIRLDisc, MLPDisc
from utils import system, collect, logger, eval

import datetime
import dateutil.tz
import json
import copy


class AIRLTrainer:
    """AIRL trainer with imitation best practices"""
    
    def __init__(self, config_dict, device):
        self.v = config_dict
        self.device = device
        
        # Environment setup
        self.env_name = self.v['env']['env_name']
        self.state_indices = self.v['env']['state_indices']
        self.seed = self.v['seed']
        self.num_expert_trajs = self.v['irl']['expert_episodes']
        
        # Create environments
        self.env_fn = lambda: gym.make(self.env_name)
        self.env = self.env_fn()
        self.test_env = self.env_fn()
        
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        
        if self.state_indices == 'all':
            self.state_indices = list(range(self.state_size))
        
        # Load expert data
        self._load_expert_data()
        
        # Normalize if needed
        if self.v['adv_irl']['normalize']:
            self._normalize_expert_data()
        
        # Initialize discriminator
        self._init_discriminator()
        
        # Replay buffer for policy
        self.replay_buffer = ReplayBuffer(
            self.state_size,
            self.action_size,
            device=device,
            size=self.v['adv_irl']['replay_buffer_size']
        )
        
        # Initialize SAC agent
        self._init_sac_agent()
        
        # Tracking
        self.max_real_return_det = -np.inf
        self.max_real_return_sto = -np.inf
        self._n_env_steps = 0
        self._n_train_steps = 0
        
        # imitation 风格的参数
        self.gamma = self.v['adv_irl'].get('gamma', 0.99)
        self.reward_scale = self.v['adv_irl'].get('reward_scale', 0.2)
        self.use_grad_pen = self.v['adv_irl'].get('use_grad_pen', False)
    
    def _load_expert_data(self):
        """Load expert trajectories and actions（修复 action 对齐）"""
        load_path = f'expert_data/states/{self.env_name}_airl.pt'
        self.expert_trajs = torch.load(load_path).numpy()[:, :, self.state_indices]
        self.expert_trajs = self.expert_trajs[:self.num_expert_trajs, :, :]
        
        self.expert_action_trajs = torch.load(
            f'expert_data/actions/{self.env_name}_airl.pt'
        ).numpy()
        self.expert_action_trajs = self.expert_action_trajs[:self.num_expert_trajs, :, :]
        
        # 关键修复：action 必须和 s_t 对齐
        obs_list, act_list, obs2_list = [], [], []
        for traj_idx in range(len(self.expert_trajs)):
            obs_list.append(self.expert_trajs[traj_idx, :-1])
            act_list.append(self.expert_action_trajs[traj_idx, :-1])  # 修复：不是 1:
            obs2_list.append(self.expert_trajs[traj_idx, 1:])
        
        self.expert_obs = np.concatenate(obs_list, axis=0)
        self.expert_act = np.concatenate(act_list, axis=0)
        self.expert_obs2 = np.concatenate(obs2_list, axis=0)
        
        print(f"Expert trajectories: {self.expert_trajs.shape}")
        print(f"Expert (s,a,s') pairs: {len(self.expert_obs)}")
    
    def _normalize_expert_data(self):
        """Normalize expert data（imitation 风格）"""
        expert_samples_flat = self.expert_obs.copy()
        self.obs_mean = expert_samples_flat.mean(0)
        self.obs_std = expert_samples_flat.std(0)
        self.obs_std[self.obs_std == 0.0] = 1.0
        
        self.expert_obs = (self.expert_obs - self.obs_mean) / self.obs_std
        self.expert_obs2 = (self.expert_obs2 - self.obs_mean) / self.obs_std
        
        print(f"Normalized obs_mean: {self.obs_mean[:5]}...")
        print(f"Normalized obs_std: {self.obs_std[:5]}...")
        
        self.env_fn = lambda: gym.make(
            self.env_name,
            obs_mean=self.obs_mean,
            obs_std=self.obs_std
        )
        self.env = self.env_fn()
        self.test_env = self.env_fn()
    
    def _init_discriminator(self):
        """Initialize discriminator（imitation 风格）"""
        disc_config = self.v['adv_irl']['disc']
        
        if disc_config['model_type'] == 'resnet_disc':
            self.reward_model = ResNetAIRLDisc(
                len(self.state_indices),
                device=self.device,
                **disc_config
            ).to(self.device)
        elif disc_config['model_type'] == 'mlp_disc':
            self.reward_model = MLPDisc(
                len(self.state_indices),
                **disc_config
            ).to(self.device)
        
        self.value_model = copy.deepcopy(self.reward_model)
        
        self.disc_optimizer = torch.optim.Adam(
            list(self.reward_model.parameters()) + list(self.value_model.parameters()),
            lr=self.v['adv_irl']['disc_lr'],
            betas=(self.v['adv_irl']['disc_momentum'], 0.999)
        )
        
        self.bce_loss = torch.nn.BCELoss().to(self.device)
        print(f"Initialized AIRL discriminator with {disc_config['model_type']}")
    
    def _init_sac_agent(self):
        """Initialize SAC agent"""
        self.sac_agent = SAC(
            self.env_fn,
            self.replay_buffer,
            steps_per_epoch=self.v['env']['T'],
            max_ep_len=self.v['env']['T'],
            seed=self.seed,
            reward_state_indices=self.state_indices,
            device=self.device,
            **self.v['sac']
        )
        print("Initialized SAC agent")
    
    def _disc_forward_airl(self, p_obs, p_act, p_obs2, e_obs, e_act, e_obs2):
        """AIRL discriminator forward pass"""
        obs = torch.cat([e_obs, p_obs], dim=0)
        act = torch.cat([e_act, p_act], dim=0)
        obs2 = torch.cat([e_obs2, p_obs2], dim=0)
        
        reward = self.reward_model(obs)
        cur_val = self.value_model(obs)
        next_val = self.value_model(obs2)
        
        log_p = reward + self.gamma * next_val - cur_val
        
        with torch.no_grad():
            log_q = self.sac_agent.ac.log_prob(obs, act).unsqueeze(1)
            baseline = torch.max(log_p, log_q)
        
        log_p = torch.clamp(log_p - baseline, -20.0, 20.0)
        log_q = torch.clamp(log_q - baseline, -20.0, 20.0)
        
        disc_logits = torch.exp(log_p) / (torch.exp(log_p) + torch.exp(log_q) + 1e-8)
        disc_logits = torch.clamp(disc_logits, 0.0, 1.0)
        
        return disc_logits
    
    def _get_airl_reward(self, obs, obs2):
        """Get reward from discriminator（imitation 风格）"""
        self.reward_model.eval()
        self.value_model.eval()
        
        with torch.no_grad():
            rewards = self.reward_model(obs)
            rewards = rewards + self.gamma * self.value_model(obs2) - self.value_model(obs)
            rewards = torch.clamp(rewards, -50.0, 50.0)
            rewards = torch.nan_to_num(rewards, nan=0.0, posinf=50.0, neginf=-50.0)
            rewards = rewards * self.reward_scale
        
        self.reward_model.train()
        self.value_model.train()
        
        return rewards
    
    def _train_discriminator(self):
        """Train discriminator（imitation 风格）"""
        batch_size = self.v['adv_irl']['disc_optim_batch_size']
        
        policy_batch = self.replay_buffer.sample_batch(batch_size)
        p_obs = policy_batch['obs']
        p_act = policy_batch['act']
        p_obs2 = policy_batch['obs2']
        
        ei = np.random.choice(len(self.expert_obs), size=batch_size)
        e_obs = torch.FloatTensor(self.expert_obs[ei]).to(self.device)
        e_act = torch.FloatTensor(self.expert_act[ei]).to(self.device)
        e_obs2 = torch.FloatTensor(self.expert_obs2[ei]).to(self.device)
        
        self.disc_optimizer.zero_grad()
        
        disc_logits = self._disc_forward_airl(p_obs, p_act, p_obs2, e_obs, e_act, e_obs2)
        
        bce_targets = torch.cat([
            torch.ones(batch_size, 1),
            torch.zeros(batch_size, 1)
        ]).to(self.device)
        
        disc_loss = self.bce_loss(disc_logits, bce_targets)
        
        disc_grad_pen_loss = 0.0
        if self.use_grad_pen:
            eps = torch.rand((batch_size, 1)).to(self.device)
            interp_obs = eps * e_obs + (1 - eps) * p_obs
            interp_obs.requires_grad_(True)
            grad_reward = torch.autograd.grad(
                outputs=self.reward_model(interp_obs).sum(),
                inputs=[interp_obs], create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            grad_penalty = ((grad_reward.norm(2, dim=1) - 1) ** 2).mean()
            
            eps2 = torch.rand((batch_size, 1)).to(self.device)
            interp_obs2 = eps2 * e_obs + (1 - eps2) * p_obs
            interp_obs2.requires_grad_(True)
            grad_value = torch.autograd.grad(
                outputs=self.value_model(interp_obs2).sum(),
                inputs=[interp_obs2], create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            grad_penalty += ((grad_value.norm(2, dim=1) - 1) ** 2).mean()
            
            disc_grad_pen_loss = grad_penalty * self.v['adv_irl'].get('grad_pen_weight', 1.0)
        
        total_loss = disc_loss + disc_grad_pen_loss
        torch.nn.utils.clip_grad_norm_(
            list(self.reward_model.parameters()) + list(self.value_model.parameters()),
            max_norm=5.0
        )
        total_loss.backward()
        self.disc_optimizer.step()
        
        with torch.no_grad():
            acc = ((disc_logits > 0.5) == bce_targets).float().mean().item()
        
        return total_loss.item(), disc_loss.item(), acc
    
    def _train_policy(self):
        """Train policy"""
        batch_size = self.v['adv_irl']['policy_optim_batch_size']
        policy_batch = self.replay_buffer.sample_batch(batch_size)
        
        obs = policy_batch['obs']
        obs2 = policy_batch['obs2']
        
        policy_batch['rew'] = self._get_airl_reward(obs, obs2)
        
        return self.sac_agent.update(policy_batch)
    
    def train(self):
        """Main training loop"""
        num_epochs = self.v['adv_irl']['num_epochs']
        num_steps_per_epoch = self.v['adv_irl']['num_steps_per_epoch']
        num_steps_between_train = self.v['adv_irl']['num_steps_between_train_calls']
        min_steps_before_train = self.v['adv_irl']['min_steps_before_training']
        
        num_disc_updates = self.v['adv_irl']['num_disc_updates_per_loop_iter']
        num_policy_updates = self.v['adv_irl']['num_policy_updates_per_loop_iter']
        num_initial_disc_iters = self.v['adv_irl'].get('num_initial_disc_iters', 100)
        
        o, ep_len = self.env.reset(), 0
        
        print("Initial discriminator training...")
        for _ in range(num_initial_disc_iters):
            if self._n_env_steps < min_steps_before_train:
                a = self.env.action_space.sample()
            else:
                a = self.sac_agent.get_action(o)
            
            o2, _, d, _ = self.env.step(a)
            ep_len += 1
            self._n_env_steps += 1
            d = False if ep_len == self.v['env']['T'] else d
            
            self.replay_buffer.store(o, a, 0.0, o2, d)
            o = o2
            
            if d or ep_len == self.v['env']['T']:
                o, ep_len = self.env.reset(), 0
        
        print("Starting main training loop...")
        for epoch in range(1, num_epochs + 1):
            for step in range(num_steps_per_epoch):
                if self._n_env_steps < min_steps_before_train:
                    a = self.env.action_space.sample()
                else:
                    a = self.sac_agent.get_action(o)
                
                o2, _, d, _ = self.env.step(a)
                ep_len += 1
                self._n_env_steps += 1
                d = False if ep_len == self.v['env']['T'] else d
                
                self.replay_buffer.store(o, a, 0.0, o2, d)
                o = o2
                
                if d or ep_len == self.v['env']['T']:
                    o, ep_len = self.env.reset(), 0
                
                if self._n_env_steps % num_steps_between_train == 0:
                    if self.replay_buffer.size >= min_steps_before_train:
                        for _ in range(num_disc_updates):
                            total_loss, disc_loss, disc_acc = self._train_discriminator()
                        for _ in range(num_policy_updates):
                            self._train_policy()
                        self._n_train_steps += 1
                        
                        logger.record_tabular("Disc Acc", round(disc_acc, 3))
            
            self._evaluate(epoch)
            
            logger.record_tabular("Epoch", epoch)
            logger.record_tabular("Env Steps", self._n_env_steps)
            logger.record_tabular("Train Steps", self._n_train_steps)
            logger.dump_tabular()
    
    def _evaluate(self, epoch):
        """Evaluate current policy"""
        samples = collect.collect_trajectories_policy_single(
            self.test_env,
            self.sac_agent,
            n=self.v['irl']['training_trajs'],
            state_indices=self.state_indices
        )
        
        agent_emp_states = samples[0]
        expert_states = self.expert_obs.reshape(-1, len(self.state_indices))
        
        metrics = eval.KL_summary(
            expert_states,
            agent_emp_states.reshape(-1, agent_emp_states.shape[2]),
            self._n_env_steps,
            "Running",
            False
        )
        
        real_return_det = eval.evaluate_real_return(
            self.sac_agent.get_action,
            self.test_env,
            self.v['irl']['eval_episodes'],
            self.v['env']['T'],
            True
        )
        metrics["Real Det Return"] = real_return_det
        logger.record_tabular("Real Det Return", round(real_return_det, 2))
        
        real_return_sto = eval.evaluate_real_return(
            self.sac_agent.get_action,
            self.test_env,
            self.v['irl']['eval_episodes'],
            self.v['env']['T'],
            False
        )
        metrics["Real Sto Return"] = real_return_sto
        logger.record_tabular("Real Sto Return", round(real_return_sto, 2))
        
        if real_return_det > self.max_real_return_det and real_return_sto > self.max_real_return_sto:
            self.max_real_return_det = real_return_det
            self.max_real_return_sto = real_return_sto
            
            save_name = os.path.join(
                logger.get_dir(),
                f"model/best_reward_model_det{real_return_det:.0f}_sto{real_return_sto:.0f}.pkl"
            )
            torch.save(self.reward_model.state_dict(), save_name)
            
            save_name = os.path.join(
                logger.get_dir(),
                f"model/best_value_model_det{real_return_det:.0f}_sto{real_return_sto:.0f}.pkl"
            )
            torch.save(self.value_model.state_dict(), save_name)


if __name__ == "__main__":
    yaml = YAML()
    v = yaml.load(open(sys.argv[1]))
    
    device = torch.device(f"cuda:{v['cuda']}" if torch.cuda.is_available() and v['cuda'] >= 0 else "cpu")
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(v['seed'])
    pid = os.getpid()
    
    env_name = v['env']['env_name']
    num_expert_trajs = v['irl']['expert_episodes']
    exp_id = f"logs/{env_name}/exp-{num_expert_trajs}/airl/{v['seed']}"
    
    if not os.path.exists(exp_id):
        os.makedirs(exp_id)
    
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    log_folder = exp_id + '/' + now.strftime('%Y_%m_%d_%H_%M_%S')
    logger.configure(dir=log_folder)
    
    print(f"Logging to: {log_folder}")
    os.system(f'cp baselines/airl.py {log_folder}')
    os.system(f'cp {sys.argv[1]} {log_folder}/variant_{pid}.yml')
    
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(v, f, indent=2, sort_keys=True)
    
    os.makedirs(os.path.join(log_folder, 'model'))
    print(f"PID: {pid}")
    
    trainer = AIRLTrainer(v, device)
    trainer.train()
