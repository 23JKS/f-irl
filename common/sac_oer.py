from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import sys
import common.sac_agent as core
import torch.nn as nn
from common.sac_agent import mlp
import torch.nn.functional as F
import logging  # 引入 logging，但不做配置，直接使用原有配置

class MLPVFunction(nn.Module):
    """V网络：只接受obs作为输入，输出标量值V(s)"""
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v = mlp([obs_dim] + list(hidden_sizes) + [1], activation, nn.Identity)
    
    def forward(self, obs):
        v = self.v(obs)
        return torch.squeeze(v, -1)  # (batch,)

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """
    def __init__(self, obs_dim, act_dim, device=torch.device('cpu'), size=int(1e6)):
        self.state = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.next_state = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.action = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.reward = np.zeros(size, dtype=np.float32)
        self.done = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

    def store_batch(self, obs, act, rew, next_obs, done):
        num = len(obs)
        full =  self.ptr + num > self.max_size
        if not full:
            self.state[self.ptr: self.ptr + num] = obs
            self.next_state[self.ptr: self.ptr + num] = next_obs
            self.action[self.ptr: self.ptr + num] = act
            self.reward[self.ptr: self.ptr + num] = rew
            self.done[self.ptr: self.ptr + num] = done
            self.ptr = self.ptr + num
        else:
            idx = np.arange(self.ptr,self.ptr+num)%self.max_size
            self.state[idx] = obs
            self.next_state[idx]=next_obs
            self.action[idx]=act
            self.reward[idx]=rew
            self.done[idx]=done
            self.ptr= (self.ptr+num)%self.max_size            
        self.size = min(self.size + num, self.max_size)

    def store(self, obs, act, rew, next_obs, done):
        self.state[self.ptr] = obs
        self.next_state[self.ptr] = next_obs
        self.action[self.ptr] = act
        self.reward[self.ptr] = rew
        self.done[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.state[idxs],
                     obs2=self.next_state[idxs],
                     act=self.action[idxs],
                     rew=self.reward[idxs],
                     done=self.done[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k,v in batch.items()}

class ROERReplayBuffer(ReplayBuffer):
    def __init__(self, obs_dim, act_dim, device=torch.device('cpu'), size=int(1e6), 
                 lambda_=0.01, beta=1.0, clip_min=1e-4, clip_max=50.0):
        super().__init__(obs_dim, act_dim, device, size)
        self.priorities = np.ones(size, dtype=np.float32)  # 初始化优先级
    def store(self, obs, act, rew, next_obs, done):
        # 新样本用当前最大优先级初始化
        max_p = self.priorities[:self.size].max() if self.size > 0 else 1.0
        self.priorities[self.ptr] = max_p
        super().store(obs, act, rew, next_obs, done)
    def store_batch(self, obs, act, rew, next_obs, done):
        num = len(obs)
        if self.ptr + num <= self.max_size:
            idxs = np.arange(self.ptr, self.ptr + num)
        else:
            overflow = (self.ptr + num) - self.max_size
            idxs = np.concatenate([
                np.arange(self.ptr, self.max_size),
                np.arange(0, overflow)
            ])
        
        # 使用当前最大优先级初始化新样本
        max_p = self.priorities.max() if self.size > 0 else 1.0
        self.priorities[idxs] = max_p
        
        super().store_batch(obs, act, rew, next_obs, done)

    def update_priorities(self, indices, new_priorities):
        """只负责更新存储"""
        if isinstance(new_priorities, torch.Tensor):
            new_priorities = new_priorities.detach().cpu().numpy()
        self.priorities[indices] = new_priorities

    def sample_batch(self, batch_size):
        """
        [关键修复] 恢复使用概率采样 (Prioritized Sampling)
        """
        # 1. 计算采样概率
        probs = self.priorities[:self.size]
        probs = probs / (probs.sum() + 1e-8)
        
        # 2. 按概率采样 (代替均匀采样)
        idxs = np.random.choice(self.size, size=batch_size, p=probs)
        
        batch = dict(obs=self.state[idxs],
                     obs2=self.next_state[idxs],
                     act=self.action[idxs],
                     rew=self.reward[idxs],
                     done=self.done[idxs])
        
        # 3. 设置权重为 1.0 (因为优先级已经体现在采样概率里了)
        weights = np.ones_like(idxs, dtype=np.float32)
        
        # 获取原始优先级用于后续更新计算
        raw_p = self.priorities[idxs]
        
        batch = {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k,v in batch.items()}
        batch['weights'] = torch.as_tensor(weights, dtype=torch.float32).to(self.device)
        batch['raw_priorities'] = torch.as_tensor(raw_p, dtype=torch.float32).to(self.device)
        batch['indices'] = torch.as_tensor(idxs, dtype=torch.long).to(self.device)
        return batch

class SAC:
    def __init__(self, env_fn, replay_buffer, k=1, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
            steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, add_time=False,
            polyak=0.995, lr=1e-3, alpha=0.2, batch_size=256, # <--- [修改] 默认 Batch Size 从 100 增加到 256
            start_steps=10000, update_num=20,
            update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
            log_step_interval=None, reward_state_indices=None,
            save_freq=1, device=torch.device("cpu"), automatic_alpha_tuning=True, reinitialize=True,
            enable_roer_priority=True, gumbel_max_clip=7.0, **kwargs):

        self.env, self.test_env = env_fn(), env_fn()
        self.env.seed(seed)
        self.test_env.seed(seed+1)
        # 设置 action_space 的种子，确保 sample() 可复现
        self.env.action_space.seed(seed)
        self.test_env.action_space.seed(seed+1)
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]
        self.max_ep_len=max_ep_len
        self.start_steps=start_steps
        self.batch_size=batch_size
        self.gamma=gamma
        
        self.polyak=polyak
        self.act_limit = self.env.action_space.high[0]
        self.steps_per_epoch = steps_per_epoch
        self.update_after = update_after
        self.update_every = update_every
        self.udpate_num = update_num
        self.num_test_episodes = num_test_episodes
        self.epochs = epochs
        
        self.ac = actor_critic(self.env.observation_space, self.env.action_space, k, add_time=add_time, device=device, **ac_kwargs)
        self.ac_targ = deepcopy(self.ac)

        for p in self.ac_targ.parameters():
            p.requires_grad = False
            
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.replay_buffer = replay_buffer
        self.var_counts = tuple(count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr)
        self.q_optimizer = Adam(self.q_params, lr=lr)

        # ROER V-Network
        obs_dim = self.env.observation_space.shape[0]
        hidden_sizes = ac_kwargs.get('hidden_sizes', (256, 256))
        activation = ac_kwargs.get('activation', nn.ReLU)
        self.value_net = MLPVFunction(obs_dim, hidden_sizes, activation).to(device)
        self.value_optimizer = Adam(self.value_net.parameters(), lr=lr)

        self.device = device
        self.automatic_alpha_tuning = automatic_alpha_tuning
        if self.automatic_alpha_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = alpha

        self.true_state_dim = self.env.observation_space.shape[0]

        if log_step_interval is None:
            log_step_interval = steps_per_epoch
        self.log_step_interval = log_step_interval
        self.reinitialize = reinitialize

        self.reward_function = None
        self.reward_state_indices = reward_state_indices
        self.enable_roer_priority = enable_roer_priority
        self.gumbel_max_clip = gumbel_max_clip 
        
        # ROER 参数
        roer_params = kwargs.get('roer', {})
        self.roer_lambda = roer_params.get('lambda_', 0.01)
        self.roer_beta = roer_params.get('beta', 1.0)
        self.roer_clip_min = roer_params.get('clip_min', 1e-4) 
        self.roer_clip_max = roer_params.get('clip_max', 50.0)
        self.gumbel_alpha = self.roer_beta 
      

        self.test_fn = self.test_agent

    def calculate_priorities(self, td_errors, old_priorities):
        """
        计算新的优先级
        """
        beta = self.roer_beta if abs(self.roer_beta) > 1e-6 else 1.0
        
        # 1. 计算 exponent
        a = td_errors / beta
        exp_a = np.exp(a)
        
        # 2. 截断逻辑
        exp_a = np.minimum(exp_a, self.roer_clip_max)
        
        # 允许降低低 Loss 样本的权重
        exp_a = np.maximum(exp_a, 1.0) 
        
        # 3. 归一化逻辑 (使用加权均值)
        weighted_mean = np.mean(old_priorities * exp_a)
        exp_a = exp_a / (weighted_mean + 1e-8)
        
        # 4. 更新公式
        new_priorities = (self.roer_lambda * exp_a + (1 - self.roer_lambda)) * old_priorities
        
        # 5. 最终截断
        new_priorities = np.maximum(new_priorities, self.roer_clip_min)
        
        # [监控] 记录优先级统计信息 (到原日志文件)
        # 随机采样 5% 的数据进行记录，防止日志过大
        if np.random.rand() < 0.05: 
            logging.info(f"[ROER Stats] Priorities: Min={new_priorities.min():.4f}, Max={new_priorities.max():.4f}, Mean={new_priorities.mean():.4f}")

        if np.isnan(new_priorities).any():
            logging.error("[ROER Error] NaN detected in priority calculation, resetting to 1.0")
            print("Warning: NaN in priority calculation, resetting to 1.0")
            new_priorities = np.ones_like(new_priorities)

        return new_priorities

    def compute_extreme_v_loss(self, obs, act):
        """计算 ExtremeV loss"""
        with torch.no_grad():
            q_values = torch.min(self.ac_targ.q1(obs, act), self.ac_targ.q2(obs, act))
        v = self.value_net(obs)
        
        alpha = self.gumbel_alpha if abs(self.gumbel_alpha) > 1e-6 else 1.0
        x = (q_values - v) / alpha
        
        z = torch.clamp_max(x, self.gumbel_max_clip)
        
        loss = torch.exp(z) - z - 1.0
        norm = torch.mean(torch.maximum(torch.ones_like(z), torch.exp(z)))
        norm = norm.detach()
    
        loss = loss / (norm + 1e-8)
        return loss.mean()
       
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        weights = data.get('weights', None)
        if weights is None:
            weights = torch.ones_like(r)

        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        with torch.no_grad():
            a2, logp_a2 = self.ac.pi(o2[:, :self.true_state_dim])
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        backup = backup.view(q1.shape)

        q1_error = (q1 - backup.detach())**2
        q2_error = (q2 - backup.detach())**2
        loss_q1 = (weights * q1_error).mean()
        loss_q2 = (weights * q2_error).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q

    def compute_loss_pi(self, data):
        o = data['obs']
        pi, logp_pi = self.ac.pi(o[:, :self.true_state_dim])
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        loss_pi = (self.alpha * logp_pi - q_pi).mean()
        return loss_pi, logp_pi

    def update(self,data):
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.backward()
        
        # [修改] Q网络梯度裁剪 + 监控
        # clip_grad_norm_ 会返回裁剪前的 Total Norm
        q_grad_norm = torch.nn.utils.clip_grad_norm_(self.q_params, max_norm=10.0)
        self.q_optimizer.step()

        for p in self.q_params:
            p.requires_grad = False

        self.pi_optimizer.zero_grad()
        loss_pi, log_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        
        # [修改] Policy网络梯度裁剪 + 监控
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.ac.pi.parameters(), max_norm=10.0)
        self.pi_optimizer.step()

        if self.automatic_alpha_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

        for p in self.q_params:
            p.requires_grad = True

        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        
        # [监控] 记录梯度和Loss信息 (到原日志文件)
        # 随机采样 2% 的数据 (约每50次 update 记录一次)
        if np.random.rand() < 0.02: 
            logging.info(f"[SAC Update] LossQ={loss_q.item():.2f}, LossPi={loss_pi.item():.2f}, "
                         f"GradNormQ={q_grad_norm:.2f}, GradNormPi={pi_grad_norm:.2f}")

        return np.array([loss_q.item(), loss_pi.item(), log_pi.detach().cpu().mean().item()])

    def get_action(self, o, deterministic=False, get_logprob=False):
        if len(o.shape) < 2:
            o = o[None, :]
        return self.ac.act(torch.as_tensor(o[:, :self.true_state_dim], dtype=torch.float32).to(self.device), 
                    deterministic, get_logprob=get_logprob)

    def get_action_batch(self, o, deterministic=False):
        if len(o.shape) < 2:
            o = o[None, :]
        return self.ac.act_batch(torch.as_tensor(o[:, :self.true_state_dim], dtype=torch.float32).to(self.device), 
                    deterministic)

    def test_agent(self):
        avg_ep_return  = 0.
        for j in range(self.num_test_episodes):
            o = self.test_env.reset()
            obs = np.zeros((self.max_ep_len, o.shape[0]))
            for t in range(self.max_ep_len):
                o, _, _, _ = self.test_env.step(self.get_action(o, True))
                obs[t] = o.copy()
            obs = torch.FloatTensor(obs).to(self.device)[:, self.reward_state_indices]
            avg_ep_return += self.reward_function(obs).sum()
        return avg_ep_return/self.num_test_episodes

    def test_agent_batch(self):
        if hasattr(self.test_env, 'eval'):
            self.test_env.eval()
        o, ep_ret = self.test_env.reset(self.num_test_episodes), np.zeros((self.num_test_episodes))
        log_pi = np.zeros(self.num_test_episodes)
        for t in range(self.max_ep_len-1):
            a, log_pi_ = self.get_action_batch(o)
            o, r, _, _ = self.test_env.step(a)
            ep_ret += r
            log_pi += log_pi_
        return ep_ret.mean(), log_pi.mean()

    def learn_mujoco(self, print_out=False, save_path=None):
        total_steps = self.steps_per_epoch * self.epochs
        start_time = time.time()
        local_time = time.time()
        best_eval = -np.inf
        o, ep_len = self.env.reset(), 0

        print(f"Training SAC for IRL agent: Total steps {total_steps:d}")
        
        test_rets, alphas, log_pis, test_time_steps = [], [], [], []

        for t in range(total_steps):
            if self.replay_buffer.size > self.start_steps:
                a = self.get_action(o)
            else:
                a = self.env.action_space.sample()

            o2, r, d, _ = self.env.step(a)
            ep_len += 1
            d = False if ep_len==self.max_ep_len else d
            self.replay_buffer.store(o, a, r, o2, d)
            o = o2

            if d or ep_len==self.max_ep_len:
                o, ep_len = self.env.reset(), 0

            # Update handling
            log_pi = 0
            
            should_update = False
            if self.reinitialize:
                if t >= self.update_after and t % self.update_every == 0:
                    should_update = True
            else:
                if self.replay_buffer.size >= self.update_after and t % self.update_every == 0:
                    should_update = True
            
            if should_update:
                for j in range(self.update_every):
                    # =======================================================
                    # 步骤 1: 使用概率采样 (Prioritized Sampling)
                    # =======================================================
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    
                    if self.reward_function is not None and self.reward_state_indices is not None:
                        obs = batch['obs'][:, self.reward_state_indices]
                        rew_vals = self.reward_function(obs)
                        rew_for_v = torch.as_tensor(
                            rew_vals, dtype=torch.float32, device=self.device,
                        ).squeeze() 
                        if rew_for_v.ndim == 0:
                             rew_for_v = rew_for_v.view(1).repeat(len(batch['obs']))
                        elif rew_for_v.ndim > 1:
                             rew_for_v = rew_for_v.view(-1)
                        
                        batch['rew'] = rew_for_v 
                    else:
                        rew_for_v = batch['rew']
                    
                    indices = batch['indices'].cpu().numpy()
                    
                    # =======================================================
                    # 步骤 2: 更新 V 网络
                    # =======================================================
                    value_loss = self.compute_extreme_v_loss(batch['obs'], batch['act'])
                    self.value_optimizer.zero_grad()
                    value_loss.backward()
                    
                    # [修改] Value网络梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=10.0)
                    self.value_optimizer.step()

                    with torch.no_grad():
                        next_v = self.value_net(batch['obs2']) 
                        target_v = rew_for_v + (1 - batch['done']) * self.gamma * next_v
                        current_v = self.value_net(batch['obs'])
                        td_error = target_v - current_v 
                    
                    # =======================================================
                    # 步骤 3: SAC 更新
                    # =======================================================
                    if not self.enable_roer_priority:
                        batch['weights'] = torch.ones_like(batch['rew'])

                    _, _, log_pi = self.update(data=batch)

                    # =======================================================
                    # 步骤 4: 更新优先级
                    # =======================================================
                    if self.enable_roer_priority:
                        td_err_np = td_error.detach().cpu().numpy()
                        if td_err_np.ndim > 1: td_err_np = td_err_np.flatten()
                        
                        old_p_np = batch['raw_priorities'].detach().cpu().numpy()
                        new_priorities = self.calculate_priorities(td_err_np, old_p_np)
                        self.replay_buffer.update_priorities(indices, new_priorities)

            if t % self.log_step_interval == 0:
                test_epret = self.test_fn()
                if print_out:
                    print(f"SAC Training | Evaluation: {test_epret:.3f} Timestep: {t+1:d} Elapsed {time.time() - local_time:.0f}s")
                if save_path is not None:
                    if test_epret > best_eval:
                        best_eval = test_epret
                        torch.save(self.ac.state_dict(), save_path)
                alphas.append(self.alpha.item() if self.automatic_alpha_tuning else self.alpha)
                test_rets.append(test_epret)
                log_pis.append(log_pi)
                test_time_steps.append(t+1)
                local_time = time.time()

        print(f"SAC Training End: time {time.time() - start_time:.0f}s")
        return [test_rets, alphas, log_pis, test_time_steps]