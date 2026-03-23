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

    def sample_batch(self, batch_size=32):
        """
        [修改] 对齐 JAX 源码：使用均匀采样 (Uniform Sampling)。
        JAX 实现中 ReplayBuffer.sample 是均匀采样，
        然后使用 batch.priority 直接加权 Loss (Importance Weighting)。
        """
        # 1. 均匀采样
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        batch = dict(obs=self.state[idxs],
                     obs2=self.next_state[idxs],
                     act=self.action[idxs],
                     rew=self.reward[idxs],
                     done=self.done[idxs])
        
        # 2. 获取优先级作为权重
        raw_p = self.priorities[idxs]
        weights = raw_p.copy() # 在 JAX 源码中，直接使用优先级作为权重 (w * loss)
        
        batch = {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k,v in batch.items()}
        batch['weights'] = torch.as_tensor(weights, dtype=torch.float32).to(self.device)
        batch['raw_priorities'] = torch.as_tensor(raw_p, dtype=torch.float32).to(self.device)
        batch['indices'] = torch.as_tensor(idxs, dtype=torch.long).to(self.device)
        return batch

class SAC:
    def __init__(self, env_fn, replay_buffer, k=1, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
            steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, add_time=False,
            polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, update_num=20,
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
        self.roer_beta = roer_params.get('beta', 4.0)
        self.roer_clip_min = roer_params.get('clip_min', 10.0)
        self.roer_clip_max = roer_params.get('clip_max', 50.0)

        self.gumbel_alpha = self.roer_beta

        self.test_fn = self.test_agent

    def calculate_priorities(self, td_errors, old_priorities):
        """对齐 JAX 源码 update_priority (avg scheme, std_normalize=True)"""
        beta = self.roer_beta if abs(self.roer_beta) > 1e-6 else 1.0

        a = td_errors / beta
        # 先截断指数输入，防止 overflow（np.exp 在 >709 时溢出）
        # clip_max 对应的最大安全输入是 ln(clip_max)
        a = np.clip(a, -500.0, np.log(self.roer_clip_max + 1e-8))
        exp_a = np.exp(a)

        # clip: max first, then enforce >= 1 (JAX 源码顺序)
        exp_a = np.minimum(exp_a, self.roer_clip_max)
        exp_a = np.maximum(exp_a, 1.0)

        if np.any(np.isnan(exp_a)) or np.any(np.isinf(exp_a)):
            print("Warning: NaN/Inf in exp_a, skipping priority update")
            return old_priorities

        # std_normalize=True: 用加权均值归一化，与 JAX 完全一致
        # JAX: exp_a = exp_a / jnp.mean(w * exp_a)
        weighted_mean = np.mean(old_priorities * exp_a)
        exp_a = exp_a / (weighted_mean + 1e-8)

        # EMA 更新: priority = (beta * exp_a + (1-beta)) * w
        # 注意：JAX 里这个 beta 是 per_beta (EMA rate)，对应我们的 roer_lambda
        new_priorities = (self.roer_lambda * exp_a + (1 - self.roer_lambda)) * old_priorities

        # 最终下界截断 (JAX: priority = maximum(priority, min_clip))
        new_priorities = np.maximum(new_priorities, self.roer_clip_min)

        return new_priorities

    def compute_extreme_v_loss(self, obs, act):
        """计算 ExtremeV loss (基于 Q(s,a) - V(s))
        对齐 JAX: log_loss=True 使用 gumbel_rescale_loss_per（带线性外推），
                  noise=True 给动作加噪声（std=0.1）
        """
        with torch.no_grad():
            # 对齐 JAX: noise=True, noise_std=0.1
            noise = torch.randn_like(act) * 0.1
            noisy_act = torch.clamp(act + noise, -self.act_limit, self.act_limit)
            # 对齐 JAX: 使用 Target Critic
            q_values = torch.min(self.ac_targ.q1(obs, noisy_act), self.ac_targ.q2(obs, noisy_act))
        v = self.value_net(obs)
        
        alpha = self.gumbel_alpha if abs(self.gumbel_alpha) > 1e-6 else 1.0
        x = (q_values - v) / alpha
        
        # 对齐 JAX: gumbel_rescale_loss_per (log_loss=True)
        # z = clip(x, max=gumbel_max_clip)
        # loss = exp(z) - z - 1  (for x <= clip)
        # linear = (x - z) * (exp(z) - 1) / alpha  (for x > clip, linear extrapolation)
        z = torch.clamp_max(x, self.gumbel_max_clip)
        loss = torch.exp(z) - z - 1.0
        linear = (x - z) * (torch.exp(z) - 1.0) / alpha
        loss = loss + linear
        
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

        # 对齐 JAX 源码：per=True 时使用 Huber loss (delta=20)，防止大 TD error 样本权重爆炸导致 Q 值崩溃
        def huber_loss(x, delta=20.0):
            abs_x = torch.abs(x)
            quadratic = torch.clamp(abs_x, max=delta)
            linear = abs_x - quadratic
            return 0.5 * quadratic ** 2 + delta * linear

        q1_error = huber_loss(q1 - backup.detach())
        q2_error = huber_loss(q2 - backup.detach())

        # 归一化 weights 保证梯度尺度稳定
        weights = weights / (weights.mean() + 1e-8)
        loss_q = (weights * q1_error).mean() + (weights * q2_error).mean()

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
        # Align with common/sac.py: standard SAC update (no gradient clipping)
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        for p in self.q_params:
            p.requires_grad = False

        self.pi_optimizer.zero_grad()
        loss_pi, log_pi = self.compute_loss_pi(data)
        loss_pi.backward()
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
                    # 步骤 1: 均匀采样
                    # =======================================================
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    
                    # =======================================================
                    # [修改] Reward Relabeling (核心修复)
                    # =======================================================
                    if self.reward_function is not None and self.reward_state_indices is not None:
                        # 提取用于计算奖励的状态部分
                        obs = batch['obs'][:, self.reward_state_indices]
                        
                        # 使用最新的 reward_function 计算奖励
                        # 注意：reward_function 应该返回 tensor 或 numpy array
                        # 如果 reward_function 是 torch module，确保它在不需要梯度模式下运行以节省内存
                        with torch.no_grad():
                             rew_vals = self.reward_function(obs)
                        
                        # 处理维度
                        rew_for_v = torch.as_tensor(
                            rew_vals, dtype=torch.float32, device=self.device
                        ).squeeze() 
                        if rew_for_v.ndim == 0:
                             rew_for_v = rew_for_v.view(1).repeat(len(batch['obs']))
                        elif rew_for_v.ndim > 1:
                             rew_for_v = rew_for_v.view(-1)
                        
                        # [防御性编程] 截断奖励值，防止极端值破坏训练
                        # Ant环境正常奖励通常在 -10 到 100 之间，IRL学出的奖励可能漂移
                        # 建议限制在 [-10, 10] 或类似范围，取决于IRL的具体设定
                        # rew_for_v = torch.clamp(rew_for_v, min=-10.0, max=10.0)

                        # 覆盖 Batch 中的旧奖励
                        batch['rew'] = rew_for_v 
                    else:
                        rew_for_v = batch['rew']
                    
                    indices = batch['indices'].cpu().numpy()
                    
                    # =======================================================
                    # 步骤 2: 计算 V-Net Error 并更新 V 网络
                    # =======================================================
                    
                    # 2.1 先更新 V 网络
                    # 注意：compute_extreme_v_loss 内部使用了 self.ac_targ，
                    # 此时 batch['rew'] 还没有被用到，因为 loss(V) = E[exp((Q_targ - V)/alpha) - ...]
                    # 但 Q_targ 的计算是在 SAC 更新里做的。
                    # ROER 的 Extreme-V Loss 是基于 Q(s,a) - V(s)，这里 Q 应该是当前的 Critic。
                    # 您之前的代码使用的是 self.ac_targ，这通常是为了稳定性，但逻辑上 Q 应该是 Critic 的估计。
                    # JAX 实现中确实常使用 Target Q 来计算 V 的目标。
                    value_loss = self.compute_extreme_v_loss(batch['obs'], batch['act'])
                    self.value_optimizer.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=10.0)
                    self.value_optimizer.step()

                    # 2.2 [修改] 使用 Relabeled Reward 计算 TD Error
                    # TD Error = (r + gamma * V(s')) - V(s)
                    # 必须使用最新的 rew_for_v
                    with torch.no_grad():
                        next_v = self.value_net(batch['obs2']) 
                        # 使用最新的奖励计算目标 V
                        target_v = rew_for_v + (1 - batch['done']) * self.gamma * next_v
                        current_v = self.value_net(batch['obs'])
                        td_error = target_v - current_v 
                    
                    # =======================================================
                    # 步骤 3: SAC 更新 (带 Importance Weighting)
                    # =======================================================
                    if not self.enable_roer_priority:
                        batch['weights'] = torch.ones_like(batch['rew'])

                    # update 方法内部会使用 batch['rew'] 计算 Q loss
                    # 此时 batch['rew'] 已经是 Relabeled 过的最新奖励
                    _, _, log_pi = self.update(data=batch)

                    # =======================================================
                    # 步骤 4: 计算并更新优先级
                    # =======================================================
                    if self.enable_roer_priority:
                        td_err_np = td_error.detach().cpu().numpy()
                        if td_err_np.ndim > 1: td_err_np = td_err_np.flatten()
                        
                        old_p_np = batch['raw_priorities'].detach().cpu().numpy()
                        new_priorities = self.calculate_priorities(td_err_np, old_p_np)
                        self.replay_buffer.update_priorities(indices, new_priorities)

            if t % self.log_step_interval == 0:
                # 评估时也需要使用最新的奖励函数
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