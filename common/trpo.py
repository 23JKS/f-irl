"""
TRPO (Trust Region Policy Optimization) for PyTorch
Simplified implementation for AIRL
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import scipy.optimize


class GaussianMLPPolicy(nn.Module):
    """Gaussian policy with MLP"""
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()
        
        layers = []
        prev_size = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(activation())
            prev_size = h
        
        self.net = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(prev_size, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        
    def forward(self, obs):
        h = self.net(obs)
        mean = self.mean_layer(h)
        std = torch.exp(self.log_std)
        return mean, std
    
    def get_action(self, obs, deterministic=False):
        with torch.no_grad():
            mean, std = self.forward(obs)
            if deterministic:
                return mean.cpu().numpy()
            else:
                dist = Normal(mean, std)
                action = dist.sample()
                return action.cpu().numpy()
    
    def get_log_prob(self, obs, actions):
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        return log_prob


class ValueFunction(nn.Module):
    """Value function V(s)"""
    def __init__(self, obs_dim, hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()
        
        layers = []
        prev_size = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(activation())
            prev_size = h
        layers.append(nn.Linear(prev_size, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class TRPO:
    """
    Trust Region Policy Optimization
    Simplified for AIRL usage
    """
    def __init__(
        self,
        env_fn,
        obs_dim,
        act_dim,
        hidden_sizes=(64, 64),
        max_kl=0.01,
        damping=0.1,
        gamma=0.99,
        lam=0.97,
        vf_lr=1e-3,
        vf_iters=5,
        entropy_weight=0.0,
        use_linear_baseline=False,
        device=torch.device('cpu'),
        seed=0,
    ):
        self.env = env_fn()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_kl = max_kl
        self.damping = damping
        self.gamma = gamma
        self.lam = lam
        self.vf_iters = vf_iters
        self.entropy_weight = entropy_weight
        self.use_linear_baseline = use_linear_baseline
        self.device = device
        
        # Policy and value function
        self.policy = GaussianMLPPolicy(obs_dim, act_dim, hidden_sizes).to(device)
        
        if use_linear_baseline:
            # Use simple linear baseline (more stable)
            from common.linear_baseline import LinearFeatureBaseline
            self.value_fn = LinearFeatureBaseline(obs_dim)
            print("Using LinearFeatureBaseline")
        else:
            # Use neural network baseline
            self.value_fn = ValueFunction(obs_dim, hidden_sizes).to(device)
            self.vf_optimizer = torch.optim.Adam(self.value_fn.parameters(), lr=vf_lr)
            print("Using neural network baseline")
        
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
    
    def get_action(self, obs, deterministic=False):
        """Get action from policy"""
        if not torch.is_tensor(obs):
            obs = torch.FloatTensor(obs).to(self.device)
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        return self.policy.get_action(obs, deterministic)[0]
    
    def collect_trajectories(self, num_steps, reward_fn=None, max_path_length=1000):
        """Collect trajectories using current policy"""
        trajectories = []
        obs_list, act_list, rew_list, done_list = [], [], [], []
        logp_list, val_list = [], []
        
        obs = self.env.reset()
        ep_len = 0
        total_steps = 0
        
        while total_steps < num_steps:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # Get action and log prob
            with torch.no_grad():
                action = self.policy.get_action(obs_tensor, deterministic=False)[0]
                logp = self.policy.get_log_prob(obs_tensor, 
                    torch.FloatTensor(action).unsqueeze(0).to(self.device)).item()
                
                # Get value
                if self.use_linear_baseline:
                    val = self.value_fn.predict(obs.reshape(1, -1))[0]
                else:
                    val = self.value_fn(obs_tensor).item()
            
            # Step environment
            next_obs, env_rew, done, _ = self.env.step(action)
            
            # Use custom reward if provided (for IRL)
            if reward_fn is not None:
                rew = reward_fn(obs)
            else:
                rew = env_rew
            
            obs_list.append(obs)
            act_list.append(action)
            rew_list.append(rew)
            done_list.append(done)
            logp_list.append(logp)
            val_list.append(val)
            
            obs = next_obs
            ep_len += 1
            total_steps += 1
            
            if done or ep_len >= max_path_length:
                trajectories.append({
                    'observations': np.array(obs_list[-ep_len:]),
                    'actions': np.array(act_list[-ep_len:]),
                    'rewards': np.array(rew_list[-ep_len:]),
                    'dones': np.array(done_list[-ep_len:]),
                    'logprobs': np.array(logp_list[-ep_len:]),
                    'values': np.array(val_list[-ep_len:]),
                })
                obs = self.env.reset()
                ep_len = 0
        
        return trajectories
    
    def compute_advantages(self, trajectories):
        """Compute GAE advantages"""
        for traj in trajectories:
            rewards = traj['rewards']
            values = traj['values']
            dones = traj['dones']
            
            advantages = np.zeros_like(rewards)
            last_adv = 0
            
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0
                else:
                    next_value = values[t + 1]
                
                delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
                advantages[t] = last_adv = delta + self.gamma * self.lam * (1 - dones[t]) * last_adv
            
            traj['advantages'] = advantages
            traj['returns'] = advantages + values
    
    def update_value_function(self, trajectories):
        """Update value function"""
        if self.use_linear_baseline:
            # Linear baseline: fit with closed-form solution
            obs = np.concatenate([t['observations'] for t in trajectories])
            returns = np.concatenate([t['returns'] for t in trajectories])
            self.value_fn.fit(obs, returns)
        else:
            # Neural network baseline: gradient descent
            obs = np.concatenate([t['observations'] for t in trajectories])
            returns = np.concatenate([t['returns'] for t in trajectories])
            
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            returns_tensor = torch.FloatTensor(returns).to(self.device)
            
            for _ in range(self.vf_iters):
                values = self.value_fn(obs_tensor)
                vf_loss = ((values - returns_tensor) ** 2).mean()
                
                self.vf_optimizer.zero_grad()
                vf_loss.backward()
                self.vf_optimizer.step()
    
    def update_policy(self, trajectories):
        """Update policy using TRPO"""
        obs = np.concatenate([t['observations'] for t in trajectories])
        acts = np.concatenate([t['actions'] for t in trajectories])
        advs = np.concatenate([t['advantages'] for t in trajectories])
        old_logps = np.concatenate([t['logprobs'] for t in trajectories])
        
        # Normalize advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        acts_tensor = torch.FloatTensor(acts).to(self.device)
        advs_tensor = torch.FloatTensor(advs).to(self.device)
        old_logps_tensor = torch.FloatTensor(old_logps).to(self.device)
        
        # Compute policy gradient
        def get_loss_and_kl():
            logps = self.policy.get_log_prob(obs_tensor, acts_tensor)
            ratio = torch.exp(logps - old_logps_tensor)
            loss = -(ratio * advs_tensor).mean()
            
            # Add entropy bonus
            if self.entropy_weight > 0:
                mean, std = self.policy(obs_tensor)
                entropy = (0.5 * torch.log(2 * np.pi * np.e * std ** 2)).sum(dim=-1).mean()
                loss = loss - self.entropy_weight * entropy
            
            # KL divergence
            with torch.no_grad():
                old_mean, old_std = self.policy(obs_tensor)
            new_mean, new_std = self.policy(obs_tensor)
            
            kl = (torch.log(new_std / old_std) + 
                  (old_std ** 2 + (old_mean - new_mean) ** 2) / (2 * new_std ** 2) - 0.5).sum(dim=-1).mean()
            
            return loss, kl
        
        # Compute gradient
        loss, _ = get_loss_and_kl()
        grads = torch.autograd.grad(loss, self.policy.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        
        # Compute Fisher vector product
        def fisher_vector_product(v):
            kl = self.compute_kl(obs_tensor)
            grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
            flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
            
            kl_v = (flat_grad_kl * v).sum()
            grads = torch.autograd.grad(kl_v, self.policy.parameters())
            flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).detach()
            
            return flat_grad_grad_kl + v * self.damping
        
        # Conjugate gradient
        stepdir = self.conjugate_gradient(fisher_vector_product, -loss_grad)
        
        # Line search
        shs = 0.5 * (stepdir * fisher_vector_product(stepdir)).sum()
        lm = torch.sqrt(shs / self.max_kl)
        fullstep = stepdir / lm
        
        self.line_search(obs_tensor, acts_tensor, advs_tensor, old_logps_tensor, fullstep)
    
    def compute_kl(self, obs):
        """Compute KL divergence"""
        with torch.no_grad():
            old_mean, old_std = self.policy(obs)
        new_mean, new_std = self.policy(obs)
        
        kl = (torch.log(new_std / old_std) + 
              (old_std ** 2 + (old_mean - new_mean) ** 2) / (2 * new_std ** 2) - 0.5).sum(dim=-1).mean()
        return kl
    
    def conjugate_gradient(self, Avp, b, nsteps=10, residual_tol=1e-10):
        """Conjugate gradient algorithm"""
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        
        for _ in range(nsteps):
            Ap = Avp(p)
            alpha = rdotr / torch.dot(p, Ap)
            x += alpha * p
            r -= alpha * Ap
            new_rdotr = torch.dot(r, r)
            if new_rdotr < residual_tol:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x
    
    def line_search(self, obs, acts, advs, old_logps, fullstep, max_backtracks=10):
        """Backtracking line search"""
        old_params = torch.cat([p.data.view(-1) for p in self.policy.parameters()])
        
        def set_params(params):
            idx = 0
            for p in self.policy.parameters():
                size = p.data.numel()
                p.data.copy_(params[idx:idx+size].view_as(p.data))
                idx += size
        
        old_loss = self.compute_policy_loss(obs, acts, advs, old_logps)
        
        for stepfrac in [0.5 ** x for x in range(max_backtracks)]:
            new_params = old_params + stepfrac * fullstep
            set_params(new_params)
            
            new_loss = self.compute_policy_loss(obs, acts, advs, old_logps)
            kl = self.compute_kl(obs)
            
            if new_loss < old_loss and kl < self.max_kl:
                return
        
        set_params(old_params)
    
    def compute_policy_loss(self, obs, acts, advs, old_logps):
        """Compute policy loss with optional entropy bonus"""
        logps = self.policy.get_log_prob(obs, acts)
        ratio = torch.exp(logps - old_logps)
        loss = -(ratio * advs).mean()
        
        # Add entropy bonus if entropy_weight > 0
        if self.entropy_weight > 0:
            mean, std = self.policy(obs)
            entropy = (0.5 * torch.log(2 * np.pi * np.e * std ** 2)).sum(dim=-1).mean()
            loss = loss - self.entropy_weight * entropy
        
        return loss.item()
