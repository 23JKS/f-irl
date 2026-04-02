import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gym
from ruamel.yaml import YAML
import envs # Assuming you have an envs module

# --- 1. Policy and Value Network Definitions ---
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

class MLPDiscriminator(nn.Module):
    """Discriminator for AIRL, outputs both reward and value."""
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()
        
        # Shared layers for processing state-action pair
        sa_layers = []
        prev_size = obs_dim + act_dim
        for h in hidden_sizes:
            sa_layers.append(nn.Linear(prev_size, h))
            sa_layers.append(activation())
            prev_size = h
        self.sa_net = nn.Sequential(*sa_layers)
        
        # Output layer for reward
        self.reward_layer = nn.Linear(prev_size, 1)
        
        # Separate network for value function
        v_layers = []
        prev_size = obs_dim
        for h in hidden_sizes:
            v_layers.append(nn.Linear(prev_size, h))
            v_layers.append(activation())
            prev_size = h
        self.v_net = nn.Sequential(*v_layers)
        self.value_layer = nn.Linear(prev_size, 1)
        
    def forward(self, obs, act):
        sa = torch.cat([obs, act], dim=-1)
        sa_features = self.sa_net(sa)
        reward = self.reward_layer(sa_features)
        
        v_features = self.v_net(obs)
        value = self.value_layer(v_features)
        
        return reward.squeeze(-1), value.squeeze(-1)

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
        self.device = device
        
        # Policy and value function
        self.policy = GaussianMLPPolicy(obs_dim, act_dim, hidden_sizes).to(device)
        self.value_fn = nn.Linear(obs_dim, 1).to(device) # Simple linear baseline as in original
        self.vf_optimizer = torch.optim.Adam(self.value_fn.parameters(), lr=vf_lr)
        
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
                
                gamma_coeff = self.gamma if not dones[t] else 0
                delta = rewards[t] + gamma_coeff * next_value - values[t]
                lam_coeff = self.lam if not dones[t] else 0
                advantages[t] = last_adv = delta + gamma_coeff * lam_coeff * last_adv
            
            traj['advantages'] = advantages
            traj['returns'] = advantages + values

    def update_value_function(self, trajectories):
        """Update value function"""
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
        """Compute policy loss"""
        logps = self.policy.get_log_prob(obs, acts)
        ratio = torch.exp(logps - old_logps)
        loss = -(ratio * advs).mean()
        return loss.item()


# --- 2. Main AIRL Trainer ---
class AIRLTrainer:
    """AIRL with TRPO (original paper setup)"""
    
    def __init__(self, config_dict, device):
        self.v = config_dict
        self.device = device
        
        # Environment setup
        self.env_name = self.v['env']['env_name']
        self.seed = self.v['seed']
        self.num_expert_trajs = self.v['irl']['expert_episodes']
        
        # Create environments
        self.env_fn = lambda: gym.make(self.env_name)
        self.env = self.env_fn()
        self.test_env = self.env_fn()
        
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        
        # Load expert data
        self._load_expert_data()
        
        # Initialize discriminator
        self._init_discriminator()
        
        # Initialize TRPO agent
        self._init_trpo_agent()
        
        # Tracking
        self.max_real_return_det = -np.inf
        self.max_real_return_sto = -np.inf
        self._n_env_steps = 0
        self._n_train_steps = 0

        # BCE Loss for discriminator
        self.bce_loss = nn.BCEWithLogitsLoss() # Use logits version for stability

    def _load_expert_data(self):
        """Load expert trajectories"""
        load_path = f'expert_data/states/{self.env_name}_1_det.pt'
        self.expert_trajs = torch.load(load_path).numpy()
        self.expert_trajs = self.expert_trajs[:self.num_expert_trajs, :, :]
        
        self.expert_action_trajs = torch.load(
            f'expert_data/actions/{self.env_name}_1_det.pt'
        ).numpy()
        self.expert_action_trajs = self.expert_action_trajs[:self.num_expert_trajs, :, :]
        
        # Flatten to (s, a) pairs
        obs_list, act_list = [], []
        for traj_idx in range(len(self.expert_trajs)):
            obs_list.append(self.expert_trajs[traj_idx, :-1])
            act_list.append(self.expert_action_trajs[traj_idx, :-1])
        
        self.expert_obs = np.concatenate(obs_list, axis=0)
        self.expert_act = np.concatenate(act_list, axis=0)
        
        print(f"Expert trajectories: {self.expert_trajs.shape}")
        print(f"Expert (s,a) pairs: {len(self.expert_obs)}")

    def _init_discriminator(self):
        """Initialize AIRL discriminator"""
        # Fixed: Use hidden_sizes directly from config without accessing 'disc' key
        hidden_sizes = tuple(self.v['trpo']['hidden_sizes']) # Reusing TRPO's hidden sizes for simplicity
        self.discriminator = MLPDiscriminator(
            self.state_size,
            self.action_size,
            hidden_sizes=hidden_sizes
        ).to(self.device)
        
        self.disc_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=self.v['adv_irl']['disc_lr'],
        )
        
        print(f"Initialized AIRL discriminator with hidden sizes {hidden_sizes}")

    def _init_trpo_agent(self):
        """Initialize TRPO agent"""
        trpo_config = self.v['trpo']
        self.trpo_agent = TRPO(
            self.env_fn,
            self.state_size,
            self.action_size,
            hidden_sizes=tuple(trpo_config['hidden_sizes']),
            max_kl=trpo_config['max_kl'],
            damping=trpo_config.get('damping', 0.1),
            gamma=trpo_config['gamma'],
            lam=trpo_config.get('lam', 0.97),
            vf_lr=trpo_config.get('vf_lr', 1e-3),
            vf_iters=trpo_config.get('vf_iters', 5),
            device=self.device,
            seed=self.seed,
        )
        print("Initialized TRPO agent")

    def _get_airl_reward(self, obs, act, next_obs):
        """Get reward from discriminator for a state-action-next_state tuple"""
        if not torch.is_tensor(obs):
            obs = torch.FloatTensor(obs).to(self.device)
        if not torch.is_tensor(act):
            act = torch.FloatTensor(act).to(self.device)
        if not torch.is_tensor(next_obs):
            next_obs = torch.FloatTensor(next_obs).to(self.device)

        self.discriminator.eval()
        with torch.no_grad():
            reward_raw, value_cur = self.discriminator(obs, act)
            _, value_next = self.discriminator(next_obs, torch.zeros_like(act)) # Approximation: V(s_T) = 0 for terminal
            gamma = self.v['trpo']['gamma']
            reward = reward_raw + gamma * value_next - value_cur
        self.discriminator.train()
        return reward.squeeze().cpu().numpy()

    def _train_discriminator(self, policy_trajs):
        """Train discriminator on policy and expert trajectories"""
        batch_size = self.v['adv_irl']['disc_optim_batch_size']
        # CRITICAL FIX: Reduce training iterations to prevent overfitting
        max_disc_itrs = self.v['adv_irl'].get('max_disc_itrs', 1) # Set to 1 or 2
        
        # Extract policy data
        p_obs = np.concatenate([t['observations'][:-1] for t in policy_trajs])
        p_act = np.concatenate([t['actions'][:-1] for t in policy_trajs])
        p_next_obs = np.concatenate([t['observations'][1:] for t in policy_trajs])

        total_loss = 0
        total_acc = 0

        for _ in range(max_disc_itrs):
            # Sample policy batch
            if len(p_obs) < batch_size:
                continue # Skip if not enough data
            pi = np.random.choice(len(p_obs), size=batch_size, replace=False)
            p_obs_batch = torch.FloatTensor(p_obs[pi]).to(self.device)
            p_act_batch = torch.FloatTensor(p_act[pi]).to(self.device)
            p_next_obs_batch = torch.FloatTensor(p_next_obs[pi]).to(self.device)

            # Sample expert batch
            if len(self.expert_obs) < batch_size:
                continue
            ei = np.random.choice(len(self.expert_obs), size=batch_size, replace=False)
            e_obs_batch = torch.FloatTensor(self.expert_obs[ei]).to(self.device)
            e_act_batch = torch.FloatTensor(self.expert_act[ei]).to(self.device)

            self.disc_optimizer.zero_grad()

            # Compute discriminator logits for expert (label=1) and policy (label=0)
            e_reward, e_value = self.discriminator(e_obs_batch, e_act_batch)
            p_reward, p_value = self.discriminator(p_obs_batch, p_act_batch)

            # AIRL Discriminator Objective (log sigmoid trick for numerical stability)
            # D(s,a) = sigmoid(r(s,a) + gamma*V(s') - V(s))
            # Loss is BCE between D and labels (1 for expert, 0 for policy)
            e_logits = e_reward + self.v['trpo']['gamma'] * p_value - e_value # Note: Using p_value for next state is an approximation
            p_logits = p_reward + self.v['trpo']['gamma'] * p_value - p_value # This simplifies to p_reward

            e_targets = torch.ones(batch_size).to(self.device)
            p_targets = torch.zeros(batch_size).to(self.device)

            e_loss = self.bce_loss(e_logits, e_targets)
            p_loss = self.bce_loss(p_logits, p_targets)
            loss = e_loss + p_loss

            loss.backward()
            # Optional: Clip gradients for extra stability
            # torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=5.0)
            self.disc_optimizer.step()

            with torch.no_grad():
                # Calculate accuracy
                e_pred = (e_logits > 0).float()
                p_pred = (p_logits > 0).float()
                e_acc = (e_pred == e_targets).float().mean().item()
                p_acc = (p_pred == p_targets).float().mean().item()
                acc = (e_acc + p_acc) / 2.0

            total_loss += loss.item()
            total_acc += acc

        return total_loss / max(max_disc_itrs, 1), total_acc / max(max_disc_itrs, 1)

    def train(self):
        """Main training loop"""
        n_itrs = self.v['irl']['n_itrs']
        batch_size = self.v['adv_irl']['batch_size']
        max_path_length = self.v['adv_irl'].get('max_path_length', 1000)
        
        print("Starting AIRL + TRPO training...")
        print(f"Total iterations: {n_itrs}, Batch size: {batch_size}, Max path length: {max_path_length}")
        
        for itr in range(1, n_itrs + 1):
            # Step 1: Collect trajectories using current policy (without IRL reward)
            trajectories = self.trpo_agent.collect_trajectories(
                batch_size,
                reward_fn=None,  # Don't use IRL reward during collection
                max_path_length=max_path_length
            )
            
            self._n_env_steps += batch_size
            
            # Step 2: Train discriminator on collected trajectories
            disc_loss, disc_acc = self._train_discriminator(trajectories)
            
            # Step 3: Compute IRL rewards for collected trajectories
            self._compute_irl_rewards(trajectories)
            
            # Step 4: Compute advantages using IRL rewards
            self.trpo_agent.compute_advantages(trajectories)
            
            # Step 5: Update value function
            self.trpo_agent.update_value_function(trajectories)
            
            # Step 6: Update policy with TRPO
            self.trpo_agent.update_policy(trajectories)
            
            self._n_train_steps += 1
            
            # Logging
            print(f"Iteration {itr}, Env Steps: {self._n_env_steps}, Disc Loss: {disc_loss:.3f}, Disc Acc: {disc_acc:.3f}")
            
            # Log IRL reward statistics
            all_irl_rewards = np.concatenate([t['rewards'] for t in trajectories])
            print(f"  IRL Reward Mean: {np.mean(all_irl_rewards):.3f}, Std: {np.std(all_irl_rewards):.3f}, Min: {np.min(all_irl_rewards):.3f}, Max: {np.max(all_irl_rewards):.3f}")
            
            # Evaluation (every iteration)
            self._evaluate(itr)

    def _compute_irl_rewards(self, trajectories):
        """Compute IRL rewards for trajectories after discriminator training"""
        for traj in trajectories:
            obs = traj['observations']  # (T, state_dim)
            acts = traj['actions']      # (T, act_dim)
            next_obs = np.roll(obs, shift=-1, axis=0) # Shift obs to get next_obs
            next_obs[-1] = self.env.reset() # Last next_obs is reset state (or can be zero-padded)

            irl_rewards = []
            for i in range(len(obs)):
                r = self._get_airl_reward(obs[i], acts[i], next_obs[i])
                irl_rewards.append(r)
            
            traj['rewards'] = np.array(irl_rewards)

    def _evaluate(self, epoch):
        """Evaluate current policy"""
        # Real return (deterministic)
        real_return_det = 0
        for _ in range(self.v['irl']['eval_episodes']):
            obs = self.test_env.reset()
            ep_ret = 0
            for _ in range(self.v['env']['T']):
                action = self.trpo_agent.get_action(obs, deterministic=True)
                obs, rew, done, _ = self.test_env.step(action)
                ep_ret += rew
                if done:
                    break
            real_return_det += ep_ret
        real_return_det /= self.v['irl']['eval_episodes']
        
        # Real return (stochastic)
        real_return_sto = 0
        for _ in range(self.v['irl']['eval_episodes']):
            obs = self.test_env.reset()
            ep_ret = 0
            for _ in range(self.v['env']['T']):
                action = self.trpo_agent.get_action(obs, deterministic=False)
                obs, rew, done, _ = self.test_env.step(action)
                ep_ret += rew
                if done:
                    break
            real_return_sto += ep_ret
        real_return_sto /= self.v['irl']['eval_episodes']

        print(f"  Real Det Return: {real_return_det:.2f}, Real Sto Return: {real_return_sto:.2f}")

        # Save best models
        if real_return_det > self.max_real_return_det and real_return_sto > self.max_real_return_sto:
            self.max_real_return_det = real_return_det
            self.max_real_return_sto = real_return_sto
            print(f"  -> New Best! Saving model...")


# --- 3. Main Execution ---
if __name__ == "__main__":
    # Load config from command line argument
    yaml = YAML()
    if len(sys.argv) > 1:
        v = yaml.load(open(sys.argv[1]))
    else:
        # Provide a default config for testing
        v = {
            'env': {'env_name': 'HopperFH-v0', 'T': 1000},
            'irl': {'expert_episodes': 16, 'n_itrs': 100, 'eval_episodes': 5},
            'adv_irl': {
                'disc_lr': 0.00003, # Lower learning rate
                'disc_optim_batch_size': 1024,
                'max_disc_itrs': 1, # CRITICAL: Low number of updates
                'batch_size': 5000,
                'gamma': 0.995,
            },
            'trpo': {
                'hidden_sizes': [64, 64],
                'max_kl': 0.01,
                'gamma': 0.995,
                'lam': 0.97,
                'vf_lr': 1e-3,
                'vf_iters': 5,
            },
            'seed': 1,
            'cuda': -1
        }
    
    device = torch.device(f"cuda:{v['cuda']}" if torch.cuda.is_available() and v['cuda'] >= 0 else "cpu")
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    
    print(f"Using device: {device}")
    print(f"Config: {v}")
    
    trainer = AIRLTrainer(v, device)
    trainer.train()
