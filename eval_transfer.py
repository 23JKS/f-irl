"""
奖励迁移评估脚本
用法:
  # 评估 f-IRL 迁移
  conda run -n roer python eval_transfer.py \
    --method firl \
    --reward_path logs/.../model/reward_model_xxx.pkl \
    --target_env AntLeg02Disabled-v0 \
    --seed 1

  # 评估 f-IRL+VAE 迁移
  conda run -n roer python eval_transfer.py \
    --method trairl \
    --reward_path logs/.../model/best_reward.pkl \
    --vae_path logs/.../model/best_vae.pkl \
    --target_env AntLeg02Disabled-v0 \
    --seed 1
"""
import argparse, os, sys, time
import numpy as np
import torch
import gym
import pandas as pd
import datetime
import dateutil.tz

from firl.models.reward import MLPReward
from common.sac import ReplayBuffer, SAC
import envs
from utils import system, eval


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['firl', 'trairl'], required=True)
    p.add_argument('--reward_path', required=True)
    p.add_argument('--vae_path', default=None, help='仅 trairl 需要')
    p.add_argument('--target_env', default='AntLeg02Disabled-v0')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--cuda', type=int, default=-1)
    p.add_argument('--T', type=int, default=500)
    p.add_argument('--epochs', type=int, default=600, help='SAC训练epoch数（总步数=epochs*T）')
    p.add_argument('--latent_dim', type=int, default=16, help='VAE潜空间维度（trairl用）')
    p.add_argument('--log_interval', type=int, default=5000)
    return p.parse_args()


def build_firl_reward_fn(reward_path, state_indices, device):
    """加载 f-IRL 奖励函数，直接在原始状态空间工作"""
    reward_func = MLPReward(
        len(state_indices),
        use_bn=False, residual=False, hid_act='relu',
        hidden_sizes=[128, 128], clamp_magnitude=10,
        lr=0.0001, weight_decay=0.001, gradient_step=1,
        momentum=0.9, anchor_lambda=0.0,
        device=device,
    ).to(device)
    reward_func.load_state_dict(torch.load(reward_path, map_location=device))
    reward_func.eval()

    def reward_fn(obs):
        with torch.no_grad():
            if not torch.is_tensor(obs):
                obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
            return reward_func.get_scalar_reward(obs)
    return reward_fn


def build_trairl_reward_fn(reward_path, vae_path, state_indices, latent_dim, device):
    """加载 f-IRL+VAE 奖励函数，需要先过 VAE 编码器"""
    # 动态导入 VAE（避免循环依赖）
    sys.path.insert(0, os.path.dirname(__file__))
    from firl__vae.irl_samples import MultiHeadVAE

    state_dim = len(state_indices)
    # 加载 VAE（num_tasks 不影响 encoder，只影响 decoder，迁移时只用 encoder）
    vae = MultiHeadVAE(
        state_dim=state_dim,
        latent_dim=latent_dim,
        num_tasks=2,  # 训练时的源任务数，encoder 权重与 num_tasks 无关
        hidden_size=256,
        device=device,
    )
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    vae.eval()

    reward_func = MLPReward(
        latent_dim,
        use_bn=False, residual=False, hid_act='relu',
        hidden_sizes=[128, 128], clamp_magnitude=10,
        lr=0.0001, weight_decay=0.001, gradient_step=1,
        momentum=0.9, anchor_lambda=0.0,
        device=device,
    ).to(device)
    reward_func.load_state_dict(torch.load(reward_path, map_location=device))
    reward_func.eval()

    def reward_fn(obs):
        with torch.no_grad():
            if not torch.is_tensor(obs):
                obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            z = vae.get_z(obs)
            return reward_func.get_scalar_reward(z)
    return reward_fn


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() and args.cuda >= 0 else 'cpu')
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(args.seed)

    env_fn = lambda: gym.make(args.target_env, T=args.T)
    gym_env = env_fn()
    state_size = gym_env.observation_space.shape[0]
    action_size = gym_env.action_space.shape[0]
    state_indices = list(range(state_size))

    # 构建奖励函数
    if args.method == 'firl':
        reward_fn = build_firl_reward_fn(args.reward_path, state_indices, device)
    else:
        assert args.vae_path is not None, '--vae_path 是 trairl 必须的参数'
        reward_fn = build_trairl_reward_fn(
            args.reward_path, args.vae_path, state_indices, args.latent_dim, device
        )

    # 用迁移的奖励函数训练目标任务上的 SAC
    train_env = gym.make(args.target_env, T=args.T, r=reward_fn)
    test_env = gym.make(args.target_env, T=args.T)

    replay_buffer = ReplayBuffer(state_size, action_size, device=device, size=1_000_000)
    sac_agent = SAC(
        env_fn, replay_buffer,
        steps_per_epoch=args.T,
        update_after=args.T * 1,
        max_ep_len=args.T,
        seed=args.seed,
        start_steps=args.T * 1,
        reward_state_indices=state_indices,
        device=device,
        epochs=args.epochs,
        log_step_interval=args.log_interval,
        update_every=1,
        random_explore_episodes=1,
        update_num=1,
        batch_size=100,
        lr=0.001,
        alpha=0.2,
        automatic_alpha_tuning=False,
        buffer_size=1_000_000,
        num_test_episodes=20,
        reinitialize=True,
        k=1,
    )
    sac_agent.env = train_env
    sac_agent.test_env = test_env
    sac_agent.test_fn = sac_agent.test_agent_ori_env

    # 日志目录
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    log_dir = f"transfer_logs/{args.target_env}/{args.method}/seed{args.seed}/{now.strftime('%Y_%m_%d_%H_%M_%S')}"
    os.makedirs(log_dir, exist_ok=True)

    print(f"[Transfer Eval] method={args.method}, target={args.target_env}")
    print(f"[Transfer Eval] reward_path={args.reward_path}")
    print(f"[Transfer Eval] log_dir={log_dir}")

    # 训练并记录
    rows = []
    local_time = time.time()
    total_steps = args.epochs * args.T
    best_det = -np.inf
    o, ep_len = sac_agent.env.reset(), 0

    for t in range(total_steps):
        if sac_agent.replay_buffer.size > sac_agent.start_steps:
            a = sac_agent.get_action(o)
        else:
            a = sac_agent.env.action_space.sample()
        o2, r, d, _ = sac_agent.env.step(a)
        ep_len += 1
        d = False if ep_len == args.T else d
        sac_agent.replay_buffer.store(o, a, r, o2, d)
        o = o2
        if d or ep_len == args.T:
            o, ep_len = sac_agent.env.reset(), 0

        if t >= sac_agent.update_after and t % sac_agent.update_every == 0:
            for _ in range(sac_agent.update_every):
                batch = sac_agent.replay_buffer.sample_batch(sac_agent.batch_size)
                sac_agent.update(data=batch)

        if t % args.log_interval == 0:
            det = eval.evaluate_real_return(sac_agent.get_action, test_env, 20, args.T, True)
            sto = eval.evaluate_real_return(sac_agent.get_action, test_env, 20, args.T, False)
            elapsed = time.time() - local_time
            print(f"Timestep: {t+1} | Det={det:.2f} Sto={sto:.2f} | {elapsed:.0f}s")
            rows.append({'timestep': t+1, 'real_det_return': det, 'real_sto_return': sto})
            local_time = time.time()
            if det > best_det:
                best_det = det
                torch.save(sac_agent.ac.state_dict(), os.path.join(log_dir, 'best_policy.pth'))

    csv_path = os.path.join(log_dir, 'progress.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n[Transfer Eval Done] Best Det Return: {best_det:.2f}")
    print(f"Log: {csv_path}")
