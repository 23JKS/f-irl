'''
使用 AIRL 学到的奖励函数，用 PPO 从头训练智能体
奖励加载和使用方式对齐 imitation/scripts/train_adversarial.py:
  - torch.load 加载完整 RewardNet 对象（reward_test.pt）
  - RewardVecEnvWrapper 替换环境奖励，调用 predict_processed
保存在 train_sac_optimal_with_logs/{env_name}/airl_ppo/{seed}
'''
import sys, os, time
import numpy as np
import torch
import gym
from ruamel.yaml import YAML
import pandas as pd

import envs  # 注册自定义环境
from utils import system

import datetime
import dateutil.tz

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from imitation.policies.base import NormalizeFeaturesExtractor
from imitation.util.networks import RunningNorm
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper


def evaluate_real_return(policy, env_fn, n_episodes, horizon, deterministic):
    returns = []
    for _ in range(n_episodes):
        env = env_fn()
        obs = env.reset()
        ret = 0.0
        for _ in range(horizon):
            action, _ = policy.predict(obs[None], deterministic=deterministic)
            obs, r, done, _ = env.step(action[0])
            ret += r
            if done:
                break
        returns.append(ret)
        env.close()
    return float(np.mean(returns))


if __name__ == "__main__":
    yaml = YAML()
    v = yaml.load(open(sys.argv[1]))

    env_name = v['env']['env_name']
    env_T = v['env']['T']
    seed = v['seed']
    device = torch.device(f"cuda:{v['cuda']}" if torch.cuda.is_available() and v['cuda'] >= 0 else "cpu")
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(seed)
    pid = os.getpid()

    # ===== 加载 AIRL 奖励网络 =====
    # 支持两种格式：
    #   .pt  — torch.save(reward_net_object)，直接加载完整对象
    #   .pkl — torch.save(state_dict)，需要先构建网络再 load_state_dict
    reward_model_path = v['reward']['pretrained']
    assert os.path.exists(reward_model_path), f"奖励模型不存在: {reward_model_path}"
    print(f"加载 AIRL 奖励模型: {reward_model_path}")

    tmp_env = gym.make(env_name, T=env_T)
    import gymnasium.spaces as gym2_spaces
    from imitation.rewards.reward_nets import BasicRewardNet
    from imitation.util.networks import RunningNorm as RunningNormCls

    if reward_model_path.endswith('.pkl'):
        # state_dict 格式，需要手动构建网络
        obs_space = gym2_spaces.Box(low=tmp_env.observation_space.low,
                                    high=tmp_env.observation_space.high, dtype=np.float32)
        act_space = gym2_spaces.Box(low=tmp_env.action_space.low,
                                    high=tmp_env.action_space.high, dtype=np.float32)
        reward_net = BasicRewardNet(obs_space, act_space,
                                    normalize_input_layer=RunningNormCls,
                                    hid_sizes=(32,)).to(device)
        reward_net.load_state_dict(torch.load(reward_model_path, map_location=device,
                                              weights_only=False))
    else:
        # 完整对象格式（reward_test.pt）
        reward_net = torch.load(reward_model_path, map_location=device, weights_only=False)
        reward_net.to(device)

    tmp_env.close()
    reward_net.eval()
    print(f"奖励模型类型: {reward_net.__class__.__name__}, 加载成功")

    # reward_fn 对齐 train_adversarial 中 RewardVecEnvWrapper 的调用方式：
    # reward_train.predict_processed(obs, act, next_obs, done)
    def reward_fn(obs: np.ndarray, act: np.ndarray,
                  next_obs: np.ndarray, done: np.ndarray) -> np.ndarray:
        return reward_net.predict_processed(obs, act, next_obs, done)

    # ===== 日志 =====
    exp_id = f"train_sac_optimal_with_logs/{env_name}/airl_ppo/{seed}"
    os.makedirs(exp_id, exist_ok=True)
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    log_folder = exp_id + '/' + now.strftime('%Y_%m_%d_%H_%M_%S')
    os.makedirs(log_folder, exist_ok=True)
    os.makedirs(os.path.join(log_folder, 'model'), exist_ok=True)
    print(f"日志目录: {log_folder}")
    os.system(f'cp {sys.argv[0]} {log_folder}')
    os.system(f'cp {sys.argv[1]} {log_folder}/variant_{pid}.yml')

    # ===== 训练环境 =====
    def make_train_env():
        e = gym.make(env_name, T=env_T)
        return Monitor(e)

    venv = DummyVecEnv([make_train_env])
    # 用 RewardVecEnvWrapper 替换奖励，对齐 train_adversarial 的做法
    venv = RewardVecEnvWrapper(venv, reward_fn)

    normalize_reward = v.get('normalize_reward', False)
    if normalize_reward:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True)
        print("已启用奖励归一化 (VecNormalize)")

    # ===== PPO =====
    ppo_cfg = v['ppo']
    policy_cfg = v.get('policy', {})
    activation_map = {'Tanh': torch.nn.Tanh, 'ReLU': torch.nn.ReLU}
    activation_fn = activation_map.get(policy_cfg.get('activation', 'ReLU'), torch.nn.ReLU)
    net_arch_size = policy_cfg.get('net_arch', [64, 64])

    policy_kwargs = dict(
        features_extractor_class=NormalizeFeaturesExtractor,
        features_extractor_kwargs=dict(normalize_class=RunningNorm),
        net_arch=dict(pi=net_arch_size, vf=net_arch_size),
        activation_fn=activation_fn,
    )
    model = PPO(
        "MlpPolicy",
        venv,
        n_steps=ppo_cfg['n_steps'],
        batch_size=ppo_cfg['batch_size'],
        n_epochs=ppo_cfg['n_epochs'],
        learning_rate=ppo_cfg['learning_rate'],
        clip_range=ppo_cfg['clip_range'],
        ent_coef=ppo_cfg['ent_coef'],
        gae_lambda=ppo_cfg['gae_lambda'],
        gamma=ppo_cfg['gamma'],
        max_grad_norm=ppo_cfg['max_grad_norm'],
        vf_coef=ppo_cfg['vf_coef'],
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=0,
    )

    # ===== 评估环境（真实奖励）=====
    real_env_fn = lambda: gym.make(env_name, T=env_T)
    n_eval = v.get('n_eval_episodes', 10)
    eval_interval = v.get('eval_interval', 8192)
    total_timesteps = v['total_timesteps']

    # ===== 训练循环 =====
    csv_path = os.path.join(log_folder, 'progress.csv')
    rows = []
    t = 0
    start_time = time.time()
    print(f"Training PPO with AIRL reward: Total steps {total_timesteps}")

    while t < total_timesteps:
        model.learn(total_timesteps=eval_interval, reset_num_timesteps=False)
        t += eval_interval

        real_det = evaluate_real_return(model.policy, real_env_fn, n_eval, env_T, True)
        real_sto = evaluate_real_return(model.policy, real_env_fn, n_eval, env_T, False)
        row = {'timestep': t, 'real_det_return': real_det, 'real_sto_return': real_sto}
        rows.append(row)
        pd.DataFrame([row]).to_csv(
            csv_path, mode='a',
            header=not os.path.exists(csv_path) or t == eval_interval,
            index=False,
        )
        print(f"Timestep: {t} | Real: Det={real_det:.2f} Sto={real_sto:.2f} | Elapsed {time.time()-start_time:.0f}s")

    model.save(os.path.join(log_folder, 'model', 'final_policy'))
    print(f"最佳真实 Det Return: {max(r['real_det_return'] for r in rows):.2f}")
    print(f"日志: {log_folder}")
