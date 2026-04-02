'''
使用 AIRL 学到的奖励函数（BasicRewardNet，r(s,a)）从头训练 SAC 智能体
仿照 train_sac_optimal_with_logging.py
'''
import sys, os, time
import numpy as np
import torch
import gym
from ruamel.yaml import YAML
import pandas as pd

import envs  # 注册自定义环境
from common.sac import ReplayBuffer, SAC
from utils import system, eval

import datetime
import dateutil.tz

# imitation 库的奖励网络
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
import gymnasium.spaces as gym2_spaces
import gym.spaces as gym1_spaces


def make_airl_reward_fn(reward_net, device):
    """把 BasicRewardNet 包装成 r(s, a, s') 的 numpy 函数，供 MujocoFH 使用。"""
    def reward_fn(obs, action, next_obs):
        reward_net.eval()
        with torch.no_grad():
            obs_t = torch.FloatTensor(np.atleast_2d(obs)).to(device)
            act_t = torch.FloatTensor(np.atleast_2d(action)).to(device)
            next_obs_t = torch.FloatTensor(np.atleast_2d(next_obs)).to(device)
            done_t = torch.zeros(obs_t.shape[0], dtype=torch.float32).to(device)
            r = reward_net(obs_t, act_t, next_obs_t, done_t)
        return r.cpu().numpy().flatten()
    return reward_fn


class SACWithLogging(SAC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_log = {
            'timestep': [],
            'real_det_return': [],
            'real_sto_return': [],
            'alpha': []
        }

    def append_progress_row(self, csv_path, row):
        pd.DataFrame([row]).to_csv(
            csv_path,
            mode='a',
            header=not os.path.exists(csv_path),
            index=False,
        )

    def learn_mujoco(self, print_out=False, save_path=None, csv_path=None):
        total_steps = self.steps_per_epoch * self.epochs
        start_time = time.time()
        local_time = time.time()
        o, ep_len = self.env.reset(), 0

        print(f"Training SAC with AIRL reward: Total steps {total_steps:d}")

        for t in range(total_steps):
            if self.replay_buffer.size > self.start_steps:
                a = self.get_action(o)
            else:
                a = self.env.action_space.sample()

            o2, r, d, _ = self.env.step(a)
            ep_len += 1
            d = False if ep_len == self.max_ep_len else d
            self.replay_buffer.store(o, a, r, o2, d)
            o = o2

            if d or ep_len == self.max_ep_len:
                o, ep_len = self.env.reset(), 0

            if self.replay_buffer.size >= self.update_after and t % self.update_every == 0:
                for _ in range(self.update_every):
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    self.update(data=batch)

            if t % self.log_step_interval == 0:
                real_det = eval.evaluate_real_return(
                    self.get_action, self.test_env,
                    self.num_test_episodes, self.max_ep_len, True
                )
                real_sto = eval.evaluate_real_return(
                    self.get_action, self.test_env,
                    self.num_test_episodes, self.max_ep_len, False
                )
                alpha_val = self.alpha.item() if self.automatic_alpha_tuning else self.alpha
                self.training_log['timestep'].append(t + 1)
                self.training_log['real_det_return'].append(real_det)
                self.training_log['real_sto_return'].append(real_sto)
                self.training_log['alpha'].append(alpha_val)
                row = {'timestep': t+1, 'real_det_return': real_det,
                       'real_sto_return': real_sto, 'alpha': alpha_val}
                if csv_path is not None:
                    self.append_progress_row(csv_path, row)
                if print_out:
                    print(f"Timestep: {t+1:d} | "
                          f"Real: Det={real_det:.2f} Sto={real_sto:.2f} | "
                          f"Elapsed {time.time()-local_time:.0f}s")
                local_time = time.time()

        print(f"SAC Training End: time {time.time()-start_time:.0f}s")
        return self.training_log


if __name__ == "__main__":
    yaml = YAML()
    v = yaml.load(open(sys.argv[1]))

    env_name = v['env']['env_name']
    env_T = v['env']['T']
    state_indices = v['env']['state_indices']
    seed = v['seed']

    device = torch.device(f"cuda:{v['cuda']}" if torch.cuda.is_available() and v['cuda'] >= 0 else "cpu")
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(seed)
    pid = os.getpid()

    gym_env = gym.make(env_name, T=env_T)
    state_size = gym_env.observation_space.shape[0]
    action_size = gym_env.action_space.shape[0]
    if state_indices == 'all':
        state_indices = list(range(state_size))

    # ===== 加载 AIRL 奖励网络 =====
    reward_model_path = v['reward']['pretrained']
    assert os.path.exists(reward_model_path), f"奖励模型不存在: {reward_model_path}"
    print(f"加载 AIRL 奖励模型: {reward_model_path}")

    # 构造和训练时相同的 observation/action space（gymnasium 格式）
    obs_space = gym2_spaces.Box(
        low=gym_env.observation_space.low,
        high=gym_env.observation_space.high,
        dtype=np.float32,
    )
    act_space = gym2_spaces.Box(
        low=gym_env.action_space.low,
        high=gym_env.action_space.high,
        dtype=np.float32,
    )

    reward_net = BasicRewardNet(
        obs_space, act_space,
        normalize_input_layer=RunningNorm,
        hid_sizes=(32,),
    ).to(device)
    reward_net.load_state_dict(torch.load(reward_model_path, map_location=device))
    reward_net.eval()
    print("AIRL 奖励模型加载成功")

    airl_reward_fn = make_airl_reward_fn(reward_net, device)

    # ===== 环境 =====
    train_env = gym.make(env_name, T=env_T, r=airl_reward_fn)
    test_env = gym.make(env_name, T=env_T)  # 真实奖励评估
    env_fn = lambda: gym.make(env_name, T=env_T, r=airl_reward_fn)

    # ===== 日志 =====
    exp_id = f"train_sac_optimal_with_logs/{env_name}/airl/{seed}"
    os.makedirs(exp_id, exist_ok=True)
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    log_folder = exp_id + '/' + now.strftime('%Y_%m_%d_%H_%M_%S')
    os.makedirs(log_folder, exist_ok=True)
    os.makedirs(os.path.join(log_folder, 'model'), exist_ok=True)
    print(f"日志目录: {log_folder}")

    os.system(f'cp {sys.argv[0]} {log_folder}')
    os.system(f'cp {sys.argv[1]} {log_folder}/variant_{pid}.yml')

    # ===== SAC =====
    replay_buffer = ReplayBuffer(state_size, action_size, device=device, size=v['sac']['buffer_size'])
    sac_agent = SACWithLogging(
        env_fn, replay_buffer,
        steps_per_epoch=env_T,
        update_after=env_T * v['sac']['random_explore_episodes'],
        max_ep_len=env_T,
        seed=seed,
        start_steps=env_T * v['sac']['random_explore_episodes'],
        reward_state_indices=state_indices,
        device=device,
        **v['sac']
    )
    sac_agent.env = train_env
    sac_agent.test_env = test_env
    sac_agent.test_fn = sac_agent.test_agent_ori_env

    csv_path = os.path.join(log_folder, 'progress.csv')
    training_log = sac_agent.learn_mujoco(print_out=True, csv_path=csv_path)

    pd.DataFrame(training_log).to_csv(csv_path, index=False)
    print(f"最佳真实 Det Return: {max(training_log['real_det_return']):.2f}")
    print(f"日志: {log_folder}")
