'''
使用启用ROER机制的SAC训练脚本（带日志记录）
基于 train_sac_optimal_with_logging.py，替换为 common.sac_roer 中的实现
'''
import sys, os, time
import numpy as np
import torch
import gym
from ruamel.yaml import YAML
import pandas as pd

from firl.models.reward import MLPReward
from common.sac_roer import ReplayBuffer, ROERReplayBuffer, SAC

import envs
from utils import system, eval

import datetime
import dateutil.tz


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
        best_eval = -np.inf
        o, ep_len = self.env.reset(), 0

        print(f"Training SAC (ROER): Total steps {total_steps:d}")

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

            # 完整的 ROER 更新逻辑（与 sac_roer.SAC.learn_mujoco 保持一致）
            should_update = False
            if self.reinitialize:
                if t >= self.update_after and t % self.update_every == 0:
                    should_update = True
            else:
                if self.replay_buffer.size >= self.update_after and t % self.update_every == 0:
                    should_update = True

            if should_update:
                for j in range(self.update_every):
                    # 步骤 1: 采样
                    batch = self.replay_buffer.sample_batch(self.batch_size)

                    # 步骤 2: train_env 已通过 r=reward_func.get_scalar_reward 在交互时计算奖励
                    # replay buffer 里存的 rew 已经是学到的奖励，无需 relabeling
                    rew_for_v = batch['rew']

                    indices = batch.get('indices', None)
                    if indices is not None:
                        indices = indices.cpu().numpy()

                    # 步骤 3: 更新 V 网络（ROER ExtremeV loss）
                    if self.enable_roer_priority:
                        value_loss = self.compute_extreme_v_loss(batch['obs'], batch['act'])
                        self.value_optimizer.zero_grad()
                        value_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=10.0)
                        self.value_optimizer.step()

                        # 步骤 4: 计算 TD error 用于优先级更新
                        with torch.no_grad():
                            next_v = self.value_net(batch['obs2'])
                            target_v = rew_for_v + (1 - batch['done']) * self.gamma * next_v
                            current_v = self.value_net(batch['obs'])
                            td_error = target_v - current_v
                    else:
                        batch['weights'] = torch.ones_like(batch['rew'])

                    # 步骤 5: SAC 更新（带 importance weighting）
                    self.update(data=batch)

                    # 步骤 6: 更新优先级
                    if self.enable_roer_priority and indices is not None:
                        td_err_np = td_error.detach().cpu().numpy()
                        if td_err_np.ndim > 1:
                            td_err_np = td_err_np.flatten()
                        old_p_np = batch['raw_priorities'].detach().cpu().numpy()
                        new_priorities = self.calculate_priorities(td_err_np, old_p_np)
                        self.replay_buffer.update_priorities(indices, new_priorities)

            # 评估
            if t % self.log_step_interval == 0:
                real_det = eval.evaluate_real_return(
                    self.get_action, self.test_env,
                    self.num_test_episodes, self.max_ep_len, True
                )
                real_sto = eval.evaluate_real_return(
                    self.get_action, self.test_env,
                    self.num_test_episodes, self.max_ep_len, False
                )

                self.training_log['timestep'].append(t + 1)
                self.training_log['real_det_return'].append(real_det)
                self.training_log['real_sto_return'].append(real_sto)
                self.training_log['alpha'].append(
                    self.alpha.item() if self.automatic_alpha_tuning else self.alpha
                )
                row = {
                    'timestep': t + 1,
                    'real_det_return': real_det,
                    'real_sto_return': real_sto,
                    'alpha': self.alpha.item() if self.automatic_alpha_tuning else self.alpha,
                }
                if csv_path is not None:
                    self.append_progress_row(csv_path, row)

                # 保存最优模型
                if save_path is not None and real_det > best_eval:
                    best_eval = real_det
                    torch.save(self.ac.state_dict(), save_path)
                    if print_out:
                        print(f"  -> 保存最优模型 (det_return={real_det:.2f})")

                if print_out:
                    print(f"Timestep: {t+1:d} | "
                          f"Real: Det={real_det:.2f} Sto={real_sto:.2f} | "
                          f"Elapsed {time.time() - local_time:.0f}s")

                local_time = time.time()

        print(f"SAC (ROER) Training End: time {time.time() - start_time:.0f}s")
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

    env_fn = lambda: gym.make(env_name)
    gym_env = env_fn()
    state_size = gym_env.observation_space.shape[0]
    action_size = gym_env.action_space.shape[0]

    if state_indices == 'all':
        state_indices = list(range(state_size))

    print(f"Environment: {env_name}")
    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Reward state indices: {state_indices}")

    pretrained_reward = v['reward'].get('pretrained', None)
    if pretrained_reward is None or not os.path.exists(pretrained_reward):
        raise ValueError(f"必须提供有效的预训练奖励模型路径! 当前路径: {pretrained_reward}")

    print(f"\n{'='*60}")
    print(f"加载预训练奖励模型: {pretrained_reward}")
    print(f"{'='*60}\n")

    reward_func = MLPReward(len(state_indices), **v['reward'], device=device).to(device)
    reward_func.load_state_dict(torch.load(pretrained_reward, map_location=device))
    reward_func.eval()
    print("奖励模型加载成功!\n")

    train_env = gym.make(env_name, T=env_T, r=reward_func.get_scalar_reward)
    test_env = gym.make(env_name, T=env_T)

    print("训练环境：使用学到的奖励函数")
    print("测试环境：使用真实环境奖励\n")

    exp_id = f"train_sac_roer_with_logs/{env_name}/sac_roer_reward/{seed}"
    os.makedirs(exp_id, exist_ok=True)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    log_folder = exp_id + '/' + now.strftime('%Y_%m_%d_%H_%M_%S')
    os.makedirs(log_folder, exist_ok=True)
    os.makedirs(os.path.join(log_folder, 'model'), exist_ok=True)

    print(f"日志目录: {log_folder}\n")

    os.system(f'cp {sys.argv[0]} {log_folder}')
    os.system(f'cp {sys.argv[1]} {log_folder}/variant_{pid}.yml')

    print(f"{'='*60}")
    print("初始化SAC（ROER）智能体")
    print(f"{'='*60}\n")

    # 选择合适的 ReplayBuffer：启用 roer 优先级时使用 ROERReplayBuffer
    use_roer = v['sac'].get('enable_roer_priority', True)
    if use_roer:
        replay_buffer = ROERReplayBuffer(
            state_size, action_size,
            device=device,
            size=v['sac']['buffer_size'],
            lambda_=v['sac'].get('roer', {}).get('lambda_', 0.01),
            beta=v['sac'].get('roer', {}).get('beta', 1.0),
            clip_min=v['sac'].get('roer', {}).get('clip_min', 1e-4),
            clip_max=v['sac'].get('roer', {}).get('clip_max', 50.0),
        )
    else:
        replay_buffer = ReplayBuffer(
            state_size, action_size,
            device=device,
            size=v['sac']['buffer_size']
        )

    sac_agent = SACWithLogging(
        env_fn, replay_buffer,
        steps_per_epoch=env_T,
        update_after=env_T * v['sac']['random_explore_episodes'],
        max_ep_len=env_T,
        seed=seed,
        start_steps=env_T * v['sac']['random_explore_episodes'],
        device=device,
        **v['sac']
    )

    sac_agent.env = train_env
    sac_agent.test_env = test_env
    sac_agent.test_fn = sac_agent.test_agent

    print(f"{'='*60}")
    print("开始训练SAC（ROER）")
    print(f"总训练步数: {v['sac']['epochs'] * env_T}")
    print(f"{'='*60}\n")

    model_save_path = os.path.join(log_folder, 'model', 'best_policy.pth')
    csv_path = os.path.join(log_folder, 'progress.csv')
    training_log = sac_agent.learn_mujoco(
        print_out=True,
        save_path=model_save_path,
        csv_path=csv_path,
    )

    # append_progress_row 已在训练中实时写入，无需再次覆盖
    # 仅在 csv 不存在时（如训练异常中断后重跑）才做兜底写入
    if not os.path.exists(csv_path):
        pd.DataFrame(training_log).to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print("训练完成!")
    if len(training_log['real_det_return']) > 0:
        print(f"最佳真实环境回报: {max(training_log['real_det_return']):.2f}")
    print(f"日志目录: {log_folder}")
    print(f"模型保存: {model_save_path}")
    print(f"训练日志: {csv_path}")
    print(f"{'='*60}\n")
