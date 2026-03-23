'''
f-IRL 多源任务版本，用于与 f-IRL+VAE 公平对比。
结构与 firl/irl_samples.py 完全一致，扩展点仅在于：
  - 每个 IRL 迭代在所有源任务上各跑一次 SAC
  - 每个源任务独立一个 Disc（状态分布不同，不能共享）
  - 对每个源任务分别算 f_div_disc_loss，再平均后更新奖励函数

用法:
  conda run -n roer python firl/irl_samples_multisource.py configs/firl_ant_transfer_multisource.yml
'''
import sys, os, time
import numpy as np
import torch
import gym
from ruamel.yaml import YAML

from firl.divs.f_div_disc import f_div_disc_loss
from firl.divs.f_div import maxentirl_loss
from firl.divs.ipm import ipm_loss
from firl.models.reward import MLPReward
from firl.models.discrim import SMMIRLDisc as Disc
from firl.models.discrim import SMMIRLCritic as Critic
from common.sac import ReplayBuffer, SAC

import envs
from utils import system, collect, logger, eval

import datetime
import dateutil.tz
import json, copy


if __name__ == "__main__":
    yaml = YAML()
    v = yaml.load(open(sys.argv[1]))

    # 解析源任务列表
    source_env_names = v.get('source_envs', [v['env']['env_name']])
    state_indices = v['env']['state_indices']
    seed = v['seed']
    num_expert_trajs = v['irl']['expert_episodes']

    device = torch.device(f"cuda:{v['cuda']}" if torch.cuda.is_available() and v['cuda'] >= 0 else "cpu")
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(seed)
    pid = os.getpid()

    assert v['obj'] in ['fkl', 'rkl', 'js', 'emd', 'maxentirl']
    assert v['IS'] == False

    # 日志目录，结构与原版一致
    primary_env = source_env_names[0]
    env_tag = primary_env if len(source_env_names) == 1 \
        else f"multi_{len(source_env_names)}src_{primary_env}"
    exp_id = f"logs/{env_tag}/exp-{num_expert_trajs}/{v['obj']}/{seed}"
    os.makedirs(exp_id, exist_ok=True)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    log_folder = exp_id + '/' + now.strftime('%Y_%m_%d_%H_%M_%S')
    logger.configure(dir=log_folder)
    print(f"Logging to directory: {log_folder}")
    os.system(f'cp {sys.argv[0]} {log_folder}')
    os.system(f'cp {sys.argv[1]} {log_folder}/variant_{pid}.yml')
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(v, f, indent=2, sort_keys=True)
    print('pid', pid)
    os.makedirs(os.path.join(log_folder, 'plt'))
    os.makedirs(os.path.join(log_folder, 'model'))

    # 初始化各源任务环境，校验状态/动作维度一致
    state_size, action_size = None, None
    source_specs = []
    for task_id, env_name in enumerate(source_env_names):
        gym_env = gym.make(env_name)
        s = gym_env.observation_space.shape[0]
        a = gym_env.action_space.shape[0]
        if state_size is None:
            state_size, action_size = s, a
        assert s == state_size and a == action_size, \
            "所有源任务必须有相同的状态/动作维度"
        source_specs.append({
            'task_id': task_id,
            'env_name': env_name,
            'gym_env': gym_env,
            'sac_agent': None,
        })

    if state_indices == 'all':
        state_indices = list(range(state_size))

    # 加载各源任务专家数据，与原版一致：优先找 {env}_1_sto.pt，fallback 到 {env}.pt
    source_expert_trajs = []
    for spec in source_specs:
        env_name = spec['env_name']
        preferred = f'expert_data/states/{env_name}_1_det.pt'
        fallback  = f'expert_data/states/{env_name}.pt'
        path = preferred if os.path.exists(preferred) else fallback
        trajs = torch.load(path).numpy()[:, :, state_indices]
        trajs = trajs[:num_expert_trajs]
        print(f"Loaded expert data from {path}: {trajs.shape}")
        source_expert_trajs.append(trajs)

    # 初始化奖励函数，与原版完全一致
    reward_func = MLPReward(len(state_indices), **v['reward'], device=device).to(device)
    pretrained_reward = v['reward'].get('pretrained', None)
    if pretrained_reward and os.path.exists(pretrained_reward):
        reward_func.load_state_dict(torch.load(pretrained_reward, map_location=device))
        print(f"Loaded pretrained reward model from: {pretrained_reward}")
    reward_optimizer = torch.optim.Adam(
        reward_func.parameters(),
        lr=v['reward']['lr'],
        weight_decay=v['reward']['weight_decay'],
        betas=(v['reward']['momentum'], 0.999),
    )

    # 每个源任务独立一个 Disc/Critic
    # 原因：两个任务状态分布不同，共享判别器会混淆专家/agent 边界
    discs, critics = [], []
    for _ in source_specs:
        if v['obj'] in ['emd']:
            critics.append(Critic(len(state_indices), **v['critic'], device=device))
            discs.append(None)
        elif v['obj'] != 'maxentirl':
            discs.append(Disc(len(state_indices), **v['disc'], device=device))
            critics.append(None)
        else:
            discs.append(None)
            critics.append(None)

    anchor_lambda = v['reward'].get('anchor_lambda', 0.0)
    max_real_return_det, max_real_return_sto = -np.inf, -np.inf

    print(f">>> [f-IRL Multi-Source] obj={v['obj']}, sources={source_env_names}")

    for itr in range(v['irl']['n_itrs']):

        # Trust region snapshot，与原版一致
        old_reward_params = {k: p.detach().clone() for k, p in reward_func.named_parameters()}

        # ===== 步骤1: 每个源任务各跑一次 SAC，与原版 sac_agent.learn_mujoco 完全一致 =====
        source_samples = []
        for spec in source_specs:
            env_name = spec['env_name']
            env_fn = lambda n=env_name: gym.make(n)

            if v['sac']['reinitialize'] or itr == 0 or spec['sac_agent'] is None:
                print("Reinitializing sac")
                replay_buffer = ReplayBuffer(
                    state_size, action_size,
                    device=device, size=v['sac']['buffer_size'],
                )
                sac_agent = SAC(
                    env_fn, replay_buffer,
                    steps_per_epoch=v['env']['T'],
                    update_after=v['env']['T'] * v['sac']['random_explore_episodes'],
                    max_ep_len=v['env']['T'],
                    seed=seed,
                    start_steps=v['env']['T'] * v['sac']['random_explore_episodes'],
                    reward_state_indices=state_indices,
                    device=device,
                    **v['sac'],
                )
                pretrained_policy = v['sac'].get('pretrained', None)
                if itr == 0 and pretrained_policy and os.path.exists(pretrained_policy):
                    sac_agent.ac.load_state_dict(torch.load(pretrained_policy, map_location=device))
                    sac_agent.ac_targ = copy.deepcopy(sac_agent.ac)
                    for p in sac_agent.ac_targ.parameters():
                        p.requires_grad = False
                    print(f"Loaded pretrained policy from: {pretrained_policy}")
                spec['sac_agent'] = sac_agent

            spec['sac_agent'].reward_function = reward_func.get_scalar_reward
            sac_info = spec['sac_agent'].learn_mujoco(print_out=True)

            # collect，与原版一致
            start = time.time()
            samples = collect.collect_trajectories_policy_single(
                spec['gym_env'], spec['sac_agent'],
                n=v['irl']['training_trajs'], state_indices=state_indices,
            )
            agent_emp_states = samples[0].copy().reshape(-1, samples[0].shape[2])
            print(f"[{env_name}] collect trajs {time.time() - start:.0f}s", flush=True)
            source_samples.append((samples, agent_emp_states))

        # ===== 步骤2: 每个源任务独立训练 disc/critic，与原版一致 =====
        start = time.time()
        for i, (spec, (samples, agent_emp_states), expert_trajs) in enumerate(
            zip(source_specs, source_samples, source_expert_trajs)
        ):
            expert_flat = expert_trajs.reshape(-1, len(state_indices))  # (N*T, d)，与原版一致
            if v['obj'] in ['emd']:
                critics[i].learn(expert_flat, agent_emp_states, iter=v['critic']['iter'])
            elif v['obj'] != 'maxentirl':
                discs[i].learn(expert_flat, agent_emp_states, iter=v['disc']['iter'])
        print(f'train disc {time.time() - start:.0f}s', flush=True)

        # ===== 步骤3: 对每个源任务分别算 loss，平均后更新奖励函数 =====
        # 不直接拼接 samples：f_div_disc_loss 的协方差计算假设轨迹来自同一分布
        reward_losses = []
        for _ in range(v['reward']['gradient_step']):
            total_loss = torch.tensor(0.0, device=device)

            for i, (spec, (samples, _), expert_trajs) in enumerate(
                zip(source_specs, source_samples, source_expert_trajs)
            ):
                # resample expert trajs，与原版完全一致
                if v['irl']['resample_episodes'] > v['irl']['expert_episodes']:
                    idx = np.random.choice(expert_trajs.shape[0], v['irl']['resample_episodes'], replace=True)
                    expert_trajs_train = expert_trajs[idx].copy()
                elif v['irl']['resample_episodes'] > 0:
                    idx = np.random.choice(expert_trajs.shape[0], v['irl']['resample_episodes'], replace=False)
                    expert_trajs_train = expert_trajs[idx].copy()
                else:
                    expert_trajs_train = None

                if v['obj'] in ['fkl', 'rkl', 'js']:
                    loss, _ = f_div_disc_loss(
                        v['obj'], v['IS'], samples, discs[i], reward_func, device,
                        expert_trajs=expert_trajs_train,
                    )
                elif v['obj'] == 'maxentirl':
                    expert_flat = expert_trajs.reshape(-1, len(state_indices))
                    loss = maxentirl_loss(v['obj'], samples, expert_flat, reward_func, device)
                elif v['obj'] == 'emd':
                    loss, _ = ipm_loss(
                        v['obj'], v['IS'], samples, critics[i].value, reward_func, device,
                        expert_trajs=expert_trajs_train,
                    )

                total_loss = total_loss + loss

            total_loss = total_loss / len(source_specs)  # 平均，保持梯度尺度与单源一致

            # Trust region，与原版一致
            if anchor_lambda > 0:
                anchor_loss = sum(
                    ((p - old_reward_params[k]) ** 2).sum()
                    for k, p in reward_func.named_parameters()
                )
                total_loss = total_loss + anchor_lambda * anchor_loss

            reward_losses.append(total_loss.item())
            print(f"{v['obj']} loss: {total_loss}")
            reward_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(reward_func.parameters(), max_norm=1.0)
            reward_optimizer.step()

        # ===== 步骤4: 评估，对每个源任务分别评估后取平均，与原版 try_evaluate 对齐 =====
        det_returns, sto_returns = [], []
        for spec in source_specs:
            env_fn_eval = lambda n=spec['env_name']: gym.make(n)
            det = eval.evaluate_real_return(
                spec['sac_agent'].get_action, env_fn_eval(),
                v['irl']['eval_episodes'], v['env']['T'], True,
            )
            sto = eval.evaluate_real_return(
                spec['sac_agent'].get_action, env_fn_eval(),
                v['irl']['eval_episodes'], v['env']['T'], False,
            )
            det_returns.append(det)
            sto_returns.append(sto)
            print(f"[{spec['env_name']}] real det return: {det:.2f}")
            logger.record_tabular(f"{spec['env_name']} Det Return", round(det, 2))
            logger.record_tabular(f"{spec['env_name']} Sto Return", round(sto, 2))

        real_return_det = float(np.mean(det_returns))
        real_return_sto = float(np.mean(sto_returns))
        print(f"real det return avg: {real_return_det:.2f}")
        logger.record_tabular("Real Det Return", round(real_return_det, 2))
        logger.record_tabular("Real Sto Return", round(real_return_sto, 2))

        if real_return_det > max_real_return_det and real_return_sto > max_real_return_sto:
            max_real_return_det, max_real_return_sto = real_return_det, real_return_sto
            best_suffix = f"itr{itr}_det{max_real_return_det:.0f}_sto{max_real_return_sto:.0f}"
            torch.save(
                reward_func.state_dict(),
                os.path.join(logger.get_dir(), f"model/reward_model_{best_suffix}.pkl"),
            )
            torch.save(
                reward_func.state_dict(),
                os.path.join(logger.get_dir(), "model/best_reward.pkl"),
            )
            # 保存每个源任务的最优策略
            for spec in source_specs:
                torch.save(
                    spec['sac_agent'].ac.state_dict(),
                    os.path.join(logger.get_dir(), f"model/policy_{spec['env_name']}_{best_suffix}.pkl"),
                )

        logger.record_tabular("Itration", itr)
        logger.record_tabular("Reward Loss", total_loss.item())
        if v['sac']['automatic_alpha_tuning']:
            logger.record_tabular("alpha", source_specs[0]['sac_agent'].alpha.item())

        logger.dump_tabular()
