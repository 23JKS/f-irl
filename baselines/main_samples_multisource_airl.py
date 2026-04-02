"""
多源 AIRL：共享 reward_model + value_model，每源独立 SAC 与 ReplayBuffer。
AIRL 目标与损失与 baselines/adv_smm.py + baselines/main_samples.py 一致（BCELoss、disc logits、get_reward）。
调度与 AdvSMM.train 对齐：
  - 全局环境步计数 _n_env_steps_total，每步后若 _n_env_steps_total %% num_steps_between_train_calls == 0
    且当前源 replay >= min_steps_before_training，则对当前源执行一次训练（首个成功触发仅 num_initial_disc_iters
    次判别更新，之后与 AdvSMM._do_training 相同）；
  - 训练开始前一次 epoch=0 评估；每个 epoch 结束再评估；
  - 每个源每一 epoch 采集 num_steps_per_epoch 步（与单源每 epoch 单环境步数一致）。

用法:
  export PYTHONPATH=${PWD}:$PYTHONPATH
  python baselines/main_samples_multisource_airl.py configs/baselines/airl_ant_transfer_multisource.yml

专家数据（每源）:
  优先 expert_data/states/{env}_airl.pt 与 expert_data/actions/{env}_airl.pt；
  不存在则用 {env}_1_det.pt（与 train_expert.py 输出一致）。
"""
import copy
import json
import os
import sys

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ruamel.yaml import YAML

import envs  # noqa: F401
from baselines.airl_multisource_core import (
    airl_disc_training_step,
    airl_policy_training_step,
    process_airl_expert_buffer,
)
from baselines.discrim import MLPDisc, ResNetAIRLDisc
from common.sac import ReplayBuffer, SAC
from utils import collect, eval, logger, system

import datetime
import dateutil.tz


def _resolve_expert_paths(env_name):
    st_airl = f"expert_data/states/{env_name}_airl.pt"
    st_det = f"expert_data/states/{env_name}_1_det.pt"
    ac_airl = f"expert_data/actions/{env_name}_airl.pt"
    ac_det = f"expert_data/actions/{env_name}_1_det.pt"
    st_path = st_airl if os.path.exists(st_airl) else st_det
    ac_path = ac_airl if os.path.exists(ac_airl) else ac_det
    if not os.path.exists(st_path):
        raise FileNotFoundError(f"缺少专家状态: {st_airl} 或 {st_det}")
    if not os.path.exists(ac_path):
        raise FileNotFoundError(f"缺少专家动作: {ac_airl} 或 {ac_det}")
    return st_path, ac_path


def _normalize_trajs(trajs, obs_mean, obs_std):
    return (trajs - obs_mean) / obs_std


def _run_training_block(
    spec,
    reward_model,
    value_model,
    disc_optimizer,
    bce,
    bce_targets,
    adv,
    gamma,
    device,
    airl_shaping,
    reward_scale,
    initial_only=False,
):
    """一次「训练触发」：仅初始判别 或 完整 disc+policy 循环。"""
    rb = spec["replay_buffer"]
    ex_obs, ex_act, ex_o2 = spec["expert_tuple"]
    sac = spec["sac_agent"]
    stats = []

    if initial_only:
        for _ in range(int(adv.get("num_initial_disc_iters", 100))):
            stats.append(
                airl_disc_training_step(
                    rb,
                    ex_obs,
                    ex_act,
                    ex_o2,
                    reward_model,
                    value_model,
                    sac,
                    disc_optimizer,
                    bce,
                    bce_targets,
                    adv["disc_optim_batch_size"],
                    gamma,
                    device,
                    adv["use_grad_pen"],
                    adv["grad_pen_weight"],
                )
            )
        return stats

    for _ in range(adv["num_update_loops_per_train_call"]):
        for __ in range(adv["num_disc_updates_per_loop_iter"]):
            stats.append(
                airl_disc_training_step(
                    rb,
                    ex_obs,
                    ex_act,
                    ex_o2,
                    reward_model,
                    value_model,
                    sac,
                    disc_optimizer,
                    bce,
                    bce_targets,
                    adv["disc_optim_batch_size"],
                    gamma,
                    device,
                    adv["use_grad_pen"],
                    adv["grad_pen_weight"],
                )
            )
        for __ in range(adv["num_policy_updates_per_loop_iter"]):
            airl_policy_training_step(
                rb,
                sac,
                adv["policy_optim_batch_size"],
                reward_model,
                value_model,
                gamma,
                reward_scale,
                airl_shaping,
            )
    return stats


if __name__ == "__main__":
    yaml = YAML()
    v = yaml.load(open(sys.argv[1]))

    assert v.get("obj") == "airl", "本脚本仅支持 obj: airl"
    assert v.get("IS") is False

    source_env_names = v.get("source_envs", [v["env"]["env_name"]])
    env_T = v["env"]["T"]
    state_indices = v["env"]["state_indices"]
    seed = v["seed"]
    num_expert_trajs = v["irl"]["expert_episodes"]
    adv = v["adv_irl"]

    device = torch.device(
        f"cuda:{v['cuda']}" if torch.cuda.is_available() and v["cuda"] >= 0 else "cpu"
    )
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(seed)
    pid = os.getpid()

    primary_env = source_env_names[0]
    env_tag = (
        primary_env
        if len(source_env_names) == 1
        else f"multi_{len(source_env_names)}src_{primary_env}"
    )
    exp_id = f"logs/{env_tag}/exp-{num_expert_trajs}/airl-multisource/{seed}"
    os.makedirs(exp_id, exist_ok=True)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    log_folder = exp_id + "/" + now.strftime("%Y_%m_%d_%H_%M_%S")
    logger.configure(dir=log_folder)
    print(f"Logging to directory: {log_folder}")
    os.system(f"cp {sys.argv[0]} {log_folder}/")
    os.system(f"cp baselines/airl_multisource_core.py {log_folder}/")
    os.system(f"cp baselines/adv_smm.py {log_folder}/")
    os.system(f"cp {sys.argv[1]} {log_folder}/variant_{pid}.yml")
    with open(os.path.join(logger.get_dir(), "variant.json"), "w") as f:
        json.dump(v, f, indent=2, sort_keys=True)
    os.makedirs(os.path.join(log_folder, "model"), exist_ok=True)

    state_size, action_size = None, None
    source_specs = []
    for task_id, env_name in enumerate(source_env_names):
        g = gym.make(env_name, T=env_T)
        s, a = g.observation_space.shape[0], g.action_space.shape[0]
        if state_size is None:
            state_size, action_size = s, a
        assert s == state_size and a == action_size, "所有源任务状态/动作维必须相同"
        g.close()
        source_specs.append({"task_id": task_id, "env_name": env_name})

    if state_indices == "all":
        state_indices = list(range(state_size))
    # 与 main_samples + AdvSMM 一致：专家经 [:,:,state_indices] 后维数须与 replay 全维观测相同，才能与 policy 批拼接进同一 reward/log_pi
    assert len(state_indices) == state_size, (
        "len(state_indices) 须等于 observation_space.shape[0]（专家观测维与环境观测维一致；可用 state_indices: all 或等价的全集下标）"
    )

    # 加载原始专家轨迹（未归一化），用于算全局 mean/std
    raw_state_trajs = []
    raw_action_trajs = []
    for spec in source_specs:
        st_path, ac_path = _resolve_expert_paths(spec["env_name"])
        st = torch.load(st_path).numpy()[:, :, state_indices][:num_expert_trajs]
        ac = torch.load(ac_path).numpy()[:num_expert_trajs]
        assert st.shape[0] == ac.shape[0], f"{spec['env_name']} states/actions 条数不一致"
        raw_state_trajs.append(st)
        raw_action_trajs.append(ac)
        print(f"Loaded expert {st_path} {st.shape}, actions {ac_path} {ac.shape}")

    obs_mean = obs_std = None
    if adv.get("normalize", False):
        concat = np.concatenate([t.reshape(-1, t.shape[-1]) for t in raw_state_trajs], axis=0)
        obs_mean = concat.mean(0)
        obs_std = concat.std(0)
        obs_std[obs_std == 0.0] = 1.0
        print("AIRL multisource: normalizing experts + envs with shared obs_mean/std")

    # 构建共享 AIRL 网络（与 main_samples.py 一致，直接用 adv_irl.disc）
    disc_cfg = adv["disc"]
    in_dim = len(state_indices)

    def _make_disc():
        if disc_cfg["model_type"] == "resnet_disc":
            return ResNetAIRLDisc(in_dim, device=device, **disc_cfg)
        if disc_cfg["model_type"] == "mlp_disc":
            return MLPDisc(in_dim, device=device, **disc_cfg)
        raise ValueError(disc_cfg["model_type"])

    reward_model = _make_disc().to(device)
    value_model = copy.deepcopy(reward_model).to(device)
    disc_optimizer = optim.Adam(
        list(reward_model.parameters()) + list(value_model.parameters()),
        lr=adv["disc_lr"],
        betas=(adv.get("disc_momentum", 0.0), 0.999),
    )
    # AdvSMM 在 mode=='airl' 时使用 BCELoss（概率 logits），见 adv_smm.py L119 / disc_forward_airl
    bce = nn.BCELoss().to(device)
    bce_targets = torch.cat(
        [
            torch.ones(adv["disc_optim_batch_size"], 1, device=device),
            torch.zeros(adv["disc_optim_batch_size"], 1, device=device),
        ],
        dim=0,
    )

    gamma = float(adv.get("gamma", 0.99))
    airl_shaping = adv.get("airl_rew_shaping", True)
    # AdvSMM.__init__ 在 airl 下强制 reward_scale=1.0、忽略 rew_clip（L139-141）
    reward_scale = 1.0

    # 每源专家 (obs,act,obs2) 与 env_fn
    for i, spec in enumerate(source_specs):
        st = raw_state_trajs[i]
        ac = raw_action_trajs[i]
        if obs_mean is not None:
            st = _normalize_trajs(st, obs_mean, obs_std)
        ex_obs, ex_act, ex_o2 = process_airl_expert_buffer(st, ac)
        spec["expert_tuple"] = (ex_obs, ex_act, ex_o2)
        en = spec["env_name"]

        def _env_fn(n=en, T=env_T, om=obs_mean, os_=obs_std):
            if om is not None:
                return gym.make(n, T=T, obs_mean=om, obs_std=os_)
            return gym.make(n, T=T)

        spec["env_fn"] = _env_fn

    n_sources = len(source_specs)
    default_steps_per_env = env_T * v["sac"]["epochs"]
    num_steps_per_epoch = int(adv.get("num_steps_per_epoch", default_steps_per_env))
    num_epochs = int(adv.get("num_epochs", v["irl"].get("n_itrs", 400)))
    steps_per_source = max(1, num_steps_per_epoch)
    actual_steps_this_epoch = steps_per_source * n_sources

    min_steps = int(adv["min_steps_before_training"])
    num_steps_between_train_calls = int(adv["num_steps_between_train_calls"])
    assert num_steps_between_train_calls > 0, "adv_irl.num_steps_between_train_calls 必须 > 0"

    # 嵌套函数无法在模块级 __main__ 使用 nonlocal，用单元素列表存可变状态
    max_det, max_sto = [-np.inf], [-np.inf]
    train_trajs = v["irl"]["training_trajs"]
    _n_env_steps_total = 0
    _n_train_steps_total = [0]
    not_done_initial_disc = [True]

    def _init_sac_for_spec(spec):
        env_name = spec["env_name"]
        print(f"[{env_name}] Initializing SAC + replay")
        rb = ReplayBuffer(
            state_size,
            action_size,
            device=device,
            size=adv["replay_buffer_size"],
        )
        sac = SAC(
            spec["env_fn"],
            rb,
            steps_per_epoch=env_T,
            update_after=env_T * v["sac"]["random_explore_episodes"],
            max_ep_len=env_T,
            seed=seed,
            start_steps=env_T * v["sac"]["random_explore_episodes"],
            reward_state_indices=state_indices,
            device=device,
            **v["sac"],
        )
        sac.test_fn = sac.test_agent_ori_env
        spec["replay_buffer"] = rb
        spec["sac_agent"] = sac
        spec["rollout_o"], spec["rollout_ep_len"] = sac.env.reset(), 0

    def _training_mode_airl(sac_agent, mode: bool):
        reward_model.train(mode)
        value_model.train(mode)
        if sac_agent is not None and getattr(sac_agent, "ac", None) is not None:
            sac_agent.ac.train(mode)

    def _try_to_train_advsmm(spec, epoch):
        """与 AdvSMM._try_to_train + _do_training 一致：先 min buffer，再 initial disc 仅一次全局。"""
        rb = spec["replay_buffer"]
        if rb.size < min_steps:
            return
        sac = spec["sac_agent"]
        _training_mode_airl(sac, True)
        if not_done_initial_disc[0]:
            _run_training_block(
                spec,
                reward_model,
                value_model,
                disc_optimizer,
                bce,
                bce_targets,
                adv,
                gamma,
                device,
                airl_shaping,
                reward_scale,
                initial_only=True,
            )
            not_done_initial_disc[0] = False
        else:
            _run_training_block(
                spec,
                reward_model,
                value_model,
                disc_optimizer,
                bce,
                bce_targets,
                adv,
                gamma,
                device,
                airl_shaping,
                reward_scale,
                initial_only=False,
            )
        _training_mode_airl(sac, False)
        _n_train_steps_total[0] += 1

    def _eval_multisource(epoch, env_steps_count, save_best: bool):
        det_list, sto_list = [], []
        for spec in source_specs:
            env_name = spec["env_name"]
            sac = spec["sac_agent"]
            test_env = spec["env_fn"]()
            try:
                det = eval.evaluate_real_return(
                    sac.get_action,
                    test_env,
                    v["irl"]["eval_episodes"],
                    env_T,
                    True,
                )
                sto = eval.evaluate_real_return(
                    sac.get_action,
                    test_env,
                    v["irl"]["eval_episodes"],
                    env_T,
                    False,
                )
            finally:
                test_env.close()
            det_list.append(det)
            sto_list.append(sto)
            print(f"[{env_name}] real det={det:.2f} sto={sto:.2f}")
            logger.record_tabular(f"{env_name} Det Return", round(det, 2))
            logger.record_tabular(f"{env_name} Sto Return", round(sto, 2))

            ex_obs, _, _ = spec["expert_tuple"]
            samp_env = spec["env_fn"]()
            try:
                samples = collect.collect_trajectories_policy_single(
                    samp_env,
                    sac,
                    n=train_trajs,
                    state_indices=np.array(state_indices, dtype=np.int64),
                )
            finally:
                samp_env.close()
            ag_st = samples[0].reshape(-1, samples[0].shape[2])
            eval.KL_summary(
                ex_obs,
                ag_st,
                env_steps_count,
                f"Running-{env_name}",
                v.get("task", {}).get("task_name", "") == "uniform",
            )

        mean_det = float(np.mean(det_list))
        mean_sto = float(np.mean(sto_list))
        logger.record_tabular("epoch", epoch)
        logger.record_tabular("Iteration", epoch)
        upd = _n_train_steps_total[0] * adv["num_update_loops_per_train_call"] * adv[
            "num_disc_updates_per_loop_iter"
        ]
        logger.record_tabular("Running Update Time", upd)
        logger.record_tabular("Running Env Steps", env_steps_count)
        logger.record_tabular("Real Det Return", round(mean_det, 2))
        logger.record_tabular("Real Sto Return", round(mean_sto, 2))

        if save_best and mean_det > max_det[0] and mean_sto > max_sto[0]:
            max_det[0], max_sto[0] = mean_det, mean_sto
            pref = os.path.join(log_folder, "model")
            torch.save(reward_model.state_dict(), os.path.join(pref, "best_reward_model.pkl"))
            torch.save(value_model.state_dict(), os.path.join(pref, "best_value_model.pkl"))
            torch.save(
                {
                    "reward_model": reward_model.state_dict(),
                    "value_model": value_model.state_dict(),
                    "obs_mean": None if obs_mean is None else obs_mean.tolist(),
                    "obs_std": None if obs_std is None else obs_std.tolist(),
                    "state_indices": state_indices,
                    "gamma": gamma,
                    "airl_rew_shaping": airl_shaping,
                    "reward_scale": reward_scale,
                },
                os.path.join(pref, "best_airl_bundle.pt"),
            )
            for spec in source_specs:
                torch.save(
                    spec["sac_agent"].ac.state_dict(),
                    os.path.join(
                        pref,
                        f"policy_{spec['env_name']}_ep{epoch}_det{mean_det:.0f}.pkl",
                    ),
                )
            print(f"  -> saved best AIRL checkpoints (mean det={mean_det:.2f})")

        logger.dump_tabular()

    print(
        f">>> [AIRL Multi-Source] sources={source_env_names}, "
        f"num_epochs={num_epochs}, num_steps_per_epoch(per_env)={num_steps_per_epoch}, "
        f"steps_this_epoch_total={actual_steps_this_epoch}, "
        f"train_trigger=全局步数%{num_steps_between_train_calls}==0 (与 AdvSMM 一致)"
    )

    for spec in source_specs:
        _init_sac_for_spec(spec)

    _eval_multisource(0, 0, save_best=False)

    for epoch in range(1, num_epochs + 1):
        for spec in source_specs:
            env_name = spec["env_name"]
            if v["sac"]["reinitialize"]:
                print(f"[{env_name}] reinitialize=True，本 epoch 重建 SAC")
                _init_sac_for_spec(spec)

            sac = spec["sac_agent"]
            rb = spec["replay_buffer"]

            for _step in range(steps_per_source):
                o = spec["rollout_o"]
                ep_len = spec["rollout_ep_len"]
                a = sac.get_action(o)
                o2, _, d, _ = sac.env.step(a)
                ep_len += 1
                d = False if ep_len == sac.max_ep_len else d
                rb.store(o, a, 0.0, o2, d)
                if d or ep_len == sac.max_ep_len:
                    spec["rollout_o"], spec["rollout_ep_len"] = sac.env.reset(), 0
                else:
                    spec["rollout_o"] = o2
                    spec["rollout_ep_len"] = ep_len

                _n_env_steps_total += 1
                if _n_env_steps_total % num_steps_between_train_calls == 0:
                    _try_to_train_advsmm(spec, epoch)

        _eval_multisource(epoch, _n_env_steps_total, save_best=True)

    print(f"Done. Best mean Real Det={max_det[0]:.2f} Sto={max_sto[0]:.2f}. Log: {log_folder}")
