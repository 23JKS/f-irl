"""
在目标环境上从零训练 SAC：使用 AIRL 保存的 reward_model + value_model，
逐步奖励：r(s)+γV(s')-V(s)（airl_rew_shaping 为 true 时）。

bundle 中含 obs_mean/obs_std 时，训练与测试环境均传入 gym.make（CustomAntEnv/MujocoFH），
与多源 AIRL 专家、replay 观测分布一致；SAC 的 batch 奖励与 env 逐步奖励一致。
airl.gamma / reward_scale / airl_rew_shaping 可由 yml 覆盖 bundle 默认值。

用法:
  export PYTHONPATH=${PWD}:$PYTHONPATH
  python train_sac_airl_transfer.py configs/baselines/sac_airl_transfer_ant.yml
"""
import datetime
import json
import os
import sys
import time

import dateutil.tz
import gym
import numpy as np
import pandas as pd
import torch
from ruamel.yaml import YAML

_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from baselines.airl_multisource_core import get_airl_reward_batch
from baselines.discrim import MLPDisc, ResNetAIRLDisc
from common.sac import ReplayBuffer, SAC

import envs  # noqa: F401
from utils import eval, system


class SACAIRLTransferWithLogging(SAC):
    """reinitialize=False 时用 obs/obs2 重算 AIRL 塑形奖励。"""

    def __init__(self, *args, airl_reward_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.airl_reward_fn = airl_reward_fn
        self.training_log = {
            "timestep": [],
            "real_det_return": [],
            "real_sto_return": [],
            "alpha": [],
        }

    def append_progress_row(self, csv_path, row):
        pd.DataFrame([row]).to_csv(
            csv_path,
            mode="a",
            header=not os.path.exists(csv_path),
            index=False,
        )

    def learn_mujoco(self, print_out=False, save_path=None, csv_path=None):
        total_steps = self.steps_per_epoch * self.epochs
        start_time = time.time()
        local_time = time.time()
        o, ep_len = self.env.reset(), 0

        print(f"Training SAC (AIRL transfer): Total steps {total_steps:d}")

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

            if self.reinitialize:
                if t >= self.update_after and t % self.update_every == 0:
                    for _ in range(self.update_every):
                        batch = self.replay_buffer.sample_batch(self.batch_size)
                        self.update(data=batch)
            else:
                if self.replay_buffer.size >= self.update_after and t % self.update_every == 0:
                    for _ in range(self.update_every):
                        batch = self.replay_buffer.sample_batch(self.batch_size)
                        batch["rew"] = self.airl_reward_fn(
                            batch["obs"], batch["obs2"]
                        )
                        self.update(data=batch)

            if t % self.log_step_interval == 0:
                real_det = eval.evaluate_real_return(
                    self.get_action,
                    self.test_env,
                    self.num_test_episodes,
                    self.max_ep_len,
                    True,
                )
                real_sto = eval.evaluate_real_return(
                    self.get_action,
                    self.test_env,
                    self.num_test_episodes,
                    self.max_ep_len,
                    False,
                )
                alp = (
                    self.alpha.item()
                    if self.automatic_alpha_tuning
                    else self.alpha
                )
                self.training_log["timestep"].append(t + 1)
                self.training_log["real_det_return"].append(real_det)
                self.training_log["real_sto_return"].append(real_sto)
                self.training_log["alpha"].append(alp)
                row = {
                    "timestep": t + 1,
                    "real_det_return": real_det,
                    "real_sto_return": real_sto,
                    "alpha": alp,
                }
                if csv_path is not None:
                    self.append_progress_row(csv_path, row)
                if print_out:
                    print(
                        f"Timestep: {t+1:d} | Real: Det={real_det:.2f} Sto={real_sto:.2f} | "
                        f"Elapsed {time.time() - local_time:.0f}s"
                    )
                local_time = time.time()

        print(f"SAC Training End: time {time.time() - start_time:.0f}s")
        return self.training_log


def _resolve_state_indices(state_indices, state_size):
    if state_indices == "all" or state_indices is None:
        return list(range(state_size))
    return list(state_indices)


def build_airl_nets(in_dim, disc_cfg, device):
    if disc_cfg["model_type"] == "resnet_disc":
        return ResNetAIRLDisc(in_dim, device=device, **disc_cfg).to(device)
    if disc_cfg["model_type"] == "mlp_disc":
        return MLPDisc(in_dim, device=device, **disc_cfg).to(device)
    raise ValueError(disc_cfg["model_type"])


def main():
    yaml = YAML()
    v = yaml.load(open(sys.argv[1]))

    env_name = v["env"]["env_name"]
    env_T = v["env"]["T"]
    state_indices = v["env"]["state_indices"]
    seed = v["seed"]
    device = torch.device(
        f"cuda:{v['cuda']}" if torch.cuda.is_available() and v["cuda"] >= 0 else "cpu"
    )
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(seed)
    pid = os.getpid()

    airl_cfg = v.get("airl", {})
    bundle_path = airl_cfg.get("bundle")
    rw_path = airl_cfg.get("reward_model")
    val_path = airl_cfg.get("value_model")

    env_probe = gym.make(env_name, T=env_T)
    state_size = env_probe.observation_space.shape[0]
    action_size = env_probe.action_space.shape[0]
    env_probe.close()

    state_indices_list = _resolve_state_indices(state_indices, state_size)
    in_dim = len(state_indices_list)

    def _bundle_str_set(p):
        return p is not None and str(p).strip() and str(p).lower() not in ("none", "null")

    if _bundle_str_set(bundle_path) and not os.path.exists(bundle_path):
        raise FileNotFoundError(
            f"airl.bundle 已填写但文件不存在: {bundle_path}\n"
            "请指向多源 AIRL 训练日志目录下的 model/best_airl_bundle.pt"
        )

    obs_mean = obs_std = None
    if _bundle_str_set(bundle_path) and os.path.exists(bundle_path):
        bundle = torch.load(bundle_path, map_location="cpu")
        gamma = float(airl_cfg.get("gamma", bundle.get("gamma", 0.99)))
        reward_scale = float(airl_cfg.get("reward_scale", bundle.get("reward_scale", 1.0)))
        if "airl_rew_shaping" in airl_cfg:
            airl_shaping = bool(airl_cfg["airl_rew_shaping"])
        else:
            airl_shaping = bool(bundle.get("airl_rew_shaping", True))
        if bundle.get("obs_mean") is not None:
            obs_mean = np.array(bundle["obs_mean"], dtype=np.float32)
            obs_std = np.array(bundle["obs_std"], dtype=np.float32)
        rw_sd = bundle["reward_model"]
        val_sd = bundle["value_model"]
    else:
        if not rw_path or not val_path:
            raise ValueError(
                "未配置 AIRL 权重。请在 configs/.../sac_airl_transfer_ant.yml 的 airl 段填写其一：\n"
                "  1) bundle: 指向训练生成的 .../model/best_airl_bundle.pt（python baselines/main_samples_multisource_airl.py）\n"
                "  2) 或同时填写 reward_model、value_model 两个 .pkl 的绝对路径\n"
                f"当前 bundle={bundle_path!r}, reward_model={rw_path!r}, value_model={val_path!r}"
            )
        if not os.path.exists(rw_path) or not os.path.exists(val_path):
            raise ValueError("reward_model / value_model 路径无效")
        gamma = float(airl_cfg.get("gamma", 0.99))
        reward_scale = float(airl_cfg.get("reward_scale", 1.0))
        airl_shaping = bool(airl_cfg.get("airl_rew_shaping", True))
        rw_sd = torch.load(rw_path, map_location="cpu")
        val_sd = torch.load(val_path, map_location="cpu")
    disc_cfg = v["adv_irl"]["disc"]
    reward_model = build_airl_nets(in_dim, disc_cfg, device)
    value_model = build_airl_nets(in_dim, disc_cfg, device)
    reward_model.load_state_dict(rw_sd)
    value_model.load_state_dict(val_sd)
    reward_model.eval()
    value_model.eval()

    def env_step_reward(s_np, s2_np):
        """s,s' 已由环境（CustomAntEnv/MujocoFH）归一化时与专家、bundle 训练一致，此处直接送网络。"""
        x = np.asarray(s_np, dtype=np.float32).reshape(1, -1)
        x2 = np.asarray(s2_np, dtype=np.float32).reshape(1, -1)
        o = torch.as_tensor(x, device=device)
        o2 = torch.as_tensor(x2, device=device)
        with torch.no_grad():
            r = get_airl_reward_batch(
                o,
                o2,
                reward_model,
                value_model,
                gamma,
                reward_scale,
                airl_shaping,
            )
        return float(r.reshape(-1).cpu().item())

    def batch_airl_reward(obs_t, obs2_t):
        with torch.no_grad():
            return get_airl_reward_batch(
                obs_t,
                obs2_t,
                reward_model,
                value_model,
                gamma,
                reward_scale,
                airl_shaping,
            )

    def _make_env(with_reward_cb: bool):
        kw = dict(T=env_T)
        if obs_mean is not None:
            kw["obs_mean"] = obs_mean
            kw["obs_std"] = obs_std
        if with_reward_cb:
            kw["r"] = env_step_reward
        return gym.make(env_name, **kw)

    train_env = _make_env(with_reward_cb=True)
    test_env = _make_env(with_reward_cb=False)

    exp_id = f"train_sac_airl_transfer/{env_name}/sac/{seed}"
    os.makedirs(exp_id, exist_ok=True)
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    log_folder = exp_id + "/" + now.strftime("%Y_%m_%d_%H_%M_%S")
    os.makedirs(log_folder, exist_ok=True)
    os.makedirs(os.path.join(log_folder, "model"), exist_ok=True)
    print(f"目标环境: {env_name}, T={env_T}, gamma={gamma}, shaping={airl_shaping}")
    print(f"日志目录: {log_folder}")

    os.system(f"cp {sys.argv[0]} {log_folder}/")
    os.system(f"cp {sys.argv[1]} {log_folder}/variant_{pid}.yml")
    with open(os.path.join(log_folder, "variant.json"), "w") as f:
        json.dump(v, f, indent=2, sort_keys=True)

    env_fn = lambda: _make_env(with_reward_cb=False)
    replay_buffer = ReplayBuffer(
        state_size,
        action_size,
        device=device,
        size=v["sac"]["buffer_size"],
    )

    sac_agent = SACAIRLTransferWithLogging(
        env_fn,
        replay_buffer,
        steps_per_epoch=env_T,
        update_after=env_T * v["sac"]["random_explore_episodes"],
        max_ep_len=env_T,
        seed=seed,
        start_steps=env_T * v["sac"]["random_explore_episodes"],
        reward_state_indices=state_indices_list,
        device=device,
        airl_reward_fn=batch_airl_reward,
        **v["sac"],
    )
    sac_agent.env = train_env
    sac_agent.test_env = test_env
    sac_agent.test_fn = sac_agent.test_agent_ori_env

    csv_path = os.path.join(log_folder, "progress.csv")
    model_path = os.path.join(log_folder, "model", "best_policy.pth")
    log = sac_agent.learn_mujoco(print_out=True, save_path=model_path, csv_path=csv_path)
    pd.DataFrame(log).to_csv(csv_path, index=False)
    print(f"完成。最佳 Det: {max(log['real_det_return']):.2f} | {log_folder}")


if __name__ == "__main__":
    main()
