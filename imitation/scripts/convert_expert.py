"""
把 f-IRL 格式的专家数据转成 imitation 库需要的 Trajectory 格式
运行：python imitation/scripts/convert_expert_antfh.py
"""
import numpy as np
import torch
import pickle
from imitation.data.types import TrajectoryWithRew

def convert(
    states_path: str,
    actions_path: str,
    output_path: str,
    n_trajs: int = 16,
):
    states = torch.load(states_path).numpy()[:n_trajs]   # (N, T, obs_dim)
    actions = torch.load(actions_path).numpy()[:n_trajs] # (N, T, act_dim)

    trajs = []
    for i in range(len(states)):
        obs = states[i]    # (T, obs_dim)
        acts = actions[i]  # (T, act_dim)

        # imitation Trajectory: obs 长度比 acts 多 1（包含终止状态）
        n_acts = len(acts) - 1
        traj = TrajectoryWithRew(
            obs=obs.astype(np.float32),              # (T, obs_dim)
            acts=acts[:-1].astype(np.float32),       # (T-1, act_dim)
            rews=np.zeros(n_acts, dtype=np.float32), # placeholder，rollout_stats 需要
            infos=None,
            terminal=True,
        )
        trajs.append(traj)

    with open(output_path, 'wb') as f:
        pickle.dump(trajs, f)

    print(f"Saved {len(trajs)} trajectories to {output_path}")
    print(f"  obs shape: {trajs[0].obs.shape}")
    print(f"  acts shape: {trajs[0].acts.shape}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="AntFH-v0")
    parser.add_argument("--n_trajs", type=int, default=16)
    parser.add_argument("--suffix", type=str, default="_airl",
                        help="文件后缀，如 _airl / _1_det / _1_sto")
    args = parser.parse_args()

    convert(
        states_path=f"expert_data/states/{args.env}{args.suffix}.pt",
        actions_path=f"expert_data/actions/{args.env}{args.suffix}.pt",
        output_path=f"expert_data/imitation/{args.env}{args.suffix}_trajs.pkl",
        n_trajs=args.n_trajs,
    )
