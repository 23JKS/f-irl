"""
AIRL + PPO (使用 imitation 库)
"""
import sys, os, json, datetime
import gym
import torch
import numpy as np
from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout
from imitation.data.types import Transitions
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.util import util
from stable_baselines3 import PPO
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
import dateutil.tz
from ruamel.yaml import YAML
from utils import logger, eval as eval_utils

# ========== 配置 ==========
yaml = YAML()
if len(sys.argv) > 1:
    v = yaml.load(open(sys.argv[1]))
else:
    raise RuntimeError("请提供配置文件路径")

env_name = v['env']['env_name']
num_expert_trajs = v['irl']['expert_episodes']
seed = v['seed']
obj = v.get('obj', 'airl')

# 日志目录
exp_id = f"logs/{env_name}/exp-{num_expert_trajs}/{obj}/{seed}"
os.makedirs(exp_id, exist_ok=True)
now = datetime.datetime.now(dateutil.tz.tzlocal())
log_folder = exp_id + '/' + now.strftime('%Y_%m_%d_%H_%M_%S')
os.makedirs(log_folder)
os.makedirs(os.path.join(log_folder, 'model'))
logger.configure(dir=log_folder)
print(f"Logging to: {log_folder}")

with open(os.path.join(log_folder, 'variant.json'), 'w') as f:
    json.dump(v, f, indent=2, sort_keys=True)
os.system(f'cp baselines/airl_ppo.py {log_folder}')
os.system(f'cp {sys.argv[1]} {log_folder}/variant_{os.getpid()}.yml')

# ========== 1. 环境 ==========
import envs

# Fix: your gym env returns bool as info, imitation needs dict
class InfoDictWrapper(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if not isinstance(info, dict):
            info = {}
        return obs, rew, done, info

env = gym.make(env_name)
state_indices = list(range(env.observation_space.shape[0]))

# ========== 2. 专家数据 ==========
states_path = f"expert_data/states/{env_name}_1_det.pt"
actions_path = f"expert_data/actions/{env_name}_1_det.pt"

if not os.path.exists(states_path):
    raise FileNotFoundError(f"专家数据不存在: {states_path}")

states = torch.load(states_path).numpy()[:num_expert_trajs]   # (N, T, obs_dim)
actions = torch.load(actions_path).numpy()[:num_expert_trajs] # (N, T, act_dim)
print(f"Expert states: {states.shape}, actions: {actions.shape}")

# 转换为 imitation 库的 Transitions 格式
obs_list, act_list, next_obs_list, done_list = [], [], [], []
for i in range(len(states)):
    T = states.shape[1]
    obs_list.append(states[i, :-1])       # (T-1, obs_dim)
    act_list.append(actions[i, :-1])      # (T-1, act_dim)
    next_obs_list.append(states[i, 1:])   # (T-1, obs_dim)
    dones = np.zeros(T - 1, dtype=bool)
    dones[-1] = True
    done_list.append(dones)

expert_transitions = Transitions(
    obs=np.concatenate(obs_list, axis=0).astype(np.float32),
    acts=np.concatenate(act_list, axis=0).astype(np.float32),
    next_obs=np.concatenate(next_obs_list, axis=0).astype(np.float32),
    dones=np.concatenate(done_list, axis=0),
    infos=np.array([{}] * (len(states) * (states.shape[1] - 1))),
)
print(f"Expert transitions: {len(expert_transitions)}")

# ========== 3. 初始化 AIRL ==========
venv = DummyVecEnv([lambda: RolloutInfoWrapper(InfoDictWrapper(gym.make(env_name)))])

gen_algo = PPO(
    "MlpPolicy",
    venv,
    verbose=0,
    n_steps=v['adv_irl'].get('steps_per_epoch', v['env']['T']),
    batch_size=v['adv_irl']['disc_optim_batch_size'],
    learning_rate=v['sac']['lr'],
    ent_coef=v['sac']['alpha'],
    seed=seed,
)

reward_net = BasicShapedRewardNet(
    observation_space=venv.observation_space,
    action_space=venv.action_space,
    normalize_input_layer=RunningNorm,
)

airl_trainer = AIRL(
    venv=venv,
    demonstrations=expert_transitions,
    demo_batch_size=v['adv_irl']['disc_optim_batch_size'],
    gen_algo=gen_algo,
    reward_net=reward_net,
    n_disc_updates_per_round=v['adv_irl']['num_disc_updates_per_loop_iter'],
    log_dir=log_folder,
    allow_variable_horizon=True,
)
print("AIRL initialized")

# ========== 4. 训练主循环 ==========
print("--- 开始训练 AIRL + PPO ---")
max_real_return_det = -np.inf
max_real_return_sto = -np.inf
num_epochs = v['adv_irl']['num_epochs']
steps_per_epoch = v['adv_irl'].get('steps_per_epoch', v['env']['T'])

for epoch in range(1, num_epochs + 1):
    # 训练一轮
    airl_trainer.train(total_timesteps=steps_per_epoch)

    # 评估真实回报（确定性）
    policy_fn = lambda obs, det: airl_trainer.policy.predict(obs, deterministic=det)[0]
    
    real_return_det = eval_utils.evaluate_real_return(
        policy_fn,
        env,
        v['irl']['eval_episodes'],
        v['env']['T'],
        True
    )

    # 评估真实回报（随机性）
    real_return_sto = eval_utils.evaluate_real_return(
        policy_fn,
        env,
        v['irl']['eval_episodes'],
        v['env']['T'],
        False
    )

    logger.record_tabular("Epoch", epoch)
    logger.record_tabular("Env Steps", epoch * steps_per_epoch)
    logger.record_tabular("Real Det Return", round(real_return_det, 2))
    logger.record_tabular("Real Sto Return", round(real_return_sto, 2))
    logger.dump_tabular()

    # 保存最优模型
    if real_return_det > max_real_return_det and real_return_sto > max_real_return_sto:
        max_real_return_det = real_return_det
        max_real_return_sto = real_return_sto
        torch.save(
            airl_trainer._reward_net.state_dict(),
            os.path.join(log_folder, f"model/best_reward_det{real_return_det:.0f}_sto{real_return_sto:.0f}.pt")
        )
        airl_trainer.policy.save(
            os.path.join(log_folder, f"model/best_policy_det{real_return_det:.0f}_sto{real_return_sto:.0f}")
        )

print("--- 训练完成！---")
