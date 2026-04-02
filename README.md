# f-IRL：基于f散度的逆强化学习

## 学习奖励函数
<img src="Ant.png" width="200" /> <img src="Ant.png" width="200" />

## 用学的奖励函数来训练智能体
<table>
  <tr>
    <td><img src="real_sto_return_mean_std_no_smooth.png" width="200" height="150"/></td>
    <td><img src="real_sto_return_mean_std_no_smooth.png" width="200" height="150"/></td>
    <td><img src="real_sto_return_mean_std_no_smooth.png" width="200" height="150"/></td>
  </tr>
</table>

## 快速开始
运行时首先执行
```bash
export PYTHONPATH=${PWD}:$PYTHONPATH
```
1. 收集专家数据

```bash
python common/train_expert.py configs/samples/experts/hopper.yml
```

2. 学习奖励函数
```bash
python common/train_expert.py configs/samples/experts/hopper.yml
```

3. 用学得的奖励函数从头训练智能体
```bash
python train_sac_airl_with_logging.py configs/sac_airl_reward_hopper.yml

python /home/qu/f-IRL/train_ppo_airl_with_logging.py configs/ppo_airl_reward_ant.yml

 python imitation/scripts/train_rl.py with total_timesteps=3000000 reward_type=RewardNet_normalized reward_path=/home/qu/f-IRL/logs/AntFH-v0/exp-16/airl_imitation/3/checkpoints/final/reward_train.pt normalize_reward=True rollout_save_final=True policy_save_final=True environment.gym_id=AntFH-v0
```
4. 绘制智能体学习时的真实奖励曲线

（1）绘制逆强化学习过程

```bash
python plot.py
```

（2）绘制智能体训练过程
```bash
python plot_real_sto_return_mean_std.py
```

5. 奖励函数迁移
- f-IRL在两个源环境中学习一个通用的奖励函数
```bash
python /home/qu/f-IRL/firl/irl_samples_multisource.py
 ```

```bash
python firl__vae/TraIRL.py configs/trairl_ant_transfer_multisource.yml
 ```
python /home/qu/f-IRL/train_sac_optimal_with_logging.py configs/sac_pretrained_reward_ant.yml
python train_sac_trairl_transfer.py configs/sac_trairl_transfer_ant.yml
## Contact

export PYTHONPATH=/home/qu/f-IRL:$PYTHONPATH
python -m imitation.scripts.train_adversarial airl with \
    airl_seals_hopper \
    environment.gym_id=HopperFH-v0 \
    "environment.parallel=False" \
    environment.num_vec=1 \
    seed=1 \
    demonstrations.source=local \
    demonstrations.path=expert_data/imitation/HopperFH-v0_1_det_trajs.pkl \
    demonstrations.n_expert_demos=16 \
    logging.log_dir=logs/HopperFH-v0/exp-16/airl_imitation/1 \
    allow_variable_horizon=true


 python -m imitation.scripts.train_adversarial airl with \
    airl_seals_ant \
    environment.gym_id=AntFH-v0 \
    "environment.parallel=False" \
    environment.num_vec=1 \
    seed=1 \
    demonstrations.source=local \
    demonstrations.path=expert_data/imitation/AntFH-v0_airl_trajs.pkl \
    demonstrations.n_expert_demos=16 \
    logging.log_dir=logs/AntFH-v0/exp-16/airl_imitation/3 \
    allow_variable_horizon=true

**quxingyang@stu.ouc.edu.cn**.
- Loaded expert data from expert_data/states/AntLeg12Disabled-v0_1_det.pt: (16, 1000, 111)
- Loaded expert data from expert_data/states/AntLeg03Disabled-v0_1_det.pt: (16, 1000, 111)

- 调参，改变z的维度
-  Reward covariance: if true, each step draw B=min(N_agent,N_expert) learner + B expert trajs (after resample), then cat.
  reward_balanced_batch: true
- 博弈方法。在正常环境中运行。然后再在转移奖励函数到新的环境

| 左对齐 | 居中对齐 | 右对齐 |
| :----- | :------: | -----: |
| 单元格 | 单元格   | 单元格 |
| 数据1  | 数据2    | 数据3  |
