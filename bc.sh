#!/bin/bash
# 复现 BC 实验
export PYTHONPATH=${PWD}:$PYTHONPATH

ENVS=("hopper" "walker2d" "halfcheetah" "ant")
EXPERT_EPISODES=(1 4 16)
SEEDS=(1 23 42)

for env in "${ENVS[@]}"; do
    for episodes in "${EXPERT_EPISODES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "Running BC for $env with $episodes expert episodes, seed $seed"
            
            # 使用Python修改YAML文件（使用与bc.py相同的python解释器）
            python << EOF
import sys
from ruamel.yaml import YAML

yaml = YAML()
yaml.preserve_quotes = True
with open("configs/samples/agents/${env}.yml") as f:
    config = yaml.load(f)

config['seed'] = $seed
config['obj'] = 'bc'
config['bc']['expert_episodes'] = $episodes

temp_config = "/tmp/${env}_${episodes}_${seed}.yml"
with open(temp_config, 'w') as f:
    yaml.dump(config, f)

print(f"Modified config saved to {temp_config}")
EOF
            
            # 检查临时文件是否创建成功
            if [ -f "/tmp/${env}_${episodes}_${seed}.yml" ]; then
                # 运行实验
                python baselines/bc.py "/tmp/${env}_${episodes}_${seed}.yml"
                
                # 清理临时文件
                rm "/tmp/${env}_${episodes}_${seed}.yml"
            else
                echo "Error: Failed to create temporary config file"
            fi
        done
    done
done