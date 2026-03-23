#!/bin/bash
# 复现 f-MAX-RKL 基线
export PYTHONPATH=${PWD}:$PYTHONPATH

ENVS=("hopper" "walker2d" "halfcheetah" "ant")
EXPERT_EPISODES=(1 4 16)
SEEDS=(1 23 42)

for env in "${ENVS[@]}"; do
    for episodes in "${EXPERT_EPISODES[@]}"; do
        # for seed in "${SEEDS[@]}"; do
            seed=1
            echo "Running f-MAX-RKL for $env with $episodes expert episodes, seed $seed"

            temp_config="/tmp/${env}_fmaxrkl_${episodes}_${seed}.yml"

            python << EOF
from ruamel.yaml import YAML

yaml = YAML()
yaml.preserve_quotes = True

with open("configs/samples/agents/${env}.yml") as f:
    config = yaml.load(f)

config['obj'] = 'f-max-rkl'
config['seed'] = $seed
if 'irl' not in config:
    config['irl'] = {}
config['irl']['expert_episodes'] = $episodes

with open("$temp_config", 'w') as f:
    yaml.dump(config, f)
EOF

            if [ -f "$temp_config" ]; then
                python baselines/main_samples.py "$temp_config"
                rm "$temp_config"
            else
                echo "Error: Failed to create temp config"
                exit 1
            fi
        done
    done
done