from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gym

def make_env(): return gym.make("CartPole-v1")
venv = DummyVecEnv([make_env])
model = PPO("MlpPolicy", venv, learning_rate=0.001)

print(f"LR before: {model.policy.optimizer.param_groups[0]['lr']}")
for _ in range(3):
    model.learn(total_timesteps=64, reset_num_timesteps=False)
    print(f"LR after learn: {model.policy.optimizer.param_groups[0]['lr']}")
print("Done")
