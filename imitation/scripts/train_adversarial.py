"""Train GAIL or AIRL."""

import functools
import logging
import os
import pathlib
from typing import Any, Mapping, Optional, Type

import numpy as np
import sacred.commands
import torch as th
from sacred.observers import FileStorageObserver

from imitation.algorithms.adversarial import airl as airl_algo
from imitation.algorithms.adversarial import common
from imitation.algorithms.adversarial import gail as gail_algo
from imitation.data import rollout
from imitation.policies import serialize
from imitation.scripts.config.train_adversarial import train_adversarial_ex
from imitation.scripts.ingredients import demonstrations, environment
from imitation.scripts.ingredients import logging as logging_ingredient
from imitation.scripts.ingredients import policy_evaluation, reward, rl
from utils import logger as firl_logger

logger = logging.getLogger("imitation.scripts.train_adversarial")


def save(trainer: common.AdversarialTrainer, save_path: pathlib.Path):
    """Save discriminator and generator."""
    save_path.mkdir(parents=True, exist_ok=True)
    th.save(trainer.reward_train, save_path / "reward_train.pt")
    th.save(trainer.reward_test, save_path / "reward_test.pt")
    serialize.save_stable_model(
        save_path / "gen_policy",
        trainer.gen_algo,
    )


def save_reward_model(trainer: common.AdversarialTrainer, model_dir: pathlib.Path,
                      suffix: str) -> None:
    """Save reward_test state_dict as .pkl，对齐 firl/irl_samples.py 的命名方式。

    保存 reward_test（剥掉 shaping wrapper 后的基础奖励网络），
    这才是 AIRL 论文中可迁移的 r(s, a, s')。
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    th.save(trainer.reward_test.state_dict(),
            model_dir / f"reward_model_{suffix}.pkl")
    th.save(trainer.gen_algo.policy.state_dict(),
            model_dir / f"policy_model_{suffix}.pkl")


def evaluate_real_return(trainer: common.AdversarialTrainer,
                         env_fn, n_episodes: int, deterministic: bool) -> float:
    """在原始环境奖励下评估当前 policy 的平均 return，使用独立环境不干扰训练。"""
    import gym
    env = env_fn()
    returns = []
    for _ in range(n_episodes):
        obs = env.reset()
        ret = 0.0
        done = False
        while not done:
            action, _ = trainer.policy.predict(obs[None] if obs.ndim == 1 else obs,
                                               deterministic=deterministic)
            obs, rew, done, _ = env.step(action[0] if hasattr(action, '__len__') else action)
            ret += float(rew)
        returns.append(ret)
    env.close()
    return float(np.mean(returns))


def _add_hook(ingredient: sacred.Ingredient) -> None:
    # This is an ugly hack around Sacred config brokenness.
    # Config hooks only apply to their current ingredient,
    # and cannot update things in nested ingredients.
    # So we have to apply this hook to every ingredient we use.
    @ingredient.config_hook
    def hook(config, command_name, logger):
        del logger
        path = ingredient.path
        if path == "train_adversarial":
            path = ""
        ingredient_config = sacred.utils.get_by_dotted_path(config, path)
        return ingredient_config["algorithm_specific"].get(command_name, {})

    # We add this so Sacred doesn't complain that algorithm_specific is unused
    @ingredient.capture
    def dummy_no_op(algorithm_specific):
        pass

    # But Sacred may then complain it isn't defined in config! So, define it.
    @ingredient.config
    def dummy_config():
        algorithm_specific = {}  # noqa: F841


for ingredient in [train_adversarial_ex, *train_adversarial_ex.ingredients]:
    _add_hook(ingredient)


@train_adversarial_ex.capture
def train_adversarial(
    _run,
    show_config: bool,
    algo_cls: Type[common.AdversarialTrainer],
    algorithm_kwargs: Mapping[str, Any],
    total_timesteps: int,
    checkpoint_interval: int,
    agent_path: Optional[str],
) -> Mapping[str, Mapping[str, float]]:
    """Train an adversarial-network-based imitation learning algorithm.

    Checkpoints:
        - AdversarialTrainer train and test RewardNets are saved to
           `f"{log_dir}/checkpoints/{step}/reward_{train,test}.pt"`
            where step is either the training round or "final".
        - Generator policies are saved to `f"{log_dir}/checkpoints/{step}/gen_policy/"`.

    Args:
        show_config: Print the merged config before starting training. This is
            analogous to the print_config command, but will show config after
            rather than before merging `algorithm_specific` arguments.
        algo_cls: The adversarial imitation learning algorithm to use.
        algorithm_kwargs: Keyword arguments for the `GAIL` or `AIRL` constructor.
        total_timesteps: The number of transitions to sample from the environment
            during training.
        checkpoint_interval: Save the discriminator and generator models every
            `checkpoint_interval` rounds and after traini`ng is complete. If 0,
            then only save weights after training is complete. If <0, then don't
            save weights at all.
        agent_path: Path to a directory containing a pre-trained agent. If
            provided, then the agent will be initialized using this stored policy
            (warm start). If not provided, then the agent will be initialized using
            a random policy.

    Returns:
        A dictionary with two keys. "imit_stats" gives the return value of
        `rollout_stats()` on rollouts test-reward-wrapped environment, using the final
        policy (remember that the ground-truth reward can be recovered from the
        "monitor_return" key). "expert_stats" gives the return value of
        `rollout_stats()` on the expert demonstrations.
    """
    # This allows to specify total_timesteps and checkpoint_interval in scientific
    # notation, which is interpreted as a float by python.
    total_timesteps = int(total_timesteps)
    checkpoint_interval = int(checkpoint_interval)

    if show_config:
        # Running `train_adversarial print_config` will show unmerged config.
        # So, support showing merged config from `train_adversarial {airl,gail}`.
        sacred.commands.print_config(_run)

    custom_logger, log_dir = logging_ingredient.setup_logging()
    expert_trajs = demonstrations.get_expert_trajectories()

    # 初始化 firl 风格的 tabular logger（写 progress.csv）
    firl_logger.configure(dir=str(log_dir))

    # 创建 model 子目录（对齐 firl/irl_samples.py）
    model_dir = log_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    with environment.make_venv() as venv:  # type: ignore[wrong-arg-count]
        reward_net = reward.make_reward_net(venv)
        relabel_reward_fn = functools.partial(
            reward_net.predict_processed,
            update_stats=False,
        )

        if agent_path is None:
            gen_algo = rl.make_rl_algo(venv, relabel_reward_fn=relabel_reward_fn)
        else:
            gen_algo = rl.load_rl_algo_from_path(
                agent_path=agent_path,
                venv=venv,
                relabel_reward_fn=relabel_reward_fn,
            )

        logger.info(f"Using '{algo_cls}' algorithm")
        algorithm_kwargs = dict(algorithm_kwargs)
        for k in ("shared", "airl", "gail"):
            if k in algorithm_kwargs:
                del algorithm_kwargs[k]
        trainer = algo_cls(
            venv=venv,
            demonstrations=expert_trajs,
            gen_algo=gen_algo,
            log_dir=log_dir,
            reward_net=reward_net,
            custom_logger=custom_logger,
            **algorithm_kwargs,
        )

        eval_interval = 10  # 每隔多少轮评估一次真实 return
        max_real_return_det, max_real_return_sto = -np.inf, -np.inf
        env_cfg = _run.config["environment"]
        gym_id = env_cfg["gym_id"]
        env_make_kwargs = env_cfg.get("env_make_kwargs", {})
        max_episode_steps = env_cfg.get("max_episode_steps")
        import gym as _gym

        def _make_eval_env():
            env = _gym.make(gym_id, **env_make_kwargs)
            if max_episode_steps is not None:
                if getattr(env, "spec", None) is None or env.spec.max_episode_steps != max_episode_steps:
                    env = _gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
            return env

        eval_env_fn = _make_eval_env

        def callback(round_num: int, /) -> None:
            nonlocal max_real_return_det, max_real_return_sto

            # 固定 checkpoint（原有逻辑）
            if checkpoint_interval > 0 and round_num % checkpoint_interval == 0:
                save(trainer, log_dir / "checkpoints" / f"{round_num:05d}")

            if round_num % eval_interval != 0:
                return

            # 评估真实环境 return，使用独立环境不干扰训练 venv
            real_return_det = evaluate_real_return(trainer, eval_env_fn, n_episodes=5,
                                                   deterministic=True)
            real_return_sto = evaluate_real_return(trainer, eval_env_fn, n_episodes=5,
                                                   deterministic=False)

            # 只在 det+sto 同时创新高时保存（对齐 firl/irl_samples.py）
            if real_return_det > max_real_return_det and real_return_sto > max_real_return_sto:
                max_real_return_det = real_return_det
                max_real_return_sto = real_return_sto
                suffix = (f"itr{round_num}"
                          f"_det{max_real_return_det:.0f}"
                          f"_sto{max_real_return_sto:.0f}")
                save_reward_model(trainer, model_dir, suffix)
                logger.info(f"New best at round {round_num}: "
                            f"det={max_real_return_det:.1f} sto={max_real_return_sto:.1f}")

            # tabular 日志（对齐 firl 的 logger.record_tabular / dump_tabular）
            env_steps = (round_num + 1) * trainer.gen_train_timesteps
            firl_logger.record_tabular("Iteration", round_num)
            firl_logger.record_tabular("Env Steps", env_steps)
            firl_logger.record_tabular("Real Det Return", round(real_return_det, 2))
            firl_logger.record_tabular("Real Sto Return", round(real_return_sto, 2))
            firl_logger.dump_tabular()

        trainer.train(total_timesteps, callback)
        imit_stats = policy_evaluation.eval_policy(trainer.policy, trainer.venv_train)

    # 保存最终 checkpoint
    if checkpoint_interval >= 0:
        save(trainer, log_dir / "checkpoints" / "final")

    return {
        "imit_stats": imit_stats,
        "expert_stats": rollout.rollout_stats(expert_trajs),
    }


@train_adversarial_ex.command
def gail():
    return train_adversarial(algo_cls=gail_algo.GAIL)


@train_adversarial_ex.command
def airl():
    return train_adversarial(algo_cls=airl_algo.AIRL)


def main_console():
    observer_path = pathlib.Path.cwd() / "output" / "sacred" / "train_adversarial"
    observer = FileStorageObserver(observer_path)
    train_adversarial_ex.observers.append(observer)
    train_adversarial_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()
