#!/usr/bin/env python

from gym.envs.registration import register

# 同时注册到 gymnasium（供 imitation 库使用），不影响现有 gym 代码
try:
    from gymnasium.envs.registration import register as gymnasium_register
    import gymnasium
    import gym

    class GymToGymnasiumWrapper(gymnasium.Env):
        """
        把 gym.Env 包装成 gymnasium.Env，保留 gym.spaces（供 SB3 1.8.0 使用）。
        reset() 返回单个 obs（SB3 DummyVecEnv 期望的格式），
        同时满足 gymnasium 接口（step 返回 5 个值）。
        """
        def __init__(self, env_id, **kwargs):
            self._env = gym.make(env_id, **kwargs)
            self.observation_space = self._env.observation_space  # gym.spaces
            self.action_space = self._env.action_space            # gym.spaces
            self.metadata = getattr(self._env, 'metadata', {})
            self.render_mode = None

        def reset(self, seed=None, **kwargs):
            # 返回单个 obs（SB3 1.8.0 DummyVecEnv 期望的格式）
            obs = self._env.reset()
            return obs

        def step(self, action):
            result = self._env.step(action)
            obs, reward, done = result[0], result[1], result[2]
            # MujocoFH 返回 (obs, r, done, done)，第4个是 bool 不是 dict
            raw_info = result[3] if len(result) > 3 else {}
            info = raw_info if isinstance(raw_info, dict) else {}
            return obs, reward, done, info

        def render(self):
            return self._env.render()

        def close(self):
            return self._env.close()

        def seed(self, seed=None):
            return self._env.seed(seed)

    def _dual_register(id, entry_point, kwargs=None):
        register(id=id, entry_point=entry_point, kwargs=kwargs or {})
        try:
            gymnasium_register(
                id=id,
                entry_point="envs:GymToGymnasiumWrapper",
                kwargs={"env_id": id},
            )
        except Exception:
            pass

except ImportError:
    def _dual_register(id, entry_point, kwargs=None):
        register(id=id, entry_point=entry_point, kwargs=kwargs or {})


_dual_register('ContinuousVecGridEnv-v0', 'envs.vectorized_grid:ContinuousGridEnv')
_dual_register('GoalGrid-v0', 'envs.goal_grid:GoalContinuousGrid')
_dual_register('ReacherDraw-v0', 'envs.reacher_trace:ReacherTraceEnv')
_dual_register('HopperFH-v0',     'envs.mujocoFH:MujocoFH', dict(env_name='Hopper-v2'))
_dual_register('Walker2dFH-v0',   'envs.mujocoFH:MujocoFH', dict(env_name='Walker2d-v2'))
_dual_register('HalfCheetahFH-v0','envs.mujocoFH:MujocoFH', dict(env_name='HalfCheetah-v2'))
_dual_register('AntFH-v0',        'envs.mujocoFH:MujocoFH', dict(env_name='Ant-v2'))
_dual_register('PointMazeRight-v0','envs.point_maze_env:PointMazeEnv', dict(sparse_reward=False, direction=1))
_dual_register('PointMazeLeft-v0', 'envs.point_maze_env:PointMazeEnv', dict(sparse_reward=False, direction=0))
_dual_register('CustomAnt-v0',    'envs.ant_env:CustomAntEnv', dict(gear=30, disabled=False))
_dual_register('DisabledAnt-v0',  'envs.ant_env:CustomAntEnv', dict(gear=30, disabled=True))
_dual_register('AntLeg12Disabled-v0', 'envs.ant_env:CustomAntEnv', dict(
    disabled_legs=(1, 2), match_standard_ant=True, disabled_action_ratio=0, lock_disabled_legs=True))
_dual_register('AntLeg03Disabled-v0', 'envs.ant_env:CustomAntEnv', dict(
    disabled_legs=(0, 3), match_standard_ant=True, disabled_action_ratio=0, lock_disabled_legs=True))
_dual_register('AntLeg02Disabled-v0', 'envs.ant_env:CustomAntEnv', dict(
    disabled_legs=(0, 2), match_standard_ant=True, disabled_action_ratio=0, lock_disabled_legs=True))
