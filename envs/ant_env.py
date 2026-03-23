# '''
# Refer[Original Code]: https://github.com/justinjfu/inverse_rl/blob/master/inverse_rl/envs/ant_env.py
# '''
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import sys
from envs.dynamic_mjc.model_builder import MJCModel


def ant_env(gear=150, eyes=True):
    mjcmodel = MJCModel('ant_maze')
    mjcmodel.root.compiler(inertiafromgeom="true", angle="degree", coordinate="local")
    mjcmodel.root.option(timestep="0.01", integrator="RK4")
    mjcmodel.root.custom().numeric(data="0.0 0.0 0.55 1.0 0.0 0.0 0.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0 1.0",name="init_qpos")
    asset = mjcmodel.root.asset()
    asset.texture(builtin="gradient",height="100",rgb1="1 1 1",rgb2="0 0 0",type="skybox",width="100")
    asset.texture(builtin="flat",height="1278",mark="cross",markrgb="1 1 1",name="texgeom",random="0.01",rgb1="0.8 0.6 0.4",rgb2="0.8 0.6 0.4",type="cube",width="127")
    asset.texture(builtin="checker",height="100",name="texplane",rgb1="0 0 0",rgb2="0.8 0.8 0.8",type="2d",width="100")
    asset.material(name="MatPlane",reflectance="0.5",shininess="1",specular="1",texrepeat="60 60",texture="texplane")
    asset.material(name="geom",texture="texgeom",texuniform="true")

    default = mjcmodel.root.default()
    default.joint(armature=1, damping=1, limited='true')
    default.geom(friction=[1.5,0.5,0.5], density=5.0, margin=0.01, condim=3, conaffinity=0, rgba="0.8 0.6 0.4 1")

    worldbody = mjcmodel.root.worldbody()
    worldbody.light(cutoff="100",diffuse=[.8,.8,.8],dir="-0 0 -1.3",directional="true",exponent="1",pos="0 0 1.3",specular=".1 .1 .1")
    worldbody.geom(conaffinity=1, condim=3, material="MatPlane",name="floor",pos="0 0 0",rgba="0.8 0.9 0.8 1",size="40 40 40",type="plane")

    ant = worldbody.body(name='torso', pos=[0, 0, 0.75])
    ant.geom(name='torso_geom', pos=[0, 0, 0], size="0.25", type="sphere")
    ant.joint(armature="0", damping="0", limited="false", margin="0.01", name="root", pos=[0, 0, 0], type="free")
    ant.camera(name="track", mode="trackcom", pos=[0, -3, 3], xyaxes=[1, 0, 0, 0, 1, 1])

    if eyes:
        eye_z = 0.1
        eye_y = -.21
        eye_x_offset = 0.07
        # eyes
        ant.geom(fromto=[eye_x_offset,0,eye_z,eye_x_offset,eye_y,eye_z], name='eye1', size='0.03', type='capsule', rgba=[1,1,1,1])
        ant.geom(fromto=[eye_x_offset,0,eye_z,eye_x_offset,eye_y-0.02,eye_z], name='eye1_', size='0.02', type='capsule', rgba=[0,0,0,1])
        ant.geom(fromto=[-eye_x_offset,0,eye_z,-eye_x_offset,eye_y,eye_z], name='eye2', size='0.03', type='capsule', rgba=[1,1,1,1])
        ant.geom(fromto=[-eye_x_offset,0,eye_z,-eye_x_offset,eye_y-0.02,eye_z], name='eye2_', size='0.02', type='capsule', rgba=[0,0,0,1])
        # eyebrows
        ant.geom(fromto=[eye_x_offset-0.03,eye_y, eye_z+0.07, eye_x_offset+0.03, eye_y, eye_z+0.1], name='brow1', size='0.02', type='capsule', rgba=[0,0,0,1])
        ant.geom(fromto=[-eye_x_offset+0.03,eye_y, eye_z+0.07, -eye_x_offset-0.03, eye_y, eye_z+0.1], name='brow2', size='0.02', type='capsule', rgba=[0,0,0,1])

    front_left_leg = ant.body(name="front_left_leg", pos=[0, 0, 0])
    front_left_leg.geom(fromto=[0.0, 0.0, 0.0, 0.2, 0.2, 0.0], name="aux_1_geom", size="0.08", type="capsule")
    aux_1 = front_left_leg.body(name="aux_1", pos=[0.2, 0.2, 0])
    aux_1.joint(axis=[0, 0, 1], name="hip_1", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
    aux_1.geom(fromto=[0.0, 0.0, 0.0, 0.2, 0.2, 0.0], name="left_leg_geom", size="0.08", type="capsule")
    ankle_1 = aux_1.body(pos=[0.2, 0.2, 0])
    ankle_1.joint(axis=[-1, 1, 0], name="ankle_1", pos=[0.0, 0.0, 0.0], range=[30, 70], type="hinge")
    ankle_1.geom(fromto=[0.0, 0.0, 0.0, 0.4, 0.4, 0.0], name="left_ankle_geom", size="0.08", type="capsule")

    front_right_leg = ant.body(name="front_right_leg", pos=[0, 0, 0])
    front_right_leg.geom(fromto=[0.0, 0.0, 0.0, -0.2, 0.2, 0.0], name="aux_2_geom", size="0.08", type="capsule")
    aux_2 = front_right_leg.body(name="aux_2", pos=[-0.2, 0.2, 0])
    aux_2.joint(axis=[0, 0, 1], name="hip_2", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
    aux_2.geom(fromto=[0.0, 0.0, 0.0, -0.2, 0.2, 0.0], name="right_leg_geom", size="0.08", type="capsule")
    ankle_2 = aux_2.body(pos=[-0.2, 0.2, 0])
    ankle_2.joint(axis=[1, 1, 0], name="ankle_2", pos=[0.0, 0.0, 0.0], range=[-70, -30], type="hinge")
    ankle_2.geom(fromto=[0.0, 0.0, 0.0, -0.4, 0.4, 0.0], name="right_ankle_geom", size="0.08", type="capsule")

    back_left_leg = ant.body(name="back_left_leg", pos=[0, 0, 0])
    back_left_leg.geom(fromto=[0.0, 0.0, 0.0, -0.2, -0.2, 0.0], name="aux_3_geom", size="0.08", type="capsule")
    aux_3 = back_left_leg.body(name="aux_3", pos=[-0.2, -0.2, 0])
    aux_3.joint(axis=[0, 0, 1], name="hip_3", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
    aux_3.geom(fromto=[0.0, 0.0, 0.0, -0.2, -0.2, 0.0], name="backleft_leg_geom", size="0.08", type="capsule")
    ankle_3 = aux_3.body(pos=[-0.2, -0.2, 0])
    ankle_3.joint(axis=[-1, 1, 0], name="ankle_3", pos=[0.0, 0.0, 0.0], range=[-70, -30], type="hinge")
    ankle_3.geom(fromto=[0.0, 0.0, 0.0, -0.4, -0.4, 0.0], name="backleft_ankle_geom", size="0.08", type="capsule")

    back_right_leg = ant.body(name="back_right_leg", pos=[0, 0, 0])
    back_right_leg.geom(fromto=[0.0, 0.0, 0.0, 0.2, -0.2, 0.0], name="aux_4_geom", size="0.08", type="capsule")
    aux_4 = back_right_leg.body(name="aux_4", pos=[0.2, -0.2, 0])
    aux_4.joint(axis=[0, 0, 1], name="hip_4", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
    aux_4.geom(fromto=[0.0, 0.0, 0.0, 0.2, -0.2, 0.0], name="backright_leg_geom", size="0.08", type="capsule")
    ankle_4 = aux_4.body(pos=[0.2, -0.2, 0])
    ankle_4.joint(axis=[1, 1, 0], name="ankle_4", pos=[0.0, 0.0, 0.0], range=[30, 70], type="hinge")
    ankle_4.geom(fromto=[0.0, 0.0, 0.0, 0.4, -0.4, 0.0], name="backright_ankle_geom", size="0.08", type="capsule")

    actuator = mjcmodel.root.actuator()
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_4", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_4", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_1", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_1", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_2", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_2", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_3", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_3", gear=gear)
    return mjcmodel


def angry_ant_crippled(gear=150, disabled_legs=(2, 3), disabled_gear_ratio=0.3):
    mjcmodel = MJCModel('ant_maze')
    mjcmodel.root.compiler(inertiafromgeom="true", angle="degree", coordinate="local")
    mjcmodel.root.option(timestep="0.01", integrator="RK4")
    mjcmodel.root.custom().numeric(data="0.0 0.0 0.55 1.0 0.0 0.0 0.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0 1.0",name="init_qpos")
    asset = mjcmodel.root.asset()
    asset.texture(builtin="gradient",height="100",rgb1="1 1 1",rgb2="0 0 0",type="skybox",width="100")
    asset.texture(builtin="flat",height="1278",mark="cross",markrgb="1 1 1",name="texgeom",random="0.01",rgb1="0.8 0.6 0.4",rgb2="0.8 0.6 0.4",type="cube",width="127")
    asset.texture(builtin="checker",height="100",name="texplane",rgb1="0 0 0",rgb2="0.8 0.8 0.8",type="2d",width="100")
    asset.material(name="MatPlane",reflectance="0.5",shininess="1",specular="1",texrepeat="60 60",texture="texplane")
    asset.material(name="geom",texture="texgeom",texuniform="true")



    default = mjcmodel.root.default()
    default.joint(armature=1, damping=1, limited='true')
    default.geom(friction=[1.5,0.5,0.5], density=5.0, margin=0.01, condim=3, conaffinity=0, rgba="0.8 0.6 0.4 1")

    worldbody = mjcmodel.root.worldbody()

    worldbody.geom(conaffinity=1, condim=3, material="MatPlane",name="floor",pos="0 0 0",rgba="0.8 0.9 0.8 1",size="40 40 40",type="plane")
    worldbody.light(cutoff="100",diffuse=[.8,.8,.8],dir="-0 0 -1.3",directional="true",exponent="1",pos="0 0 1.3",specular=".1 .1 .1")


    ant = worldbody.body(name='torso', pos=[0, 0, 0.75])
    ant.geom(name='torso_geom', pos=[0, 0, 0], size="0.25", type="sphere")
    ant.joint(armature="0", damping="0", limited="false", margin="0.01", name="root", pos=[0, 0, 0], type="free")
    ant.camera(name="track", mode="trackcom", pos=[0, -3, 3], xyaxes=[1, 0, 0, 0, 1, 1])

    eye_z = 0.1
    eye_y = -.21
    eye_x_offset = 0.07
    # eyes
    ant.geom(fromto=[eye_x_offset,0,eye_z,eye_x_offset,eye_y,eye_z], name='eye1', size='0.03', type='capsule', rgba=[1,1,1,1])
    ant.geom(fromto=[eye_x_offset,0,eye_z,eye_x_offset,eye_y-0.02,eye_z], name='eye1_', size='0.02', type='capsule', rgba=[0,0,0,1])
    ant.geom(fromto=[-eye_x_offset,0,eye_z,-eye_x_offset,eye_y,eye_z], name='eye2', size='0.03', type='capsule', rgba=[1,1,1,1])
    ant.geom(fromto=[-eye_x_offset,0,eye_z,-eye_x_offset,eye_y-0.02,eye_z], name='eye2_', size='0.02', type='capsule', rgba=[0,0,0,1])
    # eyebrows
    ant.geom(fromto=[eye_x_offset-0.03,eye_y, eye_z+0.07, eye_x_offset+0.03, eye_y, eye_z+0.1], name='brow1', size='0.02', type='capsule', rgba=[0,0,0,1])
    ant.geom(fromto=[-eye_x_offset+0.03,eye_y, eye_z+0.07, -eye_x_offset-0.03, eye_y, eye_z+0.1], name='brow2', size='0.02', type='capsule', rgba=[0,0,0,1])




    # Keep geometry close to the standard Ant; only weaken selected legs.
    thigh_length = 0.2
    ankle_length = 0.4
    dark_red = [0.8,0.3,0.3,1.0]
    normal_color = [0.8, 0.6, 0.4, 1.0]

    disabled_legs = set(disabled_legs)
    assert all(leg_idx in {0, 1, 2, 3} for leg_idx in disabled_legs), "disabled_legs must use zero-based leg ids in {0,1,2,3}"
    assert 0 < disabled_gear_ratio <= 1.0, "disabled_gear_ratio must be in (0, 1]"

    def leg_gear(leg_idx):
        if leg_idx in disabled_legs:
            return max(gear * disabled_gear_ratio, 1.0)
        return gear

    def leg_dims(leg_idx, x_sign, y_sign):
        if leg_idx in disabled_legs:
            return (
                [0.0, 0.0, 0.0, x_sign * thigh_length, y_sign * thigh_length, 0.0],
                [x_sign * thigh_length, y_sign * thigh_length, 0.0],
                [0.0, 0.0, 0.0, x_sign * ankle_length, y_sign * ankle_length, 0.0],
                dark_red,
            )
        return (
            [0.0, 0.0, 0.0, x_sign * 0.2, y_sign * 0.2, 0.0],
            [x_sign * 0.2, y_sign * 0.2, 0.0],
            [0.0, 0.0, 0.0, x_sign * 0.4, y_sign * 0.4, 0.0],
            normal_color,
        )

    front_left_thigh, front_left_ankle_pos, front_left_ankle, front_left_color = leg_dims(0, 1, 1)
    front_right_thigh, front_right_ankle_pos, front_right_ankle, front_right_color = leg_dims(1, -1, 1)
    back_left_thigh, back_left_ankle_pos, back_left_ankle, back_left_color = leg_dims(2, -1, -1)
    back_right_thigh, back_right_ankle_pos, back_right_ankle, back_right_color = leg_dims(3, 1, -1)

    front_left_leg = ant.body(name="front_left_leg", pos=[0, 0, 0])
    front_left_leg.geom(fromto=[0.0, 0.0, 0.0, 0.2, 0.2, 0.0], name="aux_1_geom", size="0.08", type="capsule",
                        rgba=front_left_color)
    aux_1 = front_left_leg.body(name="aux_1", pos=[0.2, 0.2, 0])
    aux_1.joint(axis=[0, 0, 1], name="hip_1", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
    aux_1.geom(fromto=front_left_thigh, name="left_leg_geom", size="0.08", type="capsule", rgba=front_left_color)
    ankle_1 = aux_1.body(pos=front_left_ankle_pos)
    ankle_1.joint(axis=[-1, 1, 0], name="ankle_1", pos=[0.0, 0.0, 0.0], range=[30, 70], type="hinge")
    ankle_1.geom(fromto=front_left_ankle, name="left_ankle_geom", size="0.08", type="capsule", rgba=front_left_color)

    front_right_leg = ant.body(name="front_right_leg", pos=[0, 0, 0])
    front_right_leg.geom(fromto=[0.0, 0.0, 0.0, -0.2, 0.2, 0.0], name="aux_2_geom", size="0.08", type="capsule",
                         rgba=front_right_color)
    aux_2 = front_right_leg.body(name="aux_2", pos=[-0.2, 0.2, 0])
    aux_2.joint(axis=[0, 0, 1], name="hip_2", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
    aux_2.geom(fromto=front_right_thigh, name="right_leg_geom", size="0.08", type="capsule", rgba=front_right_color)
    ankle_2 = aux_2.body(pos=front_right_ankle_pos)
    ankle_2.joint(axis=[1, 1, 0], name="ankle_2", pos=[0.0, 0.0, 0.0], range=[-70, -30], type="hinge")
    ankle_2.geom(fromto=front_right_ankle, name="right_ankle_geom", size="0.08", type="capsule", rgba=front_right_color)

    back_left_leg = ant.body(name="back_left_leg", pos=[0, 0, 0])
    back_left_leg.geom(fromto=[0.0, 0.0, 0.0, -0.2, -0.2, 0.0], name="aux_3_geom", size="0.08", type="capsule",
                       rgba=back_left_color)
    aux_3 = back_left_leg.body(name="aux_3", pos=[-0.2, -0.2, 0])
    aux_3.joint(axis=[0, 0, 1], name="hip_3", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
    aux_3.geom(fromto=back_left_thigh, name="backleft_leg_geom", size="0.08", type="capsule",
               rgba=back_left_color)
    ankle_3 = aux_3.body(pos=back_left_ankle_pos)
    ankle_3.joint(axis=[-1, 1, 0], name="ankle_3", pos=[0.0, 0.0, 0.0], range=[-70, -30], type="hinge")
    ankle_3.geom(fromto=back_left_ankle, name="backleft_ankle_geom", size="0.08", type="capsule",
                 rgba=back_left_color)

    back_right_leg = ant.body(name="back_right_leg", pos=[0, 0, 0])
    back_right_leg.geom(fromto=[0.0, 0.0, 0.0, 0.2, -0.2, 0.0], name="aux_4_geom", size="0.08", type="capsule",
                        rgba=back_right_color)
    aux_4 = back_right_leg.body(name="aux_4", pos=[0.2, -0.2, 0])
    aux_4.joint(axis=[0, 0, 1], name="hip_4", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
    aux_4.geom(fromto=back_right_thigh, name="backright_leg_geom", size="0.08", type="capsule",
               rgba=back_right_color)
    ankle_4 = aux_4.body(pos=back_right_ankle_pos)
    ankle_4.joint(axis=[1, 1, 0], name="ankle_4", pos=[0.0, 0.0, 0.0], range=[30, 70], type="hinge")
    ankle_4.geom(fromto=back_right_ankle, name="backright_ankle_geom", size="0.08", type="capsule",
                 rgba=back_right_color)

    actuator = mjcmodel.root.actuator()
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_1", gear=leg_gear(0))
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_1", gear=leg_gear(0))
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_2", gear=leg_gear(1))
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_2", gear=leg_gear(1))
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_3", gear=leg_gear(2))
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_3", gear=leg_gear(2))
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_4", gear=leg_gear(3))
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_4", gear=leg_gear(3))
    return mjcmodel


class CustomAntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    A modified ant env with lower joint gear ratios so it flips less often and learns faster.
    """
    def __init__(
        self,
        max_timesteps=500,
        r=None,
        disabled=False,
        disabled_legs=None,
        gear=150,
        disabled_gear_ratio=0.3,
        match_standard_ant=False,
        disabled_action_ratio=0.3,
        lock_disabled_legs=False,
        T=None,
    ):
        if T is not None:
            max_timesteps = int(T)
        #mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self, max_timesteps=max_timesteps, r=r, disabled=disabled,
                                disabled_legs=disabled_legs, gear=gear,
                                disabled_gear_ratio=disabled_gear_ratio,
                                match_standard_ant=match_standard_ant,
                                disabled_action_ratio=disabled_action_ratio,
                                lock_disabled_legs=lock_disabled_legs, T=T)
        self.timesteps = 0
        self.max_timesteps=max_timesteps
        self.T = T
        self.r = r
        self.prev_obs = None
        self.match_standard_ant = match_standard_ant
        self.disabled_action_ratio = disabled_action_ratio
        self.lock_disabled_legs = lock_disabled_legs
        if disabled_legs is None and disabled:
            disabled_legs = (2, 3)
        self.disabled_legs = tuple(disabled_legs) if disabled_legs is not None else None
        if disabled_legs is not None:
            if match_standard_ant:
                # Keep dynamics/model identical to standard Ant-v2 and only weaken selected leg actions.
                mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
                return
            model = angry_ant_crippled(
                gear=gear,
                disabled_legs=tuple(disabled_legs),
                disabled_gear_ratio=disabled_gear_ratio,
            )
        else:
            model = ant_env(gear=gear)
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, f.name, 5)

    def step(self, a):
        if self.match_standard_ant and self.disabled_legs is not None:
            a = np.array(a, dtype=np.float32, copy=True)
            leg_to_action_ids = {
                0: (0, 1),
                1: (2, 3),
                2: (4, 5),
                3: (6, 7),
            }
            for leg_id in self.disabled_legs:
                for idx in leg_to_action_ids[leg_id]:
                    if self.lock_disabled_legs:
                        a[idx] = 0.0
                    else:
                        a[idx] *= self.disabled_action_ratio

        xposbefore = self.get_body_com("torso")[0]
        if (self._get_obs is not None) and (self.r is not None):
            reward_network = self.r(self._get_obs().copy())
        else:
            reward_network = 0
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore) / self.dt

        ctrl_cost = .01 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        state = self.state_vector()
        healthy = np.isfinite(state).all() and (state[2] >= 0.2) and (state[2] <= 1.0)
        healthy_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + healthy_reward

        if self.r is not None:
            # print(reward_network)
            reward = reward_network
        # self.prev_obs = self._get_obs().copy()
        self.timesteps += 1
        done = (not healthy) or (self.timesteps >= self.max_timesteps)

        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=healthy_reward)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        self.timesteps = 0
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        self.prev_obs = self._get_obs().copy()
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def log_diagnostics(self, paths):
        forward_rew = np.array([np.mean(traj['env_infos']['reward_forward']) for traj in paths])
        reward_ctrl = np.array([np.mean(traj['env_infos']['reward_ctrl']) for traj in paths])
        reward_cont = np.array([np.mean(traj['env_infos']['reward_contact']) for traj in paths])
        reward_survive = np.array([np.mean(traj['env_infos']['reward_survive']) for traj in paths])

if __name__ == "__main__":
    env = CustomAntEnv(disabled=False, gear=30)

    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())
