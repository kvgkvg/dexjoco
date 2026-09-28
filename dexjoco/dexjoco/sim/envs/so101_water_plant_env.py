"""MuJoCo water_plant task driven by a single SO-101 arm (proof-of-concept swap-in
for the Panda+Allegro arm used by PandaWaterPlantGymEnv).

Unlike the Panda arm, SO-101 does not use an operational-space (Cartesian) torque
controller: its actuators are native joint-position servos, so actions here are
direct joint-angle targets rather than a Cartesian mocap pose.
"""

import random
import time
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import mujoco
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

from ..mujoco_gym_env import MujocoGymEnv
from ..rendering import MujocoRenderer

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "arena_so101_hand_plant.xml"

_SO101_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
_N_SO101 = len(_SO101_JOINT_NAMES)

_SO101_HOME = np.asarray((0.0, -1.4, 1.4, 0.8, 0.0, 0.5), dtype=np.float64)

_SPRAY_SAMPLE_LOW = np.array([-0.35, -0.25], dtype=np.float64)
_SPRAY_SAMPLE_HIGH = np.array([-0.30, -0.20], dtype=np.float64)
_PLANT_SAMPLE_LOW = np.array([-0.10, 0.15], dtype=np.float64)
_PLANT_SAMPLE_HIGH = np.array([-0.05, 0.20], dtype=np.float64)
_MAX_EPISODE_STEPS = 1000

_TABLE_LEG_NAMES = (
    "table_leg_1",
    "table_leg_2",
    "table_leg_3",
    "table_leg_4",
)


class So101WaterPlantGymEnv(MujocoGymEnv):
    """water_plant scene with a single SO-101 arm instead of Panda+Allegro.

    This is a structural proof-of-concept: the scene (table, plant, spray) and
    task objects are unchanged from PandaWaterPlantGymEnv, but the robot,
    action space, and controller are swapped. Success detection tied to the
    spray trigger is not meaningful for a parallel-jaw gripper and is omitted;
    this env only verifies that the SO-101 model loads, steps, and renders
    correctly inside the existing task scene.
    """

    def __init__(
        self,
        render_mode: Literal["rgb_array", "human", "none"] = "rgb_array",
        randomize: bool = False,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        hz: int = 30,
    ):
        self.hz = hz
        self.randomize = randomize
        self.image_obs = render_mode != "none"

        super().__init__(
            xml_path=_XML_PATH, seed=seed, control_dt=control_dt, physics_dt=physics_dt
        )

        random.seed(seed)
        np.random.seed(seed)

        self.render_mode = render_mode
        self.env_step = 0

        self._so101_dof_ids = np.asarray(
            [self._model.joint(name).id for name in _SO101_JOINT_NAMES]
        )
        self._so101_qpos_adr = np.asarray(
            [int(self._model.jnt_qposadr[jid]) for jid in self._so101_dof_ids]
        )
        self._so101_ctrl_ids = np.asarray(
            [self._model.actuator(name).id for name in _SO101_JOINT_NAMES]
        )
        self._ctrl_ranges = np.asarray(
            [self._model.actuator_ctrlrange[cid] for cid in self._so101_ctrl_ids]
        )

        self._mj_viewer = None
        if self.image_obs:
            self._mj_viewer = MujocoRenderer(self.model, self.data)
            self._mj_viewer.render(self.render_mode)

        self._front_camera_id = int(self._model.camera("front").id)
        self._wrist_camera_id = int(self._model.camera("handcam_rgb").id)

        self._table_body_z0 = float(self._model.body("table").pos[2])
        self._table_leg_half_len0 = {
            name: float(self._model.geom(name).size[1]) for name in _TABLE_LEG_NAMES
        }
        self._plant_body_z0 = float(self._model.body("plant").pos[2])
        self._spray_body_z0 = float(self._model.body("link_2").pos[2])

        image_h = int(self._model.vis.global_.offheight)
        image_w = int(self._model.vis.global_.offwidth)

        self.observation_space = spaces.Dict(
            {
                "state": spaces.Dict(
                    {
                        "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
                        "joint_pos": spaces.Box(-np.inf, np.inf, shape=(_N_SO101,), dtype=np.float64),
                        "table_delta_height": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                    }
                ),
            }
        )
        if self.image_obs:
            self.observation_space["images"] = spaces.Dict(
                {
                    "wrist": spaces.Box(0, 255, shape=(image_h, image_w, 3), dtype=np.uint8),
                    "front": spaces.Box(0, 255, shape=(image_h, image_w, 3), dtype=np.uint8),
                }
            )

        # Action: normalized [-1, 1] per joint, scaled to that joint's ctrlrange.
        self.action_space = spaces.Box(
            low=np.full(_N_SO101, -1.0, dtype=np.float32),
            high=np.full(_N_SO101, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

    def _prime_rgb_array_renderer(self):
        if self._mj_viewer is None:
            return
        self._mj_viewer.render(render_mode="rgb_array", camera_id=self._wrist_camera_id)
        self._mj_viewer.render(render_mode="rgb_array", camera_id=self._front_camera_id)

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        mujoco.mj_resetData(self._model, self._data)

        self.delta_h = float(np.random.uniform(0.0, 0.05))
        table_body_id = self._model.body("table").id
        self._model.body_pos[table_body_id, 2] = self._table_body_z0 + self.delta_h
        for lname in _TABLE_LEG_NAMES:
            lgid = self._model.geom(lname).id
            self._model.geom_size[lgid, 1] = self._table_leg_half_len0[lname] + self.delta_h

        spray_xy = np.random.uniform(_SPRAY_SAMPLE_LOW, _SPRAY_SAMPLE_HIGH)
        spray_ori_pos = self._model.body("link_2").pos
        spray_ori_pos[:2] = spray_xy
        spray_ori_pos[2] = self._spray_body_z0 + self.delta_h
        self._data.jnt("spray_root").qpos[:3] = spray_ori_pos

        plant_body_id = self._model.body("plant").id
        plant_xy = np.random.uniform(_PLANT_SAMPLE_LOW, _PLANT_SAMPLE_HIGH)
        plant_ori_pos = self._model.body("plant").pos
        plant_ori_pos[:2] = plant_xy
        plant_ori_pos[2] = self._plant_body_z0 + self.delta_h
        self._model.body_pos[plant_body_id] = plant_ori_pos

        self._data.qpos[self._so101_qpos_adr] = _SO101_HOME
        self._data.ctrl[self._so101_ctrl_ids] = _SO101_HOME
        mujoco.mj_forward(self._model, self._data)

        self.env_step = 0
        self._prime_rgb_array_renderer()

        obs = self._compute_observation()
        return obs, {"succeed": False}

    def step(self, action: np.ndarray):
        start_time = time.time()

        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        lo = self._ctrl_ranges[:, 0]
        hi = self._ctrl_ranges[:, 1]
        target = lo + (action + 1.0) * 0.5 * (hi - lo)

        for _ in range(self._n_substeps):
            self._data.ctrl[self._so101_ctrl_ids] = target
            mujoco.mj_step(self._model, self._data)

        obs = self._compute_observation()
        self.env_step += 1
        terminated = self.env_step >= _MAX_EPISODE_STEPS

        if self.render_mode == "human":
            try:
                self._mj_viewer.render("human")
            except Exception:
                pass

        dt = time.time() - start_time
        time.sleep(max(0.0, (1.0 / self.hz) - dt))

        return obs, 0.0, terminated, False, {"succeed": False}

    def render(self):
        if self._mj_viewer is None:
            raise RuntimeError("Rendering is disabled because render_mode='none'.")
        wrist_frame = self._mj_viewer.render(render_mode="rgb_array", camera_id=self._wrist_camera_id)
        front_frame = self._mj_viewer.render(render_mode="rgb_array", camera_id=self._front_camera_id)
        return [wrist_frame, front_frame]

    def _compute_observation(self) -> dict:
        obs = {}

        tcp_pos = self._data.sensor("so101/tcp_pos").data
        tcp_quat = self._data.sensor("so101/tcp_quat").data
        tcp_pose = np.concatenate([tcp_pos, tcp_quat])
        joint_pos = self._data.qpos[self._so101_qpos_adr].copy()

        if self.image_obs:
            obs["images"] = {}
            obs["images"]["wrist"], obs["images"]["front"] = self.render()

        obs["state"] = {
            "tcp_pose": tcp_pose,
            "joint_pos": joint_pos,
            "table_delta_height": np.asarray([self.delta_h], dtype=np.float64),
        }
        return obs

    def close(self):
        viewer = getattr(self, "_mj_viewer", None)
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass
        super().close()


if __name__ == "__main__":
    env = So101WaterPlantGymEnv(render_mode="human")
    obs, info = env.reset()
    for _ in range(200):
        obs, rew, done, trunc, info = env.step(np.zeros(_N_SO101, dtype=np.float32))
        if done or trunc:
            break
    env.close()
