"""Single-arm Hanoi task driven by one SO-101 arm, with the peg board and disks
scaled down (0.5x) so they fit within the SO-101 parallel-jaw gripper's grasp
width. Mounted centered behind the board, facing the middle (red/medium) peg.

Same approach as the water_plant and bimanual Hanoi swaps: direct joint-position
control instead of the Panda's Cartesian mocap + operational-space controller.
The Hanoi tower/success-detection logic is unchanged from the Panda version.
"""

import random
import time
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import mujoco
import numpy as np
from gymnasium import spaces

from ..mujoco_gym_env import GymRenderingSpec, MujocoGymEnv
from ..rendering import MujocoRenderer

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "arena_so101_single_hanoi.xml"

_SO101_JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
_N_SO101 = len(_SO101_JOINT_NAMES)
_SO101_HOME = np.asarray((0.0, -0.7, 0.9, 0.6, 0.0, 0.5), dtype=np.float64)

# Board/disk geometry is scaled 0.5x relative to the Panda version, so the base
# sampling jitter and reset tower layout are scaled down to match.
_HANOI_BASE_SAMPLING_BOUNDS = 0.5 * np.asarray([[-0.25, 0], [-0.2, 0]], dtype=np.float64)
_HANOI_RESET_TOWER_STATE = {
    "A": ["hanoi_disk_small"],
    "B": ["hanoi_disk_medium"],
    "C": ["hanoi_disk_large"],
}


class So101SingleHanoiGymEnv(MujocoGymEnv):
    """arena_so101_single_hanoi scene: one SO-101 arm, scaled-down Hanoi set."""

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 10.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = True,
        randomize: bool = False,
        hz=30,
        post_assignment_tol: float = 0.020,
        placement_height_tol: float = 0.006,
        static_qvel_tol: float = 0.35,
    ):
        self.hz = hz
        self.randomize = randomize

        super().__init__(
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )

        random.seed(seed)
        np.random.seed(seed)

        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }
        self.render_mode = render_mode
        self.image_obs = image_obs
        self.env_step = 0

        self._post_assignment_tol = float(post_assignment_tol)
        self._placement_height_tol = float(placement_height_tol)
        self._static_qvel_tol = float(static_qvel_tol)

        joint_ids = [self._model.joint(name).id for name in _SO101_JOINT_NAMES]
        self._so101_qpos_adr = np.asarray([int(self._model.jnt_qposadr[jid]) for jid in joint_ids])
        self._so101_ctrl_ids = np.asarray([self._model.actuator(name).id for name in _SO101_JOINT_NAMES])
        self._ctrl_ranges = np.asarray([self._model.actuator_ctrlrange[cid] for cid in self._so101_ctrl_ids])

        self._base_body_id = self._model.body("hanoi_base").id
        self._base_init_pos = self._model.body_pos[self._base_body_id].copy()
        self._base_body_z0 = float(self._base_init_pos[2])

        self._disk_names = ["hanoi_disk_large", "hanoi_disk_medium", "hanoi_disk_small"]
        self._disk_size_rank = {"hanoi_disk_large": 3, "hanoi_disk_medium": 2, "hanoi_disk_small": 1}
        self._disk_qpos_adr = {}
        self._disk_qvel_adr = {}
        self._disk_body_id = {}
        self._disk_init_pos = {}
        self._disk_init_quat = {}
        self._disk_half_height = {}

        for disk_name in self._disk_names:
            joint_id = self._model.joint(f"{disk_name}_joint").id
            body_id = self._model.body(disk_name).id
            self._disk_qpos_adr[disk_name] = int(self._model.jnt_qposadr[joint_id])
            self._disk_qvel_adr[disk_name] = int(self._model.jnt_dofadr[joint_id])
            self._disk_body_id[disk_name] = body_id
            self._disk_init_pos[disk_name] = self._model.body_pos[body_id].copy()
            self._disk_init_quat[disk_name] = self._model.body_quat[body_id].copy()

            top_site = self._model.site(f"{disk_name}_top_site").id
            bottom_site = self._model.site(f"{disk_name}_bottom_site").id
            self._disk_half_height[disk_name] = 0.5 * (
                float(self._model.site_pos[top_site][2]) - float(self._model.site_pos[bottom_site][2])
            )

        self._post_site_names = {"A": "hanoi_post_a_site", "B": "hanoi_post_b_site", "C": "hanoi_post_c_site"}
        self._post_site_ids = {name: self._model.site(site_name).id for name, site_name in self._post_site_names.items()}

        self._base_upper_geom_id = self._model.geom("hanoi_base_upper_collision").id
        self._base_upper_top_local_z = float(
            self._model.geom_pos[self._base_upper_geom_id, 2] + self._model.geom_size[self._base_upper_geom_id, 2]
        )

        self._table_z0 = float(self._model.body("table").pos[2])
        self._table_leg_geom_ids = [
            gid
            for gid in range(self._model.ngeom)
            if self._model.geom_bodyid[gid] == self._model.body("table").id
            and self._model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CYLINDER
        ]
        self._table_leg_half_len0 = {gid: float(self._model.geom_size[gid, 1]) for gid in self._table_leg_geom_ids}

        self._front_camera_id = int(self._model.camera("front").id)
        self._wrist_camera_id = int(self._model.camera("handcam_rgb").id)
        self.camera_id = (self._front_camera_id, self._wrist_camera_id)

        state_space = spaces.Dict(
            {
                "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "joint_pos": spaces.Box(-np.inf, np.inf, shape=(_N_SO101,)),
                "hanoi_base_ori_pos": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "table_delta_height": spaces.Box(-np.inf, np.inf, shape=(1,)),
            }
        )
        observation_space_dict = {"state": state_space}
        if self.image_obs:
            image_h = int(self._model.vis.global_.offheight)
            image_w = int(self._model.vis.global_.offwidth)
            observation_space_dict["images"] = spaces.Dict(
                {
                    "wrist": spaces.Box(0, 255, shape=(image_h, image_w, 3), dtype=np.uint8),
                    "ego": spaces.Box(0, 255, shape=(image_h, image_w, 3), dtype=np.uint8),
                }
            )
        self.observation_space = spaces.Dict(observation_space_dict)

        self.action_space = spaces.Box(
            low=np.full(_N_SO101, -1.0, dtype=np.float32),
            high=np.full(_N_SO101, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

        self._viewer = MujocoRenderer(self.model, self.data)
        try:
            self._viewer.render(self.render_mode)
        except Exception:
            pass

    def _set_free_joint_pose(self, qpos_adr, qvel_adr, pos, quat):
        self._data.qpos[qpos_adr : qpos_adr + 3] = np.asarray(pos, dtype=np.float64)
        self._data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.asarray(quat, dtype=np.float64)
        self._data.qvel[qvel_adr : qvel_adr + 6] = 0.0

    def _apply_reset_tower_state(self, tower_state, base_delta_xy):
        base_top_z = self._base_body_z0 + self.delta_h + self._base_upper_top_local_z
        post_xy = {
            post_name: self._base_init_pos[:2] + self._model.site_pos[site_id][:2]
            for post_name, site_id in self._post_site_ids.items()
        }
        for post_name, disk_names in tower_state.items():
            running_height = base_top_z
            for disk_name in disk_names:
                disk_pos = self._disk_init_pos[disk_name].copy()
                disk_pos[:2] = post_xy[post_name] + base_delta_xy
                disk_pos[2] = running_height + self._disk_half_height[disk_name]
                running_height += 2.0 * self._disk_half_height[disk_name]
                self._set_free_joint_pose(
                    self._disk_qpos_adr[disk_name], self._disk_qvel_adr[disk_name], disk_pos, self._disk_init_quat[disk_name]
                )

    def _prime_rgb_array_renderer(self):
        self._viewer.render(render_mode="rgb_array", camera_id=self._wrist_camera_id)
        self._viewer.render(render_mode="rgb_array", camera_id=self._front_camera_id)

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        mujoco.mj_resetData(self._model, self._data)

        self.delta_h = np.float64(np.random.uniform(0.0, 0.025))
        table_pos = self._model.body("table").pos
        table_pos[2] = self.delta_h + self._table_z0
        self._model.body("table").pos = table_pos
        for gid in self._table_leg_geom_ids:
            self._model.geom_size[gid, 1] = self._table_leg_half_len0[gid] + self.delta_h

        base_xy = np.random.uniform(_HANOI_BASE_SAMPLING_BOUNDS[0], _HANOI_BASE_SAMPLING_BOUNDS[1])
        base_pos = np.array([base_xy[0], base_xy[1], self._base_body_z0 + self.delta_h], dtype=np.float64)
        base_delta_xy = base_pos[:2] - self._base_init_pos[:2]
        self.base_ori_pos = base_pos
        self._model.body_pos[self._base_body_id] = self.base_ori_pos

        self._apply_reset_tower_state(_HANOI_RESET_TOWER_STATE, base_delta_xy)

        self._data.qpos[self._so101_qpos_adr] = _SO101_HOME
        self._data.ctrl[self._so101_ctrl_ids] = _SO101_HOME
        mujoco.mj_forward(self._model, self._data)

        self.env_step = 0
        self._prime_rgb_array_renderer()

        success, metrics = self._compute_success_metrics()
        obs = self._compute_observation()

        return obs, {
            "succeed": False,
            "tower_state": metrics["tower_state"],
            "disk_assignment": metrics["disk_assignment"],
        }

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        start_time = time.time()

        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        lo = self._ctrl_ranges[:, 0]
        hi = self._ctrl_ranges[:, 1]
        target = lo + (action + 1.0) * 0.5 * (hi - lo)

        for _ in range(self._n_substeps):
            self._data.ctrl[self._so101_ctrl_ids] = target
            mujoco.mj_step(self._model, self._data)

        success, metrics = self._compute_success_metrics()
        obs = self._compute_observation()
        self.env_step += 1
        terminated = self.env_step >= 1500 or success

        if self.render_mode == "human":
            try:
                self._viewer.render("human")
            except Exception:
                pass

        dt = time.time() - start_time
        time.sleep(max(0.0, (1.0 / self.hz) - dt))

        reward = 1.0 if success else 0.0
        info = {
            "succeed": success,
            "tower_state": metrics["tower_state"],
            "disk_assignment": metrics["disk_assignment"],
            "all_disks_static": metrics["all_disks_static"],
        }
        return obs, reward, terminated, False, info

    # ---- Hanoi success/tower-state logic (robot-agnostic; unchanged from Panda env) ----

    def _disk_is_static(self, disk_name: str) -> bool:
        qvel_adr = self._disk_qvel_adr[disk_name]
        return float(np.linalg.norm(self._data.qvel[qvel_adr : qvel_adr + 6])) <= self._static_qvel_tol

    def _get_post_world_positions(self) -> Dict[str, np.ndarray]:
        return {
            post_name: np.array(self._data.site_xpos[site_id], dtype=np.float64)
            for post_name, site_id in self._post_site_ids.items()
        }

    def _base_top_world_z(self) -> float:
        base_body = self._data.body("hanoi_base")
        return float(base_body.xpos[2] + self._base_upper_top_local_z)

    def _reconstruct_tower_state(self) -> Dict[str, Any]:
        post_positions = self._get_post_world_positions()
        base_top_z = self._base_top_world_z()

        disk_assignment = {}
        disk_positions = {}
        disk_static = {}
        towers = {"A": [], "B": [], "C": []}

        for disk_name in self._disk_names:
            pos = np.array(self._data.body(disk_name).xpos, dtype=np.float64)
            disk_positions[disk_name] = pos
            disk_static[disk_name] = self._disk_is_static(disk_name)

            closest_post = min(post_positions.keys(), key=lambda k: np.linalg.norm(pos[:2] - post_positions[k][:2]))
            closest_dist = float(np.linalg.norm(pos[:2] - post_positions[closest_post][:2]))

            if closest_dist <= self._post_assignment_tol:
                disk_assignment[disk_name] = closest_post
                towers[closest_post].append(disk_name)
            else:
                disk_assignment[disk_name] = "table" if pos[2] <= base_top_z + 0.03 else "floating"

        for post_name in towers:
            towers[post_name].sort(key=lambda name: disk_positions[name][2])

        return {
            "post_positions": post_positions,
            "base_top_z": float(base_top_z),
            "disk_assignment": disk_assignment,
            "disk_positions": disk_positions,
            "disk_static": disk_static,
            "towers": towers,
        }

    def _compute_success_metrics(self) -> Tuple[bool, Dict[str, Any]]:
        state = self._reconstruct_tower_state()
        towers = state["towers"]
        disk_positions = state["disk_positions"]
        disk_static = state["disk_static"]
        post_positions = state["post_positions"]
        base_top_z = state["base_top_z"]

        tower_state = {post: list(names) for post, names in towers.items()}
        all_disks_static = all(bool(v) for v in disk_static.values())

        target_stack = ["hanoi_disk_large", "hanoi_disk_medium", "hanoi_disk_small"]
        if towers["C"] == target_stack and len(towers["A"]) == 0 and len(towers["B"]) == 0:
            running_height = base_top_z
            xy_error_max = 0.0
            height_error_max = 0.0
            for disk_name in target_stack:
                expected_center_z = running_height + self._disk_half_height[disk_name]
                running_height += 2.0 * self._disk_half_height[disk_name]
                pos = disk_positions[disk_name]
                xy_error_max = max(xy_error_max, float(np.linalg.norm(pos[:2] - post_positions["C"][:2])))
                height_error_max = max(height_error_max, float(abs(pos[2] - expected_center_z)))
            success = (
                xy_error_max <= self._post_assignment_tol
                and height_error_max <= self._placement_height_tol
                and all_disks_static
            )
        else:
            success = False

        return bool(success), {
            "tower_state": tower_state,
            "disk_assignment": dict(state["disk_assignment"]),
            "all_disks_static": bool(all_disks_static),
        }

    def render(self):
        return [self._viewer.render(render_mode="rgb_array", camera_id=cam_id) for cam_id in self.camera_id]

    def _compute_observation(self) -> dict:
        tcp_pos = self._data.sensor("so101/tcp_pos").data
        tcp_quat = self._data.sensor("so101/tcp_quat").data
        tcp_pose = np.concatenate([tcp_pos, tcp_quat])
        joint_pos = self._data.qpos[self._so101_qpos_adr].copy()

        obs = {"state": {}}
        if self.image_obs:
            obs["images"] = {}
            obs["images"]["ego"], obs["images"]["wrist"] = self.render()

        obs["state"] = {
            "tcp_pose": tcp_pose,
            "joint_pos": joint_pos,
            "hanoi_base_ori_pos": self.base_ori_pos,
            "table_delta_height": self.delta_h,
        }
        return obs


if __name__ == "__main__":
    env = So101SingleHanoiGymEnv(render_mode="human")
    obs, info = env.reset()
    for _ in range(200):
        obs, rew, done, trunc, info = env.step(np.zeros(_N_SO101, dtype=np.float32))
        if done or trunc:
            break
    env.close()
