# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Sequence

import torch

from isaaclab.assets import RigidObject
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, subtract_frame_transforms
import os
import isaaclab.utils.math as math_utils

_ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../../../../../../")
)
_OBS_LOG_PATH = os.path.join(_ROOT_DIR, "obs_debug_pouring4.log")


def _append_obs_log(line: str) -> None:
    try:
        with open(_OBS_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def reset_bead_in_cup(
    env,
    env_ids: Sequence[int],
    cup_name: str = "object",
    bead_name: str = "bead",
    offset: tuple[float, float, float] = (0.0, 0.0, 0.05),
) -> None:
    """Reset bead pose to be inside the source cup."""
    cup: RigidObject = env.scene[cup_name]
    bead: RigidObject = env.scene[bead_name]

    cup_pos = cup.data.root_pos_w[env_ids]
    cup_quat = cup.data.root_quat_w[env_ids]

    offset_vec = torch.tensor(offset, device=env.device, dtype=torch.float32)
    offset_vec = offset_vec.expand(cup_pos.shape[0], 3)
    bead_pos = cup_pos + quat_apply(cup_quat, offset_vec)
    bead_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).expand(cup_pos.shape[0], 4)

    zeros = torch.zeros_like(bead_pos)
    root_state = torch.cat([bead_pos, bead_quat, zeros, zeros], dim=-1)
    bead.write_root_state_to_sim(root_state, env_ids=env_ids)


def reset_root_state_uniform_robot_frame(
    env,
    env_ids: Sequence[int],
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    base_body_name: str | None = None,
) -> None:
    """Reset asset root using a pose sampled in the robot root frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]

    root_states = asset.data.default_root_state[env_ids].clone()

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    pos_offset = rand_samples[:, 0:3]
    if base_body_name is not None and base_body_name in robot.data.body_names:
        body_idx = robot.data.body_names.index(base_body_name)
        robot_pos_w = robot.data.body_pos_w[env_ids, body_idx]
        robot_quat_w = robot.data.body_quat_w[env_ids, body_idx]
    else:
        robot_pos_w = robot.data.root_pos_w[env_ids]
        robot_quat_w = robot.data.root_quat_w[env_ids]
    positions = robot_pos_w + quat_apply(robot_quat_w, pos_offset)

    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    velocities = root_states[:, 7:13] + rand_samples

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_cup_from_tcp_offset(
    env,
    env_ids: Sequence[int],
    cup_name: str,
    tcp_body_name: str,
    offset: tuple[float, float, float] = (-0.1, 0.0, 0.05),
    yaw: float = -1.57079632679,
) -> None:
    """Reset cup so its pose makes the TCP sit at a fixed offset in the cup frame."""
    robot = env.scene["robot"]
    cup: RigidObject = env.scene[cup_name]

    body_idx = robot.data.body_names.index(tcp_body_name)
    tcp_pos_w = robot.data.body_pos_w[env_ids, body_idx]

    # fixed cup orientation (yaw about world z)
    cup_quat = math_utils.quat_from_euler_xyz(
        torch.zeros_like(tcp_pos_w[:, 0]),
        torch.zeros_like(tcp_pos_w[:, 0]),
        torch.full_like(tcp_pos_w[:, 0], yaw),
    )

    offset_vec = torch.tensor(offset, device=env.device, dtype=torch.float32).expand(tcp_pos_w.shape[0], 3)
    cup_pos_w = tcp_pos_w - quat_apply(cup_quat, offset_vec)

    zeros = torch.zeros_like(cup_pos_w)
    root_state = torch.cat([cup_pos_w, cup_quat, zeros, zeros], dim=-1)
    cup.write_root_state_to_sim(root_state, env_ids=env_ids)


def reset_robot_tcp_to_cups(
    env,
    env_ids: Sequence[int] | None,
    left_cup_name: str = "object",
    right_cup_name: str = "object2",
    left_tcp_body_name: str = "openarm_left_ee_tcp",
    right_tcp_body_name: str = "openarm_right_ee_tcp",
    left_joint_names: Sequence[str] | None = None,
    right_joint_names: Sequence[str] | None = None,
    mirror_signs: Sequence[float] | None = None,
    offset: tuple[float, float, float] = (-0.1, 0.0, 0.05),
    ik_method: str = "dls",
    ik_lambda: float = 0.5,
    ik_iters: int = 5,
    max_delta: float = 0.2,
) -> None:
    """Reset robot joints so each TCP reaches a fixed offset in each cup frame."""
    robot = env.scene["robot"]

    # Clear log once per run to keep reset diagnostics scoped to the current session.
    if not getattr(env, "_obs_log_cleared", False):
        try:
            with open(_OBS_LOG_PATH, "w", encoding="utf-8") as f:
                f.write("")
        except OSError:
            pass
        env._obs_log_cleared = True

    # Only run once after environment startup (cups are fixed).
    if getattr(env, "_tcp_reset_done", False):
        return

    if env_ids is None or isinstance(env_ids, slice):
        env_ids_t = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids_t = torch.as_tensor(env_ids, device=env.device)

    if left_joint_names is None:
        left_joint_names = [f"openarm_left_joint{i}" for i in range(1, 8)]
    if right_joint_names is None:
        right_joint_names = [f"openarm_right_joint{i}" for i in range(1, 8)]
    if mirror_signs is None:
        # default mirror based on initial pose symmetry: right = sign * left (kept for logging/backward compat)
        mirror_signs = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0]

    joint_names = robot.data.joint_names
    left_joint_ids = [joint_names.index(n) for n in left_joint_names]
    right_joint_ids = [joint_names.index(n) for n in right_joint_names]

    body_names = robot.data.body_names
    left_body_idx = body_names.index(left_tcp_body_name)
    right_body_idx = body_names.index(right_tcp_body_name)

    root_pos_w = robot.data.root_pos_w[env_ids_t]
    root_quat_w = robot.data.root_quat_w[env_ids_t]

    # desired TCP positions from cup frames (world -> root frame)
    offset_vec = torch.tensor(offset, device=env.device, dtype=torch.float32).expand(env_ids_t.shape[0], 3)

    left_cup = env.scene[left_cup_name]
    right_cup = env.scene[right_cup_name]
    left_cup_quat_w = left_cup.data.root_quat_w[env_ids_t]
    right_cup_quat_w = right_cup.data.root_quat_w[env_ids_t]
    left_des_pos_w = left_cup.data.root_pos_w[env_ids_t] + quat_apply(left_cup_quat_w, offset_vec)
    right_des_pos_w = right_cup.data.root_pos_w[env_ids_t] + quat_apply(right_cup_quat_w, offset_vec)

    left_des_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, left_des_pos_w)
    right_des_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, right_des_pos_w)

    # current TCP poses in root frame
    left_pos_w = robot.data.body_pos_w[env_ids_t, left_body_idx]
    right_pos_w = robot.data.body_pos_w[env_ids_t, right_body_idx]
    left_quat_w = robot.data.body_quat_w[env_ids_t, left_body_idx]
    right_quat_w = robot.data.body_quat_w[env_ids_t, right_body_idx]

    left_pos_b, left_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, left_pos_w, left_quat_w)
    right_pos_b, right_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, right_pos_w, right_quat_w)

    # desired TCP orientations (world): align tcp_x with world z for a stable reset pose
    world_z = torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=left_cup_quat_w.dtype)
    world_x = torch.tensor([1.0, 0.0, 0.0], device=env.device, dtype=left_cup_quat_w.dtype)
    left_x = world_z.expand(env_ids_t.shape[0], 3)
    right_x = world_z.expand(env_ids_t.shape[0], 3)
    left_z_seed = world_x.expand(env_ids_t.shape[0], 3)
    right_z_seed = world_x.expand(env_ids_t.shape[0], 3)

    def _orthonormal_from_xz(x_axis: torch.Tensor, z_seed: torch.Tensor) -> torch.Tensor:
        x = x_axis / torch.linalg.norm(x_axis, dim=-1, keepdim=True)
        z = z_seed - torch.sum(z_seed * x, dim=-1, keepdim=True) * x
        z_norm = torch.linalg.norm(z, dim=-1, keepdim=True)
        # fallback if nearly parallel
        z = torch.where(z_norm > 1e-6, z / z_norm, torch.tensor([0.0, 0.0, 1.0], device=z.device))
        y = torch.cross(z, x, dim=-1)
        y = y / torch.linalg.norm(y, dim=-1, keepdim=True)
        z = torch.cross(x, y, dim=-1)
        return torch.stack([x, y, z], dim=-1)

    left_tcp_R_w = _orthonormal_from_xz(left_x, left_z_seed)
    right_tcp_R_w = _orthonormal_from_xz(right_x, right_z_seed)
    left_des_quat_w = math_utils.quat_from_matrix(left_tcp_R_w)
    right_des_quat_w = math_utils.quat_from_matrix(right_tcp_R_w)

    # desired TCP orientations in root frame
    root_quat_inv = math_utils.quat_inv(root_quat_w)
    left_des_quat_b = math_utils.quat_mul(root_quat_inv, left_des_quat_w)
    right_des_quat_b = math_utils.quat_mul(root_quat_inv, right_des_quat_w)

    # jacobians in root frame
    jacobians_w = robot.root_physx_view.get_jacobians()[env_ids_t]
    base_rot = math_utils.quat_inv(root_quat_w)
    base_rot_matrix = math_utils.matrix_from_quat(base_rot)

    # map body index into jacobian index (fixed-base uses body_idx-1)
    if robot.is_fixed_base:
        left_jacobi_body_idx = left_body_idx - 1
        right_jacobi_body_idx = right_body_idx - 1
        left_jacobi_joint_ids = left_joint_ids
        right_jacobi_joint_ids = right_joint_ids
    else:
        left_jacobi_body_idx = left_body_idx
        right_jacobi_body_idx = right_body_idx
        left_jacobi_joint_ids = [i + 6 for i in left_joint_ids]
        right_jacobi_joint_ids = [i + 6 for i in right_joint_ids]

    left_jac_w = jacobians_w[:, left_jacobi_body_idx, :, :][:, :, left_jacobi_joint_ids]
    right_jac_w = jacobians_w[:, right_jacobi_body_idx, :, :][:, :, right_jacobi_joint_ids]
    left_jac_b = left_jac_w.clone()
    right_jac_b = right_jac_w.clone()
    left_jac_b[:, :3, :] = torch.bmm(base_rot_matrix, left_jac_b[:, :3, :])
    left_jac_b[:, 3:, :] = torch.bmm(base_rot_matrix, left_jac_b[:, 3:, :])
    right_jac_b[:, :3, :] = torch.bmm(base_rot_matrix, right_jac_b[:, :3, :])
    right_jac_b[:, 3:, :] = torch.bmm(base_rot_matrix, right_jac_b[:, 3:, :])

    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method=ik_method,
        ik_params={"lambda_val": ik_lambda},
    )
    ik = DifferentialIKController(ik_cfg, num_envs=env_ids_t.shape[0], device=env.device)

    for _ in range(max(1, ik_iters)):
        # refresh current poses
        left_pos_w = robot.data.body_pos_w[env_ids_t, left_body_idx]
        right_pos_w = robot.data.body_pos_w[env_ids_t, right_body_idx]
        left_quat_w = robot.data.body_quat_w[env_ids_t, left_body_idx]
        right_quat_w = robot.data.body_quat_w[env_ids_t, right_body_idx]
        left_pos_b, left_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, left_pos_w, left_quat_w)
        right_pos_b, right_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, right_pos_w, right_quat_w)

        # left arm IK
        left_cmd = torch.cat([left_des_pos_b, left_des_quat_b], dim=1)
        ik.set_command(left_cmd, ee_pos=left_pos_b, ee_quat=left_quat_b)
        left_joint_pos = robot.data.joint_pos[env_ids_t][:, left_joint_ids]
        left_joint_pos_des = ik.compute(left_pos_b, left_quat_b, left_jac_b, left_joint_pos)
        if max_delta is not None and max_delta > 0.0:
            left_joint_pos_des = torch.clamp(left_joint_pos_des, left_joint_pos - max_delta, left_joint_pos + max_delta)
        robot.write_joint_state_to_sim(
            left_joint_pos_des, torch.zeros_like(left_joint_pos_des), joint_ids=left_joint_ids, env_ids=env_ids_t
        )

        # right arm IK (do not mirror from left)
        right_cmd = torch.cat([right_des_pos_b, right_des_quat_b], dim=1)
        ik.set_command(right_cmd, ee_pos=right_pos_b, ee_quat=right_quat_b)
        right_joint_pos = robot.data.joint_pos[env_ids_t][:, right_joint_ids]
        right_joint_pos_des = ik.compute(right_pos_b, right_quat_b, right_jac_b, right_joint_pos)
        if max_delta is not None and max_delta > 0.0:
            right_joint_pos_des = torch.clamp(
                right_joint_pos_des, right_joint_pos - max_delta, right_joint_pos + max_delta
            )
        robot.write_joint_state_to_sim(
            right_joint_pos_des, torch.zeros_like(right_joint_pos_des), joint_ids=right_joint_ids, env_ids=env_ids_t
        )

    # symmetry log (root frame) for the first env in this reset batch
    if env_ids_t.numel() > 0:
        env0 = env_ids_t[0].item()
        left_tcp_pos_b, _ = subtract_frame_transforms(
            robot.data.root_pos_w[env0 : env0 + 1],
            robot.data.root_quat_w[env0 : env0 + 1],
            robot.data.body_pos_w[env0 : env0 + 1, left_body_idx],
        )
        right_tcp_pos_b, _ = subtract_frame_transforms(
            robot.data.root_pos_w[env0 : env0 + 1],
            robot.data.root_quat_w[env0 : env0 + 1],
            robot.data.body_pos_w[env0 : env0 + 1, right_body_idx],
        )
        left_obj_pos_b, _ = subtract_frame_transforms(
            robot.data.root_pos_w[env0 : env0 + 1],
            robot.data.root_quat_w[env0 : env0 + 1],
            left_cup.data.root_pos_w[env0 : env0 + 1],
        )
        right_obj_pos_b, _ = subtract_frame_transforms(
            robot.data.root_pos_w[env0 : env0 + 1],
            robot.data.root_quat_w[env0 : env0 + 1],
            right_cup.data.root_pos_w[env0 : env0 + 1],
        )

        # symmetry deltas: x/z should match, y should be opposite
        lt = left_tcp_pos_b[0].cpu().numpy()
        rt = right_tcp_pos_b[0].cpu().numpy()
        lo = left_obj_pos_b[0].cpu().numpy()
        ro = right_obj_pos_b[0].cpu().numpy()
        tcp_dx = lt[0] - rt[0]
        tcp_dy = lt[1] + rt[1]
        tcp_dz = lt[2] - rt[2]
        obj_dx = lo[0] - ro[0]
        obj_dy = lo[1] + ro[1]
        obj_dz = lo[2] - ro[2]
        _append_obs_log(
            "[RESET_SYM] "
            + f"tcp_L={lt} tcp_R={rt} "
            + f"obj_L={lo} obj_R={ro} "
            + f"tcp_d(x,y,z)=({tcp_dx:.4f},{tcp_dy:.4f},{tcp_dz:.4f}) "
            + f"obj_d(x,y,z)=({obj_dx:.4f},{obj_dy:.4f},{obj_dz:.4f})"
        )
        # log joint mirroring for sanity check
        lq = left_joint_pos_des[0].cpu().numpy()
        rq = right_joint_pos_des[0].cpu().numpy()
        _append_obs_log(f"[RESET_JOINTS] left={lq} right={rq} mirror_signs={list(mirror_signs)}")

    # Refresh commands after cup + TCP reset to ensure targets are near the cups on step 0.
    try:
        env.command_manager.reset(env_ids_t)
        env.command_manager.compute(0.0)
    except Exception:
        pass

    env._tcp_reset_done = True
