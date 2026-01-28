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

"""Event functions for Primitive B: Contact Force Hold."""

from __future__ import annotations

from typing import TYPE_CHECKING
import math

import torch
from pxr import Gf, Sdf, UsdGeom, Vt

from isaaclab.managers import SceneEntityCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils import math as math_utils

from isaaclab.envs import mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = [
    "reset_contact_force_hold_state",
    "apply_micro_disturbances",
    "linear_interpolate_fn",
    "delayed_linear_interpolate_fn",
    "randomize_object_com",
    "randomize_object_shape_scale",
]


def _linear_schedule(step: int, start: float, end: float, duration: int) -> float:
    if duration <= 0:
        return end
    alpha = min(float(step) / float(duration), 1.0)
    return start + (end - start) * alpha


def _reset_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg,
    pose_range: dict[str, tuple[float, float]],
):
    # Reuse standard reset logic for root pose.
    mdp.reset_root_state_uniform(
        env=env,
        env_ids=env_ids,
        pose_range=pose_range,
        velocity_range={},
        asset_cfg=object_cfg,
    )


def _reset_object_pose_from_grasp(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg,
    robot,
    grasp_frame_name: str,
    offset_range: dict[str, tuple[float, float]],
):
    """Place object relative to a grasp frame using local offset ranges."""
    try:
        grasp_id = robot.body_names.index(grasp_frame_name)
    except ValueError:
        return False

    grasp_pos_w = robot.data.body_pos_w[env_ids, grasp_id]
    grasp_quat_w = robot.data.body_quat_w[env_ids, grasp_id]

    offsets = torch.stack(
        [
            math_utils.sample_uniform(offset_range["x"][0], offset_range["x"][1], (len(env_ids),), device=env.device),
            math_utils.sample_uniform(offset_range["y"][0], offset_range["y"][1], (len(env_ids),), device=env.device),
            math_utils.sample_uniform(offset_range["z"][0], offset_range["z"][1], (len(env_ids),), device=env.device),
        ],
        dim=-1,
    )
    offsets_w = math_utils.quat_apply(grasp_quat_w, offsets)
    target_pos = grasp_pos_w + offsets_w

    obj = env.scene[object_cfg.name]
    root_state = obj.data.default_root_state[env_ids].clone()
    root_state[:, :3] = target_pos
    root_state[:, 7:] = 0.0
    obj.write_root_state_to_sim(root_state, env_ids=env_ids)
    return True


def _set_arm_targets(
    robot,
    env_ids: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_names: list[str],
    target_positions: dict[str, float],
    noise: float,
):
    joint_ids = [robot.joint_names.index(name) for name in joint_names]
    joint_ids_t = torch.tensor(joint_ids, device=robot.device, dtype=torch.long)
    joint_limits = robot.data.joint_pos_limits[:, joint_ids_t, :]
    if joint_limits.dim() == 3:
        joint_limits = joint_limits[0]
    targets = torch.tensor(
        [target_positions.get(name, robot.data.default_joint_pos[0, jid].item()) for name, jid in zip(joint_names, joint_ids)],
        device=robot.device,
    )
    targets = targets.unsqueeze(0).repeat(len(env_ids), 1)
    if noise > 0.0:
        targets = targets + math_utils.sample_uniform(-noise, noise, targets.shape, device=robot.device)
    targets = torch.clamp(targets, joint_limits[:, 0], joint_limits[:, 1])
    joint_pos[env_ids[:, None], joint_ids_t] = targets




def _set_hand_targets(
    robot,
    env_ids: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_names: list[str],
    grasp_bias: float,
    noise: float,
):
    joint_ids = [robot.joint_names.index(name) for name in joint_names]
    joint_ids_t = torch.tensor(joint_ids, device=robot.device, dtype=torch.long)
    joint_limits = robot.data.joint_pos_limits[:, joint_ids_t, :]
    if joint_limits.dim() == 3:
        joint_limits = joint_limits[0]
    low = joint_limits[:, 0]
    high = joint_limits[:, 1]
    targets = low + grasp_bias * (high - low)
    targets = targets.unsqueeze(0).repeat(len(env_ids), 1)
    if noise > 0.0:
        targets = targets + math_utils.sample_uniform(-noise, noise, targets.shape, device=robot.device)
    targets = torch.clamp(targets, low, high)
    joint_pos[env_ids[:, None], joint_ids_t] = targets


def _apply_hand_bias_delta(
    robot,
    env_ids: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_names: list[str],
    exclude_prefix: str,
    delta_bias: float,
):
    """Increase bias toward closed pose for selected joints."""
    if delta_bias <= 0.0:
        return
    joint_ids = [robot.joint_names.index(name) for name in joint_names]
    joint_ids_t = torch.tensor(joint_ids, device=robot.device, dtype=torch.long)
    joint_limits = robot.data.joint_pos_limits[:, joint_ids_t, :]
    if joint_limits.dim() == 3:
        joint_limits = joint_limits[0]
    low = joint_limits[:, 0]
    high = joint_limits[:, 1]
    bias_delta = delta_bias * (high - low)

    mask = torch.tensor(
        [not name.startswith(exclude_prefix) for name in joint_names],
        device=robot.device,
        dtype=torch.bool,
    )
    if not mask.any():
        return
    joint_pos[env_ids[:, None], joint_ids_t[mask]] = torch.clamp(
        joint_pos[env_ids[:, None], joint_ids_t[mask]] + bias_delta[mask],
        low[mask],
        high[mask],
    )


def _apply_hand_bias_prefix(
    robot,
    env_ids: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_names: list[str],
    include_prefix: str,
    delta_bias: float,
):
    """Increase bias toward closed pose for a specific prefix."""
    if delta_bias <= 0.0:
        return
    joint_ids = [robot.joint_names.index(name) for name in joint_names]
    joint_ids_t = torch.tensor(joint_ids, device=robot.device, dtype=torch.long)
    joint_limits = robot.data.joint_pos_limits[:, joint_ids_t, :]
    if joint_limits.dim() == 3:
        joint_limits = joint_limits[0]
    low = joint_limits[:, 0]
    high = joint_limits[:, 1]
    bias_delta = delta_bias * (high - low)

    mask = torch.tensor(
        [name.startswith(include_prefix) for name in joint_names],
        device=robot.device,
        dtype=torch.bool,
    )
    if not mask.any():
        return
    joint_pos[env_ids[:, None], joint_ids_t[mask]] = torch.clamp(
        joint_pos[env_ids[:, None], joint_ids_t[mask]] + bias_delta[mask],
        low[mask],
        high[mask],
    )


def _apply_hand_offsets(
    robot,
    env_ids: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_names: list[str],
    offset_targets: dict[str, float],
    offset_noise: float,
):
    """Override hand joint positions with explicit offset targets."""
    fixed_zero_or_target = {
        "lj_dg_1_1",
        "lj_dg_2_1",
        "lj_dg_3_1",
        "lj_dg_4_1",
        "lj_dg_5_1",
        "lj_dg_5_2",
    }
    joint_ids = [robot.joint_names.index(name) for name in joint_names]
    joint_ids_t = torch.tensor(joint_ids, device=robot.device, dtype=torch.long)
    joint_limits = robot.data.joint_pos_limits[:, joint_ids_t, :]
    if joint_limits.dim() == 3:
        joint_limits = joint_limits[0]
    base_targets = torch.tensor(
        [offset_targets.get(name, joint_pos[env_ids[0], jid].item()) for name, jid in zip(joint_names, joint_ids)],
        device=robot.device,
    )
    targets = base_targets.unsqueeze(0).repeat(len(env_ids), 1)
    if offset_noise > 0.0:
        noise = math_utils.sample_uniform(-offset_noise, offset_noise, targets.shape, device=robot.device)
        targets = targets + noise

        # Do not cross zero when adding noise.
        base_pos = base_targets > 0.0
        base_neg = base_targets < 0.0
        if base_pos.any():
            targets[:, base_pos] = torch.clamp(targets[:, base_pos], min=0.0)
        if base_neg.any():
            targets[:, base_neg] = torch.clamp(targets[:, base_neg], max=0.0)

    # Fixed joints: choose either base target or zero (no noise).
    fixed_mask = torch.tensor(
        [name in fixed_zero_or_target for name in joint_names],
        device=robot.device,
        dtype=torch.bool,
    )
    if fixed_mask.any():
        rand = torch.rand((len(env_ids), fixed_mask.sum().item()), device=robot.device)
        choose_base = rand < 0.5
        fixed_vals = torch.where(
            choose_base,
            base_targets[fixed_mask].expand(len(env_ids), -1),
            torch.zeros((len(env_ids), fixed_mask.sum().item()), device=robot.device),
        )
        targets[:, fixed_mask] = fixed_vals
    targets = torch.clamp(targets, joint_limits[:, 0], joint_limits[:, 1])
    joint_pos[env_ids[:, None], joint_ids_t] = targets


def reset_contact_force_hold_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    left_hand_joint_names: list[str],
    right_hand_joint_names: list[str],
    left_arm_joint_names: list[str],
    right_arm_joint_names: list[str],
    arm_joint_targets: dict[str, float],
    object_pose_b1: dict[str, tuple[float, float]],
    object_pose_b2: dict[str, tuple[float, float]],
    grasp_frame_name: str | None,
    object_offset_b1: dict[str, tuple[float, float]] | None,
    object_offset_b2: dict[str, tuple[float, float]] | None,
    b1_prob_init: float,
    b1_prob_final: float,
    b1_prob_steps: int,
    hand_bias_b1: float,
    hand_bias_b2: float,
    hand_extra_bias_b1: float,
    hand_extra_bias_b2: float,
    hand_extra_bias_lj_dg_4: float,
    left_hand_offset_targets: dict[str, float] | None,
    left_hand_offset_noise: float = 0.0,
    hand_noise: float = 0.0,
    arm_noise: float = 0.0,
    active_hand: str = "left",
    left_arm_fixed_joints: list[str] | None = None,
    left_arm_fixed_noise_deg: float = 0.0,
):
    """Reset for Primitive B with curriculum-driven contact modes.

    Mode B1: Object inside grasp, hands closed.
    Mode B2: Object near fingertips, hands slightly open.
    """
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    robot = env.scene[robot_cfg.name]
    joint_pos = robot.data.default_joint_pos[env_ids].clone()

    b1_prob = _linear_schedule(int(env.common_step_counter), b1_prob_init, b1_prob_final, b1_prob_steps)
    rand = torch.rand(len(env_ids), device=env.device)
    b1_mask = rand < b1_prob
    env_ids_b1 = env_ids[b1_mask]
    env_ids_b2 = env_ids[~b1_mask]
    env_ids_local = torch.arange(len(env_ids), device=env.device, dtype=torch.long)
    env_ids_b1_local = env_ids_local[b1_mask]
    env_ids_b2_local = env_ids_local[~b1_mask]

    active_hand = active_hand.lower().strip()
    use_left = active_hand != "right"

    if len(env_ids_b1) > 0:
        if use_left:
            _set_hand_targets(
                robot,
                env_ids_b1_local,
                joint_pos,
                left_hand_joint_names,
                hand_bias_b1,
                hand_noise,
            )
            _apply_hand_bias_delta(
                robot,
                env_ids_b1_local,
                joint_pos,
                left_hand_joint_names,
                "lj_dg_1_",
                hand_extra_bias_b1,
            )
            _apply_hand_bias_prefix(
                robot,
                env_ids_b1_local,
                joint_pos,
                left_hand_joint_names,
                "lj_dg_4_",
                hand_extra_bias_lj_dg_4,
            )
            if left_hand_offset_targets is not None:
                _apply_hand_offsets(
                    robot,
                    env_ids_b1_local,
                    joint_pos,
                    left_hand_joint_names,
                    left_hand_offset_targets,
                    left_hand_offset_noise,
                )
        else:
            _set_hand_targets(
                robot,
                env_ids_b1_local,
                joint_pos,
                right_hand_joint_names,
                hand_bias_b1,
                hand_noise,
            )

    if len(env_ids_b2) > 0:
        if use_left:
            _set_hand_targets(
                robot,
                env_ids_b2_local,
                joint_pos,
                left_hand_joint_names,
                hand_bias_b2,
                hand_noise,
            )
            _apply_hand_bias_delta(
                robot,
                env_ids_b2_local,
                joint_pos,
                left_hand_joint_names,
                "lj_dg_1_",
                hand_extra_bias_b2,
            )
            _apply_hand_bias_prefix(
                robot,
                env_ids_b2_local,
                joint_pos,
                left_hand_joint_names,
                "lj_dg_4_",
                hand_extra_bias_lj_dg_4,
            )
            if left_hand_offset_targets is not None:
                _apply_hand_offsets(
                    robot,
                    env_ids_b2_local,
                    joint_pos,
                    left_hand_joint_names,
                    left_hand_offset_targets,
                    left_hand_offset_noise,
                )
        else:
            _set_hand_targets(
                robot,
                env_ids_b2_local,
                joint_pos,
                right_hand_joint_names,
                hand_bias_b2,
                hand_noise,
            )

    # Arms/wrists to holding-ready pose for all envs.
    _set_arm_targets(robot, env_ids_local, joint_pos, left_arm_joint_names, arm_joint_targets, arm_noise)
    _set_arm_targets(robot, env_ids_local, joint_pos, right_arm_joint_names, arm_joint_targets, arm_noise)

    if left_arm_fixed_joints:
        fixed_noise = math.radians(left_arm_fixed_noise_deg)
        _set_arm_targets(robot, env_ids_local, joint_pos, left_arm_fixed_joints, arm_joint_targets, fixed_noise)

    joint_vel = torch.zeros_like(robot.data.default_joint_vel[env_ids])
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    # Keep PD targets aligned with reset pose to avoid immediate droop.
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    env.scene.write_data_to_sim()

    if len(env_ids_b1) > 0:
        if grasp_frame_name and object_offset_b1:
            used_grasp = _reset_object_pose_from_grasp(
                env, env_ids_b1, object_cfg, robot, grasp_frame_name, object_offset_b1
            )
            if not used_grasp:
                _reset_object_pose(env, env_ids_b1, object_cfg, object_pose_b1)
        else:
            _reset_object_pose(env, env_ids_b1, object_cfg, object_pose_b1)

    if len(env_ids_b2) > 0:
        if grasp_frame_name and object_offset_b2:
            used_grasp = _reset_object_pose_from_grasp(
                env, env_ids_b2, object_cfg, robot, grasp_frame_name, object_offset_b2
            )
            if not used_grasp:
                _reset_object_pose(env, env_ids_b2, object_cfg, object_pose_b2)
        else:
            _reset_object_pose(env, env_ids_b2, object_cfg, object_pose_b2)

    # Debug counters for B1/B2 ratio.
    if len(env_ids) > 0:
        env._debug_b1_total = getattr(env, "_debug_b1_total", 0) + int(env_ids_b1.numel())
        env._debug_b2_total = getattr(env, "_debug_b2_total", 0) + int(env_ids_b2.numel())


def linear_interpolate_fn(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    old_value,
    initial_value,
    final_value,
    num_steps: int,
):
    """Linearly interpolate any tuple/dict/scalar based on step count."""
    step = int(env.common_step_counter)
    alpha = min(float(step) / float(max(num_steps, 1)), 1.0)

    def _interp(a, b):
        return a + (b - a) * alpha

    if isinstance(initial_value, dict):
        return {k: (_interp(initial_value[k][0], final_value[k][0]), _interp(initial_value[k][1], final_value[k][1]))
                for k in initial_value}
    if isinstance(initial_value, (tuple, list)):
        return tuple(_interp(a, b) for a, b in zip(initial_value, final_value))
    return _interp(initial_value, final_value)


def delayed_linear_interpolate_fn(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    old_value,
    initial_value,
    final_value,
    delay_steps: int,
    num_steps: int,
):
    """Hold initial_value for delay_steps, then linearly interpolate over num_steps."""
    step = int(env.common_step_counter)
    if step < delay_steps:
        alpha = 0.0
    else:
        alpha = min(float(step - delay_steps) / float(max(num_steps, 1)), 1.0)

    def _interp(a, b):
        return a + (b - a) * alpha

    if isinstance(initial_value, dict):
        return {k: (_interp(initial_value[k][0], final_value[k][0]), _interp(initial_value[k][1], final_value[k][1]))
                for k in initial_value}
    if isinstance(initial_value, (tuple, list)):
        return tuple(_interp(a, b) for a, b in zip(initial_value, final_value))
    return _interp(initial_value, final_value)

def apply_micro_disturbances(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    robot_cfg: SceneEntityCfg,
    force_range_initial: tuple[float, float],
    force_range_final: tuple[float, float],
    torque_range_initial: tuple[float, float],
    torque_range_final: tuple[float, float],
    steps: int,
):
    """Apply small external disturbances that grow over training."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    step = int(env.common_step_counter)
    force_low = _linear_schedule(step, force_range_initial[0], force_range_final[0], steps)
    force_high = _linear_schedule(step, force_range_initial[1], force_range_final[1], steps)
    torque_low = _linear_schedule(step, torque_range_initial[0], torque_range_final[0], steps)
    torque_high = _linear_schedule(step, torque_range_initial[1], torque_range_final[1], steps)

    if force_high <= 0.0 and torque_high <= 0.0:
        return

    mdp.apply_external_force_torque(
        env=env,
        env_ids=env_ids,
        force_range=(force_low, force_high),
        torque_range=(torque_low, torque_high),
        asset_cfg=robot_cfg,
    )


def randomize_object_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    object_cfg: SceneEntityCfg,
    com_range: dict[str, tuple[float, float]],
):
    """Randomize object CoM offsets for both rigid objects and articulations."""
    asset = env.scene[object_cfg.name]
    coms = asset.root_physx_view.get_coms().clone()
    coms_device = coms.device
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=coms_device)
    else:
        env_ids = env_ids.to(device=coms_device)

    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=coms_device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=coms_device)
    if coms.dim() == 2:
        # RigidObject case: (num_envs, 3)
        coms[env_ids, :3] += rand_samples
        asset.root_physx_view.set_coms(coms, env_ids)
        return

    # Articulation case: (num_envs, num_bodies, 3)
    if object_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, device=coms_device)
    else:
        body_ids = torch.tensor(object_cfg.body_ids, device=coms_device)
    coms[env_ids[:, None], body_ids, :3] += rand_samples[:, None, :]
    asset.root_physx_view.set_coms(coms, env_ids)


def randomize_object_shape_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    object_cfg: SceneEntityCfg,
    cylinder_scale_xy_range: tuple[float, float],
    cube_scale_xy_range: tuple[float, float],
):
    """Apply continuous scale randomization to the object root based on mesh type."""
    if env.sim.is_playing():
        raise RuntimeError(
            "Randomizing scale while simulation is running leads to unpredictable behaviors. "
            "Use event mode 'prestartup' for this term."
        )

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    stage = get_current_stage()
    obj = env.scene[object_cfg.name]
    prim_paths = sim_utils.find_matching_prim_paths(obj.cfg.prim_path)

    with Sdf.ChangeBlock():
        for env_id in env_ids.tolist():
            prim_path = prim_paths[env_id]
            mesh_path = prim_path + "/geometry/mesh"
            mesh_prim = stage.GetPrimAtPath(mesh_path)
            if not mesh_prim.IsValid():
                continue
            mesh_type = mesh_prim.GetTypeName()
            if mesh_type == "Cylinder":
                scale_xy = math_utils.sample_uniform(*cylinder_scale_xy_range, (1,), device="cpu")[0].item()
            elif mesh_type == "Cube":
                scale_xy = math_utils.sample_uniform(*cube_scale_xy_range, (1,), device="cpu")[0].item()
            else:
                continue

            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)
            scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
            has_scale_attr = scale_spec is not None
            if not has_scale_attr:
                scale_spec = Sdf.AttributeSpec(prim_spec, prim_path + ".xformOp:scale", Sdf.ValueTypeNames.Double3)
            scale_spec.default = Gf.Vec3f(scale_xy, scale_xy, 1.0)
            if not has_scale_attr:
                op_order_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
                if op_order_spec is None:
                    op_order_spec = Sdf.AttributeSpec(
                        prim_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                    )
                op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])
