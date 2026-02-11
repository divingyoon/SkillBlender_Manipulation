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

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup2"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos_w = obj.data.root_pos_w[:, :3]
    obj_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, obj_pos_w)
    return obj_pos_b


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup2"),
    eef_link_name: str = "rl_dg_ee",
) -> torch.Tensor:
    """Reward the end-effector approaching the cup using a tanh kernel."""
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos_w = obj.data.root_pos_w.clone()

    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    ee_pos_w = env.scene["robot"].data.body_pos_w[:, eef_idx]

    # Dynamic approach target in cup-local frame: approach high first, then descend when xy is aligned.
    base_offset = getattr(getattr(env, "cfg", None), "grasp2g_target_offset", (0.0, 0.0, 0.08))
    if isinstance(base_offset, (list, tuple)) and len(base_offset) == 3:
        xy_dist = torch.norm(ee_pos_w[:, :2] - obj_pos_w[:, :2], dim=1)

        z_high = 0.2
        z_low = base_offset[2]
        x_hi = 0.06
        x_lo = 0.015
        u = torch.clamp((x_hi - xy_dist) / (x_hi - x_lo), 0.0, 1.0)
        u = u * u * (3.0 - 2.0 * u)  # smoothstep
        dynamic_z = z_high * (1.0 - u) + z_low * u

        offset_local = torch.zeros(obj_pos_w.shape[0], 3, device=obj_pos_w.device, dtype=obj_pos_w.dtype)
        offset_local[:, 0] = base_offset[0]
        offset_local[:, 1] = base_offset[1]
        offset_local[:, 2] = dynamic_z
        offset_w = quat_apply(obj.data.root_quat_w, offset_local)
        target_pos_w = obj_pos_w + offset_w
        _maybe_visualize_approach_target(env, target_pos_w, obj.data.root_quat_w, marker_attr="_debug_approach_target_right")
    else:
        target_pos_w = obj_pos_w

    dist = torch.norm(target_pos_w - ee_pos_w, dim=1)
    reach_reward = 1 - torch.tanh(dist / std)
    reached_stable = _is_reaching_stably_complete(env, object_cfg, eef_link_name)
    return (1.0 - reached_stable) * reach_reward


def _is_reaching_complete(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    eef_link_name: str,
    reach_threshold: float = 0.01,
) -> torch.Tensor:
    """Check if EE reached the cup-local low approach target."""
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos_w = obj.data.root_pos_w.clone()

    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    ee_pos_w = env.scene["robot"].data.body_pos_w[:, eef_idx]

    base_offset = getattr(getattr(env, "cfg", None), "grasp2g_target_offset", (0.0, 0.0, 0.08))
    if isinstance(base_offset, (list, tuple)) and len(base_offset) == 3:
        offset_local = torch.zeros(obj_pos_w.shape[0], 3, device=obj_pos_w.device, dtype=obj_pos_w.dtype)
        offset_local[:, 0] = base_offset[0]
        offset_local[:, 1] = base_offset[1]
        offset_local[:, 2] = base_offset[2]
        offset_w = quat_apply(obj.data.root_quat_w, offset_local)
        target_pos_w = obj_pos_w + offset_w
    else:
        target_pos_w = obj_pos_w

    dist = torch.norm(target_pos_w - ee_pos_w, dim=1)
    return (dist < reach_threshold).float()


def _is_reaching_stably_complete(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    eef_link_name: str,
) -> torch.Tensor:
    """Gate reaching completion after N consecutive near-target steps."""
    cfg = getattr(env, "cfg", None)
    reach_threshold = float(getattr(cfg, "reach_switch_threshold", 0.01))
    hold_steps = int(getattr(cfg, "reach_switch_hold_steps", 10))
    hold_steps = max(1, hold_steps)

    reached_now = _is_reaching_complete(env, object_cfg, eef_link_name, reach_threshold=reach_threshold)
    reached_now_i64 = reached_now.to(dtype=torch.int64)

    if not hasattr(env, "_reach_hold_counter_right_approach"):
        env._reach_hold_counter_right_approach = torch.zeros(env.num_envs, device=env.device, dtype=torch.int64)
    counter = env._reach_hold_counter_right_approach

    step_count = int(getattr(env, "common_step_counter", -1))
    if not hasattr(env, "_reach_hold_counter_right_approach_last_step"):
        env._reach_hold_counter_right_approach_last_step = -2
    if env._reach_hold_counter_right_approach_last_step != step_count:
        reset_mask = (env.episode_length_buf == 0).squeeze(-1)
        counter[reset_mask] = 0

        counter = torch.where(reached_now_i64 > 0, counter + 1, torch.zeros_like(counter))
        env._reach_hold_counter_right_approach = counter
        env._reach_hold_counter_right_approach_last_step = step_count

    return (counter >= hold_steps).to(dtype=reached_now.dtype)


def _maybe_visualize_approach_target(
    env: ManagerBasedRLEnv,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    marker_attr: str,
) -> None:
    cfg = getattr(env, "cfg", None)
    if cfg is None or not getattr(cfg, "debug_approach_target_vis", True):
        return

    interval = int(getattr(cfg, "debug_approach_target_vis_interval", 10))
    step_count = int(getattr(env, "common_step_counter", 0))
    if interval > 1 and (step_count % interval) != 0:
        return

    if not hasattr(env, marker_attr):
        marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Debug/ApproachTargetRight")
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker = VisualizationMarkers(marker_cfg)
        marker.set_visibility(True)
        setattr(env, marker_attr, marker)

    marker = getattr(env, marker_attr)
    env_id = int(getattr(cfg, "debug_approach_target_vis_env_id", 0))
    env_id = max(0, min(env.num_envs - 1, env_id))
    marker.visualize(target_pos_w[env_id : env_id + 1], target_quat_w[env_id : env_id + 1])


def eef_to_object_orientation(
    env: ManagerBasedRLEnv,
    std: float,
    eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup2"),
) -> torch.Tensor:
    """Reward loose orientation alignment between end-effector axes and object axes."""
    object_quat = env.scene[object_cfg.name].data.root_quat_w
    body_quat_w = env.scene["robot"].data.body_quat_w
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    eef_quat = body_quat_w[:, eef_idx]

    x_axis = torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)

    eef_axes = [
        quat_apply(eef_quat, x_axis),
        quat_apply(eef_quat, y_axis),
        quat_apply(eef_quat, z_axis),
    ]
    obj_axes = [
        quat_apply(object_quat, x_axis),
        quat_apply(object_quat, y_axis),
        quat_apply(object_quat, z_axis),
    ]

    max_align = torch.zeros(env.num_envs, device=env.device)
    for eef_axis in eef_axes:
        for obj_axis in obj_axes:
            align = torch.abs(torch.sum(eef_axis * obj_axis, dim=1))
            max_align = torch.maximum(max_align, align)

    error = 1.0 - max_align
    return 1 - torch.tanh(error / std)


def _object_root_displacement_from_init(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Compute displacement from per-episode initial object root position."""
    obj: RigidObject = env.scene[object_cfg.name]
    attr_name = f"_init_root_pos_{object_cfg.name}"

    if not hasattr(env, attr_name):
        init_pos = obj.data.root_pos_w.clone()
    else:
        init_pos = getattr(env, attr_name)

    if hasattr(env, "reset_buf"):
        reset_mask = env.reset_buf.unsqueeze(1)
        init_pos = torch.where(reset_mask, obj.data.root_pos_w, init_pos)

    setattr(env, attr_name, init_pos)
    return torch.linalg.norm(obj.data.root_pos_w - init_pos, dim=1)


def object_root_displacement_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup2"),
    scale: float = 1.0,
) -> torch.Tensor:
    """Penalty for moving the cup from its initial pose."""
    return _object_root_displacement_from_init(env, object_cfg) * scale


def joint_pos_target_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target: float = 0.0,
) -> torch.Tensor:
    """Penalize deviation from a target joint position (L1)."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if joint_ids is None:
        joint_ids = slice(None)
    target_t = torch.tensor(target, device=asset.data.joint_pos.device, dtype=asset.data.joint_pos.dtype)
    diff = asset.data.joint_pos[:, joint_ids] - target_t
    return torch.sum(torch.abs(diff), dim=1)


def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize deviation from default joint positions (L1)."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if joint_ids is None:
        joint_ids = slice(None)
    diff = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    return torch.sum(torch.abs(diff), dim=1)
