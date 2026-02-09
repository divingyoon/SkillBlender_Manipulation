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
    obj_pos_w = obj.data.root_pos_w

    # Interpret target offset in the cup local frame so "side approach" stays consistent
    # even if cup orientation changes.
    offset = getattr(getattr(env, "cfg", None), "grasp2g_target_offset", (0.0, 0.0, 0.0))
    if isinstance(offset, (list, tuple)) and len(offset) == 3:
        offset_local = torch.tensor(offset, device=obj_pos_w.device, dtype=obj_pos_w.dtype).unsqueeze(0)
        offset_local = offset_local.expand(obj.data.root_quat_w.shape[0], -1)
        offset_w = quat_apply(obj.data.root_quat_w, offset_local)
        obj_pos_w = obj_pos_w + offset_w
        _maybe_visualize_approach_target(env, obj_pos_w, obj.data.root_quat_w, marker_attr="_debug_approach_target_right")

    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    ee_pos_w = env.scene["robot"].data.body_pos_w[:, eef_idx]

    dist = torch.norm(obj_pos_w - ee_pos_w, dim=1)
    return 1 - torch.tanh(dist / std)


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
