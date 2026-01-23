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

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from isaaclab.utils.math import quat_apply, subtract_frame_transforms, combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_obs(
    env: ManagerBasedRLEnv,
    left_eef_link_name: str,
    right_eef_link_name: str,
    command_name: str | None = None,
    use_command_pos: bool = True,
    mask_cross_hand: bool = True,
) -> torch.Tensor:
    """Object observations in env frame with relative vectors to both end effectors."""
    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index(left_eef_link_name)
    right_eef_idx = env.scene["robot"].data.body_names.index(right_eef_link_name)
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins
    right_eef_pos = body_pos_w[:, right_eef_idx] - env.scene.env_origins

    if command_name is not None and use_command_pos:
        command = env.command_manager.get_command(command_name)
        des_pos_b = command[:, :3]
        des_quat_b = command[:, 3:7]
        robot = env.scene["robot"]
        des_pos_w, des_quat_w = combine_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b, des_quat_b
        )
        object_pos = des_pos_w - env.scene.env_origins
        object_quat = des_quat_w
        object_lin_vel = torch.zeros_like(object_pos)
        object_ang_vel = torch.zeros_like(object_pos)
    else:
        object_pos = env.scene["object"].data.root_pos_w - env.scene.env_origins
        object_quat = env.scene["object"].data.root_quat_w
        object_lin_vel = env.scene["object"].data.root_lin_vel_w
        object_ang_vel = env.scene["object"].data.root_ang_vel_w

    left_eef_to_object = object_pos - left_eef_pos
    right_eef_to_object = object_pos - right_eef_pos
    if mask_cross_hand:
        right_eef_to_object = torch.zeros_like(right_eef_to_object)
        # add a left-hand token without changing dimension
        right_eef_to_object[:, 0] = 1.0

    return torch.cat(
        (
            object_pos,
            object_quat,
            object_lin_vel,
            object_ang_vel,
            left_eef_to_object,
            right_eef_to_object,
        ),
        dim=1,
    )


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    command_name: str | None = None,
    use_command_pos: bool = False,
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    if command_name is not None and use_command_pos:
        command = env.command_manager.get_command(command_name)
        des_pos_b = command[:, :3]
        des_quat_b = command[:, 3:7]
        des_pos_w, _ = combine_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b, des_quat_b
        )
        object_pos_w = des_pos_w[:, :3]
    else:
        obj: RigidObject = env.scene[object_cfg.name]
        object_pos_w = obj.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w
    )
    return object_pos_b


def object2_obs(
    env: ManagerBasedRLEnv,
    left_eef_link_name: str,
    right_eef_link_name: str,
    command_name: str | None = None,
    use_command_pos: bool = True,
    mask_cross_hand: bool = True,
) -> torch.Tensor:
    """Object2 observations in env frame with relative vectors to both end effectors."""
    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index(left_eef_link_name)
    right_eef_idx = env.scene["robot"].data.body_names.index(right_eef_link_name)
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins
    right_eef_pos = body_pos_w[:, right_eef_idx] - env.scene.env_origins

    if command_name is not None and use_command_pos:
        command = env.command_manager.get_command(command_name)
        des_pos_b = command[:, :3]
        des_quat_b = command[:, 3:7]
        robot = env.scene["robot"]
        des_pos_w, des_quat_w = combine_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b, des_quat_b
        )
        object_pos = des_pos_w - env.scene.env_origins
        object_quat = des_quat_w
        object_lin_vel = torch.zeros_like(object_pos)
        object_ang_vel = torch.zeros_like(object_pos)
    else:
        object_pos = env.scene["object2"].data.root_pos_w - env.scene.env_origins
        object_quat = env.scene["object2"].data.root_quat_w
        object_lin_vel = env.scene["object2"].data.root_lin_vel_w
        object_ang_vel = env.scene["object2"].data.root_ang_vel_w

    left_eef_to_object = object_pos - left_eef_pos
    right_eef_to_object = object_pos - right_eef_pos
    if mask_cross_hand:
        left_eef_to_object = torch.zeros_like(left_eef_to_object)
        # add a right-hand token without changing dimension
        left_eef_to_object[:, 0] = -1.0

    return torch.cat(
        (
            object_pos,
            object_quat,
            object_lin_vel,
            object_ang_vel,
            left_eef_to_object,
            right_eef_to_object,
        ),
        dim=1,
    )


def cup_pair_obs(
    env: ManagerBasedRLEnv,
    source_name: str,
    target_name: str,
) -> torch.Tensor:
    """Cup poses and relative vector between source and target cups."""
    source_pos = env.scene[source_name].data.root_pos_w - env.scene.env_origins
    source_quat = env.scene[source_name].data.root_quat_w
    target_pos = env.scene[target_name].data.root_pos_w - env.scene.env_origins
    target_quat = env.scene[target_name].data.root_quat_w
    rel_pos = target_pos - source_pos
    return torch.cat((source_pos, source_quat, target_pos, target_quat, rel_pos), dim=1)


def bead_obs(
    env: ManagerBasedRLEnv,
    bead_name: str,
    target_name: str,
    source_name: str | None = None,
) -> torch.Tensor:
    """Bead position and relative vectors to cups."""
    bead_pos = env.scene[bead_name].data.root_pos_w - env.scene.env_origins
    target_pos = env.scene[target_name].data.root_pos_w - env.scene.env_origins
    bead_to_target = bead_pos - target_pos
    parts = [bead_pos, bead_to_target]
    if source_name is not None:
        source_pos = env.scene[source_name].data.root_pos_w - env.scene.env_origins
        bead_to_source = bead_pos - source_pos
        parts.append(bead_to_source)
    return torch.cat(parts, dim=1)


def cup_pair_compact_obs(
    env: ManagerBasedRLEnv,
    source_name: str,
    target_name: str,
) -> torch.Tensor:
    """Compact cup observation: relative position and tilt angle toward target."""
    source_pos = env.scene[source_name].data.root_pos_w - env.scene.env_origins
    target_pos = env.scene[target_name].data.root_pos_w - env.scene.env_origins
    source_quat = env.scene[source_name].data.root_quat_w
    rel_pos = target_pos - source_pos

    target_vector = torch.nn.functional.normalize(rel_pos, p=2.0, dim=-1)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=source_pos.device, dtype=source_pos.dtype).expand_as(source_pos)
    cup_z_axis = quat_apply(source_quat, z_axis)
    cup_z_axis = torch.nn.functional.normalize(cup_z_axis, p=2.0, dim=-1)
    dot_product = torch.sum(cup_z_axis * target_vector, dim=-1).clamp(-1.0, 1.0)
    tilt_angle = torch.acos(dot_product).unsqueeze(-1)
    return torch.cat((rel_pos, tilt_angle), dim=1)


def bead_to_target_obs(
    env: ManagerBasedRLEnv,
    bead_name: str,
    target_name: str,
) -> torch.Tensor:
    """Compact bead observation: vector from target cup to bead."""
    bead_pos = env.scene[bead_name].data.root_pos_w - env.scene.env_origins
    target_pos = env.scene[target_name].data.root_pos_w - env.scene.env_origins
    bead_to_target = bead_pos - target_pos
    return bead_to_target


def pour_phase_left(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Left-hand pour phase as a single-value observation."""
    if not hasattr(env, "pour_phase_left"):
        return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)
    return env.pour_phase_left.to(dtype=torch.float32).unsqueeze(-1)


def pour_phase_right(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Right-hand pour phase as a single-value observation."""
    if not hasattr(env, "pour_phase_right"):
        return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)
    return env.pour_phase_right.to(dtype=torch.float32).unsqueeze(-1)


def pour_phase_group(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Group pour phase as a single-value observation."""
    if not hasattr(env, "pour_phase"):
        return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)
    return env.pour_phase.to(dtype=torch.float32).unsqueeze(-1)


def skill_bias_reach_grasp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Bias for skill selection: [reach, grasp2g], piecewise by phase."""
    if not hasattr(env, "pour_phase_left") or not hasattr(env, "pour_phase_right"):
        return torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)
    phase_min = torch.minimum(env.pour_phase_left, env.pour_phase_right)
    bias = torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)
    # phase < 3: favor grasp2g, phase >= 3: favor reach
    bias[:, 0] = torch.where(
        phase_min < 3,
        torch.tensor(-2.0, device=env.device),
        torch.tensor(2.0, device=env.device),
    )
    bias[:, 1] = torch.where(
        phase_min < 3,
        torch.tensor(2.0, device=env.device),
        torch.tensor(-2.0, device=env.device),
    )
    return bias
