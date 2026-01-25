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
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def tcp_distance_to_target(env: ManagerBasedRLEnv, tcp_body_name: str, target_cfg: SceneEntityCfg, offset: list[float]) -> torch.Tensor:
    """Reward based on the distance between TCP and target."""
    robot = env.scene["robot"]
    tcp_body_idx = robot.data.body_names.index(tcp_body_name)
    tcp_pos_w = robot.data.body_pos_w[:, tcp_body_idx]
    
    target: RigidObject = env.scene[target_cfg.name]
    target_pos_w = target.data.root_pos_w + torch.tensor(offset, device=env.device)

    dist = torch.norm(tcp_pos_w - target_pos_w, p=2, dim=-1)
    return torch.exp(-dist)

def tcp_x_axis_alignment(env: ManagerBasedRLEnv, tcp_body_name: str, target_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward based on the alignment of the TCP's x-axis with the target's z-axis."""
    robot = env.scene["robot"]
    tcp_body_idx = robot.data.body_names.index(tcp_body_name)
    tcp_quat_w = robot.data.body_quat_w[:, tcp_body_idx]

    target: RigidObject = env.scene[target_cfg.name]
    target_quat_w = target.data.root_quat_w

    # Get the TCP's x-axis in the world frame
    x_axis_tcp = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand_as(robot.data.body_pos_w[:, tcp_body_idx])
    x_axis_world = quat_apply(tcp_quat_w, x_axis_tcp)

    # Get the target's z-axis in the world frame
    z_axis_target = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(target.data.root_pos_w)
    z_axis_world = quat_apply(target_quat_w, z_axis_target)

    # Calculate the dot product between the two vectors
    dot_product = torch.sum(x_axis_world * z_axis_world, dim=-1)
    return dot_product

def hand_joint_position(env: ManagerBasedRLEnv, joint_name: str, target_pos: float) -> torch.Tensor:
    """Reward for keeping the hand open."""
    robot = env.scene["robot"]
    joint_idx = robot.data.joint_names.index(joint_name)
    joint_pos = robot.data.joint_pos[:, joint_idx]
    
    # Reward for being close to the target position (0.0 for open)
    return torch.exp(-torch.abs(joint_pos - target_pos))