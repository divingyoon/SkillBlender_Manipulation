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
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def body_pose(env: ManagerBasedRLEnv, body_name: str) -> torch.Tensor:
    """Pose of a specific body of the robot."""
    robot = env.scene["robot"]
    body_idx = robot.data.body_names.index(body_name)
    pos_w = robot.data.body_pos_w[:, body_idx]
    quat_w = robot.data.body_quat_w[:, body_idx]
    return torch.cat((pos_w, quat_w), dim=1)


def root_pose(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Pose of the root of a rigid object."""
    obj: RigidObject = env.scene[asset_cfg.name]
    pos_w = obj.data.root_pos_w
    quat_w = obj.data.root_quat_w
    return torch.cat((pos_w, quat_w), dim=1)


def target_pos_in_tcp_frame(env: ManagerBasedRLEnv, tcp_body_name: str, target_cfg: SceneEntityCfg, offset: list[float]) -> torch.Tensor:
    """Position of the target object in the TCP frame."""
    robot = env.scene["robot"]
    tcp_body_idx = robot.data.body_names.index(tcp_body_name)
    tcp_pos_w = robot.data.body_pos_w[:, tcp_body_idx]
    tcp_quat_w = robot.data.body_quat_w[:, tcp_body_idx]
    
    target: RigidObject = env.scene[target_cfg.name]
    target_pos_w = target.data.root_pos_w + torch.tensor(offset, device=env.device)

    target_pos_tcp, _ = subtract_frame_transforms(tcp_pos_w, tcp_quat_w, target_pos_w)
    return target_pos_tcp