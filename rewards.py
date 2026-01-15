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

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for lifting the object above a minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_eef_distance(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Distance between the object and the specified end-effector link."""
    object_pos = env.scene[object_cfg.name].data.root_pos_w - env.scene.env_origins
    body_pos_w = env.scene["robot"].data.body_pos_w
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    eef_pos = body_pos_w[:, eef_idx] - env.scene.env_origins
    return torch.norm(object_pos - eef_pos, dim=1)


def object_eef_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the end-effector being close to the object using tanh-kernel."""
    distance = object_eef_distance(env, eef_link_name, object_cfg)
    return 1 - torch.tanh(distance / std)

# -----------------------------------------------------------------------------
# Additional reward helpers for pick-and-place tasks
#
# The functions below extend the existing reward library with helpers specific
# to multi-stage pick-and-place problems. They enable dense guidance for final
# object placement and discourage unnatural bimanual interactions.

def object_target_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    target_pos: list[float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for the object being near a target position using a tanh-kernel.

    The reward is close to 1 when the object is at the desired target position and
    decays smoothly as the distance increases. A larger standard deviation `std`
    results in a slower decay. `target_pos` should be a 3D position in world
    coordinates (x, y, z) representing the desired placement location on the table.

    Args:
        env: The RL environment instance.
        std: Standard deviation controlling the decay of the tanh-kernel.
        target_pos: A list of three floats specifying the target position.
        object_cfg: Scene entity configuration for the manipulated object.

    Returns:
        A tensor of shape (num_envs,) with values in [0, 1] representing the
        proximity reward to the target location.
    """
    object_pos = env.scene[object_cfg.name].data.root_pos_w - env.scene.env_origins
    # Convert the target position to a tensor on the correct device/dtype
    target = torch.tensor(target_pos, device=object_pos.device, dtype=object_pos.dtype)
    # Compute Euclidean distance between each object's position and the target
    distance = torch.norm(object_pos - target, dim=1)
    return 1 - torch.tanh(distance / std)


def hand_proximity_penalty(
    env: ManagerBasedRLEnv,
    threshold: float,
    left_eef_link_name: str,
    right_eef_link_name: str,
    penalty: float = 1.0,
) -> torch.Tensor:
    """Penalty applied when the end-effectors of two hands are too close.

    This helper returns a negative reward when the distance between the specified
    left and right end-effectors falls below the given threshold. It is
    designed to discourage unnatural hand-to-hand transfers by penalizing
    configurations where the hands overlap excessively. The penalty scales with
    the provided `penalty` weight.

    Args:
        env: The RL environment instance.
        threshold: Minimum allowed distance between the two hands before a penalty
            is applied (in meters).
        left_eef_link_name: Name of the left hand's end-effector link.
        right_eef_link_name: Name of the right hand's end-effector link.
        penalty: Magnitude of the penalty applied when hands are too close.

    Returns:
        A tensor of shape (num_envs,) containing negative penalties (or zero if
        no penalty) for each environment.
    """
    body_pos_w = env.scene["robot"].data.body_pos_w
    body_names = env.scene["robot"].data.body_names
    # Resolve indices for the specified end-effectors
    left_idx = body_names.index(left_eef_link_name)
    right_idx = body_names.index(right_eef_link_name)
    # Compute positions relative to environment origins
    left_pos = body_pos_w[:, left_idx] - env.scene.env_origins
    right_pos = body_pos_w[:, right_idx] - env.scene.env_origins
    dist = torch.norm(left_pos - right_pos, dim=1)
    # Apply penalty where hands are too close
    return torch.where(dist < threshold, -penalty * torch.ones_like(dist), torch.zeros_like(dist))
