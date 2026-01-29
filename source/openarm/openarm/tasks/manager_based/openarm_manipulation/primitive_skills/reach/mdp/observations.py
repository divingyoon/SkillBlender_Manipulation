"""
Modified observation utilities for the Reach‑v1 task.

These functions mirror the original observation helpers but incorporate a cup‑relative
offset for computing the target position in the TCP frame.  The offset is
rotated into the world frame based on the cup's orientation to ensure the
relative target position remains correct even when the cup is rotated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def body_pose(env: "ManagerBasedRLEnv", body_name: str) -> torch.Tensor:
    """Pose of a specific body of the robot.

    Returns the world position and orientation (as a quaternion) of the
    specified body.
    """
    robot = env.scene["robot"]
    body_idx = robot.data.body_names.index(body_name)
    pos_w = robot.data.body_pos_w[:, body_idx]
    quat_w = robot.data.body_quat_w[:, body_idx]
    return torch.cat((pos_w, quat_w), dim=1)


def root_pose(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Pose of the root of a rigid object.

    Returns the world position and orientation of the object's root.
    """
    obj: RigidObject = env.scene[asset_cfg.name]
    pos_w = obj.data.root_pos_w
    quat_w = obj.data.root_quat_w
    return torch.cat((pos_w, quat_w), dim=1)


def target_pos_in_tcp_frame(
    env: "ManagerBasedRLEnv",
    tcp_body_name: str,
    target_cfg: SceneEntityCfg,
    offset: list[float],
) -> torch.Tensor:
    """Position of a target point on the target object expressed in the TCP frame.

    The provided offset is specified in the target object's local coordinate
    frame.  It is rotated into the world frame using the target's orientation
    before being added to the object's root position.  The resulting point is
    then expressed relative to the TCP frame using subtract_frame_transforms.
    """
    robot = env.scene["robot"]
    tcp_body_idx = robot.data.body_names.index(tcp_body_name)
    tcp_pos_w = robot.data.body_pos_w[:, tcp_body_idx]
    tcp_quat_w = robot.data.body_quat_w[:, tcp_body_idx]

    target: RigidObject = env.scene[target_cfg.name]
    # Rotate the offset into world coordinates
    # Expand offset to match batch for quat_apply
    offset_tensor = torch.tensor(offset, device=env.device).unsqueeze(0)
    offset_tensor = offset_tensor.repeat(target.data.root_quat_w.shape[0], 1)
    offset_world = quat_apply(target.data.root_quat_w, offset_tensor)
    target_pos_w = target.data.root_pos_w + offset_world

    target_pos_tcp, _ = subtract_frame_transforms(tcp_pos_w, tcp_quat_w, target_pos_w)
    return target_pos_tcp
