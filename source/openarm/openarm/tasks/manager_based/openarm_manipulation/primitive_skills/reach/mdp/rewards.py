"""
Modified reward functions for the Reach‑v1 task to support cup‑relative offsets and
an additional alignment reward.  The original rewards computed the target
position by adding a fixed world‑frame offset to the target object's root
position.  To ensure the pre‑grasp point remains correct when the cup is
rotated, we now rotate the offset into the world frame using the cup's
orientation.  We also add a new reward that aligns the TCP's forward (z) axis
with the direction from the TCP to the target.  This encourages the end‑effector
to approach the cup head‑on rather than from an arbitrary angle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def tcp_distance_to_target(
    env: "ManagerBasedRLEnv",
    tcp_body_name: str,
    target_cfg: SceneEntityCfg,
    offset: list[float],
) -> torch.Tensor:
    """Reward based on the distance between TCP and target.

    The target position is computed by rotating the provided offset from the
    target object's local frame into the world frame and adding it to the
    object's root position.  This ensures the offset moves with the cup's
    orientation.
    """
    robot = env.scene["robot"]
    tcp_body_idx = robot.data.body_names.index(tcp_body_name)
    tcp_pos_w = robot.data.body_pos_w[:, tcp_body_idx]

    target: RigidObject = env.scene[target_cfg.name]
    # Convert the offset from the cup's local frame into the world frame
    offset_tensor = torch.tensor(offset, device=env.device).unsqueeze(0)
    offset_tensor = offset_tensor.repeat(target.data.root_quat_w.shape[0], 1)
    offset_world = quat_apply(target.data.root_quat_w, offset_tensor)
    target_pos_w = target.data.root_pos_w + offset_world

    dist = torch.norm(tcp_pos_w - target_pos_w, p=2, dim=-1)
    return torch.exp(-dist)


def tcp_x_axis_alignment(
    env: "ManagerBasedRLEnv",
    tcp_body_name: str,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward based on the alignment of the TCP's x-axis with the target's z-axis.

    This function is retained for backward compatibility but may be deprecated in
    favor of tcp_z_axis_to_target_alignment.  It computes the dot product
    between the TCP's x axis and the cup's z axis (world up).
    """
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


def tcp_z_axis_to_target_alignment(
    env: "ManagerBasedRLEnv",
    tcp_body_name: str,
    target_cfg: SceneEntityCfg,
    offset: list[float],
) -> torch.Tensor:
    """Reward that aligns the TCP's forward (z) axis with the direction to the target.

    This reward encourages the TCP to face directly toward the cup.  It is
    computed as the dot product between the TCP's local z-axis, transformed
    into the world frame, and the unit vector from the TCP to the target
    position.  The target position is computed using a cup‑relative offset.
    """
    robot = env.scene["robot"]
    tcp_body_idx = robot.data.body_names.index(tcp_body_name)
    tcp_pos_w = robot.data.body_pos_w[:, tcp_body_idx]
    tcp_quat_w = robot.data.body_quat_w[:, tcp_body_idx]

    target: RigidObject = env.scene[target_cfg.name]

    # Rotate the offset into world coordinates
    offset_tensor = torch.tensor(offset, device=env.device).unsqueeze(0)
    offset_tensor = offset_tensor.repeat(target.data.root_quat_w.shape[0], 1)
    offset_world = quat_apply(target.data.root_quat_w, offset_tensor)
    target_pos_w = target.data.root_pos_w + offset_world

    # TCP's z axis (forward) in world frame
    z_axis_tcp = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(tcp_pos_w)
    z_axis_world = quat_apply(tcp_quat_w, z_axis_tcp)

    # Direction from TCP to target (unit vector)
    dir_vec = target_pos_w - tcp_pos_w
    dir_norm = torch.norm(dir_vec, p=2, dim=-1, keepdim=True) + 1e-6
    dir_unit = dir_vec / dir_norm

    # Dot product between forward axis and direction vector
    dot_product = torch.sum(z_axis_world * dir_unit, dim=-1)
    return dot_product


def hand_joint_position(
    env: "ManagerBasedRLEnv",
    joint_name: str | list[str],
    target_pos: float,
) -> torch.Tensor:
    """Reward for keeping the hand open.

    A high reward is given when the specified joint's position is close to
    target_pos (e.g. 0.0 for an open gripper).
    """
    robot = env.scene["robot"]
    if isinstance(joint_name, str):
        joint_idx = [robot.data.joint_names.index(joint_name)]
    else:
        joint_idx = [robot.data.joint_names.index(name) for name in joint_name]
    joint_pos = robot.data.joint_pos[:, joint_idx].mean(dim=-1)

    # Reward for being close to the target position
    return torch.exp(-torch.abs(joint_pos - target_pos))
