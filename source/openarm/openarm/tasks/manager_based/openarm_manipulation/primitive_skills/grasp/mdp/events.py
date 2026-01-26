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

"""
Event functions for the Grasp-v1 task.

Key concept: Teacher-forcing reset
- Cup is placed on table with some randomization (xy offset, yaw rotation)
- Robot arm is initialized to pre-grasp pose relative to cup
- Gripper starts fully open
"""

from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    quat_apply,
    quat_mul,
    quat_from_euler_xyz,
    sample_uniform,
    quat_inv,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_cup_with_randomization(
    env,
    env_ids: Sequence[int],
    pose_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> None:
    """Reset cup on table with position and yaw randomization.

    Args:
        pose_range: Dictionary with keys 'x', 'y', 'z', 'yaw' for randomization ranges.
                   Example: {'x': (-0.03, 0.03), 'y': (-0.03, 0.03), 'yaw': (-0.26, 0.26)}
    """
    cup: RigidObject = env.scene[asset_cfg.name]

    default_state = cup.data.default_root_state[env_ids].clone()

    # Sample random offsets
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=cup.device)
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=cup.device)

    # Apply position offset (add to default position + env origin)
    positions = default_state[:, :3] + env.scene.env_origins[env_ids]
    positions[:, 0] += rand_samples[:, 0]  # x offset
    positions[:, 1] += rand_samples[:, 1]  # y offset
    positions[:, 2] += rand_samples[:, 2]  # z offset

    # Apply orientation delta (yaw only for cup)
    orient_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = quat_mul(default_state[:, 3:7], orient_delta)

    # Zero velocity
    velocities = torch.zeros((len(env_ids), 6), device=cup.device)

    cup.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    cup.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_robot_to_pregrasp_pose(
    env,
    env_ids: Sequence[int],
    pregrasp_distance: tuple[float, float],
    height_offset: tuple[float, float],
    yaw_noise: tuple[float, float],
    target_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    tcp_body_name: str = "openarm_left_ee_tcp",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset robot arm to pre-grasp pose relative to cup.

    This uses IK or direct joint positioning to place the TCP at a pre-grasp location:
    - Distance from cup: pregrasp_distance (e.g., 3-6cm in front)
    - Height: cup center height + height_offset
    - Yaw alignment: roughly aligned with some noise

    For now, we use a simplified approach: reset to default joint positions
    which should already be close to a pre-grasp pose.
    The actual pre-grasp positioning is handled by the reward structure.
    """
    robot: Articulation = env.scene[robot_cfg.name]

    # Reset to default joint positions (pre-grasp pose defined in config)
    default_joint_pos = robot.data.default_joint_pos[env_ids].clone()
    default_joint_vel = robot.data.default_joint_vel[env_ids].clone()

    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)


def reset_gripper_open(
    env,
    env_ids: Sequence[int],
    joint_names: list[str],
    open_position: float = 0.04,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset gripper joints to fully open position.

    Args:
        joint_names: List of gripper joint names to reset.
        open_position: Target position for open gripper (e.g., 0.04 for fully open).
    """
    robot: Articulation = env.scene[robot_cfg.name]

    joint_pos = robot.data.joint_pos[env_ids].clone()
    joint_vel = robot.data.joint_vel[env_ids].clone()

    for joint_name in joint_names:
        if joint_name in robot.data.joint_names:
            joint_idx = robot.data.joint_names.index(joint_name)
            joint_pos[:, joint_idx] = open_position
            joint_vel[:, joint_idx] = 0.0

    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_robot_to_pregrasp_ik(
    env,
    env_ids: Sequence[int],
    pregrasp_offset: list[float],
    offset_noise: dict[str, tuple[float, float]],
    target_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    tcp_body_name: str = "openarm_left_ee_tcp",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset robot TCP to pre-grasp pose using offset from cup position.

    This computes the desired TCP pose as:
        TCP_target = cup_pos + rotate(cup_quat, pregrasp_offset + noise)

    Then sets joint positions to achieve this pose.
    Note: This is a placeholder - actual IK solver integration needed.

    Args:
        pregrasp_offset: Base offset from cup center in cup's local frame [x, y, z].
                        e.g., [0.0, 0.0, -0.05] means 5cm behind the cup.
        offset_noise: Random noise to add to offset {'x': (min, max), 'y': ..., 'z': ...}
    """
    robot: Articulation = env.scene[robot_cfg.name]
    cup: RigidObject = env.scene[target_cfg.name]

    # Get cup pose
    cup_pos = cup.data.root_pos_w[env_ids]
    cup_quat = cup.data.root_quat_w[env_ids]

    # Compute target TCP position
    base_offset = torch.tensor(pregrasp_offset, device=env.device).unsqueeze(0)
    base_offset = base_offset.expand(len(env_ids), -1).clone()

    # Add noise
    range_list = [offset_noise.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=env.device)
    noise = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=env.device)
    offset_with_noise = base_offset + noise

    # Transform offset to world frame
    offset_world = quat_apply(cup_quat, offset_with_noise)
    target_tcp_pos = cup_pos + offset_world

    # For now, just reset to default - IK integration needed for actual positioning
    # Store target for observation/reward computation
    if not hasattr(env, "_pregrasp_target_pos"):
        env._pregrasp_target_pos = torch.zeros((env.num_envs, 3), device=env.device)
    env._pregrasp_target_pos[env_ids] = target_tcp_pos

    # Reset to default joint positions
    default_joint_pos = robot.data.default_joint_pos[env_ids].clone()
    default_joint_vel = robot.data.default_joint_vel[env_ids].clone()
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)


def reset_from_reach_terminal_states(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    terminal_states_path: str = "reach_terminal_states.pt",
    left_gripper_joint_names: list[str] = ["openarm_left_finger_joint1", "openarm_left_finger_joint2"],
    right_gripper_joint_names: list[str] = ["openarm_right_finger_joint1", "openarm_right_finger_joint2"],
    gripper_open_position: float = 0.04,
    reset_cups: bool = True,
    cup_cfg_name: str = "cup",
    cup2_cfg_name: str = "cup2",
):
    """
    Reset robot joints AND cup positions from pre-recorded terminal states of a successful reach policy.

    IMPORTANT: This now resets BOTH joint positions AND cup positions to ensure consistency.
    The terminal states contain the exact cup positions that correspond to the saved joint positions.

    Args:
        env: The environment instance.
        env_ids: Tensor of environment IDs to reset.
        terminal_states_path: Path to the .pt file containing saved terminal states.
        left_gripper_joint_names: Names of the left gripper joints.
        right_gripper_joint_names: Names of the right gripper joints.
        gripper_open_position: The position to set gripper joints to (open).
        reset_cups: Whether to also reset cup positions from terminal states.
        cup_cfg_name: Name of the left cup in scene config.
        cup2_cfg_name: Name of the right cup in scene config.
    """
    print(f"[DEBUG] reset_from_reach_terminal_states called with {len(env_ids)} env_ids")
    if len(env_ids) == 0:
        return

    # Load saved terminal states
    if not hasattr(env, "_reach_terminal_states_buffer"):
        try:
            env._reach_terminal_states_buffer = torch.load(terminal_states_path, map_location=env.device, weights_only=False)
            # Transfer all states to the correct device
            for key, value in env._reach_terminal_states_buffer.items():
                if isinstance(value, torch.Tensor):
                    env._reach_terminal_states_buffer[key] = value.to(env.device)
            print(f"[INFO] Loaded {env._reach_terminal_states_buffer['joint_pos'].shape[0]} reach terminal states")
            print(f"[INFO] Available keys: {list(env._reach_terminal_states_buffer.keys())}")
        except FileNotFoundError:
            print(f"Warning: Reach terminal states file not found at {terminal_states_path}. "
                  "Roll-out reset will not function correctly.")
            env._reach_terminal_states_buffer = None
            return

    if env._reach_terminal_states_buffer is None:
        return

    terminal_states = env._reach_terminal_states_buffer

    # Handle cases where there might not be enough stored states for all envs.
    num_available_states = terminal_states["joint_pos"].shape[0]
    if num_available_states == 0:
        print("Warning: No reach terminal states available for roll-out reset.")
        return

    # Randomly select states - STORE indices for cup reset to use same states
    state_indices = torch.randint(0, num_available_states, (len(env_ids),), device=env.device)

    # Store selected indices for this reset batch
    if not hasattr(env, "_current_reset_state_indices"):
        env._current_reset_state_indices = {}
    env._current_reset_state_indices[tuple(env_ids.tolist())] = state_indices

    # Set robot joint positions
    joint_pos_to_set = terminal_states["joint_pos"][state_indices]
    joint_vel_to_set = terminal_states["joint_vel"][state_indices] if "joint_vel" in terminal_states else torch.zeros_like(joint_pos_to_set)

    # Debug: Print first env's joint positions and EEF positions
    if len(env_ids) > 0:
        print(f"[DEBUG] Setting joint_pos[0]: {joint_pos_to_set[0, :7].tolist()}")  # First 7 joints (left arm)
        if "left_eef_pos" in terminal_states:
            eef_pos = terminal_states["left_eef_pos"][state_indices]
            print(f"[DEBUG] Expected left EEF pos: {eef_pos[0].tolist()}")

    env.scene["robot"].write_joint_state_to_sim(
        joint_pos_to_set,
        joint_vel_to_set,
        env_ids=env_ids
    )

    # Reset cup positions from terminal states (CRITICAL for consistency!)
    if reset_cups:
        # Reset left cup (cup/object)
        if "object_pos" in terminal_states and cup_cfg_name in env.scene.keys():
            cup = env.scene[cup_cfg_name]
            cup_pos = terminal_states["object_pos"][state_indices]
            cup_quat = terminal_states["object_quat"][state_indices]
            # Convert local position to world position
            world_pos = cup_pos + env.scene.env_origins[env_ids]

            # Debug: Print cup position
            if len(env_ids) > 0:
                print(f"[DEBUG] Setting cup local_pos: {cup_pos[0].tolist()}, world_pos: {world_pos[0].tolist()}")

            cup.write_root_pose_to_sim(
                torch.cat([world_pos, cup_quat], dim=-1),
                env_ids=env_ids
            )
            cup.write_root_velocity_to_sim(
                torch.zeros((len(env_ids), 6), device=env.device),
                env_ids=env_ids
            )

        # Reset right cup (cup2/object2)
        if "object2_pos" in terminal_states and cup2_cfg_name in env.scene.keys():
            cup2 = env.scene[cup2_cfg_name]
            cup2_pos = terminal_states["object2_pos"][state_indices]
            cup2_quat = terminal_states["object2_quat"][state_indices]
            # Convert local position to world position
            world_pos2 = cup2_pos + env.scene.env_origins[env_ids]
            cup2.write_root_pose_to_sim(
                torch.cat([world_pos2, cup2_quat], dim=-1),
                env_ids=env_ids
            )
            cup2.write_root_velocity_to_sim(
                torch.zeros((len(env_ids), 6), device=env.device),
                env_ids=env_ids
            )

    # Ensure grippers are open after robot reset
    reset_gripper_open(env, env_ids, left_gripper_joint_names, gripper_open_position)
    reset_gripper_open(env, env_ids, right_gripper_joint_names, gripper_open_position)

