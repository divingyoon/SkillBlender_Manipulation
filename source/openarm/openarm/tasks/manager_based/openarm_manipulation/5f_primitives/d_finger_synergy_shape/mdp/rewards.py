# Copyright 2025 Enactic, Inc.
# Licensed under the Apache License, Version 2.0

"""Reward functions for Primitive D: Finger Joint Target Tracking."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


__all__ = [
    "joint_tracking_error_l2",
    "joint_tracking_tanh",
    "stepwise_decay",
]

def joint_tracking_error_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
) -> torch.Tensor:
    """Compute L2 error between current joint positions and target.

    Args:
        env: Environment instance
        asset_cfg: Robot config with hand joint_names
        command_name: Joint position command name

    Returns:
        L2 joint error (num_envs,)
    """
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Get hand joint positions
    joint_pos = asset.data.joint_pos
    if asset_cfg.joint_ids is not None:
        joint_pos = joint_pos[:, asset_cfg.joint_ids]

    # Target joints from command
    target_joints = command[:, : joint_pos.shape[1]]

    # Debug: log per-env current vs target joint positions once per env step (per command).
    if not hasattr(env, "_debug_joint_log_last_step_by_cmd"):
        env._debug_joint_log_last_step_by_cmd = {}
        env._debug_joint_log_every = 20
    step = int(env.common_step_counter)
    if step % max(env._debug_joint_log_every, 1) == 0:
        last_step = env._debug_joint_log_last_step_by_cmd.get(command_name, -1)
        if step != last_step:
            env._debug_joint_log_last_step_by_cmd[command_name] = step
            cur = joint_pos.detach().cpu()
            tgt = target_joints.detach().cpu()
            side = "left" if "left" in command_name else "right" if "right" in command_name else command_name
            for env_id in range(cur.shape[0]):
                cur_vals = ", ".join([f"{v:+.3f}" for v in cur[env_id].tolist()])
                tgt_vals = ", ".join([f"{v:+.3f}" for v in tgt[env_id].tolist()])
                print(f"[joint_current] env={env_id} {side}_current=[{cur_vals}]")
                print(f"[joint_current] env={env_id} {side}_target=[{tgt_vals}]")
                print(f"[joint_ERROR] env={env_id} {side}_error={torch.norm(joint_pos - target_joints, dim=-1)}")

    # L2 error
    return torch.norm(joint_pos - target_joints, dim=-1)


def joint_tracking_tanh(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    std: float = 0.5,
) -> torch.Tensor:
    """Joint tracking reward with tanh kernel.

    Args:
        env: Environment instance
        asset_cfg: Robot config
        command_name: Joint command name
        std: Standard deviation for tanh

    Returns:
        Reward in [0, 1] (num_envs,)
    """
    error = joint_tracking_error_l2(env, asset_cfg, command_name)
    return 1.0 - torch.tanh(error / std)


def stepwise_decay(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    old_value: float,
    initial_value: float,
    interval_iters: int,
    decrement: float,
    min_value: float,
) -> float:
    """Stepwise decay schedule based on training iterations."""
    if interval_iters <= 0:
        return old_value

    num_envs = getattr(env.scene, "num_envs", 1)
    horizon_length = getattr(getattr(env, "cfg", None), "horizon_length", 1)
    steps_per_iter = max(int(num_envs) * int(horizon_length), 1)

    iterations = int(env.common_step_counter) // steps_per_iter
    num_decays = iterations // interval_iters
    new_value = max(initial_value - decrement * num_decays, min_value)

    if abs(new_value - old_value) < 1e-9:
        return mdp.modify_term_cfg.NO_CHANGE
    return new_value
