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

"""B1 event extensions: lift-start reset + time-varying payload disturbance."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

from openarm.tasks.manager_based.openarm_manipulation.primitives.b_contact_force_hold.mdp import (
    events as base_events,
)

__all__ = [
    "reset_contact_force_hold_state_b1",
    "apply_payload_disturbance",
    "apply_payload_mass_ramp",
]


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    """Smoothstep interpolation for ramping."""
    return x * x * (3.0 - 2.0 * x)


def reset_contact_force_hold_state_b1(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    left_hand_joint_names: list[str],
    right_hand_joint_names: list[str],
    left_arm_joint_names: list[str],
    right_arm_joint_names: list[str],
    arm_joint_targets: dict[str, float],
    object_pose_b1: dict[str, tuple[float, float]],
    object_pose_b2: dict[str, tuple[float, float]],
    grasp_frame_name: str | None,
    object_offset_b1: dict[str, tuple[float, float]] | None,
    object_offset_b2: dict[str, tuple[float, float]] | None,
    b1_prob_init: float,
    b1_prob_final: float,
    b1_prob_steps: int,
    hand_bias_b1: float,
    hand_bias_b2: float,
    hand_extra_bias_b1: float,
    hand_extra_bias_b2: float,
    hand_extra_bias_lj_dg_4: float,
    left_hand_offset_targets: dict[str, float] | None,
    left_hand_offset_noise: float = 0.0,
    hand_noise: float = 0.0,
    arm_noise: float = 0.0,
    active_hand: str = "left",
    left_arm_fixed_joints: list[str] | None = None,
    left_arm_fixed_noise_deg: float = 0.0,
    lift_prob_init: float = 0.3,
    lift_prob_final: float = 0.9,
    lift_prob_steps: int = 20000,
    lift_offset_b1: dict[str, tuple[float, float]] | None = None,
    lift_offset_b2: dict[str, tuple[float, float]] | None = None,
    base_lift_z: float = 0.0,
):
    """Reset with lift-start branch, reusing base contact-hold reset logic."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    lift_prob = base_events._linear_schedule(
        int(env.common_step_counter), lift_prob_init, lift_prob_final, lift_prob_steps
    )
    rand = torch.rand(len(env_ids), device=env.device)
    lift_mask = rand < lift_prob
    env_ids_lift = env_ids[lift_mask]
    env_ids_base = env_ids[~lift_mask]

    if len(env_ids_lift) > 0:
        base_events.reset_contact_force_hold_state(
            env=env,
            env_ids=env_ids_lift,
            object_cfg=object_cfg,
            robot_cfg=robot_cfg,
            left_hand_joint_names=left_hand_joint_names,
            right_hand_joint_names=right_hand_joint_names,
            left_arm_joint_names=left_arm_joint_names,
            right_arm_joint_names=right_arm_joint_names,
            arm_joint_targets=arm_joint_targets,
            object_pose_b1=object_pose_b1,
            object_pose_b2=object_pose_b2,
            grasp_frame_name=grasp_frame_name,
            object_offset_b1=lift_offset_b1 or object_offset_b1,
            object_offset_b2=lift_offset_b2 or object_offset_b2,
            b1_prob_init=1.0,
            b1_prob_final=1.0,
            b1_prob_steps=1,
            hand_bias_b1=hand_bias_b1,
            hand_bias_b2=hand_bias_b2,
            hand_extra_bias_b1=hand_extra_bias_b1,
            hand_extra_bias_b2=hand_extra_bias_b2,
            hand_extra_bias_lj_dg_4=hand_extra_bias_lj_dg_4,
            left_hand_offset_targets=left_hand_offset_targets,
            left_hand_offset_noise=left_hand_offset_noise,
            hand_noise=hand_noise,
            arm_noise=arm_noise,
            active_hand=active_hand,
            left_arm_fixed_joints=left_arm_fixed_joints,
            left_arm_fixed_noise_deg=left_arm_fixed_noise_deg,
        )
        if base_lift_z != 0.0:
            robot = env.scene[robot_cfg.name]
            obj = env.scene[object_cfg.name]
            robot_root = robot.data.root_state_w[env_ids_lift].clone()
            obj_root = obj.data.root_state_w[env_ids_lift].clone()
            robot_root[:, 2] += base_lift_z
            obj_root[:, 2] += base_lift_z
            robot.write_root_state_to_sim(robot_root, env_ids=env_ids_lift)
            obj.write_root_state_to_sim(obj_root, env_ids=env_ids_lift)

    if len(env_ids_base) > 0:
        base_events.reset_contact_force_hold_state(
            env=env,
            env_ids=env_ids_base,
            object_cfg=object_cfg,
            robot_cfg=robot_cfg,
            left_hand_joint_names=left_hand_joint_names,
            right_hand_joint_names=right_hand_joint_names,
            left_arm_joint_names=left_arm_joint_names,
            right_arm_joint_names=right_arm_joint_names,
            arm_joint_targets=arm_joint_targets,
            object_pose_b1=object_pose_b1,
            object_pose_b2=object_pose_b2,
            grasp_frame_name=grasp_frame_name,
            object_offset_b1=object_offset_b1,
            object_offset_b2=object_offset_b2,
            b1_prob_init=b1_prob_init,
            b1_prob_final=b1_prob_final,
            b1_prob_steps=b1_prob_steps,
            hand_bias_b1=hand_bias_b1,
            hand_bias_b2=hand_bias_b2,
            hand_extra_bias_b1=hand_extra_bias_b1,
            hand_extra_bias_b2=hand_extra_bias_b2,
            hand_extra_bias_lj_dg_4=hand_extra_bias_lj_dg_4,
            left_hand_offset_targets=left_hand_offset_targets,
            left_hand_offset_noise=left_hand_offset_noise,
            hand_noise=hand_noise,
            arm_noise=arm_noise,
            active_hand=active_hand,
            left_arm_fixed_joints=left_arm_fixed_joints,
            left_arm_fixed_noise_deg=left_arm_fixed_noise_deg,
        )


def apply_payload_disturbance(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    object_cfg: SceneEntityCfg,
    delta_mass_range: Tuple[float, float],
    ramp_delay_steps: int = 0,
    ramp_steps: int = 2000,
    period_steps: int = 200,
    phase_offset: int = 0,
):
    """Apply a smooth, time-varying downward force to emulate payload changes."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    step = int(env.common_step_counter)
    if ramp_steps <= 0:
        ramp = 1.0
    else:
        t = (step - ramp_delay_steps) / float(ramp_steps)
        t = torch.clamp(torch.tensor(t, device=env.device), 0.0, 1.0)
        ramp = _smoothstep(t).item()

    phase = 2.0 * math.pi * ((step + phase_offset) / float(max(period_steps, 1)))
    osc = 0.5 * (1.0 - math.cos(phase))
    delta_m = delta_mass_range[0] + (delta_mass_range[1] - delta_mass_range[0]) * osc
    delta_m *= ramp

    force_mag = delta_m * 9.81
    forces = torch.zeros((len(env_ids), 1, 3), device=env.device)
    forces[:, 0, 2] = -force_mag
    torques = torch.zeros_like(forces)

    obj = env.scene[object_cfg.name]
    # Apply external force to the object body in world frame.
    obj.set_external_force_and_torque(forces, torques, env_ids=env_ids, is_global=True)


def apply_payload_mass_ramp(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    object_cfg: SceneEntityCfg,
    start_mass: float,
    end_mass: float,
    delay_steps: int = 250,
    ramp_steps: int = 2000,
    recompute_inertia: bool = True,
):
    """Ramp object mass per-episode after a delay (based on episode_length_buf)."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    if env_ids.numel() == 0:
        return

    step = env.episode_length_buf[env_ids]
    t = (step - delay_steps) / float(max(ramp_steps, 1))
    t = torch.clamp(t, 0.0, 1.0)
    target_mass = start_mass + (end_mass - start_mass) * t

    obj = env.scene[object_cfg.name]
    env_ids_cpu = env_ids.cpu()
    if object_cfg.body_ids == slice(None):
        body_ids = torch.arange(obj.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(object_cfg.body_ids, dtype=torch.int, device="cpu")

    masses = obj.root_physx_view.get_masses()
    masses[env_ids_cpu[:, None], body_ids] = target_mass.detach().cpu().unsqueeze(-1)
    obj.root_physx_view.set_masses(masses, env_ids_cpu)

    if recompute_inertia:
        ratios = masses[env_ids_cpu[:, None], body_ids] / obj.data.default_mass[env_ids_cpu[:, None], body_ids]
        inertias = obj.root_physx_view.get_inertias()
        inertias[env_ids_cpu] = obj.data.default_inertia[env_ids_cpu] * ratios
        obj.root_physx_view.set_inertias(inertias, env_ids_cpu)
