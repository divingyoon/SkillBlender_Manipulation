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

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import configclass


def _quat_from_z_axis(z_axis: torch.Tensor) -> torch.Tensor:
    """Create a quaternion with the body z-axis aligned to the given direction."""
    z_axis = z_axis / torch.clamp_min(torch.linalg.norm(z_axis, dim=-1, keepdim=True), 1.0e-8)
    ref_x = torch.tensor([1.0, 0.0, 0.0], device=z_axis.device, dtype=z_axis.dtype)
    ref_y = torch.tensor([0.0, 1.0, 0.0], device=z_axis.device, dtype=z_axis.dtype)
    ref = ref_x.expand_as(z_axis)
    use_ref_y = torch.abs(torch.sum(z_axis * ref, dim=-1, keepdim=True)) > 0.95
    ref = torch.where(use_ref_y, ref_y.expand_as(z_axis), ref)
    y_axis = torch.cross(z_axis, ref, dim=-1)
    y_axis = y_axis / torch.clamp_min(torch.linalg.norm(y_axis, dim=-1, keepdim=True), 1.0e-8)
    x_axis = torch.cross(y_axis, z_axis, dim=-1)
    rot = torch.stack((x_axis, y_axis, z_axis), dim=-1)
    return math_utils.quat_from_matrix(rot)


class ConstrainedDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
    """Differential IK action with optional orientation constraint and null-space optimization."""

    cfg: "ConstrainedDifferentialInverseKinematicsActionCfg"

    def __init__(self, cfg: "ConstrainedDifferentialInverseKinematicsActionCfg", env):
        super().__init__(cfg, env)
        self._nullspace_q_ref = None
        if self.cfg.nullspace_gain > 0.0:
            self._nullspace_q_ref = self._asset.data.default_joint_pos[:, self._joint_ids]

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)

        if not self.cfg.orientation_constraint or self.cfg.orientation_command_name is None:
            return

        command = self._env.command_manager.get_command(self.cfg.orientation_command_name)
        obj_quat_b = command[:, 3:7]
        obj_axis = torch.tensor(
            self.cfg.orientation_object_axis, device=self.device, dtype=obj_quat_b.dtype
        ).repeat(obj_quat_b.shape[0], 1)
        target_z = math_utils.quat_apply(obj_quat_b, obj_axis)

        ee_quat_des = _quat_from_z_axis(target_z)
        if self.cfg.orientation_roll != 0.0:
            roll_quat = math_utils.quat_from_angle_axis(
                torch.full((ee_quat_des.shape[0],), self.cfg.orientation_roll, device=self.device, dtype=ee_quat_des.dtype),
                target_z,
            )
            ee_quat_des = math_utils.quat_mul(roll_quat, ee_quat_des)

        self._ik_controller.ee_quat_des[:] = ee_quat_des

    def apply_actions(self):
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        if ee_quat_curr.norm() == 0:
            joint_pos_des = joint_pos.clone()
        else:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)

            if self.cfg.nullspace_gain > 0.0 or self.cfg.joint_limit_avoidance_gain > 0.0:
                if "position" in self.cfg.controller.command_type:
                    jacobian_task = jacobian[:, 0:3]
                else:
                    jacobian_task = jacobian

                jacobian_pinv = self._compute_jacobian_pinv(jacobian_task)
                eye = torch.eye(jacobian_pinv.shape[1], device=self.device, dtype=jacobian_pinv.dtype)
                nullspace = eye.unsqueeze(0) - jacobian_pinv @ jacobian_task

                q_null = torch.zeros_like(joint_pos)
                if self.cfg.nullspace_gain > 0.0 and self._nullspace_q_ref is not None:
                    q_null += self.cfg.nullspace_gain * (self._nullspace_q_ref - joint_pos)

                if self.cfg.joint_limit_avoidance_gain > 0.0:
                    limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, :]
                    q_min = limits[..., 0]
                    q_max = limits[..., 1]
                    dist_lower = joint_pos - q_min
                    dist_upper = q_max - joint_pos
                    eps = self.cfg.joint_limit_eps
                    grad = (1.0 / torch.clamp(dist_lower, min=eps) ** 2) - (
                        1.0 / torch.clamp(dist_upper, min=eps) ** 2
                    )
                    if self.cfg.joint_limit_clamp is not None:
                        grad = torch.clamp(grad, -self.cfg.joint_limit_clamp, self.cfg.joint_limit_clamp)
                    q_null += self.cfg.joint_limit_avoidance_gain * grad

                joint_pos_des = joint_pos_des + (nullspace @ q_null.unsqueeze(-1)).squeeze(-1)

        self._asset.set_joint_position_target(joint_pos_des, self._joint_ids)

    def _compute_jacobian_pinv(self, jacobian: torch.Tensor) -> torch.Tensor:
        method = self.cfg.controller.ik_method
        params = self.cfg.controller.ik_params or {}

        if method == "dls":
            lambda_val = params.get("lambda_val", 0.01)
            jacobian_t = torch.transpose(jacobian, dim0=1, dim1=2)
            lambda_matrix = (lambda_val ** 2) * torch.eye(n=jacobian.shape[1], device=self.device)
            return jacobian_t @ torch.inverse(jacobian @ jacobian_t + lambda_matrix)
        if method == "svd":
            min_singular_value = params.get("min_singular_value", 1e-5)
            U, S, Vh = torch.linalg.svd(jacobian)
            S_inv = torch.where(S > min_singular_value, 1.0 / S, torch.zeros_like(S))
            return (
                torch.transpose(Vh, dim0=1, dim1=2)[:, :, : jacobian.shape[1]]
                @ torch.diag_embed(S_inv)
                @ torch.transpose(U, dim0=1, dim1=2)
            )
        return torch.linalg.pinv(jacobian)


@configclass
class ConstrainedDifferentialInverseKinematicsActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration for constrained differential IK action term."""

    class_type: type = ConstrainedDifferentialInverseKinematicsAction

    orientation_constraint: bool = False
    orientation_command_name: str | None = None
    orientation_object_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    orientation_roll: float = 0.0

    nullspace_gain: float = 0.0
    joint_limit_avoidance_gain: float = 0.0
    joint_limit_eps: float = 1.0e-3
    joint_limit_clamp: float | None = None
