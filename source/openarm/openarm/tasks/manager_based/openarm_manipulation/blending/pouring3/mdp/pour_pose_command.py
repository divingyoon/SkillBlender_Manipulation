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

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from .commands_cfg import PourPoseCommandCfg


class PourPoseCommand(CommandTerm):
    """Command generator that switches from source cup to target cup based on phase."""

    cfg: PourPoseCommandCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: PourPoseCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]
        self.source_object: RigidObject = env.scene[cfg.source_asset_cfg.name]
        self.target_object: RigidObject = env.scene[cfg.target_asset_cfg.name]

        # command in base frame: (x, y, z, qw, qx, qy, qz)
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)

    @property
    def command(self) -> torch.Tensor:
        return self.pose_command_b

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        # State-driven command; just refresh on resample.
        self._update_command(env_ids)

    def _update_command(self, env_ids: Sequence[int] | None = None):
        # pick phase source
        phase_tensor = None
        if self.cfg.phase_source == "left" and hasattr(self._env, "pour_phase_left"):
            phase_tensor = self._env.pour_phase_left
        elif self.cfg.phase_source == "right" and hasattr(self._env, "pour_phase_right"):
            phase_tensor = self._env.pour_phase_right
        elif self.cfg.phase_source == "group":
            if hasattr(self._env, "pour_phase_min"):
                phase_tensor = self._env.pour_phase_min
            elif hasattr(self._env, "pour_phase_left") and hasattr(self._env, "pour_phase_right"):
                phase_tensor = torch.minimum(self._env.pour_phase_left, self._env.pour_phase_right)
            elif hasattr(self._env, "pour_phase"):
                phase_tensor = self._env.pour_phase

        if phase_tensor is None:
            phase_tensor = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        use_target = phase_tensor >= self.cfg.switch_phase
        source_pos_w = self.source_object.data.root_pos_w
        source_quat_w = self.source_object.data.root_quat_w
        target_pos_w = self.target_object.data.root_pos_w
        target_quat_w = self.target_object.data.root_quat_w

        pre_offset = torch.tensor(self.cfg.pre_offset, device=self.device, dtype=torch.float32)
        post_offset = torch.tensor(self.cfg.post_offset, device=self.device, dtype=torch.float32)
        offset = torch.where(use_target.unsqueeze(1), post_offset, pre_offset)

        cmd_pos_w = torch.where(use_target.unsqueeze(1), target_pos_w, source_pos_w) + offset
        cmd_quat_w = torch.where(use_target.unsqueeze(1), target_quat_w, source_quat_w)

        cmd_pos_b, cmd_quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            cmd_pos_w,
            cmd_quat_w,
        )
        if env_ids is None:
            self.pose_command_b[:, :3] = cmd_pos_b
            self.pose_command_b[:, 3:] = cmd_quat_b
        else:
            self.pose_command_b[env_ids, :3] = cmd_pos_b[env_ids]
            self.pose_command_b[env_ids, 3:] = cmd_quat_b[env_ids]

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        # convert command to world frame for visualization
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        # visualize goal and current body pose
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        body_link_pose_w = self.robot.data.body_link_pose_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_link_pose_w[:, :3], body_link_pose_w[:, 3:7])


class PhaseSwitchPoseCommand(CommandTerm):
    """Command generator that switches from uniform sampling to pour pose based on phase."""

    cfg: "PhaseSwitchPoseCommandCfg"
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: "PhaseSwitchPoseCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]
        self.source_object: RigidObject = env.scene[cfg.source_asset_cfg.name]
        self.target_object: RigidObject = env.scene[cfg.target_asset_cfg.name]

        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)

        self.uniform_command_b = torch.zeros_like(self.pose_command_b)
        self.uniform_command_b[:, 3] = 1.0

    @property
    def command(self) -> torch.Tensor:
        return self.pose_command_b

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        offset = torch.zeros((len(env_ids), 3), device=self.device)
        offset[:, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        offset[:, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        offset[:, 2] = r.uniform_(*self.cfg.ranges.pos_z)

        source_pos_w = self.source_object.data.root_pos_w[env_ids]
        source_quat_w = self.source_object.data.root_quat_w[env_ids]
        cmd_pos_w = source_pos_w + offset

        cmd_quat_w = source_quat_w

        cmd_pos_b, cmd_quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w[env_ids],
            self.robot.data.root_quat_w[env_ids],
            cmd_pos_w,
            cmd_quat_w,
        )
        self.uniform_command_b[env_ids, :3] = cmd_pos_b
        self.uniform_command_b[env_ids, 3:] = cmd_quat_b

        self._update_command()

    def _update_command(self):
        phase_tensor = None
        if self.cfg.phase_source == "left" and hasattr(self._env, "pour_phase_left"):
            phase_tensor = self._env.pour_phase_left
        elif self.cfg.phase_source == "right" and hasattr(self._env, "pour_phase_right"):
            phase_tensor = self._env.pour_phase_right
        elif self.cfg.phase_source == "group":
            if hasattr(self._env, "pour_phase_min"):
                phase_tensor = self._env.pour_phase_min
            elif hasattr(self._env, "pour_phase_left") and hasattr(self._env, "pour_phase_right"):
                phase_tensor = torch.minimum(self._env.pour_phase_left, self._env.pour_phase_right)
            elif hasattr(self._env, "pour_phase"):
                phase_tensor = self._env.pour_phase

        if phase_tensor is None:
            phase_tensor = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        use_target = phase_tensor >= self.cfg.switch_phase
        target_pos_w = self.target_object.data.root_pos_w
        target_quat_w = self.target_object.data.root_quat_w
        post_offset = torch.tensor(self.cfg.post_offset, device=self.device, dtype=torch.float32)
        cmd_pos_w = target_pos_w + post_offset
        cmd_quat_w = target_quat_w

        cmd_pos_b, cmd_quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            cmd_pos_w,
            cmd_quat_w,
        )

        # Keep current TCP orientation before switching to target to avoid sudden spins.
        body_link_pose_w = self.robot.data.body_link_pose_w[:, self.body_idx]
        cur_pos_b, cur_quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            body_link_pose_w[:, :3],
            body_link_pose_w[:, 3:7],
        )

        self.pose_command_b[:, :3] = torch.where(use_target.unsqueeze(1), cmd_pos_b, self.uniform_command_b[:, :3])
        self.pose_command_b[:, 3:] = torch.where(use_target.unsqueeze(1), cmd_quat_b, cur_quat_b)

        if hasattr(self._env, "common_step_counter") and self._env.common_step_counter % 200 == 0:
            if not hasattr(self, "_last_log_step") or self._last_log_step != self._env.common_step_counter:
                phase0 = int(phase_tensor[0].item())
                cmd0 = self.pose_command_b[0, :3].detach().cpu().numpy()
                src = self.cfg.source_asset_cfg.name
                tgt = self.cfg.target_asset_cfg.name
                print(
                    f"[CMD] {self.cfg.body_name} phase={phase0} "
                    f"use_target={bool(use_target[0].item())} "
                    f"src={src} tgt={tgt} pos_b={cmd0}"
                )
                self._last_log_step = self._env.common_step_counter

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        body_link_pose_w = self.robot.data.body_link_pose_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_link_pose_w[:, :3], body_link_pose_w[:, 3:7])
