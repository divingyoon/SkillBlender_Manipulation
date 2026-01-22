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

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass
from isaaclab.managers import CommandTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

from .pour_pose_command import PourPoseCommand, PhaseSwitchPoseCommand


@configclass
class PourPoseCommandCfg(CommandTermCfg):
    """Command that switches from source-cup tracking to target-cup pouring pose."""

    asset_name: str = MISSING
    body_name: str = MISSING
    source_asset_cfg: SceneEntityCfg = MISSING
    target_asset_cfg: SceneEntityCfg = MISSING
    resampling_time_range: tuple[float, float] = (4.0, 4.0)
    pre_offset: tuple[float, float, float] = (0.0, 0.0, 0.03)
    post_offset: tuple[float, float, float] = (0.0, 0.0, 0.10)
    switch_phase: int = 2
    phase_source: Literal["left", "right", "group"] = "left"
    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pour_goal_pose"
    )
    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pour_body_pose"
    )

    goal_pose_visualizer_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)

    def __post_init__(self):
        super().__post_init__()
        self.command_dim = 7
        self.class_type = PourPoseCommand


@configclass
class PhaseSwitchPoseCommandCfg(CommandTermCfg):
    """Command that uses UniformPoseCommand before a phase switch, then PourPoseCommand."""

    asset_name: str = MISSING
    body_name: str = MISSING
    source_asset_cfg: SceneEntityCfg = MISSING
    target_asset_cfg: SceneEntityCfg = MISSING
    resampling_time_range: tuple[float, float] = (4.0, 4.0)
    pre_offset: tuple[float, float, float] = (0.0, 0.0, 0.03)
    post_offset: tuple[float, float, float] = (0.0, 0.0, 0.10)
    switch_phase: int = 2
    phase_source: Literal["left", "right", "group"] = "left"
    make_quat_unique: bool = False

    @configclass
    class Ranges:
        pos_x: tuple[float, float] = MISSING
        pos_y: tuple[float, float] = MISSING
        pos_z: tuple[float, float] = MISSING
        roll: tuple[float, float] = MISSING
        pitch: tuple[float, float] = MISSING
        yaw: tuple[float, float] = MISSING

    ranges: Ranges = MISSING

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/phase_goal_pose"
    )
    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/phase_body_pose"
    )

    goal_pose_visualizer_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)

    def __post_init__(self):
        super().__post_init__()
        self.command_dim = 7
        self.class_type = PhaseSwitchPoseCommand
