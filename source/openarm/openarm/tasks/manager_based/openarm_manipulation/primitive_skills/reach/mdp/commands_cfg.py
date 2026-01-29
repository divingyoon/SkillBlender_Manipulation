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

from isaaclab.utils import configclass
from isaaclab.managers import CommandTermCfg, SceneEntityCfg

from .reach_pose_command import ReachPoseCommand

@configclass
class ReachPoseCommandCfg(CommandTermCfg):
    """Configuration for the reach pose command term."""
    asset_name: str = MISSING
    body_name: str = MISSING
    target_asset_cfg: SceneEntityCfg = MISSING
    resampling_time_range: tuple[float, float] = (4.0, 4.0)
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    @configclass
    class Ranges:
        """Uniform distribution ranges for the pose commands (world frame)."""

        pos_x: tuple[float, float] = MISSING
        pos_y: tuple[float, float] = MISSING
        pos_z: tuple[float, float] = MISSING
        roll: tuple[float, float] = MISSING
        pitch: tuple[float, float] = MISSING
        yaw: tuple[float, float] = MISSING

    ranges: Ranges = MISSING

    def __post_init__(self):
        """Post-initialization."""
        super().__post_init__()
        self.command_dim = 7
        self.class_type = ReachPoseCommand