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

import logging
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.utils import prims as prim_utils
from isaaclab.sim.utils.stage import get_current_stage

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

logger = logging.getLogger(__name__)


def filter_left_right_arm_collisions(
    env: ManagerBasedRLEnv,
    env_ids: object,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left_path_substring: str = "openarm_left_",
    right_path_substring: str = "openarm_right_",
) -> None:
    """Filter collisions between left and right arm collision shapes.

    This applies a FilteredPairsAPI on all collision prims that belong to the left or right arm
    based on the provided path substrings. The filtering is applied for every matched robot
    instance in the scene.
    """
    try:
        from pxr import UsdPhysics
    except Exception as exc:  # pragma: no cover - only used inside Isaac Sim
        logger.warning("Failed to import pxr. Collision filtering skipped. Error: %s", exc)
        return
    if not hasattr(UsdPhysics, "FilteredPairsAPI"):
        logger.warning("UsdPhysics.FilteredPairsAPI is unavailable. Collision filtering skipped.")
        return

    stage = get_current_stage()
    asset = env.scene[asset_cfg.name]
    robot_prims = prim_utils.find_matching_prims(asset.cfg.prim_path, stage=stage)
    if not robot_prims:
        logger.warning("No robot prims matched for collision filtering: %s", asset.cfg.prim_path)
        return

    for robot_prim in robot_prims:
        left_prims = _find_collision_prims(robot_prim, left_path_substring, UsdPhysics)
        right_prims = _find_collision_prims(robot_prim, right_path_substring, UsdPhysics)
        if not left_prims or not right_prims:
            logger.warning(
                "Collision filtering skipped; missing prims (left=%d, right=%d) under %s",
                len(left_prims),
                len(right_prims),
                robot_prim.GetPath(),
            )
            continue
        _apply_filtered_pairs(left_prims, right_prims, UsdPhysics)
        _apply_filtered_pairs(right_prims, left_prims, UsdPhysics)


def _find_collision_prims(robot_prim, path_substring: str, usd_physics_module):
    def _predicate(prim) -> bool:
        if not prim.HasAPI(usd_physics_module.CollisionAPI):
            return False
        return path_substring in prim.GetPath().pathString

    return prim_utils.get_all_matching_child_prims(robot_prim.GetPath(), predicate=_predicate)


def _apply_filtered_pairs(source_prims, target_prims, usd_physics_module):
    for source_prim in source_prims:
        api = usd_physics_module.FilteredPairsAPI.Apply(source_prim)
        rel = api.GetFilteredPairsRel()
        for target_prim in target_prims:
            rel.AddTarget(target_prim.GetPath())
