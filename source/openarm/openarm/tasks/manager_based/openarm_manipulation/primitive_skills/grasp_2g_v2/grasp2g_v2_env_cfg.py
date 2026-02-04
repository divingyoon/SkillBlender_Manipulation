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
grasp_2g_v2: Role-Separated Curriculum for Bimanual Grasping

Curriculum Stages:
- Stage 0 (LEFT_ONLY): Train left arm only, right arm holds initial pose
- Stage 1 (RIGHT_ONLY): Train right arm only, left arm holds learned pose
- Stage 2 (BIMANUAL): Train both arms simultaneously

Stage transition is based on phase progression success rate.
"""

from dataclasses import MISSING
import math

import isaaclab.sim as sim_utils
from isaaclab.sim import PhysxCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg

from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers.config import FRAME_MARKER_CFG

from . import mdp


# Curriculum Stage Enum
class CurriculumStage:
    LEFT_ONLY = 0
    RIGHT_ONLY = 1
    BIMANUAL = 2


@configclass
class Grasp2gSceneCfg(InteractiveSceneCfg):
    """Scene with a bimanual robot, table, and a cube to be grasped."""

    # robots
    robot: ArticulationCfg = MISSING

    # target object
    cup: RigidObjectCfg = MISSING
    cup2: RigidObjectCfg = MISSING

    # table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.25, 0.0, 0.0], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
    )

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    left_arm_action: ActionTerm = MISSING
    left_hand_action: ActionTerm = MISSING
    right_arm_action: ActionTerm = MISSING
    right_hand_action: ActionTerm = MISSING


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    left_cup_pose = mdp.InitialObjectPoseCommandCfg(
        asset_name="robot",
        asset_cfg=SceneEntityCfg("cup"),
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        goal_offset=(0.0, 0.0, 0.2),
    )

    right_cup_pose = mdp.InitialObjectPoseCommandCfg(
        asset_name="robot",
        asset_cfg=SceneEntityCfg("cup2"),
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        goal_offset=(0.0, 0.0, 0.2),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # Left arm observations
        left_joint_pos = ObsTerm(func=mdp.left_joint_pos_rel)
        left_joint_vel = ObsTerm(func=mdp.left_joint_vel_rel)

        left_cup_position = ObsTerm(
            func=mdp.root_pose,
            params={"asset_cfg": SceneEntityCfg("cup")},
        )

        cup_to_hand_left = ObsTerm(
            func=mdp.target_pos_in_tcp_frame,
            params={
                "tcp_body_name": "openarm_left_hand",
                "target_cfg": SceneEntityCfg("cup"),
                "offset": [0.0, 0.0, 0.0],
            },
        )

        left_gripper_state = ObsTerm(
            func=mdp.gripper_state,
            params={"joint_names": ["openarm_left_finger_joint1", "openarm_left_finger_joint2"]},
        )
        left_hand_id = ObsTerm(func=mdp.constant_value, params={"value": 1.0, "size": 1})
        left_tcp_cup_distance = ObsTerm(
            func=mdp.tcp_to_cup_distance,
            params={"tcp_body_name": "openarm_left_hand", "target_cfg": SceneEntityCfg("cup")},
        )

        left_arm_action = ObsTerm(func=mdp.last_action, params={"action_name": "left_arm_action"})
        left_hand_action = ObsTerm(func=mdp.last_action, params={"action_name": "left_hand_action"})

        # Right arm observations
        right_joint_pos = ObsTerm(func=mdp.right_joint_pos_rel)
        right_joint_vel = ObsTerm(func=mdp.right_joint_vel_rel)
        right_cup2_position = ObsTerm(
            func=mdp.root_pose,
            params={"asset_cfg": SceneEntityCfg("cup2")},
        )
        cup2_to_hand_right = ObsTerm(
            func=mdp.target_pos_in_tcp_frame,
            params={
                "tcp_body_name": "openarm_right_hand",
                "target_cfg": SceneEntityCfg("cup2"),
                "offset": [0.0, 0.0, 0.0],
            },
        )
        right_gripper_state = ObsTerm(
            func=mdp.gripper_state,
            params={"joint_names": ["openarm_right_finger_joint1", "openarm_right_finger_joint2"]},
        )
        right_hand_id = ObsTerm(func=mdp.constant_value, params={"value": -1.0, "size": 1})
        right_tcp_cup_distance = ObsTerm(
            func=mdp.tcp_to_cup_distance,
            params={"tcp_body_name": "openarm_right_hand", "target_cfg": SceneEntityCfg("cup2")},
        )
        right_arm_action = ObsTerm(func=mdp.last_action, params={"action_name": "right_arm_action"})
        right_hand_action = ObsTerm(func=mdp.last_action, params={"action_name": "right_hand_action"})

        # Curriculum stage indicator (for policy awareness)
        curriculum_stage = ObsTerm(func=mdp.curriculum_stage_obs, params={})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_cup_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.25, 0.25), "y": (0.2, 0.2), "z": (0.0, 0.0),
                "yaw": (-math.pi / 2, -math.pi / 2),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cup"),
        },
    )
    reset_cup2_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.25, 0.25), "y": (-0.2, -0.2), "z": (0.0, 0.0),
                "yaw": (-math.pi / 2, -math.pi / 2),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cup2"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP.

    Role-Separated Curriculum:
    - Stage 0 (LEFT_ONLY): Only left arm rewards active
    - Stage 1 (RIGHT_ONLY): Only right arm rewards active
    - Stage 2 (BIMANUAL): Both arms + bimanual coordination rewards active
    """

    # ============================================================
    # LEFT ARM REWARDS (Active in Stage 0 and Stage 2)
    # ============================================================

    left_reaching_object = RewTerm(
        func=mdp.staged_phase_object_ee_distance_error,
        params={
            "object_cfg": SceneEntityCfg("cup"),
            "eef_link_name": "openarm_left_hand",
            "active_stages": [0, 2],  # Active in LEFT_ONLY and BIMANUAL
            "phase_weights": [1.0, 0.0, 0.0, 0.0],
            "phase_params": {
                "eef_link_name": "openarm_left_hand",
                "lift_height": 0.1,
                "reach_distance": 0.1,
                "grasp_distance": 0.07,
                "close_threshold": 0.5,
                "hold_duration": 2.0,
            },
        },
        weight=-1.0,  # Reduced penalty for easier exploration
    )

    left_reaching_object_fine = RewTerm(
        func=mdp.staged_phase_object_ee_distance_xy_then_z,
        params={
            "std_xy": 0.10,  # Reduced from 0.15 for stronger XY gradient
            "std_z": 0.06,
            "z_weight": 2.0,  # Stronger Z gradient while keeping XY-first gate
            "xy_threshold": 0.10,  # Z reward when XY < 10cm (same as reach_distance)
            "object_cfg": SceneEntityCfg("cup"),
            "eef_link_name": "openarm_left_hand",
            "active_stages": [0, 2],
            "phase_weights": [1.0, 1.0, 0.0, 0.0],
            "phase_params": {
                "eef_link_name": "openarm_left_hand",
                "lift_height": 0.1,
                "reach_distance": 0.1,
                "grasp_distance": 0.07,
                "close_threshold": 0.5,
                "hold_duration": 2.0,
            },
        },
        weight=12.0,
    )

    left_object_displacement_penalty = RewTerm(
        func=mdp.staged_phase_object_root_displacement_penalty,
        params={
            "object_cfg": SceneEntityCfg("cup"),
            "active_stages": [0, 2],
            "phase_weights": [0.0, 1.0, 0.0, 0.0],
            "phase_params": {
                "eef_link_name": "openarm_left_hand",
                "lift_height": 0.1,
                "reach_distance": 0.07,
                "grasp_distance": 0.05,
                "close_threshold": 0.5,
                "hold_duration": 2.0,
            },
            "scale": 5.0,
        },
        weight=-2.0,
    )

    left_end_effector_orientation_tracking = RewTerm(
        func=mdp.staged_phase_hand_x_align_object_z_penalty_gated,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="openarm_left_hand"),
            "command_name": "left_cup_pose",
            "eef_link_name": "openarm_left_hand",
            "object_cfg": SceneEntityCfg("cup"),
            "gate_std": 0.05,
            "active_stages": [0, 2],
            "phase_weights": [1.0, 1.0, 0.0, 0.0],
            "phase_params": {
                "eef_link_name": "openarm_left_hand",
                "lift_height": 0.1,
                "reach_distance": 0.1,
                "grasp_distance": 0.07,
                "close_threshold": 0.5,
                "hold_duration": 2.0,
            },
        },
    )

    left_grasping_object = RewTerm(
        func=mdp.staged_phase_grasp_band_reward,
        params={
            "eef_link_name": "openarm_left_hand",
            "object_cfg": SceneEntityCfg("cup"),
            "close_min": 0.35,
            "close_max": 0.75,
            "active_stages": [0, 2],
            "phase_weights": [0.0, 1.0, 0.0, 0.0],
            "phase_params": {
                "eef_link_name": "openarm_left_hand",
                "lift_height": 0.1,
                "reach_distance": 0.1,
                "grasp_distance": 0.07,
                "close_threshold": 0.5,
                "hold_duration": 2.0,
            },
        },
        weight=5.0,
    )

    left_gripper_hold = RewTerm(
        func=mdp.staged_phase_gripper_hold_reward,
        params={
            "eef_link_name": "openarm_left_hand",
            "close_threshold": 0.5,
            "hold_duration": 2.0,
            "object_cfg": SceneEntityCfg("cup"),
            "hold_decay": 1.0,
            "active_stages": [0, 2],
            "phase_weights": [0.0, 0.0, 1.0, 1.0],
            "phase_params": {
                "eef_link_name": "openarm_left_hand",
                "lift_height": 0.1,
                "reach_distance": 0.1,
                "grasp_distance": 0.07,
                "close_threshold": 0.5,
                "hold_duration": 2.0,
            },
        },
        weight=1.0,
    )

    left_lifting_object = RewTerm(
        func=mdp.staged_phase_lift_delta_reward,
        params={
            "lift_height": 0.1,
            "object_cfg": SceneEntityCfg("cup"),
            "active_stages": [0, 2],
            "phase_weights": [0.0, 0.0, 5.0, 0.0],
            "phase_params": {
                "eef_link_name": "openarm_left_hand",
                "lift_height": 0.1,
                "reach_distance": 0.1,
                "grasp_distance": 0.07,
                "close_threshold": 0.5,
                "hold_duration": 2.0,
            },
        },
        weight=5.0,
    )

    left_object_goal_tracking = RewTerm(
        func=mdp.staged_phase_object_goal_distance_with_ee,
        params={
            "std": 0.3,
            "minimal_height": 0.04,
            "command_name": "left_cup_pose",
            "object_cfg": SceneEntityCfg("cup"),
            "eef_link_name": "openarm_left_hand",
            "reach_std": 0.1,
            "active_stages": [0, 2],
            "phase_weights": [0.0, 0.0, 0.0, 1.0],
            "phase_params": {
                "eef_link_name": "openarm_left_hand",
                "lift_height": 0.1,
                "reach_distance": 0.1,
                "grasp_distance": 0.07,
                "close_threshold": 0.5,
                "hold_duration": 2.0,
            },
        },
        weight=1.0,
    )

    # ============================================================
    # RIGHT ARM REWARDS (Active in Stage 1 and Stage 2)
    # ============================================================

    right_reaching_object = RewTerm(
        func=mdp.staged_phase_object_ee_distance_error,
        params={
            "object_cfg": SceneEntityCfg("cup2"),
            "eef_link_name": "openarm_right_hand",
            "active_stages": [1, 2],  # Active in RIGHT_ONLY and BIMANUAL
            "phase_weights": [1.0, 0.0, 0.0, 0.0],
            "phase_params": {
                "eef_link_name": "openarm_right_hand",
                "lift_height": 0.1,
                "reach_distance": 0.1,
                "grasp_distance": 0.07,
                "close_threshold": 0.5,
                "hold_duration": 2.0,
            },
        },
        weight=-1.0,
    )

    right_reaching_object_fine = RewTerm(
        func=mdp.staged_phase_object_ee_distance_xy_then_z,
        params={
            "std_xy": 0.10,  # Reduced from 0.15 for stronger XY gradient
            "std_z": 0.06,
            "z_weight": 2.0,  # Stronger Z gradient while keeping XY-first gate
            "xy_threshold": 0.10,  # Z reward when XY < 10cm (same as reach_distance)
            "object_cfg": SceneEntityCfg("cup2"),
            "eef_link_name": "openarm_right_hand",
            "active_stages": [1, 2],
            "phase_weights": [1.0, 1.0, 0.0, 0.0],
            "phase_params": {
                "eef_link_name": "openarm_right_hand",
                "lift_height": 0.1,
                "reach_distance": 0.1,
                "grasp_distance": 0.07,
                "close_threshold": 0.5,
                "hold_duration": 2.0,
            },
        },
        weight=12.0,
    )

    right_object_displacement_penalty = RewTerm(
        func=mdp.staged_phase_object_root_displacement_penalty,
        params={
            "object_cfg": SceneEntityCfg("cup2"),
            "active_stages": [1, 2],
            "phase_weights": [0.0, 1.0, 0.0, 0.0],
            "phase_params": {
                "eef_link_name": "openarm_right_hand",
                "lift_height": 0.1,
                "reach_distance": 0.07,
                "grasp_distance": 0.05,
                "close_threshold": 0.5,
                "hold_duration": 2.0,
            },
            "scale": 5.0,
        },
        weight=-2.0,
    )

    right_end_effector_orientation_tracking = RewTerm(
        func=mdp.staged_phase_hand_x_align_object_z_penalty_gated,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="openarm_right_hand"),
            "command_name": "right_cup_pose",
            "eef_link_name": "openarm_right_hand",
            "object_cfg": SceneEntityCfg("cup2"),
            "gate_std": 0.05,
            "active_stages": [1, 2],
            "phase_weights": [1.0, 1.0, 0.0, 0.0],
            "phase_params": {
                "eef_link_name": "openarm_right_hand",
                "lift_height": 0.1,
                "reach_distance": 0.1,
                "grasp_distance": 0.07,
                "close_threshold": 0.5,
                "hold_duration": 2.0,
            },
        },
    )

    right_grasping_object = RewTerm(
        func=mdp.staged_phase_grasp_band_reward,
        params={
            "eef_link_name": "openarm_right_hand",
            "object_cfg": SceneEntityCfg("cup2"),
            "close_min": 0.35,
            "close_max": 0.75,
            "active_stages": [1, 2],
            "phase_weights": [0.0, 1.0, 0.0, 0.0],
            "phase_params": {
                "eef_link_name": "openarm_right_hand",
                "lift_height": 0.1,
                "reach_distance": 0.1,
                "grasp_distance": 0.07,
                "close_threshold": 0.5,
                "hold_duration": 2.0,
            },
        },
        weight=5.0,
    )

    right_gripper_hold = RewTerm(
        func=mdp.staged_phase_gripper_hold_reward,
        params={
            "eef_link_name": "openarm_right_hand",
            "close_threshold": 0.5,
            "hold_duration": 2.0,
            "object_cfg": SceneEntityCfg("cup2"),
            "hold_decay": 1.0,
            "active_stages": [1, 2],
            "phase_weights": [0.0, 0.0, 1.0, 1.0],
            "phase_params": {
                "eef_link_name": "openarm_right_hand",
                "lift_height": 0.1,
                "reach_distance": 0.1,
                "grasp_distance": 0.07,
                "close_threshold": 0.5,
                "hold_duration": 2.0,
            },
        },
        weight=1.0,
    )

    right_lifting_object = RewTerm(
        func=mdp.staged_phase_lift_delta_reward,
        params={
            "lift_height": 0.1,
            "object_cfg": SceneEntityCfg("cup2"),
            "active_stages": [1, 2],
            "phase_weights": [0.0, 0.0, 5.0, 0.0],
            "phase_params": {
                "eef_link_name": "openarm_right_hand",
                "lift_height": 0.1,
                "reach_distance": 0.1,
                "grasp_distance": 0.07,
                "close_threshold": 0.5,
                "hold_duration": 2.0,
            },
        },
        weight=5.0,
    )

    right_object_goal_tracking = RewTerm(
        func=mdp.staged_phase_object_goal_distance_with_ee,
        params={
            "std": 0.3,
            "minimal_height": 0.04,
            "command_name": "right_cup_pose",
            "object_cfg": SceneEntityCfg("cup2"),
            "eef_link_name": "openarm_right_hand",
            "reach_std": 0.1,
            "active_stages": [1, 2],
            "phase_weights": [0.0, 0.0, 0.0, 1.0],
            "phase_params": {
                "eef_link_name": "openarm_right_hand",
                "lift_height": 0.1,
                "reach_distance": 0.1,
                "grasp_distance": 0.07,
                "close_threshold": 0.5,
                "hold_duration": 2.0,
            },
        },
        weight=1.0,
    )

    # ============================================================
    # BIMANUAL COORDINATION REWARDS (Active only in Stage 2)
    # ============================================================

    bimanual_reach_min = RewTerm(
        func=mdp.staged_bimanual_reach_min_reward,
        params={
            "std": 0.15,
            "left_eef_link_name": "openarm_left_hand",
            "right_eef_link_name": "openarm_right_hand",
            "left_object_cfg": SceneEntityCfg("cup"),
            "right_object_cfg": SceneEntityCfg("cup2"),
            "active_stages": [2],  # Only in BIMANUAL stage
            "phase_weights": [1.0, 1.0, 0.0, 0.0],
            "left_phase_params": {
                "eef_link_name": "openarm_left_hand",
                "lift_height": 0.1,
                "reach_distance": 0.1,
                "grasp_distance": 0.07,
                "close_threshold": 0.5,
                "hold_duration": 2.0,
            },
            "right_phase_params": {
                "eef_link_name": "openarm_right_hand",
                "lift_height": 0.1,
                "reach_distance": 0.1,
                "grasp_distance": 0.07,
                "close_threshold": 0.5,
                "hold_duration": 2.0,
            },
        },
        weight=6.0,
    )

    bimanual_phase_lag = RewTerm(
        func=mdp.staged_bimanual_phase_lag_penalty,
        params={
            "left_eef_link_name": "openarm_left_hand",
            "right_eef_link_name": "openarm_right_hand",
            "left_object_cfg": SceneEntityCfg("cup"),
            "right_object_cfg": SceneEntityCfg("cup2"),
            "active_stages": [2],  # Only in BIMANUAL stage
            "left_phase_params": {
                "eef_link_name": "openarm_left_hand",
                "lift_height": 0.1,
                "reach_distance": 0.1,
                "grasp_distance": 0.07,
                "close_threshold": 0.5,
                "hold_duration": 2.0,
            },
            "right_phase_params": {
                "eef_link_name": "openarm_right_hand",
                "lift_height": 0.1,
                "reach_distance": 0.1,
                "grasp_distance": 0.07,
                "close_threshold": 0.5,
                "hold_duration": 2.0,
            },
        },
        weight=-0.5,  # Reduced penalty compared to v1
    )

    bimanual_grasp_and = RewTerm(
        func=mdp.staged_bimanual_grasp_and_reward,
        params={
            "left_eef_link_name": "openarm_left_hand",
            "right_eef_link_name": "openarm_right_hand",
            "left_object_cfg": SceneEntityCfg("cup"),
            "right_object_cfg": SceneEntityCfg("cup2"),
            "active_stages": [2],
            "phase_weights": [0.0, 1.0, 1.0, 1.0],
            "left_phase_params": {
                "eef_link_name": "openarm_left_hand",
                "lift_height": 0.1,
                "reach_distance": 0.1,
                "grasp_distance": 0.07,
                "close_threshold": 0.5,
                "hold_duration": 2.0,
            },
            "right_phase_params": {
                "eef_link_name": "openarm_right_hand",
                "lift_height": 0.1,
                "reach_distance": 0.1,
                "grasp_distance": 0.07,
                "close_threshold": 0.5,
                "hold_duration": 2.0,
            },
        },
        weight=8.0,
    )

    bimanual_grasp_xor = RewTerm(
        func=mdp.staged_bimanual_grasp_xor_penalty,
        params={
            "left_eef_link_name": "openarm_left_hand",
            "right_eef_link_name": "openarm_right_hand",
            "left_object_cfg": SceneEntityCfg("cup"),
            "right_object_cfg": SceneEntityCfg("cup2"),
            "active_stages": [2],
            "phase_weights": [0.0, 1.0, 1.0, 1.0],
            "left_phase_params": {
                "eef_link_name": "openarm_left_hand",
                "lift_height": 0.1,
                "reach_distance": 0.1,
                "grasp_distance": 0.07,
                "close_threshold": 0.5,
                "hold_duration": 2.0,
            },
            "right_phase_params": {
                "eef_link_name": "openarm_right_hand",
                "lift_height": 0.1,
                "reach_distance": 0.1,
                "grasp_distance": 0.07,
                "close_threshold": 0.5,
                "hold_duration": 2.0,
            },
        },
        weight=-2.0,  # Reduced penalty compared to v1
    )

    # ============================================================
    # DIAGNOSTIC TERMS (Always active)
    # ============================================================

    left_grasp2g_phase = RewTerm(
        func=mdp.grasp2g_phase_value,
        params={
            "object_cfg": SceneEntityCfg("cup"),
            "phase_params": {
                "eef_link_name": "openarm_left_hand",
                "lift_height": 0.1,
                "reach_distance": 0.1,
                "grasp_distance": 0.07,
                "close_threshold": 0.5,
                "hold_duration": 2.0,
            },
        },
        weight=0.0,
    )

    right_grasp2g_phase = RewTerm(
        func=mdp.grasp2g_phase_value,
        params={
            "object_cfg": SceneEntityCfg("cup2"),
            "phase_params": {
                "eef_link_name": "openarm_right_hand",
                "lift_height": 0.1,
                "reach_distance": 0.1,
                "grasp_distance": 0.07,
                "close_threshold": 0.5,
                "hold_duration": 2.0,
            },
        },
        weight=0.0,
    )

    # Diagnostic terms
    left_hand_closure_diag = RewTerm(
        func=mdp.hand_closure_diagnostic,
        params={"eef_link_name": "openarm_left_hand"},
        weight=0.0,
    )
    right_hand_closure_diag = RewTerm(
        func=mdp.hand_closure_diagnostic,
        params={"eef_link_name": "openarm_right_hand"},
        weight=0.0,
    )
    left_eef_dist_diag = RewTerm(
        func=mdp.eef_distance_diagnostic,
        params={"eef_link_name": "openarm_left_hand", "object_cfg": SceneEntityCfg("cup")},
        weight=0.0,
    )
    right_eef_dist_diag = RewTerm(
        func=mdp.eef_distance_diagnostic,
        params={"eef_link_name": "openarm_right_hand", "object_cfg": SceneEntityCfg("cup2")},
        weight=0.0,
    )
    left_eef_dist_delta_diag = RewTerm(
        func=mdp.eef_dist_delta_diagnostic,
        params={"eef_link_name": "openarm_left_hand", "object_cfg": SceneEntityCfg("cup")},
        weight=0.0,
    )
    right_eef_dist_delta_diag = RewTerm(
        func=mdp.eef_dist_delta_diagnostic,
        params={"eef_link_name": "openarm_right_hand", "object_cfg": SceneEntityCfg("cup2")},
        weight=0.0,
    )
    # XY/Z separated diagnostics for curriculum debugging
    left_eef_dist_xy_diag = RewTerm(
        func=mdp.eef_dist_xy_diagnostic,
        params={"eef_link_name": "openarm_left_hand", "object_cfg": SceneEntityCfg("cup")},
        weight=0.0,
    )
    left_eef_dist_z_diag = RewTerm(
        func=mdp.eef_dist_z_diagnostic,
        params={"eef_link_name": "openarm_left_hand", "object_cfg": SceneEntityCfg("cup")},
        weight=0.0,
    )
    right_eef_dist_xy_diag = RewTerm(
        func=mdp.eef_dist_xy_diagnostic,
        params={"eef_link_name": "openarm_right_hand", "object_cfg": SceneEntityCfg("cup2")},
        weight=0.0,
    )
    right_eef_dist_z_diag = RewTerm(
        func=mdp.eef_dist_z_diagnostic,
        params={"eef_link_name": "openarm_right_hand", "object_cfg": SceneEntityCfg("cup2")},
        weight=0.0,
    )
    left_object_height_diag = RewTerm(
        func=mdp.object_height_diagnostic,
        params={"object_cfg": SceneEntityCfg("cup")},
        weight=0.0,
    )
    right_object_height_diag = RewTerm(
        func=mdp.object_height_diagnostic,
        params={"object_cfg": SceneEntityCfg("cup2")},
        weight=0.0,
    )

    # Curriculum stage diagnostic
    curriculum_stage_diag = RewTerm(
        func=mdp.curriculum_stage_diagnostic,
        params={},
        weight=0.0,
    )

    # ─── Regularization ───
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    cup_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cup")},
    )
    cup2_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cup2")},
    )
    cup_tipping = DoneTerm(
        func=mdp.cup_tipped,
        params={"asset_cfg": SceneEntityCfg("cup"), "max_tilt_deg": 30.0},
    )
    cup2_tipping = DoneTerm(
        func=mdp.cup_tipped,
        params={"asset_cfg": SceneEntityCfg("cup2"), "max_tilt_deg": 30.0},
    )


@configclass
class CurriculumCfg:
    """Role-Separated Curriculum Configuration.

    Stage transitions based on phase progression:
    - Stage 0 → 1: When left arm achieves phase >= 2.0 (lifted) consistently
    - Stage 1 → 2: When right arm achieves phase >= 2.0 consistently
    """

    # Stage advancement based on phase success
    advance_stage = CurrTerm(
        func=mdp.advance_curriculum_stage,
        params={
            "left_phase_threshold": 1.5,   # Left arm must reach phase 1.5+ to advance from Stage 0
            "right_phase_threshold": 1.5,  # Right arm must reach phase 1.5+ to advance from Stage 1
            "success_rate_threshold": 0.5, # 50% of envs must succeed
            "min_steps_per_stage": 5000,   # Minimum steps before advancing
        },
    )


@configclass
class Grasp2gV2EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the bimanual grasping environment with role-separated curriculum."""

    # Debug settings
    task_name: str = "grasp_2g_v2"
    debug_enabled: bool = False
    actor_obs_split_index: int = 41  # +1 for curriculum_stage
    critic_obs_split_index: int = 41

    # Reward/phase parameters
    grasp2g_reach_std_xy: float = 0.15
    grasp2g_reach_std_z: float = 0.1
    grasp2g_reach_z_weight: float = 2.0
    grasp2g_target_offset: tuple[float, float, float] = (0.0, 0.0, 0.07)

    # Phase transition settings
    phase_stability_reach_steps: int = 2   # Further relaxed to reduce reach reset churn
    phase_stability_grasp_steps: int = 3   # Relaxed from v1's 5
    phase_stability_lift_steps: int = 2    # Relaxed from v1's 3
    phase_demotion_enabled: bool = False
    phase_demotion_margin: float = 1.5

    # ============================================================
    # CURRICULUM STAGE SETTINGS
    # ============================================================
    curriculum_stage: int = 0  # Start with LEFT_ONLY
    # 0 = LEFT_ONLY: Only left arm learns
    # 1 = RIGHT_ONLY: Only right arm learns
    # 2 = BIMANUAL: Both arms learn with coordination

    # Stage transition thresholds
    stage_advance_left_phase: float = 1.5   # Left must reach phase 1.5 to advance
    stage_advance_right_phase: float = 1.5  # Right must reach phase 1.5 to advance
    stage_advance_success_rate: float = 0.5 # 50% of envs
    stage_advance_min_steps: int = 5000     # Min steps per stage

    # Action masking for inactive arm
    mask_inactive_arm_actions: bool = True  # Zero out actions for inactive arm

    scene: Grasp2gSceneCfg = Grasp2gSceneCfg(num_envs=10**3, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0  # Longer than v1's 8.0 for easier learning
        self.sim.dt = 1.0 / 100.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.observations.policy.concatenate_terms = True

        # Command goal visualization
        left_vis = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/left_cup_pose_goal")
        right_vis = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/right_cup_pose_goal")
        left_vis.markers["frame"].scale = (0.08, 0.08, 0.08)
        right_vis.markers["frame"].scale = (0.08, 0.08, 0.08)
        self.commands.left_cup_pose.goal_pose_visualizer_cfg = left_vis
        self.commands.right_cup_pose.goal_pose_visualizer_cfg = right_vis

        if not self.debug_enabled:
            self.commands.left_cup_pose.debug_vis = False
            self.commands.right_cup_pose.debug_vis = False

        self.sim.physx = PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_max_num_partitions=8,
            gpu_collision_stack_size=2**24,
        )
