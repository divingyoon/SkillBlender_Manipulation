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
# from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp


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
        # target_cup_position = ObsTerm(
        #     func=mdp.generated_commands, params={"command_name": "left_cup_pose"}
        # )
        # target_cup2_position = ObsTerm(
        #     func=mdp.generated_commands, params={"command_name": "right_cup_pose"}
        # )

        # Gripper state
        left_gripper_state = ObsTerm(
            func=mdp.gripper_state,
            params={"joint_names": ["openarm_left_finger_joint1", "openarm_left_finger_joint2"]},
        )
        # Hand identity (swap-stable cue)
        left_hand_id = ObsTerm(func=mdp.constant_value, params={"value": 1.0, "size": 1})
        # Distance to cup (scalar, for easier learning)
        left_tcp_cup_distance = ObsTerm(
            func=mdp.tcp_to_cup_distance,
            params={"tcp_body_name": "openarm_left_hand", "target_cfg": SceneEntityCfg("cup")},
        )

        left_arm_action = ObsTerm(func=mdp.last_action, params={"action_name": "left_arm_action"})
        left_hand_action = ObsTerm(func=mdp.last_action, params={"action_name": "left_hand_action"})

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
    """Reward terms for the MDP."""

    # Phase 정의 (손별):
    # 0: reach (물체 접근)
    # 1: grasp (그리퍼 닫힘/접근 게이트 통과)
    # 2: lift (물체 높이가 lift_height 이상)
    # 3: hold/goal (lift 이후 추적 단계)

    # Phase 0: coarse reaching (오차 기반 보상)
    # Curriculum: 초기 -2.0 → 3000 step 후 -0.5로 복귀
    left_reaching_object = RewTerm(
        func=mdp.phase_object_ee_distance_error,
        params={
            "object_cfg": SceneEntityCfg("cup"),
            "eef_link_name": "openarm_left_hand",
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
        weight=-2.0,
    )
    right_reaching_object = RewTerm(
        func=mdp.phase_object_ee_distance_error,
        params={
            "object_cfg": SceneEntityCfg("cup2"),
            "eef_link_name": "openarm_right_hand",
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
        weight=-2.0,
    )

    # Phase 0-1: fine reaching (XY/Z 분리)
    # std 확대: 0.05→0.15(xy), 0.05→0.10(z) - EEF 초기거리 0.2~0.3m 커버
    # Curriculum: 초기 8.0 → 3000 step 후 5.0으로 복귀
    left_reaching_object_fine = RewTerm(
        func=mdp.phase_object_ee_distance_xyz_weighted,
        params={
            "std_xy": 0.15,
            "std_z": 0.10,
            "z_weight": 2.0,
            "object_cfg": SceneEntityCfg("cup"),
            "eef_link_name": "openarm_left_hand",
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
        weight=8.0,
    )
    right_reaching_object_fine = RewTerm(
        func=mdp.phase_object_ee_distance_xyz_weighted,
        params={
            "std_xy": 0.15,
            "std_z": 0.10,
            "z_weight": 2.0,
            "object_cfg": SceneEntityCfg("cup2"),
            "eef_link_name": "openarm_right_hand",
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
        weight=8.0,
    )

    # Bimanual reach: min(reach_L, reach_R)
    bimanual_reach_min = RewTerm(
        func=mdp.bimanual_reach_min_reward,
        params={
            "std": 0.15,
            "left_eef_link_name": "openarm_left_hand",
            "right_eef_link_name": "openarm_right_hand",
            "left_object_cfg": SceneEntityCfg("cup"),
            "right_object_cfg": SceneEntityCfg("cup2"),
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

    # Bimanual phase lag penalty: |phase_L - phase_R|
    bimanual_phase_lag = RewTerm(
        func=mdp.bimanual_phase_lag_penalty,
        params={
            "left_eef_link_name": "openarm_left_hand",
            "right_eef_link_name": "openarm_right_hand",
            "left_object_cfg": SceneEntityCfg("cup"),
            "right_object_cfg": SceneEntityCfg("cup2"),
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
        weight=-1.0,
    )

    # Phase 0-1: grasp/lift 전에 물체 이동을 억제.
    left_object_displacement_penalty = RewTerm(
        func=mdp.phase_object_root_displacement_penalty,
        params={
            "object_cfg": SceneEntityCfg("cup"),
            "phase_weights": [1.0, 1.0, 0.0, 0.0],
            "phase_params": {
                "eef_link_name": "openarm_left_hand",
                "lift_height": 0.1,
                "reach_distance": 0.07,
                "grasp_distance": 0.05,
                "close_threshold": 0.5,
                "hold_duration": 2.0,
            },
            "scale": 10.0,
        },
        weight=-5.0,
    )
    # Phase 0-1: grasp/lift 전에 물체 이동을 억제.
    right_object_displacement_penalty = RewTerm(
        func=mdp.phase_object_root_displacement_penalty,
        params={
            "object_cfg": SceneEntityCfg("cup2"),
            "phase_weights": [1.0, 1.0, 0.0, 0.0],
            "phase_params": {
                "eef_link_name": "openarm_right_hand",
                "lift_height": 0.1,
                "reach_distance": 0.07,
                "grasp_distance": 0.05,
                "close_threshold": 0.5,
                "hold_duration": 2.0,
            },
            "scale": 10.0,  #컵이 2cm(0.02m) 움직였고 scale=1.0, weight=-1.0이면 보상에 -0.02 추가
        },
        weight=-5.0,
    )

    # Phase 0: reach 동안 방향 정렬.
    left_end_effector_orientation_tracking = RewTerm(
        func=mdp.phase_hand_x_align_object_z_penalty_gated,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="openarm_left_hand"),
            "command_name": "left_cup_pose",
            "eef_link_name": "openarm_left_hand",
            "object_cfg": SceneEntityCfg("cup"),
            "gate_std": 0.1,
            "phase_weights": [1.0, 0.3, 0.0, 0.0],
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
    right_end_effector_orientation_tracking = RewTerm(
        func=mdp.phase_hand_x_align_object_z_penalty_gated,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="openarm_right_hand"),
            "command_name": "right_cup_pose",
            "eef_link_name": "openarm_right_hand",
            "object_cfg": SceneEntityCfg("cup2"),
            "gate_std": 0.1,
            "phase_weights": [1.0, 0.3, 0.0, 0.0],
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
    # Phase 1: grasp 보상 (거리 + 그리퍼 닫힘 연속 보상)
    left_grasping_object = RewTerm(
        func=mdp.phase_grasp_band_reward,
        params={
            "eef_link_name": "openarm_left_hand",
            "object_cfg": SceneEntityCfg("cup"),
            "close_min": 0.35,
            "close_max": 0.75,
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
    right_grasping_object = RewTerm(
        func=mdp.phase_grasp_band_reward,
        params={
            "eef_link_name": "openarm_right_hand",
            "object_cfg": SceneEntityCfg("cup2"),
            "close_min": 0.35,
            "close_max": 0.75,
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

    # Bimanual grasp AND / XOR (simultaneous success shaping)
    bimanual_grasp_and = RewTerm(
        func=mdp.bimanual_grasp_and_reward,
        params={
            "left_eef_link_name": "openarm_left_hand",
            "right_eef_link_name": "openarm_right_hand",
            "left_object_cfg": SceneEntityCfg("cup"),
            "right_object_cfg": SceneEntityCfg("cup2"),
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
        func=mdp.bimanual_grasp_xor_penalty,
        params={
            "left_eef_link_name": "openarm_left_hand",
            "right_eef_link_name": "openarm_right_hand",
            "left_object_cfg": SceneEntityCfg("cup"),
            "right_object_cfg": SceneEntityCfg("cup2"),
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
        weight=-4.0,
    )
    # Phase 2-3: grasp 이후 그리퍼 닫힘 유지 보상
    left_gripper_hold = RewTerm(
        func=mdp.phase_gripper_hold_reward,
        params={
            "eef_link_name": "openarm_left_hand",
            "close_threshold": 0.5,
            "hold_duration": 2.0,
            "object_cfg": SceneEntityCfg("cup"),
            "hold_decay": 1.0,
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
    right_gripper_hold = RewTerm(
        func=mdp.phase_gripper_hold_reward,
        params={
            "eef_link_name": "openarm_right_hand",
            "close_threshold": 0.5,
            "hold_duration": 2.0,
            "object_cfg": SceneEntityCfg("cup2"),
            "hold_decay": 1.0,
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
    # Phase 2-3: grasp 이후 lift 진행도.
    left_lifting_object = RewTerm(
        func=mdp.phase_lift_delta_reward,
        params={
            "lift_height": 0.1,
            "object_cfg": SceneEntityCfg("cup"),
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
    # Phase 2-3: grasp 이후 lift 진행도.
    right_lifting_object = RewTerm(
        func=mdp.phase_lift_delta_reward,
        params={
            "lift_height": 0.1,
            "object_cfg": SceneEntityCfg("cup2"),
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

    # Phase 3: lift 이후 목표 추적 (현재 weight=0으로 비활성).
    left_object_goal_tracking = RewTerm(
        func=mdp.phase_object_goal_distance_with_ee,
        params={
            "std": 0.3,
            "minimal_height": 0.04,
            "command_name": "left_cup_pose",
            "object_cfg": SceneEntityCfg("cup"),
            "eef_link_name": "openarm_left_hand",
            "reach_std": 0.1,
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
    # Phase 3: lift 이후 목표 추적 (현재 weight=0으로 비활성).
    right_object_goal_tracking = RewTerm(
        func=mdp.phase_object_goal_distance_with_ee,
        params={
            "std": 0.3,
            "minimal_height": 0.04,
            "command_name": "right_cup_pose",
            "object_cfg": SceneEntityCfg("cup2"),
            "eef_link_name": "openarm_right_hand",
            "reach_std": 0.1,
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

    # Phase 3: 정밀 목표 추적 (현재 weight=0으로 비활성).
    left_object_goal_tracking_fine_grained = RewTerm(
        func=mdp.phase_object_goal_distance_with_ee,
        params={
            "std": 0.05,
            "minimal_height": 0.04,
            "command_name": "left_cup_pose",
            "object_cfg": SceneEntityCfg("cup"),
            "eef_link_name": "openarm_left_hand",
            "reach_std": 0.1,
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
    # Phase 3: 정밀 목표 추적 (현재 weight=0으로 비활성).
    right_object_goal_tracking_fine_grained = RewTerm(
        func=mdp.phase_object_goal_distance_with_ee,
        params={
            "std": 0.05,
            "minimal_height": 0.04,
            "command_name": "right_cup_pose",
            "object_cfg": SceneEntityCfg("cup2"),
            "eef_link_name": "openarm_right_hand",
            "reach_std": 0.1,
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

    # Phase 값 로깅/진단용.
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
    # Phase 값 로깅/진단용.
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
    """Curriculum: 초기에 reaching 강화, 이후 원래 값으로 복귀.

    동작 방식: modify_reward_weight는 num_steps 이후에 weight 값으로 변경.
    RewardsCfg의 초기값이 강화된 값이고, curriculum이 target으로 복귀시킴.
    - left/right_reaching_object: -2.0 → -0.5 (3000 step 후)
    - left/right_reaching_object_fine: 8.0 → 5.0 (3000 step 후)
    """
    left_reaching_object = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "left_reaching_object", "weight": -0.5, "num_steps": 3000},
    )
    right_reaching_object = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "right_reaching_object", "weight": -0.5, "num_steps": 3000},
    )
    left_reaching_object_fine = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "left_reaching_object_fine", "weight": 5.0, "num_steps": 3000},
    )
    right_reaching_object_fine = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "right_reaching_object_fine", "weight": 5.0, "num_steps": 3000},
    )


@configclass
class Grasp2gEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the bimanual grasping environment."""

    # Debug settings (used by reward functions)
    task_name: str = "grasp_2g_v1"
    debug_grasp_left: bool = True
    debug_grasp_left_interval: int = 200
    debug_grasp_right: bool = True
    debug_grasp_right_interval: int = 200
    debug_grasp_target_vis: bool = True
    debug_grasp_target_vis_interval: int = 10
    debug_reach_reward: bool = True
    debug_reach_reward_interval: int = 200
    debug_obs_split: bool = True
    debug_reward_mapping: bool = True
    actor_obs_split_index: int = 40
    critic_obs_split_index: int = 40

    # reach 보상 스케일 파라미터 (로그/해석용)
    grasp2g_reach_std_xy: float = 0.15
    grasp2g_reach_std_z: float = 0.1
    grasp2g_reach_z_weight: float = 2.0

    # 보상/페이즈 거리 계산 기준 오프셋 (컵 루트 기준)
    # 예: (0, 0, 0.05) -> 컵 바닥면 기준 +5cm 지점을 목표로 사용
    grasp2g_target_offset: tuple[float, float, float] = (0.0, 0.0, 0.07)

    enable_gripper_hold: bool = False

    # Phase transition 안정성 설정 (N-step 연속 조건)
    phase_stability_reach_steps: int = 10   # Phase 0→1: N step 연속 reach_distance 이내
    phase_stability_grasp_steps: int = 5    # Phase 1→2: N step 연속 grasp 조건 충족
    phase_stability_lift_steps: int = 3     # Phase 2→3: N step 연속 lift_height 이상
    phase_demotion_enabled: bool = False    # 역전환 토글 (기본 비활성)
    phase_demotion_margin: float = 1.5      # 역전환 거리 배수 (reach_distance * margin)

    # 디버그 토글 (로그/시각화 공통)
    debug_enabled: bool = False

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
        self.episode_length_s = 8.0
        self.sim.dt = 1.0 / 100.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (3.5, 3.5, 3.5)
        # RSL-RL ActorCritic expects 1D observations.
        self.observations.policy.concatenate_terms = True
        # 커맨드 목표 시각화 프림 경로를 좌/우로 분리
        left_vis = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/left_cup_pose_goal")
        right_vis = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/right_cup_pose_goal")
        left_vis.markers["frame"].scale = (0.08, 0.08, 0.08)
        right_vis.markers["frame"].scale = (0.08, 0.08, 0.08)
        self.commands.left_cup_pose.goal_pose_visualizer_cfg = left_vis
        self.commands.right_cup_pose.goal_pose_visualizer_cfg = right_vis

        # 디버그 토글 적용 (콘솔 로그/시각화 공통)
        if not self.debug_enabled:
            self.debug_grasp_left = False
            self.debug_grasp_right = False
            self.debug_reach_reward = False
            self.debug_grasp_target_vis = False
            self.commands.left_cup_pose.debug_vis = False
            self.commands.right_cup_pose.debug_vis = False

        # assign a default physx material to all scene geometries
        # we can also do this per-asset in the scene definition
        self.sim.physx = PhysxCfg(
            solver_type=1,  # TGS
            max_position_iteration_count=192,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            # increase buffers to prevent overflow errors
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_max_num_partitions=8,
            gpu_collision_stack_size=2**24,
            # set default material properties
            # default_material=RigidBodyMaterialCfg(
            #     static_friction=1.0,
            #     dynamic_friction=1.0,
            #     restitution=0.0,
            # ),
        )
