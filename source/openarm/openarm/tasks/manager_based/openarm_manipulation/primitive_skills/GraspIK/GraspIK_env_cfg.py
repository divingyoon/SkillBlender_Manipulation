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
#from isaaclab.sim.schemas.schemas_cfg import RigidBodyMaterialCfg
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
# from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp


@configclass
class GraspIKSceneCfg(InteractiveSceneCfg):
    """Scene with a bimanual robot, table, and a cup to be grasped."""

    # robots
    robot: ArticulationCfg = MISSING

    # target object
    object_source: AssetBaseCfg = MISSING
    object: RigidObjectCfg = MISSING
    object2_source: AssetBaseCfg = MISSING
    object2: RigidObjectCfg = MISSING

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
class GraspIKCommandsCfg:
    """Command terms for the MDP."""

    left_object_pose = mdp.ObjectPoseCommandCfg(
        asset_name="robot",
        asset_cfg=SceneEntityCfg("object"),
        resampling_time_range=(5.0, 5.0),
        pre_grasp_offset=(0.0, 0.0, 0.0),
        hold_offset=(0.0, 0.0, 0.20),
        lift_threshold_z=0.05,
        max_target_z=0.05,
    )

    right_object_pose = mdp.ObjectPoseCommandCfg(
        asset_name="robot",
        asset_cfg=SceneEntityCfg("object2"),
        resampling_time_range=(5.0, 5.0),
        pre_grasp_offset=(0.0, 0.0, 0.0),
        hold_offset=(0.0, -0.0, 0.20),
        lift_threshold_z=0.05,
        max_target_z=0.05,
    )



@configclass
class GraspIKActionsCfg:
    """Action specifications for the MDP.

    DualHead를 위해 Left → Right 순서로 배치:
    [0:N] left_arm, [N:M] left_hand, [M:P] right_arm, [P:Q] right_hand
    """

    left_arm_action: ActionTerm = MISSING
    left_hand_action: ActionTerm = MISSING
    right_arm_action: ActionTerm = MISSING
    right_hand_action: ActionTerm = MISSING


@configclass
class GraspIKObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        target_object_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "left_object_pose"}
        )
        target_object2_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "right_object_pose"}
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("object")},
        )
        object2_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("object2")},
        )
        object_obs = ObsTerm(
            func=mdp.object_obs,
            params={
                "left_eef_link_name": "openarm_left_hand",
                "right_eef_link_name": "openarm_right_hand",
            },
        )
        object2_obs = ObsTerm(
            func=mdp.object2_obs,
            params={
                "left_eef_link_name": "openarm_left_hand",
                "right_eef_link_name": "openarm_right_hand",
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        # [방법4] 팔 구분자 (Arm Identifier) - 양손 비대칭 학습 문제 해결용
        # 네트워크가 명시적으로 좌/우 팔을 구분할 수 있도록 one-hot 벡터 추가
        # enable_arm_identifier=False로 설정하면 비활성화됨
        left_arm_id = ObsTerm(func=mdp.left_arm_identifier)
        right_arm_id = ObsTerm(func=mdp.right_arm_identifier)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class PolicyLowCfg(ObsGroup):
        """Observations for low-level skills (legacy shape)."""

        target_object_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "left_object_pose"}
        )
        target_object2_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "right_object_pose"}
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("object")},
        )
        object2_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("object2")},
        )
        object_obs = ObsTerm(
            func=mdp.object_obs,
            params={
                "left_eef_link_name": "openarm_left_hand",
                "right_eef_link_name": "openarm_right_hand",
            },
        )
        object2_obs = ObsTerm(
            func=mdp.object2_obs,
            params={
                "left_eef_link_name": "openarm_left_hand",
                "right_eef_link_name": "openarm_right_hand",
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy_low: PolicyLowCfg = PolicyLowCfg()


@configclass
class GraspIKEventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform_world,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.35, 0.35), "y": (0.25, 0.25), "z": (0.0, 0.0),
                "yaw": (-math.pi / 2, -math.pi / 2),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    reset_object2_position = EventTerm(
        func=mdp.reset_root_state_uniform_world,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.35, 0.35), "y": (-0.25, -0.25), "z": (0.0, 0.0),
                "yaw": (-math.pi / 2, -math.pi / 2),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object2"),
        },
    )
    reset_robot_tcp_to_cups = EventTerm(
        func=mdp.reset_robot_tcp_to_cups,
        mode="reset",
        params={
            "left_cup_name": "object",
            "right_cup_name": "object2",
            # Target the palm links when resetting TCP to cups.  These frames
            # correspond to the 2‑finger gripper's palm rather than the finger-tip TCP.
            "left_tcp_body_name": "openarm_left_hand",
            "right_tcp_body_name": "openarm_right_hand",
            "offset": (-0.1, 0.0, 0.05),
            "ik_iters": 7,
            "ik_lambda": 0.5,
            "max_delta": 0.15,
        },
    )


@configclass
class GraspIKRewardsCfg:
    """Reward terms for the MDP.

    [보상 스케일 밸런싱]
    Phase 전환 시 보상 급변으로 인한 학습 붕괴 방지를 위해 스케일 조정:
    - Phase 0 (Reaching) 총합: ~11.5 (reaching 5.0 + orientation 2.5 + gripper_open 4.0)
    - Phase 2 (Lifting) 총합: ~38.0 (lifting 16.0 + tracking 16.0 + fine 6.0)
    - 비율: ~3.3배 (기존 ~20배에서 완화)
    """

    # ==========================================================================
    # Phase 0: Reaching 보상 (물체에 접근)
    # ==========================================================================
    left_reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1, "object_cfg": SceneEntityCfg("object"), "ee_frame_cfg": SceneEntityCfg("left_ee_frame")},
        weight=5.0,  # 기존 1.0 → 2.5 (Phase 밸런싱)
    )
    right_reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1, "object_cfg": SceneEntityCfg("object2"), "ee_frame_cfg": SceneEntityCfg("right_ee_frame")},
        weight=5.0,  # 기존 1.0 → 2.5 (Phase 밸런싱)
    )

    # Orientation 보상 (손 방향 정렬)
    left_end_effector_orientation_tracking = RewTerm(
        func=mdp.phase_hand_x_align_object_z_reward,
        weight=0.2,  # 기존 0.5 → 0.75 (Phase 밸런싱)
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="openarm_left_hand"),
            "command_name": "left_object_pose",
            "object_cfg": SceneEntityCfg("object"),
            "phase_weights": [1.0, 0.5, 0.0, 0.0],  # Phase 1-3에서만 활성화
            "phase_params": {
                "eef_link_name": "openarm_left_hand",
                "lift_height": 0.03,
                "reach_distance": 0.05,
                "align_threshold": 0.0,
                "grasp_distance": 0.02,
                "close_threshold": 0.6,
                "hold_duration": 2.0,
            },
        },
    )
    right_end_effector_orientation_tracking = RewTerm(
        func=mdp.phase_hand_x_align_object_z_reward,
        weight=0.2,  # 기존 0.5 → 0.75 (Phase 밸런싱)
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="openarm_right_hand"),
            "command_name": "right_object_pose",
            "object_cfg": SceneEntityCfg("object2"),
            "phase_weights": [1.0, 0.5, 0.0, 0.0],  # Phase 1-3에서만 활성화
            "phase_params": {
                "eef_link_name": "openarm_right_hand",
                "lift_height": 0.03,
                "reach_distance": 0.05,
                "align_threshold": 0.0,
                "grasp_distance": 0.02,
                "close_threshold": 0.6,
                "hold_duration": 2.0,
            },
        },
    )
    left_end_effector_z_to_object_y = RewTerm(
        func=mdp.phase_hand_z_align_object_y_reward,
        weight=0.2,  # 기존 0.25 → 0.5 (Phase 밸런싱)
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="openarm_left_hand"),
            "command_name": "left_object_pose",
            "object_cfg": SceneEntityCfg("object"),
            "phase_weights": [1.0, 0.0, 0.0, 0.0],  # Phase 1-3에서만 활성화
            "phase_params": {
                "eef_link_name": "openarm_left_hand",
                "lift_height": 0.03,
                "reach_distance": 0.05,
                "align_threshold": 0.0,
                "grasp_distance": 0.02,
                "close_threshold": 0.6,
                "hold_duration": 2.0,
            },
        },
    )
    right_end_effector_z_to_object_y = RewTerm(
        func=mdp.phase_hand_z_align_object_y_reward,
        weight=0.2,  # 기존 0.25 → 0.5 (Phase 밸런싱)
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="openarm_right_hand"),
            "command_name": "right_object_pose",
            "object_cfg": SceneEntityCfg("object2"),
            "phase_weights": [1.0, 0.0, 0.0, 0.0],  # Phase 1-3에서만 활성화
            "phase_params": {
                "eef_link_name": "openarm_right_hand",
                "lift_height": 0.03,
                "reach_distance": 0.05,
                "align_threshold": 0.0,
                "grasp_distance": 0.02,
                "close_threshold": 0.6,
                "hold_duration": 2.0,
            },
        },
    )

    # [해결A] 그리퍼 열기 보상 (Phase 0-1에서 그리퍼를 열어두도록 유도)
    # 오른손이 너무 일찍 그리퍼를 닫는 문제 해결
    left_gripper_open = RewTerm(
        func=mdp.phase_gripper_open_reward,
        weight=1.0,
        params={
            "eef_link_name": "openarm_left_hand",
            "object_cfg": SceneEntityCfg("object"),
            "phase_weights": [1.0, 0.5, 0.0, 0.0],  # Phase 0: 100%, Phase 1: 50%, Phase 2-3: 비활성화
            "phase_params": {
                "eef_link_name": "openarm_left_hand",
                "lift_height": 0.03,
                "reach_distance": 0.05,
                "align_threshold": 0.0,
                "grasp_distance": 0.02,
                "close_threshold": 0.6,
                "hold_duration": 2.0,
            },
        },
    )
    right_gripper_open = RewTerm(
        func=mdp.phase_gripper_open_reward,
        weight=1.0,
        params={
            "eef_link_name": "openarm_right_hand",
            "object_cfg": SceneEntityCfg("object2"),
            "phase_weights": [1.0, 0.5, 0.0, 0.0],  # Phase 0: 100%, Phase 1: 50%, Phase 2-3: 비활성화
            "phase_params": {
                "eef_link_name": "openarm_right_hand",
                "lift_height": 0.03,
                "reach_distance": 0.05,
                "align_threshold": 0.0,
                "grasp_distance": 0.02,
                "close_threshold": 0.6,
                "hold_duration": 2.0,
            },
        },
    )

    # [해결B] 그리퍼 닫기 보상 (Phase 2-3에서 그리퍼를 닫고 유지하도록 유도)
    # Phase 2-3에서 물체를 들고 있을 때 그리퍼를 열어버리면 떨어뜨림
    # 닫고 있으면 보상 → 유지하도록 학습
    left_gripper_close = RewTerm(
        func=mdp.phase_gripper_close_reward,
        weight=1.0,
        params={
            "eef_link_name": "openarm_left_hand",
            "object_cfg": SceneEntityCfg("object"),
            "phase_weights": [0.0, 0.5, 1.0, 1.0],  # Phase 0: 비활성화, Phase 1: 50%, Phase 2-3: 100%
            "phase_params": {
                "eef_link_name": "openarm_left_hand",
                "lift_height": 0.03,
                "reach_distance": 0.05,
                "align_threshold": 0.0,
                "grasp_distance": 0.02,
                "close_threshold": 0.6,
                "hold_duration": 2.0,
            },
        },
    )
    right_gripper_close = RewTerm(
        func=mdp.phase_gripper_close_reward,
        weight=1.0,
        params={
            "eef_link_name": "openarm_right_hand",
            "object_cfg": SceneEntityCfg("object2"),
            "phase_weights": [0.0, 0.5, 1.0, 1.0],  # Phase 0: 비활성화, Phase 1: 50%, Phase 2-3: 100%
            "phase_params": {
                "eef_link_name": "openarm_right_hand",
                "lift_height": 0.03,
                "reach_distance": 0.05,
                "align_threshold": 0.0,
                "grasp_distance": 0.02,
                "close_threshold": 0.6,
                "hold_duration": 2.0,
            },
        },
    )

    # ==========================================================================
    # Phase 2-3: Lifting & Goal Tracking 보상 (들어올리기)
    # ==========================================================================
    left_lifting_object = RewTerm(
        func=mdp.phase_lift_reward,
        params={
            "lift_height": 0.03,
            "object_cfg": SceneEntityCfg("object"),
            "phase_weights": [0.0, 0.0, 1.0, 1.0],  # Phase 2-3에서만 활성화
            "phase_params": {
                "eef_link_name": "openarm_left_hand",
                "lift_height": 0.03,
                "reach_distance": 0.05,
                "align_threshold": 0.0,
                "grasp_distance": 0.02,
                "close_threshold": 0.6,
                "hold_duration": 2.0,
            },
        },
        weight=8.0,  # 기존 15.0 → 8.0 (Phase 밸런싱)
    )
    right_lifting_object = RewTerm(
        func=mdp.phase_lift_reward,
        params={
            "lift_height": 0.03,
            "object_cfg": SceneEntityCfg("object2"),
            "phase_weights": [0.0, 0.0, 1.0, 1.0],  # Phase 2-3에서만 활성화
            "phase_params": {
                "eef_link_name": "openarm_right_hand",
                "lift_height": 0.03,
                "reach_distance": 0.05,
                "align_threshold": 0.0,
                "grasp_distance": 0.02,
                "close_threshold": 0.6,
                "hold_duration": 2.0,
            },
        },
        weight=8.0,  # 기존 15.0 → 8.0 (Phase 밸런싱)
    )
    # ─── Object goal tracking (coarse) ──────────────────────────────────────────
    left_object_goal_tracking = RewTerm(
        func=mdp.phase_object_goal_distance_with_ee,
        params={
            "std": 0.3,
            "minimal_height": 0.04,
            "command_name": "left_object_pose",
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("left_ee_frame"),
            "reach_std": 0.1,
            "phase_weights": [0.0, 0.0, 1.0, 1.0],  # Phase 2-3에서만 활성화
            "phase_params": {
                "eef_link_name": "openarm_left_hand",
                "lift_height": 0.03,
                "reach_distance": 0.05,
                "align_threshold": 0.0,
                "grasp_distance": 0.02,
                "close_threshold": 0.6,
                "hold_duration": 2.0,
            },
        },
        weight=8.0,  # 기존 16.0 → 8.0 (Phase 밸런싱)
    )
    right_object_goal_tracking = RewTerm(
        func=mdp.phase_object_goal_distance_with_ee,
        params={
            "std": 0.3,
            "minimal_height": 0.04,
            "command_name": "right_object_pose",
            "object_cfg": SceneEntityCfg("object2"),
            "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
            "reach_std": 0.1,
            "phase_weights": [0.0, 0.0, 1.0, 1.0],  # Phase 2-3에서만 활성화
            "phase_params": {
                "eef_link_name": "openarm_right_hand",
                "lift_height": 0.03,
                "reach_distance": 0.05,
                "align_threshold": 0.0,
                "grasp_distance": 0.02,
                "close_threshold": 0.6,
                "hold_duration": 2.0,
            },
        },
        weight=8.0,  # 기존 16.0 → 8.0 (Phase 밸런싱)
    )
    
    # ─── Object goal tracking (fine-grained) ────────────────────────────────────
    left_object_goal_tracking_fine_grained = RewTerm(
        func=mdp.phase_object_goal_distance_with_ee,
        params={
            "std": 0.05,
            "minimal_height": 0.04,
            "command_name": "left_object_pose",
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("left_ee_frame"),
            "reach_std": 0.1,
            "phase_weights": [0.0, 0.0, 1.0, 1.0],  # Phase 2-3에서만 활성화
            "phase_params": {
                "eef_link_name": "openarm_left_hand",
                "lift_height": 0.03,
                "reach_distance": 0.07,
                "align_threshold": 0.0,
                "grasp_distance": 0.02,
                "close_threshold": 0.6,
                "hold_duration": 2.0,
            },
        },
        weight=3.0,  # 기존 5.0 → 3.0 (Phase 밸런싱)
    )
    right_object_goal_tracking_fine_grained = RewTerm(
        func=mdp.phase_object_goal_distance_with_ee,
        params={
            "std": 0.05,
            "minimal_height": 0.04,
            "command_name": "right_object_pose",
            "object_cfg": SceneEntityCfg("object2"),
            "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
            "reach_std": 0.1,
            "phase_weights": [0.0, 0.0, 1.0, 1.0],  # Phase 2-3에서만 활성화
            "phase_params": {
                "eef_link_name": "openarm_right_hand",
                "lift_height": 0.03,
                "reach_distance": 0.07,
                "align_threshold": 0.0,
                "grasp_distance": 0.02,
                "close_threshold": 0.6,
                "hold_duration": 2.0,
            },
        },
        weight=3.0,  # 기존 5.0 → 3.0 (Phase 밸런싱)
    )

    # ==========================================================================
    # Phase 로깅용 (weight=0.0이므로 학습에 영향 없음, 모니터링용)
    # ==========================================================================
    left_Grasp_phase = RewTerm(
        func=mdp.Grasp_phase_value,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "phase_params": {
                "eef_link_name": "openarm_left_hand",
                "lift_height": 0.03,
                "reach_distance": 0.07,
                "align_threshold": 0.0,
                "grasp_distance": 0.02,
                "close_threshold": 0.6,
                "hold_duration": 2.0,
            },
        },
        weight=0.0,  # 로깅 전용
    )
    right_Grasp_phase = RewTerm(
        func=mdp.Grasp_phase_value,
        params={
            "object_cfg": SceneEntityCfg("object2"),
            "phase_params": {
                "eef_link_name": "openarm_right_hand",
                "lift_height": 0.03,
                "reach_distance": 0.07,
                "align_threshold": 0.0,
                "grasp_distance": 0.02,
                "close_threshold": 0.6,
                "hold_duration": 2.0,
            },
        },
        weight=0.0,  # 로깅 전용
    )

    # [방법6] 잘못된 목표 페널티 (Wrong Target Penalty)
    # 좌측 손이 우측 물체로 향하거나, 우측 손이 좌측 물체로 향할 때 페널티 부여
    # 이를 통해 "전략적 모방" (한 손이 다른 손의 목표를 따라가는 행동)을 방지
    # enable_wrong_target_penalty=False로 설정하면 비활성화됨
    wrong_target_penalty = RewTerm(
        func=mdp.wrong_target_penalty_soft,
        weight=0.5,
        params={
            "left_eef_link_name": "openarm_left_hand",
            "right_eef_link_name": "openarm_right_hand",
            "left_object_cfg": SceneEntityCfg("object"),
            "right_object_cfg": SceneEntityCfg("object2"),
            "std": 0.1,
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

@configclass
class GraspIKTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_tipped = DoneTerm(
        func=mdp.cup_tipped,
        params={"object_name": "object", "min_upright_dot": 0.5},
    )
    object2_tipped = DoneTerm(
        func=mdp.cup_tipped,
        params={"object_name": "object2", "min_upright_dot": 0.5},
    )
    joint_limit_near = DoneTerm(
        func=mdp.joint_limit_near_or_min_margin,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "margin_threshold": 0.001,
            "near_rate_threshold": 0.98,
            "warmup_steps": 1000,
        },
    )

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )
    object2_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object2")},
    )


@configclass
class GraspIKCurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -0.0001, "num_steps": 10000},
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -0.0001, "num_steps": 10000},
    )


@configclass
class GraspIKEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the bimanual grasping environment."""

    scene: GraspIKSceneCfg = GraspIKSceneCfg(num_envs=2048*1, env_spacing=2.5)
    observations: GraspIKObservationsCfg = GraspIKObservationsCfg()
    actions: GraspIKActionsCfg = GraspIKActionsCfg()
    rewards: GraspIKRewardsCfg = GraspIKRewardsCfg()
    terminations: GraspIKTerminationsCfg = GraspIKTerminationsCfg()
    events: GraspIKEventCfg = GraspIKEventCfg()

    commands: GraspIKCommandsCfg = GraspIKCommandsCfg()
    curriculum: GraspIKCurriculumCfg = GraspIKCurriculumCfg()

    # [방법4] 팔 구분자 토글 플래그
    # True: 좌/우 팔 구분자 관측 활성화 (비대칭 학습 문제 해결에 도움)
    # False: 팔 구분자 비활성화 (기존 동작)
    enable_arm_identifier: bool = True

    # [방법6] 잘못된 목표 페널티 토글 플래그
    # True: 손이 잘못된 물체로 향할 때 페널티 활성화 (전략적 모방 방지)
    # False: 잘못된 목표 페널티 비활성화 (기존 동작)
    enable_wrong_target_penalty: bool = True

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 1.0 / 100.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (3.5, 3.5, 3.5)
        # Disable IK-based TCP reset; keep init_state pose on reset.
        self.events.reset_robot_tcp_to_cups = None
        # Reach-only bootstrap: disable grasp/lift/hold rewards.
        #self.rewards.left_gripper_open = None
        #self.rewards.right_gripper_open = None
        # self.rewards.left_gripper_close = None
        # self.rewards.right_gripper_close = None
        # self.rewards.left_lifting_object = None
        # self.rewards.right_lifting_object = None
        self.rewards.left_object_goal_tracking = None
        self.rewards.right_object_goal_tracking = None
        self.rewards.left_object_goal_tracking_fine_grained = None
        self.rewards.right_object_goal_tracking_fine_grained = None
        # self.rewards.left_Grasp_phase = None
        # self.rewards.right_Grasp_phase = None

        # Command-only high-level observations (drop real object positions).
        self.observations.policy.object_position = None
        self.observations.policy.object2_position = None

        # [방법4] 팔 구분자 토글 처리
        # enable_arm_identifier=False이면 팔 구분자 관측 비활성화
        if not self.enable_arm_identifier:
            self.observations.policy.left_arm_id = None
            self.observations.policy.right_arm_id = None

        # [방법6] 잘못된 목표 페널티 토글 처리
        # enable_wrong_target_penalty=False이면 잘못된 목표 페널티 비활성화
        if not self.enable_wrong_target_penalty:
            self.rewards.wrong_target_penalty = None

        self.sim.physx = PhysxCfg(
            solver_type=1,  # TGS
            max_position_iteration_count=192,
            max_velocity_iteration_count=1,
            enable_ccd=True,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            # increase buffers to prevent overflow errors
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_max_num_partitions=8,
            gpu_collision_stack_size=2**24,
        )
