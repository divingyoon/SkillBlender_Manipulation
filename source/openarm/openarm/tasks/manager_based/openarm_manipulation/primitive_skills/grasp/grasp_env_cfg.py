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
Grasp-v1 Environment Configuration.

This task focuses on the grasp primitive skill:
- Initial state: Robot in pre-grasp pose (TCP 3-6cm from cup), gripper open
- Goal: Close gripper to grasp cup, lift slightly to confirm stable grasp
- Cup position has small randomization to require fine adjustment
"""

from dataclasses import MISSING
import math

import isaaclab.sim as sim_utils
from isaaclab.sim import PhysxCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
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
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp


@configclass
class GraspSceneCfg(InteractiveSceneCfg):
    """Scene with a robot, table, and cups to be grasped."""

    robot: ArticulationCfg = MISSING
    cup: RigidObjectCfg = MISSING
    cup2: RigidObjectCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.25, 0.0, 0.0], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )

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
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        tcp_pose = ObsTerm(func=mdp.body_pose, params={"body_name": "openarm_left_ee_tcp"})
        tcp_pose_right = ObsTerm(func=mdp.body_pose, params={"body_name": "openarm_right_ee_tcp"})
        cup_pose = ObsTerm(func=mdp.root_pose, params={"asset_cfg": SceneEntityCfg("cup")})
        cup2_pose = ObsTerm(func=mdp.root_pose, params={"asset_cfg": SceneEntityCfg("cup2")})
        # Relative position from TCP to cup (grasp point)
        tcp_to_cup_pos = ObsTerm(
            func=mdp.target_pos_in_tcp_frame,
            params={
                "tcp_body_name": "openarm_left_ee_tcp",
                "target_cfg": SceneEntityCfg("cup"),
                "offset": [0.0, 0.0, 0.0],  # No offset - target is cup center
            },
        )
        tcp_to_cup2_pos = ObsTerm(
            func=mdp.target_pos_in_tcp_frame,
            params={
                "tcp_body_name": "openarm_right_ee_tcp",
                "target_cfg": SceneEntityCfg("cup2"),
                "offset": [0.0, 0.0, 0.0],
            },
        )
        # Gripper state
        left_gripper_state = ObsTerm(
            func=mdp.gripper_state,
            params={"joint_names": ["openarm_left_finger_joint1", "openarm_left_finger_joint2"]},
        )
        right_gripper_state = ObsTerm(
            func=mdp.gripper_state,
            params={"joint_names": ["openarm_right_finger_joint1", "openarm_right_finger_joint2"]},
        )
        # Distance to cup (scalar, for easier learning)
        left_tcp_cup_distance = ObsTerm(
            func=mdp.tcp_to_cup_distance,
            params={"tcp_body_name": "openarm_left_ee_tcp", "target_cfg": SceneEntityCfg("cup")},
        )
        right_tcp_cup_distance = ObsTerm(
            func=mdp.tcp_to_cup_distance,
            params={"tcp_body_name": "openarm_right_ee_tcp", "target_cfg": SceneEntityCfg("cup2")},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # 1. Reset scene to default state first
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # 2. Roll-out reset: Set robot joints AND cup positions from reach terminal states
    # This now handles BOTH robot and cup reset for consistency
    # reset_from_reach = EventTerm(
    #     func=mdp.reset_from_reach_terminal_states,
    #     mode="reset",
    #     params={
    #         "terminal_states_path": "/home/user/rl_ws/SkillBlender_Manipulation/data/reach_terminal_states.pt",
    #         "left_gripper_joint_names": ["openarm_left_finger_joint1", "openarm_left_finger_joint2"],
    #         "right_gripper_joint_names": ["openarm_right_finger_joint1", "openarm_right_finger_joint2"],
    #         "gripper_open_position": 0.04,
    #         "reset_cups": True,
    #         "cup_cfg_name": "cup",
    #         "cup2_cfg_name": "cup2",
    #     },
    # )
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_cup_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.15, 0.15), "y": (0.1, 0.1), "z": (0.0, 0.0),
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
                "x": (0.15, 0.15), "y": (-0.1, -0.1), "z": (0.0, 0.0),
                "yaw": (-math.pi / 2, -math.pi / 2),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cup2"),
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP (grasp_2g style phase-based rewards).

    Phase-based reward structure:
    - Phase 0: Reaching (EEF approaching object)
    - Phase 1: Grasping (closing gripper when near)
    - Phase 2: Lifting (object being lifted)
    - Phase 3: Holding (stable grasp maintained)
    """

    # === REACHING REWARDS (Phase 0, 1 active) ===
    left_reaching_object = RewTerm(
        func=mdp.object_ee_distance_tanh,
        params={
            "std": 0.1,
            "eef_link_name": "openarm_left_hand",
            "object_cfg": SceneEntityCfg("cup"),
        },
        weight=5.0,
    )
    right_reaching_object = RewTerm(
        func=mdp.object_ee_distance_tanh,
        params={
            "std": 0.1,
            "eef_link_name": "openarm_right_hand",
            "object_cfg": SceneEntityCfg("cup2"),
        },
        weight=5.0,
    )

    # === LIFTING REWARDS (Phase 2, 3 active) ===
    left_lifting_object = RewTerm(
        func=mdp.phase_lift_reward,
        params={
            "lift_height": 0.03,
            "object_cfg": SceneEntityCfg("cup"),
            "phase_weights": [0.0, 0.0, 1.0, 1.0],
            "phase_params": {
                "eef_link_name": "openarm_left_ee_tcp",
                "lift_height": 0.03,
                "reach_distance": 0.05,
                "grasp_distance": 0.02,
                "close_threshold": 0.6,
            },
        },
        weight=20.0,
    )
    right_lifting_object = RewTerm(
        func=mdp.phase_lift_reward,
        params={
            "lift_height": 0.03,
            "object_cfg": SceneEntityCfg("cup2"),
            "phase_weights": [0.0, 0.0, 1.0, 1.0],
            "phase_params": {
                "eef_link_name": "openarm_right_ee_tcp",
                "lift_height": 0.03,
                "reach_distance": 0.05,
                "grasp_distance": 0.02,
                "close_threshold": 0.6,
            },
        },
        weight=20.0,
    )

    # === GRASP REWARDS (Phase 1, 2 active) ===
    left_grasp = RewTerm(
        func=mdp.phase_grasp_reward,
        params={
            "eef_link_name": "openarm_left_hand",
            "object_cfg": SceneEntityCfg("cup"),
            "reach_radius": 0.05,
            "close_threshold": 0.4,
            "closure_max": 0.44,
            "phase_weights": [0.0, 1.0, 1.0, 0.5],
            "phase_params": {
                "eef_link_name": "openarm_left_ee_tcp",
                "lift_height": 0.1,
                "reach_distance": 0.05,
                "grasp_distance": 0.02,
                "close_threshold": 0.6,
            },
        },
        weight=5.0,
    )
    right_grasp = RewTerm(
        func=mdp.phase_grasp_reward,
        params={
            "eef_link_name": "openarm_right_hand",
            "object_cfg": SceneEntityCfg("cup2"),
            "reach_radius": 0.05,
            "close_threshold": 0.4,
            "closure_max": 0.44,
            "phase_weights": [0.0, 1.0, 1.0, 0.5],
            "phase_params": {
                "eef_link_name": "openarm_right_ee_tcp",
                "lift_height": 0.1,
                "reach_distance": 0.05,
                "grasp_distance": 0.02,
                "close_threshold": 0.6,
            },
        },
        weight=5.0,
    )

    # === GRIPPER CLOSURE PENALTY ===
    left_closure_penalty = RewTerm(
        func=mdp.closure_amount_penalty,
        weight=3.0,
        params={
            "eef_link_name": "openarm_left_ee_tcp",
            "threshold": 0.44,
            "penalty_scale": -1.0,
        },
    )
    right_closure_penalty = RewTerm(
        func=mdp.closure_amount_penalty,
        weight=3.0,
        params={
            "eef_link_name": "openarm_right_ee_tcp",
            "threshold": 0.44,
            "penalty_scale": -1.0,
        },
    )

    # === PHASE LOGGING (weight=0) ===
    left_grasp_phase = RewTerm(
        func=mdp.grasp_phase_value,
        params={
            "object_cfg": SceneEntityCfg("cup"),
            "phase_params": {
                "eef_link_name": "openarm_left_ee_tcp",
                "lift_height": 0.1,
                "reach_distance": 0.05,
                "grasp_distance": 0.02,
                "close_threshold": 0.6,
            },
        },
        weight=0.0,
    )
    right_grasp_phase = RewTerm(
        func=mdp.grasp_phase_value,
        params={
            "object_cfg": SceneEntityCfg("cup2"),
            "phase_params": {
                "eef_link_name": "openarm_right_ee_tcp",
                "lift_height": 0.1,
                "reach_distance": 0.05,
                "grasp_distance": 0.02,
                "close_threshold": 0.6,
            },
        },
        weight=0.0,
    )

    # === FINAL SUCCESS REWARDS ===
    lift_success = RewTerm(
        func=mdp.grasp_success_with_hold,
        weight=10.0,
        params={
            "lift_threshold": 0.1,
            "eef_link_name_left": "openarm_left_ee_tcp",
            "eef_link_name_right": "openarm_right_ee_tcp",
            "object_cfg_left": SceneEntityCfg("cup"),
            "object_cfg_right": SceneEntityCfg("cup2"),
            "contact_distance": 0.05,
            "min_closure": 0.5,
        }
    )

    hold_reward = RewTerm(
        func=mdp.continuous_hold_reward,
        weight=1.0,
        params={
            "lift_threshold": 0.1,
            "eef_link_name_left": "openarm_left_ee_tcp",
            "eef_link_name_right": "openarm_right_ee_tcp",
            "object_cfg_left": SceneEntityCfg("cup"),
            "object_cfg_right": SceneEntityCfg("cup2"),
            "contact_distance": 0.05,
            "min_closure": 0.5,
        }
    )

    # === REGULARIZATION ===
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Penalty when a cup tips over.
    cup_tip_penalty = RewTerm(
        func=mdp.cup_tipped,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("cup"), "max_tilt_deg": 45.0},
    )
    cup2_tip_penalty = RewTerm(
        func=mdp.cup_tipped,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("cup2"), "max_tilt_deg": 45.0},
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
        params={"asset_cfg": SceneEntityCfg("cup"), "max_tilt_deg": 45.0},
    )
    cup2_tipping = DoneTerm(
        func=mdp.cup_tipped,
        params={"asset_cfg": SceneEntityCfg("cup2"), "max_tilt_deg": 45.0},
    )


@configclass
class GraspEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the grasp task."""

    _task_id = "Grasp-v1"
    # Debug settings (used by reward functions)
    debug_grasp_left: bool = True
    debug_grasp_left_interval: int = 200
    debug_grasp_right: bool = True
    debug_grasp_right_interval: int = 200

    scene: GraspSceneCfg = GraspSceneCfg(num_envs=2048*1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None

    # Roll-out setting from markdown - using the simplified approach
    # use_rollout_reset: bool = True
    # reach_checkpoint_path: str = "/home/user/rl_ws/IsaacLab/logs/rsl_rl/openarm_bi_reach/test7_5080/model_9999.pt"

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 8.0
        self.sim.dt = 1.0 / 100.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.physx = PhysxCfg(
            solver_type=1,  # TGS
            max_position_iteration_count=192,
            max_velocity_iteration_count=1,
            enable_ccd=True,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_collision_stack_size=1000000000,
        )
