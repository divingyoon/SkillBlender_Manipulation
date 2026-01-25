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

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Reset cups to fixed positions (same as reach task)
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
    """Reward terms for the MDP.

    Grasp reward structure:
    1. Fine positioning: Get TCP precisely to grasp point
    2. Finger closing: Close gripper when in position
    3. Lift confirmation: Small lift to verify stable grasp
    """

    # Fine positioning reward (sharp exponential for precise positioning)
    left_tcp_to_cup = RewTerm(
        func=mdp.tcp_distance_to_cup,
        params={
            "tcp_body_name": "openarm_left_ee_tcp",
            "target_cfg": SceneEntityCfg("cup"),
            "offset": [0.0, 0.0, 0.0],  # Target cup center
        },
        weight=1.0,
    )
    right_tcp_to_cup2 = RewTerm(
        func=mdp.tcp_distance_to_cup,
        params={
            "tcp_body_name": "openarm_right_ee_tcp",
            "target_cfg": SceneEntityCfg("cup2"),
            "offset": [0.0, 0.0, 0.0],
        },
        weight=1.0,
    )

    # Grasp reward (close gripper when near object)
    left_grasp = RewTerm(
        func=mdp.grasp_reward,
        params={
            "eef_link_name": "openarm_left_ee_tcp",
            "object_cfg": SceneEntityCfg("cup"),
            "reach_radius": 0.03,
            "close_threshold": 0.5,
        },
        weight=2.0,
    )
    right_grasp = RewTerm(
        func=mdp.grasp_reward,
        params={
            "eef_link_name": "openarm_right_ee_tcp",
            "object_cfg": SceneEntityCfg("cup2"),
            "reach_radius": 0.03,
            "close_threshold": 0.5,
        },
        weight=2.0,
    )

    # Stable grasp reward (lifted while grasping)
    left_stable_grasp = RewTerm(
        func=mdp.stable_grasp_reward,
        params={
            "eef_link_name": "openarm_left_ee_tcp",
            "object_cfg": SceneEntityCfg("cup"),
            "lift_height": 0.03,
            "grasp_distance": 0.03,
            "close_threshold": 0.5,
        },
        weight=5.0,
    )
    right_stable_grasp = RewTerm(
        func=mdp.stable_grasp_reward,
        params={
            "eef_link_name": "openarm_right_ee_tcp",
            "object_cfg": SceneEntityCfg("cup2"),
            "lift_height": 0.03,
            "grasp_distance": 0.03,
            "close_threshold": 0.5,
        },
        weight=5.0,
    )

    # Lift progress reward
    left_lift = RewTerm(
        func=mdp.lift_reward,
        params={
            "object_cfg": SceneEntityCfg("cup"),
            "lift_height": 0.05,
            "table_height": 0.0,
        },
        weight=1.0,
    )
    right_lift = RewTerm(
        func=mdp.lift_reward,
        params={
            "object_cfg": SceneEntityCfg("cup2"),
            "lift_height": 0.05,
            "table_height": 0.0,
        },
        weight=1.0,
    )

    # Regularization
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


@configclass
class GraspEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the grasp task."""

    _task_id = "Grasp-v1"

    scene: GraspSceneCfg = GraspSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None

    def __post_init__(self):
        self.decimation = 2
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
            gpu_max_num_partitions=8,
            gpu_collision_stack_size=1000000000,
        )
