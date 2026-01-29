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
class ReachSceneCfg(InteractiveSceneCfg):
    """Scene with a robot, table, and a cup."""

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
class CommandsCfg:
    """Command terms for the MDP."""

    tcp_pose = mdp.ReachPoseCommandCfg(
        asset_name="robot",
        body_name="openarm_left_ee_tcp",
        target_asset_cfg=SceneEntityCfg("cup"),
        resampling_time_range=(4.0, 4.0),
        offset=(0.0, 0.0, 0.02),
        ranges=mdp.ReachPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.5),
            pos_y=(0.1, 0.3),
            pos_z=(0.25, 0.55),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        debug_vis=False,
    )
    tcp_pose2 = mdp.ReachPoseCommandCfg(
        asset_name="robot",
        body_name="openarm_left_ee_tcp",
        target_asset_cfg=SceneEntityCfg("cup2"),
        resampling_time_range=(4.0, 4.0),
        offset=(0.0, 0.0, 0.02),
        ranges=mdp.ReachPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.5),
            pos_y=(-0.3, -0.1),
            pos_z=(0.25, 0.55),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        debug_vis=False,
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
        tcp_to_cup_pos = ObsTerm(
            func=mdp.target_pos_in_tcp_frame,
            params={
                "tcp_body_name": "openarm_left_ee_tcp",
                "target_cfg": SceneEntityCfg("cup"),
                "offset": [0.0, 0.0, 0.02],
            },
        )
        tcp_to_cup2_pos = ObsTerm(
            func=mdp.target_pos_in_tcp_frame,
            params={
                "tcp_body_name": "openarm_right_ee_tcp",
                "target_cfg": SceneEntityCfg("cup2"),
                "offset": [0.0, 0.0, 0.02],
            },
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
    """Reward terms for the MDP."""

    reaching_left_cup = RewTerm(
        func=mdp.tcp_distance_to_target,
        params={"tcp_body_name": "openarm_left_hand", "target_cfg": SceneEntityCfg("cup"), "offset": [0.0, 0.0, 0.02]},
        weight=1.0,
    )
    
    reaching_right_cup2 = RewTerm(
        func=mdp.tcp_distance_to_target,
        params={"tcp_body_name": "openarm_right_hand", "target_cfg": SceneEntityCfg("cup2"), "offset": [0.0, 0.0, 0.02]},
        weight=1.0,
    )

    tcp_forward_alignment_left = RewTerm(
        func=mdp.tcp_z_axis_to_target_alignment,
        params={"tcp_body_name": "openarm_left_hand", "target_cfg": SceneEntityCfg("cup"), "offset": [0.0, 0.0, 0.02]},
        weight=0.5,
    )
    tcp_forward_alignment_right = RewTerm(
        func=mdp.tcp_z_axis_to_target_alignment,
        params={"tcp_body_name": "openarm_right_hand", "target_cfg": SceneEntityCfg("cup2"), "offset": [0.0, 0.0, 0.02]},
        weight=0.5,
    )

    open_hand_reward = RewTerm(
        func=mdp.hand_joint_position,
        params={
            "joint_name": ["openarm_left_finger_joint1", "openarm_left_finger_joint2"],
            "target_pos": 0.048,  # Open position (average of 0.044 and 0.052)
        },
        weight=0.6,
    )
    open_hand_reward_right = RewTerm(
        func=mdp.hand_joint_position,
        params={
            "joint_name": ["openarm_right_finger_joint1", "openarm_right_finger_joint2"],
            "target_pos": 0.048,  # Open position (average of 0.044 and 0.052)
        },
        weight=0.6,
    )

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
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach task."""
    _task_id = "Reach-v1"

    scene: ReachSceneCfg = ReachSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
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
