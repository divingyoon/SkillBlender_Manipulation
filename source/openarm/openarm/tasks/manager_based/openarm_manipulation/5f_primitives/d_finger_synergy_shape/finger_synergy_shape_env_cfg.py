# Copyright 2025 Enactic, Inc.
# Licensed under the Apache License, Version 2.0

"""Primitive D: Finger Joint Target Tracking - Environment Configuration."""

from dataclasses import MISSING
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
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
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

from ..common.robot_cfg import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, LEFT_HAND_JOINTS, RIGHT_HAND_JOINTS


_HAND_JOINT_LIMITS_DEG = {
    "dg_1_1": (-51.0, 22.0),
    "dg_1_2": (0.0, 180.0),
    "dg_1_3": (-90.0, 90.0),
    "dg_1_4": (-90.0, 90.0),
    "dg_2_1": (-35.0, 24.0),
    "dg_2_2": (0.0, 115.0),
    "dg_2_3": (-90.0, 90.0),
    "dg_2_4": (-90.0, 90.0),
    "dg_3_1": (-35.0, 35.0),
    "dg_3_2": (0.0, 112.0),
    "dg_3_3": (-90.0, 90.0),
    "dg_3_4": (-90.0, 90.0),
    "dg_4_1": (-24.0, 35.0),
    "dg_4_2": (0.0, 109.0),
    "dg_4_3": (-90.0, 90.0),
    "dg_4_4": (-90.0, 90.0),
    "dg_5_1": (-60.0, 1.0),
    "dg_5_2": (-35.0, 24.0),
    "dg_5_3": (-90.0, 90.0),
    "dg_5_4": (-90.0, 90.0),
}


def _hand_joint_ranges(prefix: str) -> dict[str, tuple[float, float]]:
    ranges = {}
    for suffix, (low_deg, high_deg) in _HAND_JOINT_LIMITS_DEG.items():
        joint_name = f"{prefix}_{suffix}"
        ranges[joint_name] = (math.radians(low_deg), math.radians(high_deg))
    return ranges


@configclass
class FingerSynergyShapeSceneCfg(InteractiveSceneCfg):
    """Scene configuration for finger synergy shape primitive."""

    robot: ArticulationCfg = MISSING

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


@configclass
class CommandsCfg:
    """Command terms: target joint positions."""

    left_hand_joint_target = mdp.JointPositionCommandCfg(
        resampling_time_range=(4.0, 6.0),
        debug_vis=False,
        joint_names=LEFT_HAND_JOINTS,
        ranges=_hand_joint_ranges("lj"),
        base_targets={
            "lj_dg_1_2": 1.22173,
            "lj_dg_2_2": 0.10,
            "lj_dg_3_2": 0.10,
            "lj_dg_4_2": 0.10,
            "lj_dg_5_2": 0.10,
            "lj_dg_1_3": 0.25,
            "lj_dg_2_3": 0.7164,
            "lj_dg_3_3": 0.7164,
            "lj_dg_4_3": 0.7164,
            "lj_dg_5_3": 0.5419,
            "lj_dg_1_4": 0.15,
            "lj_dg_2_4": 0.3245,
            "lj_dg_3_4": 0.3245,
            "lj_dg_4_4": 0.3245,
            "lj_dg_5_4": 0.1500,
        },
        fixed_zero_or_target=[
            "lj_dg_1_1",
            "lj_dg_2_1",
            "lj_dg_3_1",
            "lj_dg_4_1",
            "lj_dg_5_1",
            "lj_dg_5_2",
        ],
        noise=0.1,
        clamp_zero_crossing=True,
    )

    right_hand_joint_target = mdp.JointPositionCommandCfg(
        resampling_time_range=(4.0, 6.0),
        debug_vis=False,
        joint_names=RIGHT_HAND_JOINTS,
        ranges={name: (0.0, 0.0) for name in RIGHT_HAND_JOINTS},
    )


@configclass
class ActionsCfg:
    """Action specifications."""

    left_arm_action: ActionTerm = MISSING
    right_arm_action: ActionTerm = MISSING
    left_hand_action: ActionTerm = MISSING
    right_hand_action: ActionTerm = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations."""

        left_hand_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_HAND_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        left_hand_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_HAND_JOINTS)},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        right_hand_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_HAND_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        right_hand_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_HAND_JOINTS)},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        # Joint limit margins
        left_limit_margins = ObsTerm(
            func=mdp.joint_limit_margins,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_HAND_JOINTS)},
        )

        right_limit_margins = ObsTerm(
            func=mdp.joint_limit_margins,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_HAND_JOINTS)},
        )

        # Target joint positions
        left_joint_target = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "left_hand_joint_target"},
        )

        right_joint_target = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "right_hand_joint_target"},
        )

        left_actions = ObsTerm(func=mdp.last_action, params={"action_name": "left_hand_action"})
        right_actions = ObsTerm(func=mdp.last_action, params={"action_name": "right_hand_action"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Event configuration."""

    reset_hand_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )

    # Keep arms fixed at their default pose every step (B-style lock).
    lock_arm_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms.

    Weights:
    - Joint tracking: -0.3 (L2 error in joint space)
    - Joint tracking fine: 0.2 (tanh reward for being close)
    - Self-collision penalty: -0.2
    - Joint limit penalty: -0.15
    - Action smoothness: -0.001
    """

    # Joint tracking (error in joint space)
    # left_joint_track = RewTerm(
    #     func=mdp.joint_tracking_error_l2,
    #     weight=-0.3,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_HAND_JOINTS),
    #         "command_name": "left_hand_joint_target",
    #     },
    # )

    # right_joint_track = RewTerm(
    #     func=mdp.joint_tracking_error_l2,
    #     weight=-0.3,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_HAND_JOINTS),
    #         "command_name": "right_hand_joint_target",
    #     },
    # )

    # Fine-grained tracking with tanh
    left_joint_track_fine = RewTerm(
        func=mdp.joint_tracking_tanh,
        weight=10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_HAND_JOINTS),
            "command_name": "left_hand_joint_target",
            "std": 0.5,
        },
    )

    right_joint_track_fine = RewTerm(
        func=mdp.joint_tracking_tanh,
        weight=10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_HAND_JOINTS),
            "command_name": "right_hand_joint_target",
            "std": 0.5,
        },
    )

    # Joint limit penalty
    # left_joint_limits = RewTerm(
    #     func=mdp.joint_pos_limits,
    #     weight=-0.05,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_HAND_JOINTS)},
    # )

    # right_joint_limits = RewTerm(
    #     func=mdp.joint_pos_limits,
    #     weight=-0.05,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_HAND_JOINTS)},
    # )

    # Action smoothness
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.0001)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    left_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-2.5e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_HAND_JOINTS)},
    )

    right_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-2.5e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_HAND_JOINTS)},
    )


@configclass
class TerminationsCfg:
    """Termination terms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Optional: success when joint target reached
    # joint_target_reached = DoneTerm(
    #     func=mdp.joint_target_reached,
    #     params={"threshold": 0.1, "num_steps": 20},
    # )


@configclass
class CurriculumCfg:
    """Curriculum configuration."""

    left_joint_track_fine_std = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.left_joint_track_fine.params.std",
            "modify_fn": mdp.stepwise_decay,
            "modify_params": {
                "initial_value": 0.5,
                "interval_iters": 300,
                "decrement": 0.1,
                "min_value": 0.3,
            },
        },
    )


@configclass
class FingerSynergyShapeEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for finger synergy shape environment."""

    scene: FingerSynergyShapeSceneCfg = FingerSynergyShapeSceneCfg(num_envs=4096, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 6.0
        self.sim.dt = 1.0 / 100.0
        self.sim.render_interval = self.decimation
        # Physics settings for large contact counts.
        self.sim.physx.gpu_max_rigid_patch_count = 4_000_000
        self.sim.physx.gpu_max_rigid_contact_count = 16_000_000
        self.sim.physx.gpu_found_lost_pairs_capacity = 4_000_000
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 8_000_000
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 256 * 1024
        self.sim.physx.gpu_collision_stack_size = 128 * 1024 * 1024
        self.sim.physx.gpu_heap_capacity = 128 * 1024 * 1024
        self.sim.physx.gpu_temp_buffer_capacity = 32 * 1024 * 1024
        self.viewer.eye = (1.0, 1.0, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.3)
