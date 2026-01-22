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

import tempfile
import torch
import carb

from pink.tasks import DampingTask, FrameTask

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.pink_ik import NullSpacePostureTask, PinkIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from openarm.tasks.manager_based.openarm_manipulation import OPENARM_ROOT_DIR
from . import mdp

##
# Scene definition
##
@configclass
class PickAndPlaceIKSceneCfg(InteractiveSceneCfg):
    """Scene with a bimanual robot, table, and pick-and-place object."""

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
    )

    # Object
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.8]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    # Robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{OPENARM_ROOT_DIR}/usds/openarm_bimanual/openarm_tesollo_t1.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],
            rot=[0.707, 0, 0, 0.707],
            joint_pos={
                "openarm_.*": 0.0,
                "lj_dg_.*": 0.0,
                "rj_dg_.*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    bimanual_ik = PinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=[
            "openarm_left_joint1", "openarm_left_joint2", "openarm_left_joint3", "openarm_left_joint4", "openarm_left_joint5", "openarm_left_joint6", "openarm_left_joint7",
            "openarm_right_joint1", "openarm_right_joint2", "openarm_right_joint3", "openarm_right_joint4", "openarm_right_joint5", "openarm_right_joint6", "openarm_right_joint7",
        ],
        hand_joint_names=[
            "lj_dg_.*", "rj_dg_.*"
        ],
        target_eef_link_names={
            "left_hand": "ll_dg_ee",
            "right_hand": "rl_dg_ee",
        },
        asset_name="robot",
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="world", # Assuming the robot is fixed to the world
            num_hand_joints=40, # 20 joints per hand approx.
            show_ik_warnings=False,
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                FrameTask(
                    "ll_dg_ee",
                    position_cost=1.0, orientation_cost=1.0,
                    lm_damping=0.01, gain=0.5,
                ),
                FrameTask(
                    "rl_dg_ee",
                    position_cost=1.0, orientation_cost=1.0,
                    lm_damping=0.01, gain=0.5,
                ),
                DampingTask(cost=0.01),
            ],
            fixed_input_tasks=[],
        ),
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        object_pose = ObsTerm(func=mdp.root_state, params={"asset_cfg": SceneEntityCfg("object")})
        left_eef_pose = ObsTerm(func=mdp.body_state, params={"asset_cfg": SceneEntityCfg("robot", body_names="ll_dg_ee")})
        right_eef_pose = ObsTerm(func=mdp.body_state, params={"asset_cfg": SceneEntityCfg("robot", body_names="rl_dg_ee")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": -0.5, "asset_cfg": SceneEntityCfg("object")})

@configclass
class EventCfg:
    """Configuration for events."""
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.1, 0.1], "y": [-0.1, 0.1], "z": [0.0, 0.0]},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

@configclass
class PapIkEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Pick and Place IK environment."""

    scene: PickAndPlaceIKSceneCfg = PickAndPlaceIKSceneCfg(num_envs=1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    rewards = None # No rewards for this IK-based task
    commands = None
    curriculum = None

    temp_urdf_dir = tempfile.gettempdir()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 60.0

        # Convert USD to URDF for PinkIK
        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            self.scene.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=True
        )
        # Set the URDF and mesh paths for the IK controller
        self.actions.bimanual_ik.controller.urdf_path = temp_urdf_output_path
        self.actions.bimanual_ik.controller.mesh_path = temp_urdf_meshes_output_path
