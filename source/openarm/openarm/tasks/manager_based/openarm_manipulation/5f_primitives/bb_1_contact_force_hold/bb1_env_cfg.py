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

"""BB1 wrapper env cfg: lift-start + time-varying payload, reusing Primitive BB specs."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from openarm.tasks.manager_based.openarm_manipulation.primitives.bb_contact_force_hold.contact_force_hold_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    CurriculumCfg,
    EventCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)
from openarm.tasks.manager_based.openarm_manipulation.primitives.bb_contact_force_hold.config.joint_pos_env_cfg import (
    OpenArmContactForceHoldEnvCfg,
)

from .mdp import events as bb1_mdp


@configclass
class BB1EventCfg(EventCfg):
    """Event extensions for BB1: lift-start + payload disturbance."""

    payload_disturbance = EventTerm(
        func=bb1_mdp.apply_payload_disturbance,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "object_cfg": SceneEntityCfg("object"),
            "delta_mass_range": (0.0, 0.4),
            "ramp_delay_steps": 0,
            "ramp_steps": 2000,
            "period_steps": 200,
            "phase_offset": 0,
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.reset_contact_state = self.reset_contact_state.replace(
            func=bb1_mdp.reset_contact_force_hold_state_bb1,
            params={
                **self.reset_contact_state.params,
                "lift_prob_init": 0.0,
                "lift_prob_final": 0.0,
                "lift_prob_steps": 0,
                "lift_offset_b1": {"x": (0.09, 0.09), "y": (0.005, 0.005), "z": (0.12, 0.12)},
                "lift_offset_b2": {"x": (0.10, 0.10), "y": (0.005, 0.005), "z": (0.10, 0.10)},
                "base_lift_z": 0.0,
            },
        )


@configclass
class OpenArmContactForceHoldBB1EnvCfg(OpenArmContactForceHoldEnvCfg):
    """BB1 environment: same obs/act/reward/termination as BB, only events differ."""

    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    events: BB1EventCfg = BB1EventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.table = None
        self.scene.num_envs = 4096
        self.events.object_mass.params["mass_distribution_params"] = (0.2, 0.2)
        self.curriculum.object_mass_range = None


@configclass
class OpenArmContactForceHoldBB1EnvCfg_PLAY(OpenArmContactForceHoldBB1EnvCfg):
    """Play/evaluation configuration for BB1."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
