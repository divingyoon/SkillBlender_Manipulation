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

##
# Register Gym environments.
##

import importlib

from isaaclab_tasks.utils import import_packages

# Register SkillBlender custom policies with RSL-RL if available.
try:
    from sbm.rl import register_rsl_rl

    register_rsl_rl()
except ImportError:
    pass

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils", ".mdp"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)

# Explicitly import the new 'approach' task config to ensure registration
import openarm.tasks.manager_based.openarm_manipulation.pipeline.hand.both.approach.config

# bimanual/reach
import openarm.tasks.manager_based.openarm_manipulation.pipeline.gripper.both.reach.config
# bimanual/grasp,grasp2g
import openarm.tasks.manager_based.openarm_manipulation.pipeline.hand.both.grasp.config
import openarm.tasks.manager_based.openarm_manipulation.pipeline.gripper.both.grasp_2g.config
import openarm.tasks.manager_based.openarm_manipulation.pipeline.gripper.both.grasp_2g.grasp_2g_env_cfg

#primitive skills/grasp_2g_v1
import openarm.tasks.manager_based.openarm_manipulation.primitive_skills.grasp_2g_v1.config
import openarm.tasks.manager_based.openarm_manipulation.primitive_skills.grasp_2g_v1.grasp2g_v1_env_cfg

#primitive skills/reach_ik,grasp_ik,transfer_ik,pour_ik
import openarm.tasks.manager_based.openarm_manipulation.primitive_skills.ReachIK.config
import openarm.tasks.manager_based.openarm_manipulation.primitive_skills.GraspIK.config
import openarm.tasks.manager_based.openarm_manipulation.primitive_skills.TransferIK.config
import openarm.tasks.manager_based.openarm_manipulation.primitive_skills.PourIK.config

# pipeline/gripper/left/2g_grasp_left_v1
# NOTE: module segment starts with a digit, so standard `import ...` syntax is invalid.
importlib.import_module(
    "openarm.tasks.manager_based.openarm_manipulation.pipeline.gripper.left.2g_grasp_left_v1.config"
)
importlib.import_module(
    "openarm.tasks.manager_based.openarm_manipulation.pipeline.gripper.right.2g_grasp_right_v1.config"
)

# blending/pouring,pouring1,pouring2,pouring3,pouring4
import openarm.tasks.manager_based.openarm_manipulation.blending.pouring.config
import openarm.tasks.manager_based.openarm_manipulation.blending.pouring1.config
import openarm.tasks.manager_based.openarm_manipulation.blending.pouring2.config
import openarm.tasks.manager_based.openarm_manipulation.blending.pouring3.config
import openarm.tasks.manager_based.openarm_manipulation.blending.pouring4.config
