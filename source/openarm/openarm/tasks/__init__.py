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
import openarm.tasks.manager_based.openarm_manipulation.bimanual.reach.config
import openarm.tasks.manager_based.openarm_manipulation.bimanual.reach_ik.config
import openarm.tasks.manager_based.openarm_manipulation.bimanual.approach.config
import openarm.tasks.manager_based.openarm_manipulation.bimanual.grasp.config
import openarm.tasks.manager_based.openarm_manipulation.bimanual.grasp_2g_ik.config
import openarm.tasks.manager_based.openarm_manipulation.blending.pouring.config
import openarm.tasks.manager_based.openarm_manipulation.blending.pouring1.config
import openarm.tasks.manager_based.openarm_manipulation.blending.pouring2.config
import openarm.tasks.manager_based.openarm_manipulation.blending.pouring3.config
