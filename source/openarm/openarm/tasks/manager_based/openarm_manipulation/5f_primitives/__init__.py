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

"""Bimanual dexterous manipulation primitives for OpenArm.

This module contains task-agnostic, goal-conditioned motor skill primitives:
- a_arm_transport: EE pose tracking without contact
- b_contact_force_hold: Maintaining stable grasp with force control
- c_tangential_compliance: Controlled slip in allowed direction
- d_finger_synergy_shape: Hand configuration via synergy space
- e_contact_transition_option: Procedural contact set transitions
"""

from . import common
from . import a_approach
from . import b_contact_force_hold
from . import b_1_contact_force_hold
from . import bb_contact_force_hold
from . import bb_1_contact_force_hold
from . import c_tangential_compliance
from . import d_finger_synergy_shape
from . import e_contact_transition_option
from . import f_grasp_pose
