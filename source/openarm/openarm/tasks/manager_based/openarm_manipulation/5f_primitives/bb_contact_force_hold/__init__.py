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

"""Primitive B: Contact Force Hold - Maintaining stable grasp with force control.

This primitive provides goal-conditioned grip force maintenance for dexterous hands.
It is task-agnostic and does NOT include GWS/epsilon or task success metrics.

Joints scope: wrist + hand (20 DOF per hand + optional wrist)
Action space: joint position targets
Control rate: 50 Hz

Observations:
- State: hand joint positions/velocities, contact flags per fingertip,
         normal force proxy, slip proxy
- Goal: grip margin target band (low, high force limits)

Rewards:
- Contact persistence (maintain required contacts)
- Slip penalty (prevent tangential slip)
- Force spike penalty (avoid impacts)
- Overgrip penalty (avoid excessive squeeze)
- Action smoothness

Terminations:
- Success: stable contact + low slip for N consecutive steps
- Failure: object drop or sustained large slip
- Timeout: 8 seconds
"""

# Ensure gym registration happens when this package is imported.
from . import config  # noqa: F401
