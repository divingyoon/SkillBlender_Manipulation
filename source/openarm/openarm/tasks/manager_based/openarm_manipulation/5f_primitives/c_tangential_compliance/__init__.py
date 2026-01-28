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

"""Primitive C: Tangential Compliance - Controlled slip in allowed direction.

This primitive provides goal-conditioned controlled slip for dexterous manipulation.
It allows slip in one direction while maintaining contact and preventing slip
in perpendicular directions.

Joints scope: wrist + hand
Action space: joint position targets
Control rate: 50 Hz

Observations:
- State: hand joints, contact flags, normal forces, slip velocity vector
- Goal: allowed slip direction, compliance scale, slip speed band

Rewards:
- Keep contact (prevent contact loss)
- Non-allowed slip penalty (perpendicular to allowed direction)
- Allowed slip band reward (smooth slip within bounds)
- Force stability (avoid spikes during slip)
- Action smoothness

Terminations:
- Success: contact maintained + slip constraints for N steps
- Failure: contact loss or uncontrolled slip
- Timeout: 8 seconds
"""

from .tangential_compliance_env_cfg import TangentialComplianceEnvCfg

# Ensure gym registration happens when this package is imported.
from . import config  # noqa: F401
