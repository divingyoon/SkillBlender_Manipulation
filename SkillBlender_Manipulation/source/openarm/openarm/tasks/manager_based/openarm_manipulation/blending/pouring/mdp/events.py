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

from __future__ import annotations

from typing import Sequence

import torch

from isaaclab.assets import RigidObject
from isaaclab.utils.math import quat_apply


def reset_bead_in_cup(
    env,
    env_ids: Sequence[int],
    cup_name: str = "object",
    bead_name: str = "bead",
    offset: tuple[float, float, float] = (0.0, 0.0, 0.05),
) -> None:
    """Reset bead pose to be inside the source cup."""
    cup: RigidObject = env.scene[cup_name]
    bead: RigidObject = env.scene[bead_name]

    cup_pos = cup.data.root_pos_w[env_ids]
    cup_quat = cup.data.root_quat_w[env_ids]

    offset_vec = torch.tensor(offset, device=env.device, dtype=torch.float32)
    offset_vec = offset_vec.expand(cup_pos.shape[0], 3)
    bead_pos = cup_pos + quat_apply(cup_quat, offset_vec)
    bead_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).expand(cup_pos.shape[0], 4)

    zeros = torch.zeros_like(bead_pos)
    root_state = torch.cat([bead_pos, bead_quat, zeros, zeros], dim=-1)
    bead.write_root_state_to_sim(root_state, env_ids=env_ids)
