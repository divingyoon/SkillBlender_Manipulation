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

"""Robot configuration constants for OpenArm bimanual dexterous manipulation.

This module centralizes all robot-specific constants used across primitives.
"""

from typing import List

__all__ = [
    "CONTROL_RATE_HZ",
    "LEFT_ARM_JOINTS",
    "RIGHT_ARM_JOINTS",
    "LEFT_HAND_JOINTS",
    "RIGHT_HAND_JOINTS",
    "LEFT_EE_FRAME",
    "RIGHT_EE_FRAME",
    "LEFT_GRASP_FRAME",
    "RIGHT_GRASP_FRAME",
    "LEFT_CONTACT_LINKS",
    "RIGHT_CONTACT_LINKS",
    "get_hand_joint_names",
    "get_finger_joint_names",
]

# Control rate
CONTROL_RATE_HZ = 50

# Arm joint names
LEFT_ARM_JOINTS: List[str] = [
    "openarm_left_joint1",
    "openarm_left_joint2",
    "openarm_left_joint3",
    "openarm_left_joint4",
    "openarm_left_joint5",
    "openarm_left_joint6",
    "openarm_left_joint7",
]

RIGHT_ARM_JOINTS: List[str] = [
    "openarm_right_joint1",
    "openarm_right_joint2",
    "openarm_right_joint3",
    "openarm_right_joint4",
    "openarm_right_joint5",
    "openarm_right_joint6",
    "openarm_right_joint7",
]

# Hand joint names (5 fingers x 4 joints = 20 joints per hand)
LEFT_HAND_JOINTS: List[str] = [
    f"lj_dg_{finger}_{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
]

RIGHT_HAND_JOINTS: List[str] = [
    f"rj_dg_{finger}_{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
]

# Frame names
LEFT_EE_FRAME = "ll_dg_mount"
RIGHT_EE_FRAME = "rl_dg_mount"
LEFT_GRASP_FRAME = "ll_dg_3_1"
RIGHT_GRASP_FRAME = "rl_dg_3_1"

# Contact links (t3 sensor_link prims)
LEFT_CONTACT_LINKS: List[str] = [
    "ll_dg_1_3",
    "ll_dg_1_4",
    "ll_dg_2_3",
    "ll_dg_2_4",
    "ll_dg_3_3",
    "ll_dg_3_4",
    "ll_dg_4_3",
    "ll_dg_4_4",
    "ll_dg_5_3",
    "ll_dg_5_4",
]

RIGHT_CONTACT_LINKS: List[str] = [
    "rl_dg_1_3",
    "rl_dg_1_4",
    "rl_dg_2_3",
    "rl_dg_2_4",
    "rl_dg_3_3",
    "rl_dg_3_4",
    "rl_dg_4_3",
    "rl_dg_4_4",
    "rl_dg_5_3",
    "rl_dg_5_4",
]



def get_hand_joint_names(side: str = "left") -> List[str]:
    """Get all joint names for a hand.

    Args:
        side: "left" or "right"

    Returns:
        List of joint names for the specified hand
    """
    return LEFT_HAND_JOINTS if side == "left" else RIGHT_HAND_JOINTS


def get_finger_joint_names(side: str = "left", finger: int = 1) -> List[str]:
    """Get joint names for a specific finger.

    Args:
        side: "left" or "right"
        finger: Finger index (1-5, where 1=thumb, 5=pinky)

    Returns:
        List of 4 joint names for the specified finger
    """
    prefix = "lj" if side == "left" else "rj"
    return [f"{prefix}_dg_{finger}_{joint}" for joint in range(1, 5)]
