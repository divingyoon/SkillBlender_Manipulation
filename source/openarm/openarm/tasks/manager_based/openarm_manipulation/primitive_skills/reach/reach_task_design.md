# Reach-v1 Task Design Document

This document details the design of the `Reach-v1` task, including its command, observation space, and reward function.

## 1. Introduction

The `Reach-v1` task involves a unimanual robot learning to move its Tool Center Point (TCP) to a specific target pose relative to a cup. The primary goal is to align the TCP with a point 0.02 meters above the cup's root, ensuring the TCP's x-axis is aligned with the cup's z-axis (pointing upwards) while keeping the gripper open.

## 2. Command Design

The command in this task is designed to provide the robot with a target pose to reach.

*   **Name:** `tcp_pose`
*   **Configuration Class:** `mdp.commands_cfg.ReachPoseCommandCfg`
*   **Implementation Class:** `mdp.reach_pose_command.ReachPoseCommand`
*   **Purpose:** To generate a dynamic target pose (position and orientation) for the robot's TCP in the robot's base frame. This target pose is derived from the current pose of the cup and a specified offset.
*   **Input to Command Generation Logic:**
    *   `cup_pos_w`: Current position of the cup in the world frame.
    *   `cup_quat_w`: Current orientation of the cup in the world frame.
    *   `offset`: A fixed 3D vector `[0.0, 0.0, 0.02]` (from `reach_env_cfg.py`).
*   **Output (Command Space):** `pose_command_b`
    *   **Dimension:** 7-dimensional vector.
    *   **Components:** `[x, y, z, qw, qx, qy, qz]` (3D position and 4D quaternion representing orientation).
    *   **Frame:** Robot's base frame.

*   **Command Generation Logic (`mdp.reach_pose_command.ReachPoseCommand`):**
    1.  **Target Position Calculation:**
        `target_pos_w = cup_pos_w + offset`
        (The target position is 0.02 meters directly above the cup's root.)

    2.  **Target Orientation Calculation:**
        The goal is to align the TCP's x-axis with the cup's z-axis (world-up). The TCP's default x-axis typically points forward. To make it point upwards (like the world z-axis), a rotation of -90 degrees around the TCP's y-axis is applied. This rotation is applied relative to the cup's current orientation.
        `rot_quat = quat_from_euler_xyz([0, -pi/2, 0])` (approximate, actual implementation uses `torch.ones` for batching).
        `target_quat_w = quat_mul(cup_quat_w, rot_quat)`

    3.  **Frame Transformation:**
        The calculated world-frame target pose `(target_pos_w, target_quat_w)` is transformed into the robot's base frame `(cmd_pos_b, cmd_quat_b)` using `isaaclab.utils.math.subtract_frame_transforms`. This `(cmd_pos_b, cmd_quat_b)` constitutes the 7D command vector.

## 3. Observation Design

The observation space provides the agent with information about the robot's state, the cup's state, and the relative relationship between them.

*   **Overall Configuration:** `ObservationsCfg.PolicyCfg`
*   **Combined Observation Dimension:** Sum of individual observation dimensions.
*   **Individual Observation Terms:**

    1.  **`joint_pos`**
        *   **Function:** `mdp.joint_pos_rel` (from `isaaclab.envs.mdp`)
        *   **Purpose:** Provides the relative position of each joint in the robot.
        *   **Dimension:** Number of controllable robot joints (e.g., 7 for arm + 2 for gripper = 9).

    2.  **`joint_vel`**
        *   **Function:** `mdp.joint_vel_rel` (from `isaaclab.envs.mdp`)
        *   **Purpose:** Provides the relative velocity of each joint in the robot.
        *   **Dimension:** Number of controllable robot joints (e.g., 9).

    3.  **`tcp_pose`**
        *   **Function:** `mdp.observations.body_pose`
        *   **Purpose:** Provides the absolute position and orientation of the robot's TCP in the world frame.
        *   **Dimension:** 7 (3D position `[x, y, z]` + 4D quaternion `[qw, qx, qy, qz]`).

    4.  **`cup_pose`**
        *   **Function:** `mdp.observations.root_pose`
        *   **Purpose:** Provides the absolute position and orientation of the cup's root in the world frame.
        *   **Dimension:** 7 (3D position `[x, y, z]` + 4D quaternion `[qw, qx, qy, qz]`).

    5.  **`tcp_to_cup_pos`**
        *   **Function:** `mdp.observations.target_pos_in_tcp_frame`
        *   **Purpose:** Provides the position of the target point (cup's root + `offset`) relative to the robot's TCP frame. This directly tells the robot how far and in which direction it needs to move its TCP.
        *   **Dimension:** 3 (relative position `[x, y, z]`).

    6.  **`actions`**
        *   **Function:** `mdp.last_action` (from `isaaclab.envs.mdp`)
        *   **Purpose:** Provides the action taken in the previous time step. This helps the policy learn based on its past decisions.
        *   **Dimension:** Total action space dimension (e.g., 9).

## 4. Reward Function Design

The reward function guides the agent towards the desired behavior. It is composed of several terms with assigned weights.

*   **Overall Configuration:** `RewardsCfg`
*   **Total Reward:** Sum of all weighted individual reward terms.
*   **Individual Reward Terms:**

    1.  **`reaching_object`**
        *   **Function:** `mdp.rewards.tcp_distance_to_target`
        *   **Purpose:** Rewards the agent for bringing the TCP closer to the target position (cup's root + `offset`).
        *   **Equation:** `exp(-||TCP_pos_w - (Cup_root_pos_w + Offset)||_2)`
            *   `TCP_pos_w`: TCP's position in world frame.
            *   `Cup_root_pos_w`: Cup's root position in world frame.
            *   `Offset`: `[0.0, 0.0, 0.02]`.
            *   `||.||_2`: L2-norm (Euclidean distance).
        *   **Weight:** `1.0`

    2.  **`tcp_align_reward`**
        *   **Function:** `mdp.rewards.tcp_x_axis_alignment`
        *   **Purpose:** Rewards the agent for aligning the TCP's x-axis with the cup's z-axis (which effectively means aligning TCP's x-axis with world's z-axis due to the target orientation).
        *   **Equation:** `dot_product(TCP_x_axis_world, Cup_z_axis_world)`
            *   `TCP_x_axis_world`: TCP's local x-axis vector transformed into the world frame.
            *   `Cup_z_axis_world`: Cup's local z-axis vector transformed into the world frame.
        *   **Weight:** `0.5`

    3.  **`open_hand_reward`**
        *   **Function:** `mdp.rewards.hand_joint_position`
        *   **Purpose:** Rewards the agent for keeping the gripper in an open state.
        *   **Equation:** `exp(-|Gripper_joint_pos - Target_open_pos|)`
            *   `Gripper_joint_pos`: Current position of the gripper joint (e.g., `gripper_joint`).
            *   `Target_open_pos`: The joint position corresponding to a fully open gripper (assumed to be `0.0`).
        *   **Weight:** `0.2`

    4.  **`action_rate`**
        *   **Function:** `mdp.action_rate_l2` (from `isaaclab.envs.mdp`)
        *   **Purpose:** Penalizes large differences between consecutive actions, encouraging smoother movements.
        *   **Weight:** `-1e-4`

    5.  **`joint_vel`**
        *   **Function:** `mdp.joint_vel_l2` (from `isaaclab.envs.mdp`)
        *   **Purpose:** Penalizes high joint velocities, promoting controlled motion.
        *   **Weight:** `-1e-4`

## 5. Termination Conditions

The episode terminates under the following conditions:

1.  **`time_out`**
    *   **Function:** `mdp.time_out` (from `isaaclab.envs.mdp`)
    *   **Purpose:** Terminates the episode if the maximum episode length is reached. Configured in `ReachEnvCfg` with `episode_length_s = 8.0` seconds.

2.  **`cup_dropping`**
    *   **Function:** `mdp.root_height_below_minimum` (from `isaaclab.envs.mdp`)
    *   **Purpose:** Terminates the episode if the cup falls below a minimum height, indicating failure to maintain the object's stability.
    *   **Parameter:** `minimum_height = -0.05` meters.
