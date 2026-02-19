# 5G Lift Left v1 Rewards Guide (Latest)

업데이트 기준: 2026-02-19

기준 코드:
- `SkillBlender_Manipulation/source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand/left/5g_lift_left_v1/lift_left_env_cfg.py`
- `SkillBlender_Manipulation/source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand/left/5g_lift_left_v1/mdp/rewards.py`
- `SkillBlender_Manipulation/source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand/left/5g_lift_left_v1/mdp/actions.py`
- `SkillBlender_Manipulation/source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand/left/5g_lift_left_v1/mdp/terminations.py`

## 1) Phase Trigger (DexPour style)

- `λ` (approach): `λ = 1[ ||p_ee - p_target_static|| < 0.05 ]`
- `μ` (grasp): `μ = λ * 1[ n_contact >= 3 ]`
- `ν` (lift): `ν = μ * 1[ z_cup >= z_init + 0.04 ]`

여기서:
- `n_contact`: 기하학 기반 fingertip contact 개수
- `p_target_static`: `grasp2g_target_offset`를 적용한 정적 grasp target

## 2) Grasp Target / Reach Params

- `grasp2g_target_offset = (0.01, -0.06, 0.08)`
- 동적 접근 Z 파라미터:
  - `reach_dynamic_z_high=0.25`
  - `reach_dynamic_xy_hi=0.10`
  - `reach_dynamic_xy_lo=0.03`
  - `reach_dynamic_xy_gate=0.03`
  - `reach_dynamic_z_descent_rate=0.001`
- 접근 중 컵 밀림 억제:
  - `reach_displacement_free_threshold=0.015`
  - `reach_displacement_suppress_scale=0.03`

## 3) Reward Terms (Current)

| Term | Function | Weight | Key Params |
|---|---|---:|---|
| reaching_object | `object_ee_distance` | 8.0 | `std=0.15` |
| reaching_object_fine | `object_ee_distance_fine` | 10.0 | `std=0.065` |
| end_effector_orientation | `eef_z_perpendicular_object_z` | 4.0 | `std=0.3` |
| thumb_grasp | `thumb_grasp_reward` | 15.0 | `std=0.05` |
| pinky_grasp | `pinky_grasp_reward` | 12.0 | `std=0.05` |
| synergy_grip | `synergy_grip_reward` | 20.0 | `surface-gated` |
| finger_tip_to_cup | `finger_wrap_cylinder_reward` | 0.0 | `target_radius=0.045` |
| finger_wrap_coverage | `finger_wrap_coverage_reward` | 0.0 | - |
| finger_tip_orientation | `finger_tip_orientation_reward` | 5.0 | `std=0.5` |
| lifting_object | `object_is_lifted` | 10.0 | `minimal_height=0.04` |
| cup_lift_progress | `cup_lift_progress_reward` | 20.0 | `std=0.05` |
| object_goal_tracking | `object_goal_distance` | 20.0 | `std=0.3, minimal_height=0.04` |
| object_goal_tracking_fine_grained | `object_goal_distance` | 10.0 | `std=0.1, minimal_height=0.04` |
| object_displacement | `object_displacement_penalty` | -5.0 | `threshold=0.01` |
| finger_normal_range | `finger_normal_range_penalty` | -2.0 | - |
| thumb_reaching_pose | `thumb_reaching_pose_reward` | 0.5 | `std=1.0` |
| pinky_reaching_pose | `pinky_reaching_pose_reward` | 0.5 | `std=1.0` |
| synergy_reaching_pose | `synergy_reaching_pose_reward` | 0.5 | `std=5.0` |
| thumb_tip_z | `thumb_tip_z_reward` | 8.0 | `std=0.10` |
| synergy_tip_z | `synergy_tip_z_reward` | 8.0 | `std=0.06, cup_height=0.09` |
| ee_descent | `ee_descent_reward` | 10.0 | `std=0.04, target_z_offset=0.04` |
| action_rate | `action_rate_l2` | -0.0001 | - |
| joint_vel | `joint_vel_l2` | -0.0001 | left arm+hand |

## 4) 보상 함수 계산식

공통:
- `tanh-kernel(d,s) = 1 - tanh(d/s)`
- 최종 step 보상: `R = Σ_i w_i * r_i`

항목별:
- `reaching_object`
  - `r = tanh-kernel(||p_target_dyn - p_ee||, std) * exp(-max(d_xy - d_free, 0)/s_disp)`
- `reaching_object_fine`
  - `r = tanh-kernel(||p_target_static - p_ee||, std)`
- `end_effector_orientation`
  - `r = tanh-kernel(|cos(theta_ee_z,obj_z)|, std)`
- `thumb_grasp`
  - `r_surface = tanh-kernel(| ||p_thumb_xy-p_cup_xy|| - r_cup |, std)`
  - `r_penetration = tanh(max(r_cup-||p_thumb_xy-p_cup_xy||,0)/0.01)`
  - `z_gate = 1 - tanh(max(z_thumb-z_f2,0)/0.03)`
  - `r = λ * (r_surface - r_penetration) * z_gate`
- `pinky_grasp`
  - `r = λ * (tanh-kernel(| ||p_pinky_xy-p_cup_xy|| - r_cup |, std) - tanh(max(r_cup-||...||,0)/0.01))`
- `synergy_grip`
  - `close = clamp((a_synergy+1)/2, 0, 1)`
  - `g_surface = tanh-kernel(mean_k | ||p_f{k}_xy-p_cup_xy|| - r_cup |, proximity_std), k∈{2,3,4}`
  - `r = λ * g_surface * close`
- `finger_tip_to_cup` (현재 weight 0)
  - `r = λ * mean_j tanh-kernel(||p_tip_j_xy-p_cup_xy||, std)`
- `finger_wrap_coverage` (현재 weight 0)
  - `r = λ * mean_{i<j} ((1 - cos(theta_ij))/2)`
- `finger_tip_orientation`
  - `r = λ * mean_j clamp(dot(n_tip_j_xy, dir_tip_to_cup_j_xy), 0, 1)`
- `lifting_object`
  - `r = μ * 1[z_cup > z_init + h_min]`
- `cup_lift_progress`
  - `r = μ * tanh(max(z_cup-z_init, 0)/std)`
- `object_goal_tracking`, `object_goal_tracking_fine_grained`
  - `r = ν * tanh-kernel(||p_obj - p_goal||, std)`
- `object_displacement`
  - `d = ||p_cup_xy - p_cup_xy_init||`
  - `p = (max(d-th,0)/scale)^power`
  - `p = p * ((1-g_mix) + g_mix*g_grasp_progress) * (1-μ)`
  - `r = p` (음수 weight로 패널티 적용)
- `finger_normal_range`
  - `r = Σ_j [max(lo_j-q_j,0) + max(q_j-hi_j,0)]`
- `thumb_reaching_pose`
  - `e = Σ_j (q_j-q*_j)^2, j∈thumb`
  - `r = (1-λ) * tanh-kernel(e, std)`
- `pinky_reaching_pose`
  - `e = Σ_j (q_j-q*_j)^2, j∈pinky`
  - `r = (1-λ) * tanh-kernel(e, std)`
- `synergy_reaching_pose`
  - `e = Σ_j (q_j-0)^2, j∈{f2,f3,f4 joints}`
  - `r = (1-λ) * tanh-kernel(e, std)`
- `thumb_tip_z`
  - `z_term = tanh-kernel(max(z_thumb-z_f2,0), std)`
  - `xy_gate = exp(-||p_ee_xy-p_cup_xy||/xy_std)`
  - `r = λ * z_term * xy_gate`
- `synergy_tip_z`
  - `z_term = tanh-kernel(|z_f2_tip - (z_cup+cup_height)|, std)`
  - `xy_gate = exp(-||p_ee_xy-p_cup_xy||/xy_std)`
  - `r = λ * z_term * xy_gate`
- `ee_descent`
  - `z_term = tanh-kernel(|z_ee-(z_cup+z_offset)|, std)`
  - `xy_gate = exp(-||p_ee_xy-p_cup_xy||/xy_std)`
  - `r = λ * (1-μ) * z_term * xy_gate`
- `action_rate`
  - `r = ||a_t - a_{t-1}||^2`
- `joint_vel`
  - `r = ||qdot_left_arm_hand||^2`

## 5) Recent Behavior-Critical Points

- `synergy_grip_reward = λ * surface_proximity_gate * close_reward`
- 시너지 close pose에서 spread 조인트(`lj_dg_2_1`, `lj_dg_3_1`, `lj_dg_4_1`)는 `0.0`
- reaching 계열(`reaching_object`, `reaching_object_fine`, `end_effector_orientation`)은 현재 하드 `(1-λ)` 게이트 없이 동작

## 6) Termination

- `cup_dropping`: `z_cup < -0.05`
- `cup_tipping`: `dot(z_cup_axis, z_world) < cos(90°)`
