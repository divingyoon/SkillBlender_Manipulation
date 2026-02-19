# 5G Lift Left v2 Rewards Guide (Latest)

업데이트 기준: 2026-02-19

기준 코드:
- `SkillBlender_Manipulation/source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand/left/5g_lift_left_v2/lift_left_env_cfg.py`
- `SkillBlender_Manipulation/source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand/left/5g_lift_left_v2/mdp/rewards.py`
- `SkillBlender_Manipulation/source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand/left/5g_lift_left_v2/config/joint_pos_env_cfg.py`
- `SkillBlender_Manipulation/source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand/left/5g_lift_left_v2/mdp/actions.py`
- `SkillBlender_Manipulation/source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand/left/5g_lift_left_v2/mdp/terminations.py`

v2는 v1 구조를 유지하면서 `left_contact_sensor` 기반 contact 판단을 주요 게이트(μ, ν 관련)에 연결한 버전입니다.

## 1) Phase Trigger (DexPour style)

- `λ` (approach): `λ = 1[ ||p_ee - p_target_static|| < 0.05 ]`
- `μ` (grasp): `μ = λ * 1[ n_contact(sensor) >= 3 ]`
- `ν` (lift): `ν = μ * 1[ z_cup >= z_init + 0.04 ]`

여기서 `n_contact(sensor)`는 `sensor_cfg`가 주어지면 contact sensor 기반, 아니면 기하학 fallback입니다.

## 2) Contact Sensor Setup (v2 핵심)

- 센서 엔티티: `left_contact_sensor`
- 설정 파일: `.../5g_lift_left_v2/config/joint_pos_env_cfg.py`
- 현재 설정:
  - `prim_path="{ENV_REGEX_NS}/Robot/tesollo_left_.*_sensor_link"`
  - `filter_prim_paths_expr=["{ENV_REGEX_NS}/Cup"]`
  - `history_length=1`
  - `track_air_time=False`
- env 설정:
  - `require_filtered_contact_matrix=True`

## 3) Reward Terms (Current)

| Term | Function | Weight | Key Params |
|---|---|---:|---|
| reaching_object | `object_ee_distance` | 8.0 | `std=0.15` |
| reaching_object_fine | `object_ee_distance_fine` | 10.0 | `std=0.065` |
| end_effector_orientation | `eef_z_perpendicular_object_z` | 4.0 | `std=0.3` |
| thumb_grasp | `thumb_grasp_reward` | 15.0 | `std=0.05`, `sensor_cfg` |
| pinky_grasp | `pinky_grasp_reward` | 12.0 | `std=0.05`, `sensor_cfg` |
| synergy_grip | `synergy_grip_reward` | 20.0 | `surface-gated` |
| finger_tip_to_cup | `finger_wrap_cylinder_reward` | 0.0 | `target_radius=0.045` |
| finger_wrap_coverage | `finger_wrap_coverage_reward` | 0.0 | - |
| finger_tip_orientation | `finger_tip_orientation_reward` | 5.0 | `std=0.5` |
| lifting_object | `object_is_lifted` | 10.0 | `minimal_height=0.04`, `sensor_cfg` |
| cup_lift_progress | `cup_lift_progress_reward` | 20.0 | `std=0.05`, `sensor_cfg` |
| object_goal_tracking | `object_goal_distance` | 20.0 | `std=0.3`, `sensor_cfg` |
| object_goal_tracking_fine_grained | `object_goal_distance` | 10.0 | `std=0.1`, `sensor_cfg` |
| object_displacement | `object_displacement_penalty` | -5.0 | `threshold=0.01`, `penalty_max=2.0` |
| finger_normal_range | `finger_normal_range_penalty` | -2.0 | - |
| thumb_reaching_pose | `thumb_reaching_pose_reward` | 0.5 | `std=1.0` |
| pinky_reaching_pose | `pinky_reaching_pose_reward` | 0.5 | `std=1.0` |
| synergy_reaching_pose | `synergy_reaching_pose_reward` | 0.5 | `std=5.0` |
| thumb_tip_z | `thumb_tip_z_reward` | 10.0 | `std=0.03` |
| synergy_tip_z | `synergy_tip_z_reward` | 10.0 | `std=0.06`, `cup_height=0.09` |
| ee_descent | `ee_descent_reward` | 15.0 | `std=0.04`, `target_z_offset=0.04` |
| action_rate | `action_rate_l2` | -0.0001 | - |
| joint_vel | `joint_vel_l2` | -0.0001 | left arm+hand |

## 4) 보상 함수 계산식

공통:
- `tanh-kernel(d,s) = 1 - tanh(d/s)`
- 최종 step 보상: `R = Σ_i w_i * r_i`

항목별(수식 형태는 v1과 동일, 센서 기반 게이트만 추가):
- `reaching_object`
  - `r = tanh-kernel(||p_target_dyn - p_ee||, std) * exp(-max(d_xy - d_free, 0)/s_disp)`
- `reaching_object_fine`
  - `r = tanh-kernel(||p_target_static - p_ee||, std)`
- `end_effector_orientation`
  - `r = tanh-kernel(|cos(theta_ee_z,obj_z)|, std)`
- `thumb_grasp`
  - `r = λ * (tanh-kernel(| ||p_thumb_xy-p_cup_xy|| - r_cup |, std) - tanh(max(r_cup-||...||,0)/0.01)) * z_gate`
  - `z_gate = 1 - tanh(max(z_thumb-z_f2,0)/0.03)`
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
  - `r = μ(sensor) * 1[z_cup > z_init + h_min]`
- `cup_lift_progress`
  - `r = μ(sensor) * tanh(max(z_cup-z_init, 0)/std)`
- `object_goal_tracking`, `object_goal_tracking_fine_grained`
  - `r = ν(sensor) * tanh-kernel(||p_obj - p_goal||, std)`
- `object_displacement`
  - `d = ||p_cup_xy - p_cup_xy_init||`
  - `p = (max(d-th,0)/scale)^power`
  - `p = clamp(p, max=penalty_max)`
  - `p = p * ((1-g_mix) + g_mix*g_grasp_progress) * (1-μ(sensor))`
  - `r = p` (음수 weight로 패널티 적용)
- `finger_normal_range`
  - `r = Σ_j [max(lo_j-q_j,0) + max(q_j-hi_j,0)]`
- `thumb_reaching_pose`
  - `r = (1-λ) * tanh-kernel(Σ_j(q_j-q*_j)^2, std)`
- `pinky_reaching_pose`
  - `r = (1-λ) * tanh-kernel(Σ_j(q_j-q*_j)^2, std)`
- `synergy_reaching_pose`
  - `r = (1-λ) * tanh-kernel(Σ_j(q_j-0)^2, std)`
- `thumb_tip_z`
  - `r = λ * tanh-kernel(max(z_thumb-z_f2,0), std) * exp(-||p_ee_xy-p_cup_xy||/xy_std)`
- `synergy_tip_z`
  - `r = λ * tanh-kernel(|z_f2_tip-(z_cup+cup_height)|, std) * exp(-||p_ee_xy-p_cup_xy||/xy_std)`
- `ee_descent`
  - `r = λ * (1-μ(sensor)) * tanh-kernel(|z_ee-(z_cup+z_offset)|, std) * exp(-||p_ee_xy-p_cup_xy||/xy_std)`
- `action_rate`
  - `r = ||a_t - a_{t-1}||^2`
- `joint_vel`
  - `r = ||qdot_left_arm_hand||^2`

## 5) Recent Behavior-Critical Points

- `synergy_grip_reward = λ * surface_proximity_gate * close_reward`
- 시너지 close pose spread 조인트:
  - `lj_dg_2_1 = lj_dg_3_1 = lj_dg_4_1 = 0.0`
- v2는 `μ`, `ν` 경로에 `sensor_cfg`를 전달하여 센서 기반 판정 사용

## 6) Termination

- `cup_dropping`: `z_cup < -0.05`
- `cup_tipping`: `dot(z_cup_axis, z_world) < cos(90°)`
- `cup_xy_out_of_bounds` (v2 전용): `||p_cup_xy - p_cup_xy_init|| > 0.10`
