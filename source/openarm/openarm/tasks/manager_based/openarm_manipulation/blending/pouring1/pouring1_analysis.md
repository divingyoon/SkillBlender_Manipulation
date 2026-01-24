# pouring1_analysis

## 1. 전체 작동 방식 (blending)
- ManagerBasedRLEnv 기반 bimanual pouring task.
- 구성 요소: 로봇(양팔), 컵 2개(object/object2), 비드(bead), 테이블/그라운드/라이트.
- 명령(Command)은 PhaseSwitchPoseCommand를 사용.
  - phase 이전에는 Uniform 샘플 기반 목표(컵 주변 pose), phase 이후에는 컵/목표 컵 포즈 + offset으로 전환.
- 학습 흐름은 per-hand phase(0~3) + group phase(4~5)로 진행.
  - 0: reach, 1: grasp, 2: lift, 3: hold, 4: transfer, 5: pour.
- 보상은 phase 게이팅이 강하게 걸려 있으며, reach/grasp/lift/align/goal/hold 등이 단계별로 활성화.

## 2. env_cfg.py 의 reward 작동 방식, phase 및 gating
- RewardsCfg는 대부분 mdp.rewards의 phase_* 함수에 의해 게이트됨.
- 주요 보상 항목
  - left/right_reaching_object: grasp2g_mdp.object_ee_distance (std=0.1) * weight 5.0
  - left/right_tcp_align_reward: phase_tcp_x_axis_alignment * weight 0.5 (phase_min<=1)
  - left/right_grasp_reward: phase_grasp_reward * weight 3.0 (phase_min==1)
  - left/right_lifting_object: grasp2g_mdp.phase_lift_reward * weight 15.0 (phase_min==2)
  - left/right_object_goal_tracking(+fine): phase_object_goal_distance_with_ee * weight 5.0 (phase_min>=2)
  - left/right_hold_offset: hold_at_offset_reward * weight 5.0 (hold_duration 기반)
  - left/right_command_tracking: tcp_to_command_distance_reward * weight 4.0
  - action_rate/joint_vel: L2 페널티
- phase_tracker는 디버그용(가중치 0)
- Terminations: bead_spill, object_dropping, object_tipped, object_out_of_reach, time_out

## 3. rewards.py 계산 방식 및 수식
- 거리 기반 shaping
  - eef_to_object_distance = 1 - tanh(dist / std)
- grasp_reward
  - dist_score = sigmoid((reach_radius - dist)/dist_scale)
  - close_score = sigmoid((closure_amount - close_center)/close_scale)
  - grasp_reward = dist_score * close_score
- tcp_to_command_distance_reward
  - command pose (robot root -> world) 와 TCP 위치 거리 기반: 1 - tanh(dist/std)
- phase gating
  - phase_reach_reward: phase_min==0
  - phase_grasp_reward: phase_min==1
  - phase_lift_reward: phase_min==2
  - phase_hold_reward: phase_min==3
- pour phase 업데이트
  - _update_pour_hand_phase: reach_ok(거리+정렬), grasp_ok(거리+close), lift_ok(z)
  - _update_pour_phase: 양손 phase>=3이면 group phase=4, cup align ok이면 5
- cup/비드 관련 보상
  - cup_xy_alignment, cup_z_alignment, cup_tilt_reward
  - bead_in_target_reward, bead_to_target_distance_reward
  - bead_spill_penalty

## 4. joint_pos_env_cfg 에서의 세팅
- 로봇: openarm_bimanual.usd, 중력 비활성, contact sensor 비활성.
- 관절 초기값 고정, 암/그리퍼 actuator 설정(arm: stiffness 400/damping 80, gripper: stiffness 2e3/damping 1e2).
- 컵/비드 USD 로드, 물리 속성 지정.
- 액션 스케일
  - arm: scale=0.1
  - hand: scale=0.2
- FrameTransformer로 left/right ee_frame 정의.
- RightOnly/Play 설정 제공.

## 5. observations.py 관측 항목
- policy 관측(ObsGroup)
  - target_object_position / target_object2_position (generated_commands)
  - joint_pos, joint_vel (relative)
  - object_position, object2_position (robot root frame)
  - object_obs, object2_obs (컵 pose/vel + eef->object 벡터)
  - actions (last_action)
- object_obs/object2_obs
  - robot root frame 기준으로 object pose/vel 변환
  - left/right eef 위치와 상대벡터 포함
  - 오른손/왼손 토큰 삽입 (x축 1.0 / -1.0)
- 추가 obs 유틸
  - cup_pair_obs, cup_pair_compact_obs, bead_obs, bead_to_target_obs
  - pour_phase_left/right/group, skill_bias_reach_grasp

## 6. mdp 폴더 파일 구성/역할
- commands_cfg.py: PourPoseCommandCfg, PhaseSwitchPoseCommandCfg 정의 (명령 차원 7, offset/phase switch 설정)
- pour_pose_command.py: CommandTerm 구현(phase 전환에 따라 source/target cup pose로 목표 생성)
- events.py: reset 로직(컵/비드 리셋, IK 기반 TCP 리셋, robot frame 기반 리셋 등)
- observations.py: 관측 구성 및 디버그 로깅
- rewards.py: phase 계산, 보상 수식, cup/bead 관련 shaping
- terminations.py: 시간초과, 컵/비드 실패 조건
- __init__.py: mdp 함수들을 export
