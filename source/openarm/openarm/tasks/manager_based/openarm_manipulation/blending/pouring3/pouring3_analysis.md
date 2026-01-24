# pouring3_analysis

## 1. 전체 작동 방식 (blending)
- pouring2의 구조를 기반으로 한 변형이며, 명령(command) 기반 관측을 적극 사용.
- PhaseSwitchPoseCommand (switch_phase=2)로 pre_offset → post_offset 전환.
- policy + policy_low 관측 그룹 제공(저수준 스킬 입력 분리 목적).
- phase 기반 reach→grasp→lift→hold, 이후 group phase(transfer/pour) 유지.

## 2. env_cfg.py 의 reward 작동 방식, phase 및 gating
- RewardsCfg는 대부분 phase 게이팅을 사용.
- 주요 차이점
  - left/right_reaching_object: phase_reach_xy_reward 사용 (XY 거리 기반, phase_min==0)
  - left/right_tcp_align_reward: weight 3.0 (phase_min==1)
  - 나머지(phase_grasp_reward, phase_lift_reward, phase_object_goal_distance_with_ee, hold_at_offset)는 pouring2와 유사
- Terminations는 pouring2와 동일

## 3. rewards.py 계산 방식 및 수식
- command 기반 object pose 사용
  - _get_object_pose_w: 명령이 존재하면 command pose를 world로 변환하여 object pose로 사용
  - 거리/정렬/보상 계산이 실제 object 상태가 아니라 명령 기반일 수 있음
- phase_reach_xy_reward
  - d_xy = ||object_xy - eef_xy||
  - reward = 1 - tanh(d_xy / std), phase_min==0에서만 활성
- grasp_reward
  - dist_score = sigmoid((reach_radius - dist)/dist_scale)
  - close_score = sigmoid((closure_amount - close_center)/close_scale)
  - grasp_reward = dist_score * close_score
- tcp_z_axis_to_cup_y_alignment
  - TCP +Z 와 cup +Y 정렬 보상 (phase_tcp_z_to_cup_y_alignment로 사용 가능)
- pour phase 로직/보상은 pouring2와 동일 구조

## 4. joint_pos_env_cfg 에서의 세팅
- OPEN_ARM_HIGH_PD_CFG 사용(고 PD 설정).
- 암 액션은 Differential IK (DifferentialInverseKinematicsActionCfg)
  - command_type="pose", relative 모드, scale=0.3
- 그리퍼는 JointPositionActionCfg (scale=0.2)
- object_frame / object2_frame FrameTransformer 추가 (컵 프레임 시각화)

## 5. observations.py 관측 항목
- policy 관측
  - command 기반 object_obs/object2_obs (use_command_pos=True)
  - joint_pos/vel + object_position + actions 포함
- policy_low 관측
  - 정책과 동일한 항목이지만 corruption 비활성화(저수준용 입력)
- cup/bead/phase 관측 유틸은 pouring2 계열과 동일

## 6. mdp 폴더 파일 구성/역할
- commands_cfg.py / pour_pose_command.py: phase 전환 명령 생성
- events.py: reset 및 IK 기반 TCP 리셋, 비드 초기화
- observations.py: command 기반 object pose 사용, 정책/저수준 관측
- rewards.py: command pose 기반 거리 계산, phase_reach_xy_reward, alignment 함수
- terminations.py: 실패/종료 조건
- __init__.py: mdp export
