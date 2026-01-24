# pouring4_analysis

## 1. 전체 작동 방식 (blending)
- pouring2를 기반으로 한 재설계 버전.
- Reach/Grasp/Lift와 Pour 단계를 robosuite 스타일의 staged reward(max-stage)로 구성.
- policy / policy_low / high_level 3개 ObsGroup 사용.
  - policy: 상위 정책 관측
  - policy_low: 저수준 스킬과 동일 의미의 관측
  - high_level: skill bias용 보조 관측

## 2. env_cfg.py 의 reward 작동 방식, phase 및 gating
- Reach/Grasp/Lift 보상: staged_reward_bimanual (max stage) 사용
  - weight=10.0, reach_mult=0.2
  - staged_reach/grasp/lift는 weight=0.0으로 로그만 확인
- Pour 보상: staged_pour_reward (align→tilt→pour→success) 사용
  - weight=6.0
  - staged_pour_align/tilt/flow/success는 weight=0.0
- phase 기반 보상은 최소화됨 (align reward만 phase_tcp_x_axis_alignment로 약하게 유지)
- Terminations는 기존과 동일: bead_spill, object_dropping, object_tipped, out_of_reach, time_out

## 3. rewards.py 계산 방식 및 수식
- grasp_reward (pouring2 기반)
  - cup surface distance + penetration 플래그 사용
  - dist_score = sigmoid((reach_radius - surface_dist)/dist_scale)
  - close_score = sigmoid((closure_amount - close_center)/close_scale)
  - penetration이면 0
- staged_reward_bimanual
  - 좌/우 각각 reach/grasp/lift 계산 → min(left,right)로 결합
  - 최종 보상 = max(reach, grasp, lift)
- staged_pour_reward
  - align = min(cup_xy_alignment, cup_z_alignment)
  - tilt: align>threshold 조건 하에 0.5 + pour_scale*tilt
  - pour(flow): align+tilt 만족 시 0.75 + pour_scale*bead_to_target
  - success: bead_in_target면 1.0
  - 최종 = max(align, tilt, flow, success)

## 4. joint_pos_env_cfg 에서의 세팅
- pouring2와 동일한 joint position action 구성
  - arm scale=0.1, hand scale=0.8
- contact sensor 비활성
- FrameTransformer left/right ee_frame 사용
- RIGHT_ONLY / ACTION_ZERO / PLAY 변형 제공

## 5. observations.py 관측 항목
- policy
  - target_object_position/target_object2_position
  - joint_pos/vel
  - object_position/object2_position (robot root frame)
  - object_obs/object2_obs (use_command_pos=False, mask_cross_hand=False)
  - actions
- policy_low
  - 저수준 스킬과 일치하도록 use_command_pos=True, mask_cross_hand=True
- high_level
  - skill_bias_reach_grasp (phase 0에서 reach 선호로 조정됨)
- cup/bead/phase 관측 유틸은 기존과 동일

## 6. mdp 폴더 파일 구성/역할
- commands_cfg.py / pour_pose_command.py: phase 전환 명령 생성
- events.py: reset 및 TCP 초기화, 비드 리셋
- observations.py: policy/policy_low/high_level 분리, 스킬 바이어스 제공
- rewards.py: staged reward(Reach/Grasp/Lift + Pour) 및 로그용 세부 단계 제공
- terminations.py: 실패/종료 조건
- __init__.py: mdp export
