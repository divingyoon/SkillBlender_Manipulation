# pouring2_analysis

## 1. 전체 작동 방식 (blending)
- pouring1과 동일한 bimanual pouring 구조 + phase 기반 진행.
- PhaseSwitchPoseCommand로 컵 주변 목표를 샘플링 → phase 이후 post_offset으로 전환.
- 관측은 env frame(월드-환경 원점 기준) 기반으로 구성됨.
- per-hand phase(0~3) + group phase(4~5) 유지.

## 2. env_cfg.py 의 reward 작동 방식, phase 및 gating
- RewardsCfg 구성은 pouring1과 유사하나, weight와 일부 함수가 조정됨.
- 주요 보상 항목
  - left/right_reaching_object: grasp2g_mdp.object_ee_distance (std=0.1) * weight 3.0
  - phase_tcp_x_axis_alignment: weight 0.5
  - phase_grasp_reward: weight 3.0
  - grasp2g_mdp.phase_lift_reward: weight 15.0
  - phase_object_goal_distance_with_ee / fine_grained: weight 5.0
  - hold_at_offset_reward: weight 5.0
  - action_rate / joint_vel 패널티
- phase_tracker는 디버그용(가중치 0)
- Terminations 동일: bead_spill, drop, tipped, out_of_reach, time_out

## 3. rewards.py 계산 방식 및 수식
- grasp_reward가 pouring1 대비 강화됨
  - 컵을 원통으로 가정하여 surface distance + penetration 플래그 계산
  - dist_score = sigmoid((reach_radius - surface_dist)/dist_scale)
  - close_score = sigmoid((closure_amount - close_center)/close_scale)
  - penetration이면 보상 차단
- phase_hand_open_reward / hand_open_reward 존재(phase<=1, object과 멀면 open 보상)
- _update_pour_hand_phase
  - reach_ok: dist < reach_distance AND align > align_threshold
  - grasp_ok: dist < grasp_distance AND close > close_threshold AND not penetration
  - hand별 threshold 튜닝(왼/오른쪽 grasp_dist, close_threshold 조정)
- _update_pour_phase
  - 양손 phase>=3이면 group phase=4, cup align ok면 5
- cup/bead 보상 및 spill penalty는 pouring1과 동일 구조

## 4. joint_pos_env_cfg 에서의 세팅
- 로봇/컵/비드 설정은 pouring1과 동일
- 액션 스케일 차이
  - arm: scale=0.1
  - hand: scale=0.8 (pouring1보다 큼)
- RIGHT_ONLY / ACTION_ZERO / PLAY 변형 제공

## 5. observations.py 관측 항목
- policy 관측 구성은 pouring1과 동일한 항목 구성이지만 좌표 기준이 다름
  - object_obs/object2_obs는 env frame(월드-환경 원점) 기준
  - use_command_pos=False로 실제 object 상태 사용
- 로그 출력(OBS_STATS, command 거리 등) 강화
- cup_pair_obs / bead_obs / cup_pair_compact_obs / bead_to_target_obs / phase 관측 동일

## 6. mdp 폴더 파일 구성/역할
- commands_cfg.py / pour_pose_command.py: phase 전환 명령 생성
- events.py: reset 및 IK 기반 TCP 리셋, 비드 위치 초기화
- observations.py: env frame 기반 관측 + 진단 로그
- rewards.py: surface penetration 포함 grasp 계산, phase 업데이트 로직
- terminations.py: 실패/종료 조건
- __init__.py: mdp export
