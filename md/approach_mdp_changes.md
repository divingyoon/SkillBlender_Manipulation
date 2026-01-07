# bimanual approach MDP 변경 요약

아래는 요청하신 1) 도착/정지 조건(termination), 2) 도착 후 매끄러움 제약(near‑goal smoothness), 3) 손가락 완전 펼침 목표(보상 강화) 를 적용하기 위해 **추가/수정한 위치와 내용**입니다.

## 1) 도착/정지 조건 추가 (success termination)

### 추가된 파일
- `openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/approach/mdp/terminations.py`

### 추가된 함수
- `ee_reached_and_stopped(...)`
  - 양쪽 EE가 목표 위치/자세에 도달했고, 로봇의 관절 속도가 충분히 작을 때 종료(True)로 반환합니다.
  - 사용 파라미터:
    - `pos_threshold`: 위치 오차 임계값
    - `ori_threshold`: 자세 오차 임계값
    - `joint_vel_threshold`: 관절 속도 임계값

### 동작 요약
- 커맨드 포즈(`left_ee_pose`, `right_ee_pose`)를 world 기준으로 변환해 현재 EE 포즈와 비교
- 양쪽 EE의 위치/자세 오차가 기준 이하이고,
- 관절 속도 최대값이 기준 이하이면 성공 종료

### 사용/연결
- `openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/approach/mdp/__init__.py`
  - `from .terminations import *` 추가

- `openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/approach/approach_env_cfg.py`
  - `TerminationsCfg`에 `success` 항목 추가
  - 예시 파라미터:
    - `pos_threshold=0.01`, `ori_threshold=0.1`, `joint_vel_threshold=0.05`


## 2) 도착 후 매끄러움 제약 (near‑goal smoothness)

### 수정된 파일
- `openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/approach/mdp/rewards.py`

### 추가된 함수
- `near_goal_joint_vel_l2(...)`
  - 양쪽 EE가 목표 위치/자세에 충분히 근접한 경우에만 관절 속도 L2 패널티를 부과합니다.
  - near‑goal 구간 외에서는 패널티를 0으로 반환합니다.

### 연결된 보상 항목
- `openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/approach/approach_env_cfg.py`
  - `RewardsCfg`에 `near_goal_joint_vel` 추가
  - 기본 가중치: `weight=-0.01`
  - 파라미터:
    - `pos_threshold=0.01`, `ori_threshold=0.1`


## 3) 손가락 “완전 펼침” 목표 강화

### 수정된 파일
- `openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/approach/approach_env_cfg.py`

### 추가된 보상 항목
- `left_hand_joint_open`
- `right_hand_joint_open`

### 내용
- 기존 `left_hand_joint_target`, `right_hand_joint_target`과 동일한 방식으로
  **손가락 관절을 target=0.0**으로 유지하도록 추가 패널티를 줍니다.
- 기본 가중치: `weight=-0.05`


## 참고한 코드 패턴/레퍼런스
- 종료 조건 구현 예시:
  - `openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/unimanual/lift/mdp/terminations.py`
- 기본 보상 함수 패턴:
  - `IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py`


## 변경된 파일 목록
- `openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/approach/mdp/terminations.py`
- `openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/approach/mdp/__init__.py`
- `openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/approach/mdp/rewards.py`
- `openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/approach/approach_env_cfg.py`


## 이후 튜닝 포인트
- 성공 종료 임계값 조절:
  - `pos_threshold`, `ori_threshold`, `joint_vel_threshold`
- near‑goal smoothness 강화/약화:
  - `near_goal_joint_vel`의 weight
- 손가락 완전 펼침의 목표값 변경:
  - `target` 값 수정 (현재 0.0)
  - `left_hand_joint_open`/`right_hand_joint_open` 가중치 변경
