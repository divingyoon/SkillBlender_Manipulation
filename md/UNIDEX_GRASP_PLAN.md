# UniDex Grasp Integration Summary (OpenArm Isaac Lab)

## Overview
This integrates UniDexGrasp2 **objects + pc_feat** into an OpenArm bimanual grasp task, and adds **contact-based grasp rewards**, **stability reward**, and **grasp-prior pose sampling** (when the UniDex posedata file is available).

Task ID:
- `Isaac-Grasp-OpenArm-Bi-UniDex-v0`

## What Is Implemented
- **UniDex assets**: URDF objects for 5 scales (006/008/010/012/015) are spawned as a `RigidObjectCollection`.
- **pc_feat observation**: 64D feature per object is added to policy observations.
- **Contact-based grasp rewards**:
  - contact presence near end-effector
  - grasp success = contact + lift
- **Stability reward**: encourages low object velocity while lifted.
- **Grasp prior pose sampling**:
  - loads `datasetv4.1_posedata.npy` when present
  - uses `object_euler_xy` (roll/pitch) and `object_init_z` to bias object reset pose

## Key Files
- Env config: `openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/grasp/grasp_env_cfg.py`
- OpenArm/Tesollo setup: `openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/grasp/config/joint_pos_env_cfg.py`
- MDP logic: `openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/grasp/mdp/*`

## How It Works

### Contact sensor + grasp rewards
- A contact sensor is attached to the robot bodies.
- Contact is filtered to object prims: `{ENV_REGEX_NS}/Object_.*`.
- Reward logic uses a body-name regex (`.*dg.*`) to focus on gripper bodies; if it finds no matches, it falls back to all bodies.

Rewards added:
- `object_contact` (contact + proximity)
- `grasp_success` (contact + lift)
- `object_stability` (low object speed while lifted)

### Grasp prior pose sampling
If `datasetv4.1_posedata.npy` exists:
- For each reset, sample roll/pitch and z-offset from UniDex priors.
- Yaw is still sampled uniformly.
- This biases object pose toward UniDex grasps.

## Assets / Paths
- UniDex assets directory:
  - `open_source/UniDexGrasp2/assets`
- Expected link:
  - `openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/assets/unidexgrasp_assets`

## Environment Variables
- `UNIDEXGRASP_ASSET_DIR`:
  - overrides the UniDex asset root
- `UNIDEXGRASP_POSEDATA`:
  - path to `datasetv4.1_posedata.npy`

## Run
Recommended (OpenArm script):
```
/home/user/rl_ws/IsaacLab/isaaclab.sh -p /home/user/rl_ws/openarm_isaac_lab/scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Grasp-OpenArm-Bi-UniDex-v0 --num_envs 2048
```

## Known Limitations
- UniDex posedata is **ShadowHand-based**; only object pose priors are used (no hand qpos mapping yet).
- If `datasetv4.1_posedata.npy` is missing, priors fall back to uniform sampling.
- Finger link names are inferred by regex (`.*dg.*`); if Tesollo hand uses different naming, update the pattern.

## Next Optional Improvements
- Map UniDex hand pose prior to Tesollo hand (requires kinematic mapping).
- Add goal-conditioned observation from UniDex grasp target.
- Add slip-based reward using relative object–EEF velocity.

