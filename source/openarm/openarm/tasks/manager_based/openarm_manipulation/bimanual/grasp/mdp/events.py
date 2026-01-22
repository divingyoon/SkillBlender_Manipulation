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

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObjectCollection
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def load_unidex_pc_feat(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    pc_feat_paths: list[str] | tuple[str, ...],
    object_names: list[str] | None = None,
):
    """Load pc_feat vectors for UniDexGrasp assets and cache on the env."""

    if hasattr(env, "unidex_pc_feat"):
        return

    feats = []
    for feat_path in pc_feat_paths:
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"pc_feat file not found: {feat_path}")
        feat = np.load(feat_path).astype(np.float32).reshape(-1)
        feats.append(torch.tensor(feat, device=env.device))

    env.unidex_pc_feat = torch.stack(feats, dim=0)
    env.unidex_pc_feat_dim = env.unidex_pc_feat.shape[1]
    env.unidex_object_names = object_names or [os.path.basename(path) for path in pc_feat_paths]
    env.unidex_object_ids = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)


def load_unidex_grasp_prior(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    posedata_path: str,
    object_code: str,
    scale_values: list[float] | tuple[float, ...],
):
    """Load UniDex grasp priors for object pose initialization."""
    if hasattr(env, "unidex_pose_prior"):
        return

    if not posedata_path or not os.path.exists(posedata_path):
        print(f"[WARN] UniDex grasp prior file not found: {posedata_path}")
        env.unidex_pose_prior = None
        return

    data = np.load(posedata_path, allow_pickle=True).item()
    if object_code not in data:
        print(f"[WARN] UniDex grasp prior missing object code: {object_code}")
        env.unidex_pose_prior = None
        return

    obj_data = data[object_code]
    scale_key_map = {}
    for key in obj_data.keys():
        try:
            scale_key_map[float(key)] = key
        except (TypeError, ValueError):
            continue

    pose_prior = []
    for scale_val in scale_values:
        key = scale_key_map.get(scale_val)
        if key is None:
            pose_prior.append(None)
            continue
        scale_entry = obj_data.get(key, {})
        euler_xy = scale_entry.get("object_euler_xy", None)
        init_z = scale_entry.get("object_init_z", None)
        if euler_xy is None or init_z is None:
            pose_prior.append(None)
            continue
        euler_xy = torch.tensor(np.asarray(euler_xy, dtype=np.float32), device=env.device)
        init_z = torch.tensor(np.asarray(init_z, dtype=np.float32).reshape(-1), device=env.device)
        pose_prior.append({"euler_xy": euler_xy, "init_z": init_z})

    env.unidex_pose_prior = pose_prior


def reset_unidex_objects(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    pose_range: dict[str, tuple[float, float]],
    object_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
    parking_pos: tuple[float, float, float] = (0.0, 0.0, -2.0),
):
    """Reset a selected object and park the rest for each environment."""

    objects: RigidObjectCollection = env.scene[object_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=objects.device)

    num_envs = len(env_ids)
    num_objects = objects.num_objects

    if not hasattr(env, "unidex_object_ids"):
        env.unidex_object_ids = torch.zeros(env.scene.num_envs, device=objects.device, dtype=torch.long)

    # sample object indices per env
    obj_ids = torch.randint(0, num_objects, (num_envs,), device=objects.device)
    env.unidex_object_ids[env_ids] = obj_ids

    # start from default state
    object_state = objects.data.default_object_state[env_ids].clone()

    # park all objects away from the workspace
    parking_pos_tensor = torch.tensor(parking_pos, device=objects.device)
    object_state[..., 0:3] = env.scene.env_origins[env_ids].unsqueeze(1) + parking_pos_tensor
    object_state[..., 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=objects.device)
    object_state[..., 7:13] = 0.0

    # sample a pose for the selected objects
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=objects.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_envs, 6), device=objects.device)

    if hasattr(env, "unidex_pose_prior") and env.unidex_pose_prior:
        for i in range(num_envs):
            obj_id = int(obj_ids[i].item())
            prior = env.unidex_pose_prior[obj_id] if obj_id < len(env.unidex_pose_prior) else None
            if prior is None:
                continue
            num_samples = prior["euler_xy"].shape[0]
            if num_samples == 0:
                continue
            sample_idx = torch.randint(0, num_samples, (1,), device=objects.device)
            rand_samples[i, 2] = prior["init_z"][sample_idx]
            rand_samples[i, 3] = prior["euler_xy"][sample_idx, 0]
            rand_samples[i, 4] = prior["euler_xy"][sample_idx, 1]

    default_selected = objects.data.default_object_state[env_ids, obj_ids].clone()
    positions = default_selected[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientation_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(default_selected[:, 3:7], orientation_delta)

    env_idx = torch.arange(num_envs, device=objects.device)
    object_state[env_idx, obj_ids, 0:3] = positions
    object_state[env_idx, obj_ids, 3:7] = orientations
    object_state[env_idx, obj_ids, 7:13] = default_selected[:, 7:13]

    objects.write_object_state_to_sim(object_state, env_ids=env_ids)
