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

"""Debug visualization helpers for Primitive B."""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG
from isaaclab.utils import math as math_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def _quat_from_x_to_vec(vec: torch.Tensor) -> torch.Tensor:
    """Compute quaternion rotating +X to the given vector (vec in world frame)."""
    eps = 1e-6
    vec_norm = torch.norm(vec, dim=-1, keepdim=True)
    vec_unit = vec / (vec_norm + eps)
    x_axis = torch.zeros_like(vec_unit)
    x_axis[..., 0] = 1.0
    dot = torch.clamp((x_axis * vec_unit).sum(dim=-1), -1.0, 1.0)
    axis = torch.cross(x_axis, vec_unit, dim=-1)
    axis_norm = torch.norm(axis, dim=-1)
    axis = axis / (axis_norm.unsqueeze(-1) + eps)
    angle = torch.acos(dot)
    quat = math_utils.quat_from_angle_axis(angle, axis)
    # Handle near-parallel cases.
    parallel = axis_norm < eps
    if parallel.any():
        quat[parallel] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=vec.device)
        opposite = parallel & (dot < 0.0)
        if opposite.any():
            quat[opposite] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=vec.device)
    return quat


def debug_tip_force_vectors(
    env,
    env_ids: torch.Tensor,
    robot_cfg,
    sensor_names: list[str],
    link_names: list[str],
    env_id: int = 0,
    max_force: float = 0.1,
    min_scale: float = 0.02,
    enabled: bool = True,
    log_force_vectors: bool = True,
    log_every: int = 50,
):
    """Visualize per-tip contact force vectors as arrows in each tip link frame.

    Logs env0 (or specified env_id) only, but visualizes all envs in env_ids.
    Arrows are aligned to the link-frame force vector and scaled by magnitude
    (clipped by max_force).
    """
    if not enabled:
        return
    if env_id >= env.num_envs:
        return

    if env_ids is None or len(env_ids) == 0:
        return

    if not hasattr(env, "_debug_tip_force_markers"):
        cfg = RED_ARROW_X_MARKER_CFG.replace(
            prim_path="/Visuals/Debug/LeftTipForceVectors",
            markers={
                "arrow": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.1, 0.1, 1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
                )
            },
        )
        env._debug_tip_force_markers = VisualizationMarkers(cfg)
        env._debug_tip_force_markers.set_visibility(True)

    robot = env.scene[robot_cfg.name]
    body_ids, _ = robot.find_bodies(link_names, preserve_order=True)
    body_ids_t = torch.tensor(body_ids, device=robot.device, dtype=torch.long)
    env_ids_t = env_ids.to(device=robot.device, dtype=torch.long)

    tip_pos_w = robot.data.body_pos_w[env_ids_t[:, None], body_ids_t[None, :]]
    tip_quat_w = robot.data.body_quat_w[env_ids_t[:, None], body_ids_t[None, :]]

    force_vecs = []
    for sensor_name in sensor_names:
        contact_sensor = env.scene[sensor_name]
        if getattr(contact_sensor.data, "force_matrix_w", None) is not None:
            force_matrix_w = contact_sensor.data.force_matrix_w
            if force_matrix_w.dim() == 4:
                force_slice = force_matrix_w[env_ids_t, 0]
                mags = torch.norm(force_slice, dim=-1)
                max_idx = mags.argmax(dim=-1)
                force_vec = force_slice[torch.arange(force_slice.shape[0], device=robot.device), max_idx]
            else:
                force_vec = force_matrix_w[env_ids_t, 0, 0]
        else:
            net_forces_w = contact_sensor.data.net_forces_w
            if net_forces_w.dim() == 3:
                force_vec = net_forces_w[env_ids_t, 0]
            else:
                force_vec = net_forces_w[env_ids_t]
        force_vecs.append(force_vec)
    force_vecs = torch.stack(force_vecs, dim=1)

    force_link = math_utils.quat_apply_inverse(tip_quat_w, force_vecs)

    # Use only link-frame Z component for arrow direction/scale.
    z_comp = force_link[..., 2]
    dir_link = torch.zeros_like(force_link)
    dir_link[..., 2] = torch.sign(z_comp)

    if log_force_vectors:
        if not hasattr(env, "_debug_tip_force_step_count"):
            env._debug_tip_force_step_count = 0
        env._debug_tip_force_step_count += 1
        if env._debug_tip_force_step_count % max(log_every, 1) == 0:
            if (env_ids_t == env_id).any():
                idx = (env_ids_t == env_id).nonzero(as_tuple=False)[0, 0]
                force_list = [f"{name}: ({vec[0]:+.3f}, {vec[1]:+.3f}, {vec[2]:+.3f})" for name, vec in zip(link_names, force_link[idx])]
                # print(f"[tip_force_link] env={env_id} " + " | ".join(force_list))

    mags = torch.abs(z_comp)
    scale_x = torch.clamp(mags / max_force, min=0.0, max=1.0)
    # Hide arrows for zero vectors by zeroing scales.
    scale_x = torch.where(mags > 1e-6, scale_x, torch.zeros_like(scale_x))
    scale_x = torch.where(scale_x > 0.0, torch.clamp(scale_x, min=min_scale), scale_x)
    scales = torch.stack([scale_x, torch.full_like(scale_x, 0.1), torch.full_like(scale_x, 0.1)], dim=-1)

    tip_quat_flat = tip_quat_w.reshape(-1, 4)
    dir_flat = dir_link.reshape(-1, 3)
    orient_local = _quat_from_x_to_vec(dir_flat)
    orientations = math_utils.quat_mul(tip_quat_flat, orient_local)

    env._debug_tip_force_markers.visualize(
        translations=tip_pos_w.reshape(-1, 3),
        orientations=orientations,
        scales=scales.reshape(-1, 3),
    )


__all__ = ["debug_tip_force_vectors"]
