# Copyright 2025 Enactic, Inc.
# Licensed under the Apache License, Version 2.0

"""Custom command term for sampling joint position targets."""

from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.managers import CommandTerm
from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass


class JointPositionCommand(CommandTerm):
    """Command generator that samples joint position targets uniformly."""

    cfg: "JointPositionCommandCfg"

    def __init__(self, cfg: "JointPositionCommandCfg", env):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, len(cfg.joint_names), device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self):
        # No metrics for this command.
        return

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        use_base = bool(self.cfg.base_targets)
        fixed_set = set(self.cfg.fixed_zero_or_target)

        for idx, name in enumerate(self.cfg.joint_names):
            if use_base:
                base = float(self.cfg.base_targets.get(name, 0.0))
                if name in fixed_set:
                    choose_base = torch.rand(len(env_ids_t), device=self.device) < 0.5
                    values = torch.where(
                        choose_base,
                        torch.full_like(env_ids_t, base, dtype=torch.float),
                        torch.zeros_like(env_ids_t, dtype=torch.float),
                    )
                else:
                    if base == 0.0 and self.cfg.clamp_zero_crossing:
                        values = torch.zeros_like(env_ids_t, dtype=torch.float)
                    else:
                        noise = torch.empty(len(env_ids_t), device=self.device).uniform_(
                            -self.cfg.noise, self.cfg.noise
                        )
                        values = torch.full_like(env_ids_t, base, dtype=torch.float) + noise
                        if self.cfg.clamp_zero_crossing:
                            if base > 0.0:
                                values = torch.clamp(values, min=0.0)
                            elif base < 0.0:
                                values = torch.clamp(values, max=0.0)

                if name in self.cfg.ranges:
                    low, high = self.cfg.ranges[name]
                    values = torch.clamp(values, min=low, max=high)
                self._command[env_ids_t, idx] = values
            else:
                rng = self.cfg.ranges.get(name, (-1.0, 1.0))
                self._command[env_ids_t, idx].uniform_(*rng)

    def _update_command(self):
        # Commands are held constant between resamples.
        return


@configclass
class JointPositionCommandCfg(CommandTermCfg):
    """Uniformly-sampled joint position target command."""

    class_type: type = JointPositionCommand
    resampling_time_range: tuple[float, float] = (4.0, 6.0)
    debug_vis: bool = False
    joint_names: list[str] = []
    ranges: dict[str, tuple[float, float]] = {}
    base_targets: dict[str, float] = {}
    fixed_zero_or_target: list[str] = []
    noise: float = 0.0
    clamp_zero_crossing: bool = False
