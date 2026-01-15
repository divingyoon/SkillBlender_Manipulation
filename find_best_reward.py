#실행방법
#/home/user/rl_ws/IsaacLab/_isaac_sim/python.sh /home/user/rl_ws/find_best_reward.py

"""Find best checkpoint by Train/mean_reward from a log directory."""

from __future__ import annotations

import os
import sys
from typing import Iterable

# Set default log directory here if you do not want to use env vars.
LOG_DIR = "/home/user/rl_ws/IsaacLab/logs/rsl_rl/openarm_bi_grasp_2g/2026-01-15_13-35-04"


def _parse_step(name: str) -> int | None:
    if not (name.startswith("model_") and name.endswith(".pt")):
        return None
    stem = name.split("_", 1)[1].rsplit(".", 1)[0]
    try:
        return int(stem)
    except ValueError:
        return None


def _closest_checkpoint(log_dir: str, target_step: int) -> str | None:
    steps = []
    for name in os.listdir(log_dir):
        step = _parse_step(name)
        if step is not None:
            steps.append(step)
    if not steps:
        return None
    closest = min(steps, key=lambda s: abs(s - target_step))
    return os.path.join(log_dir, f"model_{closest}.pt")


def _get_event_tags(acc) -> Iterable[str]:
    return acc.Tags().get("scalars", [])


def _find_event_files(root: str) -> list[str]:
    events = []
    for base, _, files in os.walk(root):
        for name in files:
            if name.startswith("events.out.tfevents"):
                events.append(os.path.join(base, name))
    return events


def main() -> int:
    log_dir = os.environ.get("REWARD_LOG_DIR") or os.environ.get("LOG_DIR") or LOG_DIR
    if not log_dir:
        print("Set REWARD_LOG_DIR (or LOG_DIR) to your run folder.", file=sys.stderr)
        return 1
    if not os.path.isdir(log_dir):
        print(f"Log dir not found: {log_dir}", file=sys.stderr)
        return 1

    tag = os.environ.get("REWARD_TAG", "Train/mean_reward")

    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception as exc:
        print(f"TensorBoard not available: {exc}", file=sys.stderr)
        return 1

    acc = event_accumulator.EventAccumulator(
        log_dir, size_guidance={event_accumulator.SCALARS: 0}
    )
    acc.Reload()

    tags = list(_get_event_tags(acc))
    if tag not in tags:
        event_files = _find_event_files(log_dir)
        if event_files:
            acc = event_accumulator.EventAccumulator(
                event_files[0], size_guidance={event_accumulator.SCALARS: 0}
            )
            acc.Reload()
            tags = list(_get_event_tags(acc))
        if tag not in tags:
            print(f"Tag not found: {tag}", file=sys.stderr)
            print("Available tags:")
            for t in tags:
                print(f"- {t}")
            if not tags:
                print("No scalar tags found. Check LOG_DIR points to a run folder with events.out.tfevents.", file=sys.stderr)
            return 1

    scalars = acc.Scalars(tag)
    if not scalars:
        print(f"No scalar data for tag: {tag}", file=sys.stderr)
        return 1

    best = max(scalars, key=lambda e: e.value)
    ckpt = _closest_checkpoint(log_dir, best.step)

    print(f"Tag: {tag}")
    print(f"Best step: {best.step}")
    print(f"Best value: {best.value}")
    if ckpt:
        print(f"Closest checkpoint: {ckpt}")
        print(f"Exists: {os.path.exists(ckpt)}")
    else:
        print("No model_*.pt checkpoints found.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
