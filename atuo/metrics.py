from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ScalarSummary:
    last: float | None
    mean_last_n: float | None
    max_value: float | None
    min_value: float | None
    last_step: int | None


def _sha256_of_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_latest_event_file(log_dir: Path) -> Path | None:
    events = sorted(log_dir.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return events[0] if events else None


def _get_all_tags(event_file: Path) -> list[str]:
    """Get all scalar tags from a tensorboard event file."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception:
        return []
    acc = event_accumulator.EventAccumulator(str(event_file), size_guidance={event_accumulator.SCALARS: 0})
    acc.Reload()
    return acc.Tags().get("scalars", [])


def _read_scalar_series(event_file: Path, tag: str) -> list[tuple[int, float]]:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception:
        return []

    acc = event_accumulator.EventAccumulator(str(event_file), size_guidance={event_accumulator.SCALARS: 0})
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return []
    return [(e.step, float(e.value)) for e in acc.Scalars(tag)]


def _summarize_series(series: list[tuple[int, float]], n: int = 100) -> ScalarSummary:
    if not series:
        return ScalarSummary(last=None, mean_last_n=None, max_value=None, min_value=None, last_step=None)
    last_step, last = series[-1]
    values = [v for _, v in series]
    tail = [v for _, v in series[-n:]] if n > 0 else values
    mean_last_n = sum(tail) / len(tail) if tail else None
    return ScalarSummary(
        last=last,
        mean_last_n=mean_last_n,
        max_value=max(values) if values else None,
        min_value=min(values) if values else None,
        last_step=last_step,
    )


def _tag_to_key(tag: str) -> str:
    """Convert tensorboard tag to a flat key.

    Examples:
        'Train/mean_reward' -> 'mean_reward'
        'Episode_Reward/left_reaching_object' -> 'reward_left_reaching_object'
        'Loss/entropy' -> 'entropy'
        'Loss/surrogate' -> 'surrogate'
        'Loss/value_function' -> 'value_function'
        'Curriculum/left_reaching_object' -> 'curriculum_left_reaching_object'
    """
    if "/" not in tag:
        return tag
    prefix, name = tag.split("/", 1)
    if prefix == "Train":
        return name
    elif prefix == "Episode_Reward":
        return f"reward_{name}"
    elif prefix == "Loss":
        return name
    elif prefix == "Curriculum":
        return f"curriculum_{name}"
    elif prefix == "Episode_Termination":
        return f"termination_{name}"
    elif prefix == "Policy":
        return f"policy_{name}"
    elif prefix == "Perf":
        return f"perf_{name}"
    else:
        return f"{prefix.lower()}_{name}"


def summarize_train_metrics(log_dir: str) -> dict:
    log_path = Path(log_dir)
    event_file = _find_latest_event_file(log_path)
    if event_file is None:
        return {"event_file": None, "scalars": {}}

    all_tags = _get_all_tags(event_file)

    scalars = {}
    for tag in all_tags:
        key = _tag_to_key(tag)
        series = _read_scalar_series(event_file, tag)
        summary = _summarize_series(series, n=100)
        scalars[key] = {
            "last": summary.last,
            "mean_last_100": summary.mean_last_n,
            "max": summary.max_value,
            "min": summary.min_value,
            "last_step": summary.last_step,
        }

    env_yaml = log_path / "params" / "env.yaml"
    agent_yaml = log_path / "params" / "agent.yaml"
    config_hash = {
        "env": _sha256_of_file(env_yaml),
        "agent": _sha256_of_file(agent_yaml),
    }

    return {
        "event_file": str(event_file),
        "scalars": scalars,
        "config_hash": config_hash,
    }


def write_metrics_json(path: str, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
