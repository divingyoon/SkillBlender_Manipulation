from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm import call_openai_chat, call_ollama_chat


@dataclass
class AnalysisResult:
    issues: list[str]
    observations: list[str]
    llm_summary: str | None
    llm_overrides: list[str]
    applied_overrides: list[str]


def _mean_pair(a: float | None, b: float | None) -> float:
    a = a if a is not None else 0.0
    b = b if b is not None else 0.0
    return (a + b) / 2.0


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def rule_based_issues(payload: dict, thresholds: dict) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    observations: list[str] = []

    train = payload.get("train", {}).get("scalars", {})
    eval_metrics = payload.get("eval", {})

    mean_reward = train.get("mean_reward", {})
    entropy = train.get("entropy", {})

    mean_reward_last = _safe_float(mean_reward.get("mean_last_100"))
    mean_reward_max = _safe_float(mean_reward.get("max"))
    entropy_last = _safe_float(entropy.get("last"))

    lift_success = _mean_pair(
        _safe_float(eval_metrics.get("lift_success_left")),
        _safe_float(eval_metrics.get("lift_success_right")),
    )
    goal_track_success = _mean_pair(
        _safe_float(eval_metrics.get("goal_track_success_left")),
        _safe_float(eval_metrics.get("goal_track_success_right")),
    )
    goal_dist_mean = _mean_pair(
        _safe_float(eval_metrics.get("goal_dist_min_left_mean")),
        _safe_float(eval_metrics.get("goal_dist_min_right_mean")),
    )

    if mean_reward_last is None:
        issues.append("no_train_metrics")
        observations.append("train.mean_reward.mean_last_100 missing")

    if mean_reward_last is not None:
        min_reward = float(thresholds.get("min_train_reward", 0.0))
        if mean_reward_last < min_reward:
            issues.append("no_learning")
            observations.append(f"train.mean_reward.mean_last_100={mean_reward_last:.3f} < {min_reward}")

    if mean_reward_last is not None and mean_reward_max is not None:
        collapse_ratio = float(thresholds.get("collapse_ratio", 0.7))
        if mean_reward_max > 0.0 and mean_reward_last < mean_reward_max * collapse_ratio:
            issues.append("training_collapse")
            observations.append(
                f"train.mean_reward.mean_last_100={mean_reward_last:.3f} < {collapse_ratio} * max({mean_reward_max:.3f})"
            )

    if entropy_last is not None:
        entropy_min = float(thresholds.get("entropy_min", 0.0))
        if entropy_last < entropy_min:
            issues.append("entropy_collapse")
            observations.append(f"train.entropy.last={entropy_last:.4f} < {entropy_min}")

    lift_min = float(thresholds.get("lift_success_min", 0.0))
    if lift_success < lift_min:
        issues.append("low_lift_success")
        observations.append(f"eval.lift_success_mean={lift_success:.3f} < {lift_min}")

    goal_track_min = float(thresholds.get("goal_track_success_min", 0.0))
    if goal_track_success < goal_track_min:
        issues.append("tracking_fail")
        observations.append(f"eval.goal_track_success_mean={goal_track_success:.3f} < {goal_track_min}")

    goal_dist_max = float(thresholds.get("goal_dist_mean_max", 1e9))
    if goal_dist_mean > goal_dist_max:
        issues.append("tracking_dist_high")
        observations.append(f"eval.goal_dist_min_mean={goal_dist_mean:.4f} > {goal_dist_max}")

    return issues, observations


def _format_prompt(
    payload: dict,
    issues: list[str],
    observations: list[str],
    allowed_overrides: list[str],
    suggested_overrides: dict,
) -> str:
    return (
        "You are analyzing a grasp2g-v1 RL run. Output JSON only with keys: "
        "summary, issues, overrides.\n\n"
        f"Issues (rule-based): {issues}\n"
        f"Observations: {observations}\n\n"
        f"Rule-based override suggestions: {json.dumps(suggested_overrides)}\n\n"
        "Metrics payload (JSON):\n"
        + json.dumps(payload, indent=2)
        + "\n\n"
        "Allowed override keys (prefix match before '='):\n"
        + json.dumps(allowed_overrides)
        + "\n\n"
        "Return overrides as a list of 'key=value' strings. If no change, return an empty list."
    )


def _parse_llm_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        return {"summary": text.strip(), "issues": [], "overrides": []}


def _filter_overrides(overrides: list[str], allowed_overrides: list[str]) -> list[str]:
    if not allowed_overrides:
        return overrides
    filtered = []
    for item in overrides:
        if "=" not in item:
            continue
        key = item.split("=", 1)[0].strip()
        if any(key.startswith(prefix) for prefix in allowed_overrides):
            filtered.append(item)
    return filtered


def analyze(
    payload: dict,
    thresholds: dict,
    llm_cfg: dict,
    allowed_overrides: list[str],
    rules: dict,
) -> AnalysisResult:
    issues, observations = rule_based_issues(payload, thresholds)
    issue_to_overrides = rules.get("issue_to_overrides", {})
    suggested_overrides = {k: issue_to_overrides.get(k, []) for k in issues}

    llm_summary = None
    llm_overrides: list[str] = []
    applied_overrides: list[str] = []

    if llm_cfg.get("enabled", False):
        prompt = _format_prompt(payload, issues, observations, allowed_overrides, suggested_overrides)
        provider = str(llm_cfg.get("provider", "openai"))
        if provider == "ollama":
            response = call_ollama_chat(
                prompt=prompt,
                model=str(llm_cfg.get("model", "qwen2.5:14b")),
                temperature=float(llm_cfg.get("temperature", 0.2)),
                api_base=str(llm_cfg.get("api_base", "http://localhost:11434")),
            )
        else:
            response = call_openai_chat(
                prompt=prompt,
                model=str(llm_cfg.get("model", "gpt-4o-mini")),
                temperature=float(llm_cfg.get("temperature", 0.2)),
                max_tokens=int(llm_cfg.get("max_tokens", 600)),
                api_base=str(llm_cfg.get("api_base", "https://api.openai.com/v1")),
            )
        parsed = _parse_llm_json(response)
        llm_summary = str(parsed.get("summary", ""))
        llm_overrides = parsed.get("overrides", []) if isinstance(parsed.get("overrides", []), list) else []
        llm_overrides = [str(x) for x in llm_overrides]
        applied_overrides = _filter_overrides(llm_overrides, allowed_overrides)

        payload["analysis_prompt"] = prompt
        payload["analysis_response_raw"] = response

    if not llm_cfg.get("enabled", False):
        default_overrides: list[str] = []
        for issue in issues:
            default_overrides.extend(issue_to_overrides.get(issue, []))
        applied_overrides = _filter_overrides(default_overrides, allowed_overrides)

    return AnalysisResult(
        issues=issues,
        observations=observations,
        llm_summary=llm_summary,
        llm_overrides=llm_overrides,
        applied_overrides=applied_overrides,
    )
