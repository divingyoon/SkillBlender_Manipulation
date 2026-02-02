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


def _get_scalar(scalars: dict, key: str, field: str = "mean_last_100") -> float | None:
    """Safely get a scalar metric value."""
    entry = scalars.get(key, {})
    if isinstance(entry, dict):
        return _safe_float(entry.get(field))
    return _safe_float(entry)


def rule_based_issues(payload: dict, thresholds: dict) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    observations: list[str] = []

    train = payload.get("train", {}).get("scalars", {})
    eval_metrics = payload.get("eval", {})

    # ── 기존 aggregate metrics ──
    mean_reward_last = _get_scalar(train, "mean_reward")
    mean_reward_max = _get_scalar(train, "mean_reward", "max")
    entropy_last = _get_scalar(train, "entropy", "last")

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
            observations.append(f"mean_reward={mean_reward_last:.3f} < {min_reward}")

    if mean_reward_last is not None and mean_reward_max is not None:
        collapse_ratio = float(thresholds.get("collapse_ratio", 0.7))
        if mean_reward_max > 0.0 and mean_reward_last < mean_reward_max * collapse_ratio:
            issues.append("training_collapse")
            observations.append(
                f"mean_reward={mean_reward_last:.3f} < {collapse_ratio} * max({mean_reward_max:.3f})"
            )

    if entropy_last is not None:
        entropy_min = float(thresholds.get("entropy_min", 0.0))
        if entropy_last < entropy_min:
            issues.append("entropy_collapse")
            observations.append(f"entropy={entropy_last:.4f} < {entropy_min}")

    lift_min = float(thresholds.get("lift_success_min", 0.0))
    if lift_success < lift_min:
        issues.append("low_lift_success")
        observations.append(f"lift_success={lift_success:.3f} < {lift_min}")

    goal_track_min = float(thresholds.get("goal_track_success_min", 0.0))
    if goal_track_success < goal_track_min:
        issues.append("tracking_fail")
        observations.append(f"goal_track_success={goal_track_success:.3f} < {goal_track_min}")

    goal_dist_max = float(thresholds.get("goal_dist_mean_max", 1e9))
    if goal_dist_mean > goal_dist_max:
        issues.append("tracking_dist_high")
        observations.append(f"goal_dist_mean={goal_dist_mean:.4f} > {goal_dist_max}")

    # ── 개별 reward term 기반 진단 (새로 추가) ──
    _diagnose_per_reward(train, issues, observations, thresholds)

    return issues, observations


def _diagnose_per_reward(scalars: dict, issues: list[str], observations: list[str], thresholds: dict) -> None:
    """개별 reward term 기반 고급 진단."""

    # 1. Hand inactivity: 그리퍼가 전혀 움직이지 않음
    left_hand_closure = _get_scalar(scalars, "reward_left_hand_closure_diag")
    right_hand_closure = _get_scalar(scalars, "reward_right_hand_closure_diag")
    left_hand_norm = _get_scalar(scalars, "reward_left_hand_action_norm_diag")
    right_hand_norm = _get_scalar(scalars, "reward_right_hand_action_norm_diag")

    hand_inactive_threshold = float(thresholds.get("hand_closure_min", 0.05))
    hand_inactive = False
    if left_hand_closure is not None and left_hand_closure < hand_inactive_threshold:
        hand_inactive = True
        observations.append(f"left_hand_closure={left_hand_closure:.4f} < {hand_inactive_threshold} (gripper not closing)")
    if right_hand_closure is not None and right_hand_closure < hand_inactive_threshold:
        hand_inactive = True
        observations.append(f"right_hand_closure={right_hand_closure:.4f} < {hand_inactive_threshold} (gripper not closing)")
    if left_hand_norm is not None and left_hand_norm < 0.01:
        hand_inactive = True
        observations.append(f"left_hand_action_norm={left_hand_norm:.4f} (hand not actuated)")
    if right_hand_norm is not None and right_hand_norm < 0.01:
        hand_inactive = True
        observations.append(f"right_hand_action_norm={right_hand_norm:.4f} (hand not actuated)")
    if hand_inactive:
        issues.append("hand_inactive")

    # 2. Reaching plateau: EEF 거리가 높은 상태로 고착
    left_dist = _get_scalar(scalars, "reward_left_eef_dist_diag")
    right_dist = _get_scalar(scalars, "reward_right_eef_dist_diag")
    reaching_stuck_threshold = float(thresholds.get("reaching_stuck_dist", 0.12))

    reaching_stuck = False
    if left_dist is not None and left_dist > reaching_stuck_threshold:
        reaching_stuck = True
        observations.append(f"left_eef_dist={left_dist:.4f} > {reaching_stuck_threshold} (not reaching close enough)")
    if right_dist is not None and right_dist > reaching_stuck_threshold:
        reaching_stuck = True
        observations.append(f"right_eef_dist={right_dist:.4f} > {reaching_stuck_threshold} (not reaching close enough)")
    if reaching_stuck:
        issues.append("reaching_stuck")

    # 3. Phase stuck: phase 값이 낮은 상태로 고착 (대부분 phase 0)
    left_phase = _get_scalar(scalars, "reward_left_grasp2g_phase")
    right_phase = _get_scalar(scalars, "reward_right_grasp2g_phase")
    phase_stuck_threshold = float(thresholds.get("phase_stuck_max", 0.5))

    phase_stuck = False
    if left_phase is not None and left_phase < phase_stuck_threshold:
        phase_stuck = True
        observations.append(f"left_phase={left_phase:.3f} < {phase_stuck_threshold} (stuck in early phase)")
    if right_phase is not None and right_phase < phase_stuck_threshold:
        phase_stuck = True
        observations.append(f"right_phase={right_phase:.3f} < {phase_stuck_threshold} (stuck in early phase)")
    if phase_stuck:
        issues.append("phase_stuck")

    # 4. Reward conflict: displacement penalty가 reaching을 상쇄
    left_displace = _get_scalar(scalars, "reward_left_object_displacement_penalty")
    right_displace = _get_scalar(scalars, "reward_right_object_displacement_penalty")
    left_reach_fine = _get_scalar(scalars, "reward_left_reaching_object_fine")
    right_reach_fine = _get_scalar(scalars, "reward_right_reaching_object_fine")

    conflict_ratio = float(thresholds.get("reward_conflict_ratio", 0.5))
    reward_conflict = False

    if left_displace is not None and left_reach_fine is not None and left_reach_fine > 0:
        ratio = abs(left_displace) / left_reach_fine
        if ratio > conflict_ratio:
            reward_conflict = True
            observations.append(
                f"left displacement_penalty |{left_displace:.3f}| / reaching_fine {left_reach_fine:.3f} "
                f"= {ratio:.2f} > {conflict_ratio} (penalty fighting reaching)"
            )
    if right_displace is not None and right_reach_fine is not None and right_reach_fine > 0:
        ratio = abs(right_displace) / right_reach_fine
        if ratio > conflict_ratio:
            reward_conflict = True
            observations.append(
                f"right displacement_penalty |{right_displace:.3f}| / reaching_fine {right_reach_fine:.3f} "
                f"= {ratio:.2f} > {conflict_ratio} (penalty fighting reaching)"
            )
    if reward_conflict:
        issues.append("reward_conflict_displacement_vs_reaching")

    # 5. Grasping never triggered: grasping reward = 0
    left_grasp = _get_scalar(scalars, "reward_left_grasping_object")
    right_grasp = _get_scalar(scalars, "reward_right_grasping_object")
    if left_grasp is not None and right_grasp is not None:
        if left_grasp < 0.001 and right_grasp < 0.001:
            issues.append("grasp_never_triggered")
            observations.append(
                f"grasp rewards near zero (L={left_grasp:.4f}, R={right_grasp:.4f}), "
                "likely not reaching phase 1"
            )


def _format_prompt(
    payload: dict,
    issues: list[str],
    observations: list[str],
    allowed_overrides: list[str],
) -> str:
    scalars = payload.get("train", {}).get("scalars", {})

    # ── Reward Terms Table ──
    reward_rows = []
    for key, val in sorted(scalars.items()):
        if key.startswith("reward_") and not key.endswith("_diag"):
            name = key[len("reward_"):]
            if isinstance(val, dict) and val.get("mean_last_100") is not None:
                reward_rows.append(
                    f"| {name} | {val['mean_last_100']:.4f} "
                    f"| {val.get('last', 0):.4f} "
                    f"| {val.get('max', 0):.4f} "
                    f"| {val.get('min', 0):.4f} |"
                )
    reward_table = (
        "| Term | Mean | Last | Max | Min |\n"
        "|------|------|------|-----|-----|\n"
        + "\n".join(reward_rows)
        if reward_rows else "(no reward data)"
    )

    # ── Diagnostic Metrics Table ──
    diag_keys = [
        "reward_left_hand_closure_diag", "reward_right_hand_closure_diag",
        "reward_left_eef_dist_diag", "reward_right_eef_dist_diag",
        "reward_left_hand_action_norm_diag", "reward_right_hand_action_norm_diag",
        "reward_left_arm_action_norm_diag", "reward_right_arm_action_norm_diag",
    ]
    diag_rows = []
    for key in diag_keys:
        val = scalars.get(key, {})
        if isinstance(val, dict) and val.get("mean_last_100") is not None:
            name = key[len("reward_"):]
            diag_rows.append(f"| {name} | {val['mean_last_100']:.4f} |")
    diag_table = (
        "| Metric | Value |\n|--------|-------|\n" + "\n".join(diag_rows)
        if diag_rows else "(no diagnostic data)"
    )

    # ── Phase Status Table ──
    phase_rows = []
    for side in ("left", "right"):
        phase_val = scalars.get(f"reward_{side}_grasp2g_phase", {})
        if isinstance(phase_val, dict) and phase_val.get("mean_last_100") is not None:
            phase_rows.append(
                f"| {side} | {phase_val['mean_last_100']:.3f} | {phase_val.get('max', 0):.3f} |"
            )
    phase_table = (
        "| Side | Mean | Max |\n|------|------|-----|\n" + "\n".join(phase_rows)
        if phase_rows else "(no phase data)"
    )

    # ── Aggregate Metrics ──
    mean_reward = _get_scalar(scalars, "mean_reward")
    entropy = _get_scalar(scalars, "entropy", "last")
    aggregate = (
        "| Metric | Value |\n|--------|-------|\n"
        f"| mean_reward | {mean_reward:.3f} |\n" if mean_reward is not None else ""
    )
    if entropy is not None:
        aggregate += f"| entropy | {entropy:.4f} |\n"

    observations_block = "\n".join(f"- {o}" for o in observations) if observations else "(none)"

    return (
        "You are analyzing a grasp2g-v1 bimanual RL training run.\n"
        "Output JSON with keys: analysis, overrides.\n"
        "The 'analysis' field should contain your step-by-step reasoning as a string.\n"
        "The 'overrides' field should be a list of 'key=value' strings.\n\n"
        "## Task Description\n"
        "The robot has two arms (left/right) that must reach, grasp, lift, and track cups.\n"
        "Phase gating controls reward activation: Phase 0=reach, 1=grasp, 2=lift, 3=hold/goal.\n"
        "Each reward term has phase_weights=[w0,w1,w2,w3] controlling per-phase activation.\n\n"
        "## Reward Structure Reference\n"
        "- reaching_object (weight<0): L2 distance error, phase_weights=[1,0,0,0]\n"
        "- reaching_object_fine (weight>0): tanh(XY/Z), phase_weights=[1,1,0,0]\n"
        "- object_displacement_penalty (weight<0): penalizes cup movement, phase_weights=[?,1,0,0]\n"
        "- grasping_object: closure band reward, phase_weights=[0,1,0,0]\n"
        "- lifting_object: lift delta, phase_weights=[0,0,5,0]\n"
        "- bimanual_reach_min: min(reach_L, reach_R)\n"
        "- bimanual_phase_lag: |phase_L - phase_R| penalty\n"
        "- bimanual_grasp_and: both hands grasp simultaneously\n\n"
        "## Analysis Steps (follow these in order)\n\n"
        "Step 1: Reward Balance Analysis\n"
        "- List the top 5 positive and negative reward terms by magnitude\n"
        "- For each negative reward, check if it cancels a positive reward > 50%\n"
        "- If cancellation exists, identify which phase_weights cause the conflict\n\n"
        "Step 2: Phase Progression Analysis\n"
        "- Check if agents progress beyond phase 0\n"
        "- If stuck in phase 0: is reaching reward gradient sufficient? Is a penalty blocking approach?\n"
        "- If stuck in phase 1: is grasping reward activated? Is the gripper closing?\n\n"
        "Step 3: Action Analysis\n"
        "- Check hand_action_norm: if near 0, gripper is not actuating\n"
        "- Check arm_action_norm: if very high but dist not decreasing, there may be a conflict\n\n"
        "Step 4: Propose Overrides\n"
        "- For each identified issue, propose specific parameter changes\n"
        "- Calculate appropriate values based on the magnitude of conflicts in the data\n"
        "- Format: 'rewards.{side}_{name}.weight=VALUE' or 'rewards.{side}_{name}.params.KEY=VALUE'\n"
        "- Always apply symmetric changes to both left and right sides\n\n"
        "## Override Format Examples\n"
        "- rewards.left_reaching_object.weight=-2.0\n"
        "- rewards.left_object_displacement_penalty.weight=-2.0\n"
        "- rewards.left_object_displacement_penalty.params.phase_weights=[0,1,0,0]\n"
        "- rewards.left_object_displacement_penalty.params.scale=5.0\n\n"
        "## Example Analysis\n\n"
        "Given data:\n"
        "  left_reaching_object_fine: mean=0.877\n"
        "  left_object_displacement_penalty: mean=-0.843\n"
        "  left_grasp2g_phase: mean=0.226\n"
        "  left_hand_closure_diag: 0.0\n\n"
        "Step 1: displacement_penalty (-0.843) cancels 96% of reaching_fine (0.877).\n"
        "  Root cause: displacement_penalty.params.phase_weights=[1,1,0,0], active in phase 0.\n"
        "Step 2: phase stuck at 0.226, agents barely reaching phase 1.\n"
        "Step 3: hand_closure=0.0, gripper never actuates (can't reach phase 1).\n"
        "Step 4: Overrides:\n"
        "  rewards.left_object_displacement_penalty.params.phase_weights=[0,1,0,0]\n"
        "  rewards.right_object_displacement_penalty.params.phase_weights=[0,1,0,0]\n"
        "  rewards.left_object_displacement_penalty.weight=-2.0\n"
        "  rewards.right_object_displacement_penalty.weight=-2.0\n"
        "  rewards.left_object_displacement_penalty.params.scale=5.0\n"
        "  rewards.right_object_displacement_penalty.params.scale=5.0\n\n"
        "---\n\n"
        "## Raw Training Data\n\n"
        f"### Reward Terms (Episode Reward, mean_last_100)\n{reward_table}\n\n"
        f"### Diagnostic Metrics\n{diag_table}\n\n"
        f"### Phase Status\n{phase_table}\n\n"
        f"### Aggregate\n{aggregate}\n\n"
        f"## Detected Issues (rule-based)\n{json.dumps(issues, indent=2)}\n\n"
        f"## Observations (rule-based)\n{observations_block}\n\n"
        f"## Allowed override keys (prefix match):\n{json.dumps(allowed_overrides)}\n\n"
        "Now analyze the raw training data step-by-step following the 4 steps above. "
        "Calculate appropriate override values based on the actual magnitude of conflicts. "
        "Do NOT simply copy the example values—derive values from this run's data. "
        "Output JSON with keys: analysis (string), overrides (list of 'key=value' strings)."
    )


def _parse_llm_json(text: str) -> dict:
    # Try direct JSON parse first
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try extracting JSON from markdown code block
    import re
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Try finding first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    return {"analysis": text.strip(), "overrides": []}


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

    llm_summary = None
    llm_overrides: list[str] = []
    applied_overrides: list[str] = []

    # Collect rule-based overrides
    rule_overrides: list[str] = []
    for issue in issues:
        rule_overrides.extend(issue_to_overrides.get(issue, []))
    rule_overrides = _filter_overrides(rule_overrides, allowed_overrides)

    if llm_cfg.get("enabled", False):
        prompt = _format_prompt(payload, issues, observations, allowed_overrides)
        provider = str(llm_cfg.get("provider", "openai"))
        if provider == "ollama":
            response = call_ollama_chat(
                prompt=prompt,
                model=str(llm_cfg.get("model", "qwen2.5:14b")),
                temperature=float(llm_cfg.get("temperature", 0.3)),
                api_base=str(llm_cfg.get("api_base", "http://localhost:11434")),
            )
        else:
            response = call_openai_chat(
                prompt=prompt,
                model=str(llm_cfg.get("model", "gpt-4o-mini")),
                temperature=float(llm_cfg.get("temperature", 0.3)),
                max_tokens=int(llm_cfg.get("max_tokens", 2048)),
                api_base=str(llm_cfg.get("api_base", "https://api.openai.com/v1")),
            )
        parsed = _parse_llm_json(response)
        llm_summary = str(parsed.get("analysis", parsed.get("summary", "")))
        raw_overrides = parsed.get("overrides", [])
        llm_overrides = [str(x) for x in raw_overrides] if isinstance(raw_overrides, list) else []
        filtered_llm = _filter_overrides(llm_overrides, allowed_overrides)

        # LLM priority + rule-based fallback
        if len(filtered_llm) > 0:
            applied_overrides = filtered_llm
        else:
            applied_overrides = rule_overrides

        # Consistency log: overlap between LLM and rule-based
        llm_keys = {item.split("=", 1)[0].strip() for item in filtered_llm if "=" in item}
        rule_keys = {item.split("=", 1)[0].strip() for item in rule_overrides if "=" in item}
        overlap = sorted(llm_keys & rule_keys)
        payload["llm_rule_overlap"] = overlap
        payload["llm_override_count"] = len(filtered_llm)
        payload["rule_override_count"] = len(rule_overrides)
        payload["override_source"] = "llm" if len(filtered_llm) > 0 else "rule_fallback"

        payload["analysis_prompt"] = prompt
        payload["analysis_response_raw"] = response
    else:
        applied_overrides = rule_overrides

    return AnalysisResult(
        issues=issues,
        observations=observations,
        llm_summary=llm_summary,
        llm_overrides=llm_overrides,
        applied_overrides=applied_overrides,
    )
