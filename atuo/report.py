from __future__ import annotations

import json
from pathlib import Path


def _fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _mean_pair(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return (a + b) / 2.0


def build_report_md(report: dict, metrics: dict) -> str:
    train = metrics.get("train", {})
    eval_metrics = metrics.get("eval", {})
    decision = report.get("decision", {})
    analysis = report.get("analysis", {})

    mean_reward = train.get("scalars", {}).get("mean_reward", {})
    entropy = train.get("scalars", {}).get("entropy", {})

    lift_mean = _mean_pair(
        eval_metrics.get("lift_success_left"), eval_metrics.get("lift_success_right")
    )
    goal_track_mean = _mean_pair(
        eval_metrics.get("goal_track_success_left"), eval_metrics.get("goal_track_success_right")
    )
    success_rate_mean = _mean_pair(
        eval_metrics.get("success_rate_left"), eval_metrics.get("success_rate_right")
    )
    goal_dist_mean = _mean_pair(
        eval_metrics.get("goal_dist_min_left_mean"), eval_metrics.get("goal_dist_min_right_mean")
    )

    lines = []
    lines.append(f"# Experiment Report: {report.get('run_id', 'unknown')}")
    lines.append("")

    lines.append("## Summary")
    lines.append(f"- status: {decision.get('status', 'unknown')}")
    lines.append(f"- task: {report.get('task', 'grasp2g-v1')}")
    lines.append(f"- agent: {report.get('agent', 'rsl_rl_dual_cfg_entry_point')}")
    lines.append("")

    lines.append("## Training")
    lines.append(f"- log_dir: {report.get('train', {}).get('log_dir', 'n/a')}")
    lines.append(f"- seed: {report.get('train', {}).get('seed', 'n/a')}")
    lines.append(f"- mean_reward_last_100: {_fmt_float(mean_reward.get('mean_last_100'))}")
    lines.append(f"- mean_reward_max: {_fmt_float(mean_reward.get('max'))}")
    lines.append(f"- entropy_last: {_fmt_float(entropy.get('last'))}")
    lines.append(f"- total_iterations: {report.get('train', {}).get('total_iterations', 'n/a')}")
    lines.append("")

    lines.append("## Evaluation")
    lines.append(f"- success_rate_mean: {_fmt_float(success_rate_mean)}")
    lines.append(f"- lift_success_mean: {_fmt_float(lift_mean)}")
    lines.append(f"- goal_track_success_mean: {_fmt_float(goal_track_mean)}")
    lines.append(f"- goal_dist_mean: {_fmt_float(goal_dist_mean)}")
    lines.append("")

    if analysis:
        lines.append("## Analysis")
        if analysis.get("issues"):
            lines.append(f"- issues: {', '.join(analysis.get('issues', []))}")
        if analysis.get("observations"):
            lines.append(f"- observations: {', '.join(analysis.get('observations', []))}")
        if analysis.get("llm_summary"):
            lines.append(f"- llm_summary: {analysis.get('llm_summary')}")
        lines.append("")

    if analysis.get("applied_overrides"):
        lines.append("## Applied Overrides")
        for item in analysis.get("applied_overrides", []):
            lines.append(f"- {item}")
        lines.append("")

    lines.append("## Files")
    lines.append(f"- report.json: {report.get('path_report_json', 'report.json')}")
    lines.append(f"- metrics.json: {report.get('path_metrics_json', 'metrics.json')}")
    if analysis.get("llm_summary"):
        lines.append(f"- analysis_prompt.txt: {report.get('path_analysis_prompt', 'analysis_prompt.txt')}")
        lines.append(f"- analysis_response.txt: {report.get('path_analysis_response', 'analysis_response.txt')}")
    lines.append("")

    return "\n".join(lines)


def write_report_md(report_path: str, metrics_path: str, out_path: str) -> None:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    metrics_file = Path(metrics_path)
    if metrics_file.is_file():
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    else:
        metrics = {}

    report["path_report_json"] = str(Path(report_path).resolve())
    report["path_metrics_json"] = str(metrics_file.resolve())
    report["path_analysis_prompt"] = str(Path(report_path).parent.joinpath("analysis_prompt.txt").resolve())
    report["path_analysis_response"] = str(Path(report_path).parent.joinpath("analysis_response.txt").resolve())

    md = build_report_md(report, metrics)
    Path(out_path).write_text(md, encoding="utf-8")
