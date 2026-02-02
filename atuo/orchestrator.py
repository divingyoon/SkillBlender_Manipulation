from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import subprocess
import threading

from analyzer import analyze, rule_based_issues
from metrics import summarize_train_metrics, write_metrics_json
from report import write_report_md
from runner import run_train


@dataclass
class EvalResult:
    returncode: int
    metrics_path: str | None
    stdout_path: str
    stderr_path: str


def load_config(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_path(value: str, base_dir: Path) -> str:
    if not value:
        return value
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


def _task_prefix(task: str) -> str:
    return task.split("-")[0]


def _find_checkpoint(log_dir: Path) -> Path | None:
    best = log_dir / "model_best.pt"
    if best.is_file():
        return best
    models = []
    for entry in log_dir.iterdir():
        if entry.name.startswith("model_") and entry.name.endswith(".pt"):
            models.append(entry)
    if not models:
        return None

    def _step(p: Path) -> int:
        name = p.name
        try:
            return int(name.split("_")[1].split(".")[0])
        except Exception:
            return -1

    models.sort(key=_step, reverse=True)
    return models[0]


def _mean_pair(a: float, b: float) -> float:
    return (a + b) / 2.0


def _extract_reward_keys(env_yaml_path: Path) -> list[str]:
    if not env_yaml_path.is_file():
        return []
    keys: list[str] = []
    in_rewards = False
    for raw in env_yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not in_rewards:
            if line == "rewards:":
                in_rewards = True
            continue
        if not line or (not line.startswith("  ")):
            break
        if line.startswith("    "):
            continue
        if line.endswith(":"):
            name = line.strip()[:-1]
            if name:
                keys.append(name)
    return keys


def decide_success(eval_metrics: dict, criteria: dict) -> tuple[bool, dict]:
    lift = _mean_pair(eval_metrics.get("lift_success_left", 0.0), eval_metrics.get("lift_success_right", 0.0))
    hold = _mean_pair(eval_metrics.get("hold_success_left", 0.0), eval_metrics.get("hold_success_right", 0.0))
    goal = _mean_pair(eval_metrics.get("goal_success_left", 0.0), eval_metrics.get("goal_success_right", 0.0))
    goal_track = _mean_pair(
        eval_metrics.get("goal_track_success_left", 0.0), eval_metrics.get("goal_track_success_right", 0.0)
    )
    goal_dist_mean = _mean_pair(
        eval_metrics.get("goal_dist_min_left_mean", 0.0), eval_metrics.get("goal_dist_min_right_mean", 0.0)
    )

    success_rate = _mean_pair(
        eval_metrics.get("success_rate_left", 0.0), eval_metrics.get("success_rate_right", 0.0)
    )

    ok = (
        lift >= criteria.get("lift_success_min", 1.0)
        and hold >= criteria.get("hold_success_min", 1.0)
        and goal >= criteria.get("goal_success_min", 1.0)
        and goal_track >= criteria.get("goal_track_success_min", 1.0)
        and goal_dist_mean <= criteria.get("goal_dist_mean_max", float("inf"))
        and success_rate >= criteria.get("success_rate_min", 0.0)
    )

    return ok, {
        "lift": lift,
        "hold": hold,
        "goal": goal,
        "goal_track": goal_track,
        "goal_dist_mean": goal_dist_mean,
        "success_rate": success_rate,
    }


def run_eval(
    isaaclab_root: str,
    eval_script: str,
    task: str,
    agent: str,
    checkpoint: str,
    num_episodes: int,
    num_envs: int,
    goal_threshold: float,
    lift_threshold: float,
    goal_dist_threshold: float,
    seed: int | None,
    metrics_path: str,
    output_dir: str,
    extra_env: dict | None,
    stream_logs: bool,
    base_args: list[str],
) -> EvalResult:
    isaaclab_root = str(Path(isaaclab_root).resolve())
    eval_script = str(Path(eval_script).resolve())
    output_dir_path = Path(output_dir).resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "./isaaclab.sh",
        "-p",
        eval_script,
        "--task",
        task,
        "--agent",
        agent,
        "--checkpoint",
        checkpoint,
        "--num_episodes",
        str(num_episodes),
        "--num_envs",
        str(num_envs),
        "--goal_threshold",
        str(goal_threshold),
        "--lift_threshold",
        str(lift_threshold),
        "--goal_dist_threshold",
        str(goal_dist_threshold),
        "--output",
        metrics_path,
    ] + list(base_args)
    if seed is not None:
        cmd += ["--seed", str(seed)]

    stdout_path = output_dir_path / "eval.stdout.txt"
    stderr_path = output_dir_path / "eval.stderr.txt"

    env = os.environ.copy()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
        cmd = ["env"] + [f"{k}={v}" for k, v in extra_env.items()] + cmd

    def _tee_stream(stream, fh, prefix: str):
        for line in iter(stream.readline, b""):
            text = line.decode(errors="replace")
            fh.write(text)
            fh.flush()
            if stream_logs:
                print(prefix + text, end="", file=sys.stdout)
                sys.stdout.flush()

    with open(stdout_path, "w", encoding="utf-8") as stdout_fh, open(
        stderr_path, "w", encoding="utf-8"
    ) as stderr_fh:
        if stream_logs:
            proc = subprocess.Popen(
                cmd, cwd=isaaclab_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
            )
            threads = [
                threading.Thread(target=_tee_stream, args=(proc.stdout, stdout_fh, "[eval][out] ")),
                threading.Thread(target=_tee_stream, args=(proc.stderr, stderr_fh, "[eval][err] ")),
            ]
            for t in threads:
                t.daemon = True
                t.start()
            returncode = proc.wait()
            for t in threads:
                t.join(timeout=1)
        else:
            returncode = subprocess.run(
                cmd, cwd=isaaclab_root, stdout=stdout_fh, stderr=stderr_fh, check=False, env=env
            ).returncode

    return EvalResult(returncode, metrics_path, str(stdout_path), str(stderr_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment OS orchestrator (MVP).")
    parser.add_argument("--config", required=True, help="Path to experiment.json")
    parser.add_argument("--gpu", "--GPU", dest="gpu", type=str, default=None, help="CUDA_VISIBLE_DEVICES override")
    parser.add_argument("--num_envs", type=int, default=None, help="Override train --num_envs")
    parser.add_argument("--headless", action="store_true", help="Force headless mode")
    parser.add_argument("--gui", action="store_true", help="Disable headless mode")
    parser.add_argument("--task", type=str, default=None, help="Override task")
    parser.add_argument("--agent", type=str, default=None, help="Override agent")
    parser.add_argument("--max_iterations", type=int, default=None, help="Override max_iterations")
    parser.add_argument("--resume_from", type=str, default=None, help="Override resume load_run")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Override resume checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_dir = Path(args.config).resolve().parent
    project = cfg["project"]
    project["isaaclab_root"] = resolve_path(project["isaaclab_root"], config_dir)
    project["skillblender_root"] = resolve_path(project["skillblender_root"], config_dir)
    project["log_root"] = resolve_path(project["log_root"], config_dir)
    train = cfg["train"]
    train["train_script"] = resolve_path(train["train_script"], config_dir)
    eval_cfg = cfg["eval"]
    if args.task is not None:
        train["task"] = args.task
    if args.agent is not None:
        train["agent"] = args.agent
    if args.max_iterations is not None:
        train["max_iterations"] = int(args.max_iterations)
    if args.resume_from is not None:
        train["resume_from"] = args.resume_from
    if args.resume_checkpoint is not None:
        train["resume_checkpoint"] = args.resume_checkpoint
    eval_cfg["eval_script"] = resolve_path(eval_cfg["eval_script"], config_dir)
    criteria = cfg.get("success_criteria", {})
    analysis_thresholds = cfg.get("analysis_thresholds", {})
    analysis_rules_path = resolve_path(cfg.get("analysis_rules_path", ""), config_dir)
    analysis_rules = {}
    env_vars = cfg.get("env", {})
    if args.gpu is not None:
        env_vars = dict(env_vars)
        env_vars["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if analysis_rules_path:
        try:
            analysis_rules = json.loads(Path(analysis_rules_path).read_text(encoding="utf-8"))
        except Exception:
            analysis_rules = {}
    llm_cfg = cfg.get("llm", {})
    override_policy = cfg.get("override_policy", {})
    policy = cfg.get("run_policy", {})

    max_runs = int(policy.get("max_runs", 1))
    stop_on_success = bool(policy.get("stop_on_success", True))
    train_chunk_iterations = policy.get("train_chunk_iterations", None)
    max_total_iterations = policy.get("max_total_iterations", None)
    stop_on_collapse = bool(policy.get("stop_on_collapse", True))
    stream_logs = bool(policy.get("stream_logs", True))

    seed = train.get("seed", None)
    if args.num_envs is not None:
        train = dict(train)
        train["num_envs"] = int(args.num_envs)

    pending_overrides: list[str] = list(train.get("hydra_overrides", []))

    base_args = list(train.get("base_args", []))
    if args.headless:
        if "--headless" not in base_args:
            base_args.append("--headless")
    if args.gui:
        base_args = [arg for arg in base_args if arg != "--headless"]
    train["base_args"] = base_args

    for run_idx in range(max_runs):
        run_id = time.strftime("%Y%m%d_%H%M%S") + f"_run{run_idx+1}"
        run_dir = Path(__file__).resolve().parent / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        train_result = None
        last_log_dir = None
        total_iterations = 0
        chunk = int(train_chunk_iterations) if train_chunk_iterations else None
        max_total = int(max_total_iterations) if max_total_iterations else None

        early_stop_reason = None
        while True:
            resume_from = Path(last_log_dir).name if last_log_dir else train.get("resume_from", None)
            resume_checkpoint = "model_best.pt" if last_log_dir else train.get("resume_checkpoint", None)
            target_iterations = train.get("max_iterations", None)
            if chunk is not None:
                target_iterations = chunk

            train_result = run_train(
                isaaclab_root=project["isaaclab_root"],
                train_script=train["train_script"],
                task=train["task"],
                agent=train["agent"],
                base_args=train.get("base_args", []),
                hydra_overrides=pending_overrides,
                num_envs=train.get("num_envs", None),
                seed=seed,
                max_iterations=target_iterations,
                resume_from=resume_from,
                resume_checkpoint=resume_checkpoint,
                extra_env=env_vars,
                stream_logs=stream_logs,
                log_root=project["log_root"],
                output_dir=str(run_dir),
            )

            last_log_dir = train_result.log_dir
            total_iterations += target_iterations if target_iterations is not None else 0

            if train_result.returncode != 0:
                break
            if chunk is None:
                break
            if max_total is not None and total_iterations >= max_total:
                break

            if last_log_dir:
                train_metrics = summarize_train_metrics(str(last_log_dir))
                probe_payload = {"train": train_metrics, "eval": {}}
                issues, _ = rule_based_issues(probe_payload, analysis_thresholds)
                if stop_on_collapse and ("training_collapse" in issues or "entropy_collapse" in issues):
                    early_stop_reason = "training_collapse" if "training_collapse" in issues else "entropy_collapse"
                    break

        report = {
            "run_id": run_id,
            "train": {
                "returncode": train_result.returncode,
                "log_dir": train_result.log_dir,
                "stdout": train_result.stdout_path,
                "stderr": train_result.stderr_path,
                "seed": seed,
                "hydra_overrides": list(pending_overrides),
                "total_iterations": total_iterations,
                "chunk_iterations": chunk,
                "early_stop_reason": early_stop_reason,
            },
            "eval": {},
            "decision": {},
        }

        if train_result.returncode != 0 or train_result.log_dir is None:
            report["decision"] = {"status": "train_failed"}
            (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            return 1

        log_dir = Path(train_result.log_dir)
        checkpoint = _find_checkpoint(log_dir)
        if checkpoint is None:
            report["decision"] = {"status": "no_checkpoint"}
            (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            return 1

        metrics_path = str(log_dir / "metrics.json")
        eval_result = run_eval(
            isaaclab_root=project["isaaclab_root"],
            eval_script=eval_cfg["eval_script"],
            task=train["task"],
            agent=train["agent"],
            checkpoint=str(checkpoint),
            num_episodes=int(eval_cfg.get("num_episodes", 100)),
            num_envs=int(eval_cfg.get("num_envs", 1)),
            goal_threshold=float(eval_cfg.get("goal_threshold", 0.8)),
            lift_threshold=float(eval_cfg.get("lift_threshold", 0.1)),
            goal_dist_threshold=float(eval_cfg.get("goal_dist_threshold", 0.05)),
            seed=eval_cfg.get("seed", None),
            metrics_path=metrics_path,
            output_dir=str(run_dir),
            extra_env=env_vars,
            stream_logs=stream_logs,
            base_args=list(eval_cfg.get("base_args", ["--headless"])),
        )

        train_metrics = summarize_train_metrics(str(log_dir))
        eval_metrics = {}
        if eval_result.returncode == 0 and os.path.isfile(metrics_path):
            eval_metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))

        payload = {
            "run_id": run_id,
            "task": train["task"],
            "agent": train["agent"],
            "log_dir": str(log_dir),
            "checkpoint": str(checkpoint),
            "train": train_metrics,
            "eval": eval_metrics,
        }
        write_metrics_json(metrics_path, payload)

        success, aggregated = decide_success(eval_metrics, criteria)
        analysis_result = None
        applied_overrides = []
        if not success:
            env_yaml = Path(log_dir) / "params" / "env.yaml"
            reward_names = _extract_reward_keys(env_yaml)
            reward_override_keys = []
            for name in reward_names:
                reward_override_keys.append(f"rewards.{name}.weight")
                reward_override_keys.append(f"rewards.{name}.params.")
            analysis_result = analyze(
                payload=payload,
                thresholds=analysis_thresholds,
                llm_cfg=llm_cfg,
                allowed_overrides=reward_override_keys or override_policy.get("allowed_overrides", []),
                rules=analysis_rules,
            )
            applied_overrides = analysis_result.applied_overrides
            pending_overrides = list(applied_overrides)
        report["eval"] = {
            "returncode": eval_result.returncode,
            "metrics_path": metrics_path,
            "stdout": eval_result.stdout_path,
            "stderr": eval_result.stderr_path,
        }
        report["decision"] = {
            "status": "success" if success else "failed",
            "aggregated": aggregated,
            "criteria": criteria,
        }
        if analysis_result is not None:
            report["analysis"] = {
                "issues": analysis_result.issues,
                "observations": analysis_result.observations,
                "llm_summary": analysis_result.llm_summary,
                "llm_overrides": analysis_result.llm_overrides,
                "applied_overrides": applied_overrides,
            }
            (run_dir / "analysis_prompt.txt").write_text(
                payload.get("analysis_prompt", ""), encoding="utf-8"
            )
            (run_dir / "analysis_response.txt").write_text(
                payload.get("analysis_response_raw", ""), encoding="utf-8"
            )
            (run_dir / "overrides.json").write_text(
                json.dumps({"overrides": applied_overrides}, indent=2), encoding="utf-8"
            )

        report_path = run_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        write_report_md(
            report_path=str(report_path),
            metrics_path=str(metrics_path),
            out_path=str(run_dir / "report.md"),
        )

        if success and stop_on_success:
            break

        if seed is not None:
            seed += 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
