from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Patterns for progress lines to display on terminal
_RSL_RL_ITER = re.compile(r"Learning iteration \d+/\d+")
_RSL_RL_TIME = re.compile(r"Iteration time:\s*\S+")
_RSL_RL_ELAPSED = re.compile(r"Time elapsed:\s*\S+")
_RSL_RL_ETA = re.compile(r"ETA:\s*\S+")
_SKRL_PROGRESS = re.compile(r"\d+%\|.*?it/s\]")


@dataclass
class TrainResult:
    returncode: int
    log_dir: str | None
    stdout_path: str
    stderr_path: str


def _task_prefix(task: str) -> str:
    return task.split("-")[0]


def _find_latest_log_dir(log_root: Path, task: str) -> str | None:
    task_root = log_root / _task_prefix(task)
    if not task_root.is_dir():
        return None
    candidates = []
    for entry in task_root.iterdir():
        if entry.is_dir() and entry.name.startswith("test"):
            candidates.append(entry)
    if not candidates:
        return None

    def _score(p: Path) -> tuple[int, float]:
        suffix = p.name[4:]
        num = int(suffix) if suffix.isdigit() else -1
        return (num, p.stat().st_mtime)

    candidates.sort(key=_score, reverse=True)
    return str(candidates[0])


def run_train(
    isaaclab_root: str,
    train_script: str,
    task: str,
    agent: str,
    base_args: Iterable[str],
    hydra_overrides: Iterable[str],
    num_envs: int | None,
    seed: int | None,
    max_iterations: int | None,
    resume_from: str | None,
    resume_checkpoint: str | None,
    extra_env: dict | None,
    stream_logs: bool,
    log_root: str,
    output_dir: str,
) -> TrainResult:
    isaaclab_root = str(Path(isaaclab_root).resolve())
    train_script = str(Path(train_script).resolve())
    log_root_path = Path(log_root).resolve()
    output_dir_path = Path(output_dir).resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    cmd = ["./isaaclab.sh", "-p", train_script, "--task", task, "--agent", agent]
    if num_envs is not None:
        cmd += ["--num_envs", str(num_envs)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if max_iterations is not None:
        cmd += ["--max_iterations", str(max_iterations)]
    if resume_from:
        cmd += ["--resume", "--load_run", str(resume_from)]
    if resume_checkpoint:
        cmd += ["--checkpoint", str(resume_checkpoint)]
    cmd += list(base_args)
    cmd += list(hydra_overrides)

    stdout_path = output_dir_path / "train.stdout.txt"
    stderr_path = output_dir_path / "train.stderr.txt"

    env = os.environ.copy()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
        cmd = ["env"] + [f"{k}={v}" for k, v in extra_env.items()] + cmd

    # Accumulator for rsl_rl multi-line progress
    rsl_parts = {"iter": "", "time": "", "elapsed": "", "eta": ""}

    def _flush_rsl_progress():
        """Print accumulated rsl_rl progress as one line."""
        if rsl_parts["iter"]:
            line = rsl_parts["iter"]
            if rsl_parts["time"]:
                line += ", " + rsl_parts["time"]
            if rsl_parts["elapsed"]:
                line += ", " + rsl_parts["elapsed"]
            if rsl_parts["eta"]:
                line += ", " + rsl_parts["eta"]
            print(f"\r[train][out] {line}          ", end="", file=sys.stdout)
            sys.stdout.flush()
            rsl_parts["iter"] = ""
            rsl_parts["time"] = ""
            rsl_parts["elapsed"] = ""
            rsl_parts["eta"] = ""

    def _tee_stream(stream, fh, prefix: str):
        buf = b""
        while True:
            chunk = stream.read(1)
            if not chunk:
                if buf:
                    text = buf.decode(errors="replace")
                    fh.write(text)
                    fh.flush()
                _flush_rsl_progress()
                break
            buf += chunk
            # split on \n or \r (tqdm uses \r without \n)
            if chunk in (b"\n", b"\r"):
                text = buf.decode(errors="replace")
                buf = b""
                fh.write(text)
                fh.flush()
                if stream_logs:
                    stripped = text.strip()
                    if not stripped:
                        continue
                    # skrl tqdm progress
                    if _SKRL_PROGRESS.search(stripped):
                        print(f"\r{prefix}{stripped}          ", end="", file=sys.stdout)
                        sys.stdout.flush()
                        continue
                    # rsl_rl: collect parts then print as one line
                    m = _RSL_RL_ITER.search(stripped)
                    if m:
                        _flush_rsl_progress()
                        rsl_parts["iter"] = m.group()
                        continue
                    m = _RSL_RL_TIME.search(stripped)
                    if m:
                        rsl_parts["time"] = m.group()
                        continue
                    m = _RSL_RL_ELAPSED.search(stripped)
                    if m:
                        rsl_parts["elapsed"] = m.group()
                        continue
                    m = _RSL_RL_ETA.search(stripped)
                    if m:
                        rsl_parts["eta"] = m.group()
                        _flush_rsl_progress()
                        continue

    with open(stdout_path, "w", encoding="utf-8") as stdout_fh, open(
        stderr_path, "w", encoding="utf-8"
    ) as stderr_fh:
        if stream_logs:
            proc = subprocess.Popen(
                cmd, cwd=isaaclab_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
            )
            threads = [
                threading.Thread(target=_tee_stream, args=(proc.stdout, stdout_fh, "[train][out] ")),
                threading.Thread(target=_tee_stream, args=(proc.stderr, stderr_fh, "[train][err] ")),
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

    log_dir = _find_latest_log_dir(log_root_path, task)

    meta = {
        "cmd": cmd,
        "returncode": returncode,
        "log_dir": log_dir,
        "hydra_overrides": list(hydra_overrides),
        "resume_from": resume_from,
        "resume_checkpoint": resume_checkpoint,
    }
    (output_dir_path / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return TrainResult(returncode, log_dir, str(stdout_path), str(stderr_path))
