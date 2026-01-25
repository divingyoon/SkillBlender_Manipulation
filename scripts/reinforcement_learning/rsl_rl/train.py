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

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument("--able_early_stop", action="store_true", default=False, help="Enable early stopping.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
parser.add_argument(
    "--swap_lr",
    action="store_true",
    default=False,
    help="Enable left/right swapping in the RSL-RL wrapper for data augmentation.",
)
parser.add_argument(
    "--swap_lr_prob", type=float, default=0.5, help="Probability to swap each environment per episode."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import os
import shutil
import time
import torch
from datetime import datetime
from typing import Any

import omni
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from sbm.rl import register_rsl_rl

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import openarm.tasks  # noqa: F401

# PLACEHOLDER: Extension template (do not remove this comment)

register_rsl_rl()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

EARLY_STOP_PATIENCE_ITERS = 1000
EARLY_STOP_MIN_DELTA = 1.0


class EarlyStopError(RuntimeError):
    """Raised when early stopping criteria are met."""

    def __init__(self, message: str, iteration: int, best_reward: float, best_iter: int):
        super().__init__(message)
        self.iteration = iteration
        self.best_reward = best_reward
        self.best_iter = best_iter


def _save_best_checkpoint(log_dir: str) -> None:
    """Save best checkpoint by Train/mean_reward to model_best.pt."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception as exc:
        print(f"[WARN] TensorBoard not available; best checkpoint not saved: {exc}")
        return

    tag = "Train/mean_reward"
    try:
        acc = event_accumulator.EventAccumulator(log_dir, size_guidance={event_accumulator.SCALARS: 0})
        acc.Reload()
    except Exception as exc:
        print(f"[WARN] Failed to read TensorBoard logs: {exc}")
        return

    tags = acc.Tags().get("scalars", [])
    if tag not in tags:
        print(f"[WARN] Tag '{tag}' not found; best checkpoint not saved.")
        return

    scalars = acc.Scalars(tag)
    if not scalars:
        print(f"[WARN] No scalar data for '{tag}'; best checkpoint not saved.")
        return

    best = max(scalars, key=lambda e: e.value)
    models = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
    steps = []
    for name in models:
        try:
            step = int(name.split("_")[1].split(".")[0])
        except ValueError:
            continue
        steps.append(step)
    if not steps:
        print("[WARN] No model_*.pt checkpoints found; best checkpoint not saved.")
        return

    closest = min(steps, key=lambda s: abs(s - best.step))
    src = os.path.join(log_dir, f"model_{closest}.pt")
    dst = os.path.join(log_dir, "model_best.pt")
    shutil.copy2(src, dst)
    print(
        f"[INFO] Best checkpoint saved: {dst} (tag={tag}, best_step={best.step}, "
        f"best_value={best.value:.4f}, closest_step={closest})"
    )


def _attach_early_stop(runner: OnPolicyRunner | DistillationRunner, enabled: bool) -> None:
    """Wrap runner.log with early stopping on Train/mean_reward."""
    if not enabled:
        return
    if EARLY_STOP_PATIENCE_ITERS <= 0:
        return
    if getattr(runner, "disable_logs", False):
        return

    original_log = runner.log
    state: dict[str, Any] = {"best": None, "best_it": None}

    def _log_with_early_stop(locs: dict, width: int = 80, pad: int = 35):
        result = original_log(locs, width=width, pad=pad)
        rewbuffer = locs.get("rewbuffer", [])
        if not rewbuffer:
            return result
        mean_reward = sum(rewbuffer) / len(rewbuffer)
        it = int(locs.get("it", 0))

        if state["best"] is None or mean_reward > (state["best"] + EARLY_STOP_MIN_DELTA):
            state["best"] = float(mean_reward)
            state["best_it"] = it
        if state["best_it"] is not None and (it - state["best_it"]) >= EARLY_STOP_PATIENCE_ITERS:
            raise EarlyStopError(
                f"Early stop: no improvement > {EARLY_STOP_MIN_DELTA} for {EARLY_STOP_PATIENCE_ITERS} iterations.",
                iteration=it,
                best_reward=state["best"],
                best_iter=state["best_it"],
            )
        return result

    runner.log = _log_with_early_stop


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    task_name = args_cli.task.split("-")[0]
    sbm_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    log_root_path = os.path.join(sbm_root, "log", "rsl_rl", task_name)
    log_root_path = os.path.abspath(log_root_path)
    os.makedirs(log_root_path, exist_ok=True)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: testN (auto-increment)
    existing = []
    for name in os.listdir(log_root_path):
        if name.startswith("test"):
            suffix = name[4:]
            if suffix.isdigit():
                existing.append(int(suffix))
    next_idx = (max(existing) + 1) if existing else 1
    log_dir = f"test{next_idx}"
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # log policy_low observation term order/dims for pouring (or any task that exposes policy_low)
    if hasattr(env.unwrapped, "observation_manager"):
        obs_mgr = env.unwrapped.observation_manager
        if "policy_low" in obs_mgr.active_terms:
            term_names = obs_mgr.active_terms["policy_low"]
            term_dims = obs_mgr.group_obs_term_dim["policy_low"]
            print("[INFO] policy_low obs term order/dims:")
            for name, dims in zip(term_names, term_dims):
                print(f"  - {name}: {tuple(dims)}")
            if obs_mgr.group_obs_concatenate.get("policy_low", False):
                print(f"[INFO] policy_low concatenated dim: {obs_mgr.group_obs_dim['policy_low']}")

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(
        env,
        clip_actions=agent_cfg.clip_actions,
        swap_prob=args_cli.swap_lr_prob,
    )

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    _attach_early_stop(runner, enabled=args_cli.able_early_stop)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO] Resume settings: load_run='{agent_cfg.load_run}', load_checkpoint='{agent_cfg.load_checkpoint}'")
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    start_time = time.time()
    interrupted = False
    early_stopped = False
    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    except KeyboardInterrupt:
        interrupted = True
        iteration = getattr(runner, "current_learning_iteration", "interrupt")
        interrupt_path = os.path.join(log_dir, f"model_interrupt_{iteration}.pt")
        print(f"[WARN] Training interrupted. Saving checkpoint to: {interrupt_path}")
        runner.save(interrupt_path)
    except EarlyStopError as exc:
        early_stopped = True
        early_path = os.path.join(log_dir, f"model_early_stop_{exc.iteration}.pt")
        print(
            f"[WARN] {exc} Saving checkpoint to: {early_path} "
            f"(best_reward={exc.best_reward:.4f} at iter {exc.best_iter})"
        )
        runner.save(early_path)
    finally:
        _save_best_checkpoint(log_dir)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    if interrupted:
        print("[INFO] Interrupted training checkpoint saved.")
    if early_stopped:
        print("[INFO] Early stopping checkpoint saved.")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
