# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--diagnostics_interval", type=int, default=200, help="Write readable training diagnostics every N env steps. Set <= 0 to disable.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
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

import csv
import gymnasium as gym
import logging
import os
import random

import numpy as np
import torch
from datetime import datetime

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

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

# import logger
logger = logging.getLogger(__name__)

import snake_project.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class TrainingDiagnosticsWrapper(gym.Wrapper):
    """Write compact CSV diagnostics from the IsaacLab environment during training."""

    def __init__(self, env, log_dir: str, interval: int):
        super().__init__(env)
        self.interval = max(0, int(interval))
        self.step_count = 0
        self.csv_file = None
        self.csv_writer = None
        if self.interval > 0:
            diag_dir = os.path.join(log_dir, "diagnostics")
            os.makedirs(diag_dir, exist_ok=True)
            self.csv_file = open(os.path.join(diag_dir, "train_monitor.csv"), "w", newline="")
            self.csv_writer = csv.DictWriter(
                self.csv_file,
                fieldnames=[
                    "step",
                    "sim_time_s",
                    "reward_mean",
                    "reward_std",
                    "done_rate",
                    "cmd_vx_mean",
                    "cmd_vy_mean",
                    "act_vx_mean",
                    "act_vy_mean",
                    "act_wz_abs_mean",
                    "vel_xy_mae_mean",
                    "raw_action_abs_mean",
                    "joint_pos_abs_mean",
                    "joint_vel_abs_mean",
                ],
            )
            self.csv_writer.writeheader()
            self.csv_file.flush()

    @staticmethod
    def _to_float(value) -> float:
        if value is None:
            return float("nan")
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return float("nan")
            return float(value.detach().mean().cpu().item())
        return float(value)

    def get_observations(self):
        return self.env.get_observations()

    def step(self, action):
        output = self.env.step(action)
        self.step_count += 1
        if self.csv_writer is not None and self.step_count % self.interval == 0:
            self._write_row(output)
        return output

    def _write_row(self, output) -> None:
        try:
            if len(output) == 5:
                _, rewards, terminated, truncated, _ = output
                dones = torch.logical_or(torch.as_tensor(terminated), torch.as_tensor(truncated))
            else:
                _, rewards, dones, _ = output
                dones = torch.as_tensor(dones)
            rewards_t = torch.as_tensor(rewards, dtype=torch.float32)
            base_env = self.unwrapped
            row = {
                "step": self.step_count,
                "sim_time_s": self.step_count * float(getattr(base_env, "step_dt", 0.0)),
                "reward_mean": self._to_float(rewards_t.mean()),
                "reward_std": self._to_float(rewards_t.std(unbiased=False)),
                "done_rate": self._to_float(dones.float().mean()),
                "cmd_vx_mean": float("nan"),
                "cmd_vy_mean": float("nan"),
                "act_vx_mean": float("nan"),
                "act_vy_mean": float("nan"),
                "act_wz_abs_mean": float("nan"),
                "vel_xy_mae_mean": float("nan"),
                "raw_action_abs_mean": float("nan"),
                "joint_pos_abs_mean": float("nan"),
                "joint_vel_abs_mean": float("nan"),
            }
            try:
                command_term = base_env.command_manager.get_term("base_velocity")
                command = command_term.command.detach()
                row["cmd_vx_mean"] = self._to_float(command[:, 0])
                row["cmd_vy_mean"] = self._to_float(command[:, 1])
                if hasattr(command_term, "_compute_virtual_state"):
                    _, _, lin_vel_vc, ang_vel_z_vc = command_term._compute_virtual_state()
                    row["act_vx_mean"] = self._to_float(lin_vel_vc[:, 0])
                    row["act_vy_mean"] = self._to_float(lin_vel_vc[:, 1])
                    row["act_wz_abs_mean"] = self._to_float(torch.abs(ang_vel_z_vc))
                    row["vel_xy_mae_mean"] = self._to_float(torch.linalg.norm(command[:, :2] - lin_vel_vc[:, :2], dim=1))
            except Exception:
                pass
            try:
                action_term = base_env.action_manager.get_term("joint_pos")
                row["raw_action_abs_mean"] = self._to_float(torch.abs(action_term.raw_actions))
            except Exception:
                pass
            try:
                robot = base_env.scene["robot"]
                row["joint_pos_abs_mean"] = self._to_float(torch.abs(robot.data.joint_pos))
                row["joint_vel_abs_mean"] = self._to_float(torch.abs(robot.data.joint_vel))
            except Exception:
                pass
            self.csv_writer.writerow(row)
            self.csv_file.flush()
        except Exception as exc:
            print(f"[WARN] Failed to write training diagnostics: {exc}")

    def close(self):
        try:
            if self.csv_file is not None:
                self.csv_file.close()
        finally:
            return super().close()


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

    # fix all random seeds for reproducibility
    random.seed(agent_cfg.seed)
    np.random.seed(agent_cfg.seed)
    torch.manual_seed(agent_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(agent_cfg.seed)
        torch.cuda.manual_seed_all(agent_cfg.seed)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

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

    # write lightweight CSV diagnostics before adapting the environment to rsl-rl
    if args_cli.diagnostics_interval > 0:
        env = TrainingDiagnosticsWrapper(env, log_dir=log_dir, interval=args_cli.diagnostics_interval)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
