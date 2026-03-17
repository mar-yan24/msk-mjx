#!/usr/bin/env python
"""CLI entry-point for PPO training on musculoskeletal locomotion.

Usage:
    python scripts/train.py                                # default config
    python scripts/train.py --config configs/running.yaml  # custom config
    python scripts/train.py --task running_flat             # preset factory
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from msk_mjx.envs.locomotion import LocomotionEnv
from msk_mjx.envs.task_config import PRESETS, TaskConfig
from msk_mjx.training.ppo import TrainConfig, train


def _build_task_config(cfg: dict) -> TaskConfig:
    """Build a TaskConfig from YAML ``task`` + ``rewards`` sections."""
    task_sec = cfg.get("task", {})
    rew_sec = cfg.get("rewards", {})
    merged = {**task_sec, **rew_sec}
    # Filter to only valid TaskConfig fields.
    valid = {f.name for f in TaskConfig.__dataclass_fields__.values()}
    return TaskConfig(**{k: v for k, v in merged.items() if k in valid})


def _build_train_config(cfg: dict) -> TrainConfig:
    """Build a TrainConfig from the YAML ``training`` section."""
    t = cfg.get("training", {})
    # Convert list → tuple for hidden layers.
    for k in ("policy_hidden_layers", "value_hidden_layers"):
        if k in t and isinstance(t[k], list):
            t[k] = tuple(t[k])
    valid = set(TrainConfig._fields)
    return TrainConfig(**{k: v for k, v in t.items() if k in valid})


def main():
    parser = argparse.ArgumentParser(description="Train MSK locomotion policy with PPO")
    parser.add_argument("--config", type=str, default="configs/locomotion.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--task", type=str, default=None,
                        choices=list(PRESETS.keys()),
                        help="Use a preset task config (overrides YAML task/rewards)")
    parser.add_argument("--n-envs", type=int, default=None,
                        help="Override number of parallel environments")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Override total training timesteps")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    args = parser.parse_args()

    # Load YAML.
    config_path = Path(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Task config.
    if args.task:
        task_config = PRESETS[args.task]()
        print(f"Using preset: {args.task}")
    else:
        task_config = _build_task_config(cfg)

    # Training config.
    train_config = _build_train_config(cfg)
    # Apply CLI overrides.
    overrides = {}
    if args.n_envs is not None:
        overrides["n_envs"] = args.n_envs
    if args.total_timesteps is not None:
        overrides["total_timesteps"] = args.total_timesteps
    if args.seed is not None:
        overrides["seed"] = args.seed
    if overrides:
        train_config = train_config._replace(**overrides)

    # Environment.
    model_path = cfg["env"]["model_path"]
    n_frames = cfg["env"].get("n_frames", 5)
    print(f"Model: {model_path}")
    print(f"Task:  target_vel={task_config.target_velocity} m/s, "
          f"fwd_w={task_config.forward_reward_weight}, "
          f"vel_track_w={task_config.velocity_tracking_weight}")

    env = LocomotionEnv(model_path, task_config=task_config, n_frames=n_frames)

    # Train.
    results = train(env, train_config)
    print(f"\nFinal avg reward: {results['metrics_history'][-1]['avg_reward']:.3f}")


if __name__ == "__main__":
    main()
