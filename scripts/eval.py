#!/usr/bin/env python
"""Evaluate a trained PPO checkpoint on a musculoskeletal locomotion env.

Usage:
    python scripts/eval.py --checkpoint logs/checkpoints/ppo_final.pkl
    python scripts/eval.py --checkpoint logs/checkpoints/ppo_final.pkl --n-episodes 10
    python scripts/eval.py --checkpoint logs/checkpoints/ppo_final.pkl --viewer
"""

from __future__ import annotations

import argparse
import functools
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import yaml

from msk_mjx.envs.locomotion import LocomotionEnv
from msk_mjx.envs.task_config import TaskConfig
from msk_mjx.training.networks import make_networks
from msk_mjx.training.normalization import NormalizerState, normalize


def load_checkpoint(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def run_eval(
    env: LocomotionEnv,
    params: dict,
    network,
    norm_state: NormalizerState,
    n_episodes: int = 5,
    max_steps: int = 1000,
    normalize_obs: bool = True,
):
    """Run deterministic evaluation episodes and print metrics."""

    @jax.jit
    def policy_step(params, norm_state, obs):
        obs_norm = normalize(obs, norm_state) if normalize_obs else obs
        (mean, _log_std), value = network.apply(params, obs_norm)
        # Deterministic: use mean action.
        return jnp.clip(mean, 0.0, 1.0), value

    rng = jax.random.PRNGKey(42)
    all_rewards = []
    all_lengths = []

    for ep in range(n_episodes):
        rng, rng_reset = jax.random.split(rng)
        state = jax.jit(env.reset)(rng_reset)
        total_reward = 0.0
        step = 0

        for step in range(max_steps):
            action, _ = policy_step(params, norm_state, state.obs)
            state = jax.jit(env.step)(state, action)
            total_reward += float(state.reward)
            if float(state.done) > 0.5:
                break

        all_rewards.append(total_reward)
        all_lengths.append(step + 1)
        print(f"  Episode {ep+1}: reward={total_reward:.2f}, length={step+1}")

    avg_r = sum(all_rewards) / len(all_rewards)
    avg_l = sum(all_lengths) / len(all_lengths)
    print(f"\nAverage over {n_episodes} episodes: reward={avg_r:.2f}, length={avg_l:.1f}")
    return all_rewards, all_lengths


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained MSK policy")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pkl checkpoint file")
    parser.add_argument("--config", type=str, default="configs/locomotion.yaml",
                        help="Path to YAML config (for env setup)")
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--viewer", action="store_true",
                        help="Launch MuJoCo viewer (requires display)")
    args = parser.parse_args()

    # Load checkpoint.
    ckpt = load_checkpoint(args.checkpoint)
    params = ckpt["params"]
    norm_state = NormalizerState(**ckpt["normalizer"])
    obs_size = ckpt["obs_size"]
    action_size = ckpt["action_size"]
    ckpt_config = ckpt.get("config", {})

    # Rebuild network.
    policy_layers = tuple(ckpt_config.get("policy_hidden_layers", [256, 256]))
    value_layers = tuple(ckpt_config.get("value_hidden_layers", [256, 256]))
    network, _ = make_networks(obs_size, action_size, policy_layers, value_layers)

    # Build env from YAML.
    config_path = Path(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_path = cfg["env"]["model_path"]
    n_frames = cfg["env"].get("n_frames", 5)

    # Reconstruct task config from checkpoint's stored config.
    task_sec = cfg.get("task", {})
    rew_sec = cfg.get("rewards", {})
    merged = {**task_sec, **rew_sec}
    valid = {f.name for f in TaskConfig.__dataclass_fields__.values()}
    task_config = TaskConfig(**{k: v for k, v in merged.items() if k in valid})

    env = LocomotionEnv(model_path, task_config=task_config, n_frames=n_frames)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Model: {model_path}, obs={obs_size}, act={action_size}")
    print(f"Task: vel={task_config.target_velocity} m/s")

    normalize_obs = ckpt_config.get("normalize_observations", True)
    run_eval(env, params, network, norm_state,
             n_episodes=args.n_episodes, max_steps=args.max_steps,
             normalize_obs=normalize_obs)

    if args.viewer:
        print("\nViewer mode not yet integrated — use scripts/play.py with loaded policy.")


if __name__ == "__main__":
    main()
