"""Toggleable task configuration for locomotion environments.

Each parameter controls a reward term or environment behaviour.
Zero-weight terms are still computed (no JIT-unfriendly conditionals)
but contribute nothing to the total reward.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class TaskConfig:
    """Immutable task specification for LocomotionEnv."""

    # --- target ---
    target_velocity: float = 1.3  # m/s (walk ~1.3, run ~3.0, stand ~0.0)

    # --- reward weights ---
    forward_reward_weight: float = 1.25
    velocity_tracking_weight: float = 0.0  # >0 uses Gaussian speed-tracking
    healthy_reward: float = 5.0
    ctrl_cost_weight: float = 0.1
    energy_cost_weight: float = 0.0
    symmetry_reward_weight: float = 0.0
    joint_limit_penalty_weight: float = 0.0
    posture_reward_weight: float = 0.0  # upright trunk

    # --- termination ---
    min_z: float = 0.8
    max_z: float = 2.0

    # --- reset ---
    reset_noise_scale: float = 0.01


# ------------------------------------------------------------------
# Preset factories
# ------------------------------------------------------------------

def walking_flat() -> TaskConfig:
    """Default flat-ground walking at ~1.3 m/s."""
    return TaskConfig()


def running_flat() -> TaskConfig:
    """Fast running at ~3.0 m/s with velocity tracking."""
    return TaskConfig(
        target_velocity=3.0,
        forward_reward_weight=0.5,
        velocity_tracking_weight=1.5,
        healthy_reward=5.0,
        ctrl_cost_weight=0.05,
    )


def standing_balance() -> TaskConfig:
    """Stand still with upright posture."""
    return TaskConfig(
        target_velocity=0.0,
        forward_reward_weight=0.0,
        velocity_tracking_weight=1.0,
        healthy_reward=5.0,
        ctrl_cost_weight=0.1,
        posture_reward_weight=1.0,
    )


def energy_efficient() -> TaskConfig:
    """Walking with a metabolic energy penalty."""
    return TaskConfig(
        target_velocity=1.3,
        forward_reward_weight=1.25,
        energy_cost_weight=0.001,  # small: actuator_force * ctrl can be large
        ctrl_cost_weight=0.1,
    )


# Map preset name → factory for CLI convenience.
PRESETS: dict[str, callable] = {
    "walking_flat": walking_flat,
    "running_flat": running_flat,
    "standing_balance": standing_balance,
    "energy_efficient": energy_efficient,
}
