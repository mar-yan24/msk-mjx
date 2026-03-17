"""Observation builders for musculoskeletal environments."""

import jax.numpy as jnp
from mujoco import mjx


def get_observations(data: mjx.Data) -> jnp.ndarray:
    """Default observation: joint state + muscle activations.

    Omits global x/y translation for translation invariance.
    """
    return jnp.concatenate([
        data.qpos[2:],   # joint positions (skip root x, y)
        data.qvel,       # joint velocities
        data.act,        # muscle activation states
    ])


def get_proprioceptive_obs(data: mjx.Data) -> jnp.ndarray:
    """Joint-only observations (no muscle activation feedback)."""
    return jnp.concatenate([
        data.qpos[2:],
        data.qvel,
    ])


def get_full_observations(
    data: mjx.Data,
    target_velocity: float,
) -> jnp.ndarray:
    """Task-aware observation: default obs + target velocity scalar.

    The velocity target is always appended (even when zero) so that the
    observation size is consistent across task configs.
    """
    return jnp.concatenate([
        data.qpos[2:],
        data.qvel,
        data.act,
        jnp.array([target_velocity]),
    ])
