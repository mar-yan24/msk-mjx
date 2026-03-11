"""Reward component functions for locomotion tasks.

Each function computes a single scalar reward term.
The environment combines them with configurable weights.
"""

import jax.numpy as jnp


def forward_velocity(
    prev_com: jnp.ndarray,
    curr_com: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """Reward proportional to forward (x-axis) centre-of-mass velocity."""
    return (curr_com[0] - prev_com[0]) / dt


def healthy(com_z: jnp.ndarray, min_z: float, max_z: float) -> jnp.ndarray:
    """Binary reward for keeping the CoM height within bounds."""
    return jnp.where((com_z > min_z) & (com_z < max_z), 1.0, 0.0)


def ctrl_cost(action: jnp.ndarray) -> jnp.ndarray:
    """Quadratic control cost (sum of squared excitations)."""
    return jnp.sum(jnp.square(action))


def mechanical_work(
    force: jnp.ndarray,
    velocity: jnp.ndarray,
) -> jnp.ndarray:
    """Simplified metabolic proxy: total |F * v| across muscles.

    For a proper metabolic model use Umberger (2010) or Bhargava (2004).
    """
    return jnp.sum(jnp.abs(force * velocity))
