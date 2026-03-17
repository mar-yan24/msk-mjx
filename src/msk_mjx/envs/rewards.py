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


# ------------------------------------------------------------------
# Additional reward components for diverse locomotion tasks
# ------------------------------------------------------------------

def velocity_tracking(
    prev_com: jnp.ndarray,
    curr_com: jnp.ndarray,
    dt: float,
    target_vel: float,
    sigma: float = 0.25,
) -> jnp.ndarray:
    """Gaussian speed-tracking reward.

    Returns ``exp(-(v - target)^2 / sigma^2)`` so that the agent is
    rewarded most when matching the target velocity exactly.
    """
    vel = (curr_com[0] - prev_com[0]) / dt
    return jnp.exp(-jnp.square(vel - target_vel) / jnp.square(sigma))


def posture_reward(pelvis_tilt: jnp.ndarray, k: float = 4.0) -> jnp.ndarray:
    """Reward for keeping the trunk upright.

    ``pelvis_tilt`` is the pitch angle (rad) of the pelvis/trunk.
    Returns ``exp(-k * tilt^2)`` — peaks at zero tilt.
    """
    return jnp.exp(-k * jnp.square(pelvis_tilt))


def symmetry_reward(
    qpos: jnp.ndarray,
    qvel: jnp.ndarray,
    left_ids: jnp.ndarray,
    right_ids: jnp.ndarray,
) -> jnp.ndarray:
    """Negative squared asymmetry between left and right limbs.

    Returns ``-(||q_L - q_R||^2 + ||v_L - v_R||^2)`` so that perfectly
    symmetric motion scores zero (best).
    """
    q_diff = qpos[left_ids] - qpos[right_ids]
    v_diff = qvel[left_ids] - qvel[right_ids]
    return -(jnp.sum(jnp.square(q_diff)) + jnp.sum(jnp.square(v_diff)))


def joint_limit_penalty(
    qpos: jnp.ndarray,
    jnt_range: jnp.ndarray,
    jnt_limited: jnp.ndarray,
) -> jnp.ndarray:
    """Penalty for joints approaching or exceeding their limits.

    Computes the squared violation for each limited joint and returns
    the negative sum. ``jnt_range`` is shape ``(nj, 2)`` with (lo, hi),
    ``jnt_limited`` is a boolean mask of limited joints.

    Returns ``-sum(violation^2)`` where violation is the distance
    past the limit (zero when within range).
    """
    lo = jnt_range[:, 0]
    hi = jnt_range[:, 1]
    below = jnp.clip(lo - qpos, 0.0, None)
    above = jnp.clip(qpos - hi, 0.0, None)
    violation = below + above
    # Only count limited joints.
    violation = jnp.where(jnt_limited, violation, 0.0)
    return -jnp.sum(jnp.square(violation))
