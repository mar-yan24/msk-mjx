"""Muscle-tendon parameter extraction from MuJoCo models."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import mujoco


class MuscleParameters(NamedTuple):
    """Per-muscle parameters.  All arrays have shape (n_muscles,)."""

    max_isometric_force: jnp.ndarray    # F_max  (N)
    optimal_fiber_length: jnp.ndarray   # l_m_opt  (m)
    tendon_slack_length: jnp.ndarray    # l_t_slack  (m)
    pennation_angle: jnp.ndarray        # alpha_opt  (rad)
    max_contraction_velocity: jnp.ndarray  # v_max  (l_opt/s), typically 10


def from_mjmodel(model: mujoco.MjModel) -> MuscleParameters:
    """Extract muscle parameters from a MuJoCo model.

    MuJoCo stores muscle-specific parameters in ``actuator_gainprm``
    and ``actuator_biasprm``.  The exact layout depends on how the
    model was authored (native MuJoCo XML vs. converted from OpenSim).

    This function provides a *skeleton* — you will likely need to
    adjust the field mappings for your particular model.

    Args:
        model: MuJoCo model with ``muscle`` type actuators.

    Returns:
        MuscleParameters with one entry per actuator.
    """
    n = model.nu

    # MuJoCo muscle actuators store force range in gainprm[0:2].
    # The gain (force scale) is in actuator_gear[:,0] for general actuators.
    # Adapt these mappings to your model's convention.
    force = jnp.array(model.actuator_gear[:n, 0])

    return MuscleParameters(
        max_isometric_force=jnp.abs(force),
        optimal_fiber_length=jnp.ones(n),            # placeholder
        tendon_slack_length=jnp.ones(n),              # placeholder
        pennation_angle=jnp.zeros(n),                 # placeholder
        max_contraction_velocity=jnp.full(n, 10.0),   # default 10 l_opt/s
    )
