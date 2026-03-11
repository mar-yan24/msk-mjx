"""MJX pipeline with swappable muscle-tendon actuation.

For rigid tendons this is a thin wrapper around ``mjx.step``.
For elastic tendons the actuation stage is replaced with a custom
MTU solver while kinematics, constraints, and integration stay in MJX.
"""

from __future__ import annotations

import jax.numpy as jnp
from mujoco import mjx

from msk_mjx.muscle.base import MTUSolver, MTUState


def init(sys: mjx.Model, qpos: jnp.ndarray, qvel: jnp.ndarray) -> mjx.Data:
    """Create and forward-initialise an MJX physics state."""
    data = mjx.make_data(sys)
    data = data.replace(qpos=qpos, qvel=qvel)
    data = mjx.forward(sys, data)
    return data


def step_rigid(sys: mjx.Model, data: mjx.Data, ctrl: jnp.ndarray) -> mjx.Data:
    """Standard MJX step — rigid tendons via built-in muscle actuators."""
    data = data.replace(ctrl=ctrl)
    return mjx.step(sys, data)


def step_elastic(
    sys: mjx.Model,
    data: mjx.Data,
    mtu_state: MTUState,
    excitation: jnp.ndarray,
    solver: MTUSolver,
    dt: float,
) -> tuple[mjx.Data, MTUState]:
    """Physics step with elastic tendon actuation.

    To properly replace only the actuation stage you need access to
    individual MJX pipeline functions.  The implementation should be::

        data = mjx._src.forward.fwd_position(sys, data)
        data = mjx._src.forward.fwd_velocity(sys, data)
        data, state = solver.apply(sys, data, mtu_state, excitation, dt)
        data = mjx._src.forward.fwd_acceleration(sys, data)
        data = mjx._src.constraint.fwd_constraint(sys, data)
        data = mjx._src.euler.euler(sys, data)

    The exact import paths depend on your ``mujoco-mjx`` version.
    See https://github.com/google-deepmind/mujoco/tree/main/mjx
    """
    raise NotImplementedError(
        "Elastic pipeline step requires splitting MJX into individual "
        "stages. See docstring for the implementation outline."
    )
