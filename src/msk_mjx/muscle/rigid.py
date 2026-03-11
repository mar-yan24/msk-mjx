"""Rigid tendon solver — delegates entirely to MJX's built-in muscle model."""

import jax.numpy as jnp
from mujoco import mjx

from msk_mjx.muscle.base import MTUState


class RigidTendonSolver:
    """Pass-through solver for rigid-tendon muscle actuators.

    MuJoCo already implements activation dynamics and force-length /
    force-velocity curves for its ``muscle`` actuator type.  This solver
    simply maps excitation signals to ``data.ctrl`` and lets MJX do the rest.

    Satisfies the :class:`MTUSolver` protocol.
    """

    def __init__(self, n_muscles: int):
        self._n_muscles = n_muscles

    @property
    def n_muscles(self) -> int:
        return self._n_muscles

    def init_state(self, sys: mjx.Model, data: mjx.Data) -> MTUState:
        n = self._n_muscles
        return MTUState(
            activation=jnp.zeros(n),
            fiber_length=jnp.ones(n),
            fiber_velocity=jnp.zeros(n),
        )

    def apply(
        self,
        sys: mjx.Model,
        data: mjx.Data,
        state: MTUState,
        excitation: jnp.ndarray,
        dt: float,
    ) -> tuple[mjx.Data, MTUState]:
        """Set ctrl = excitation and let MJX handle forces internally."""
        ctrl = jnp.clip(excitation, 0.0, 1.0)
        data = data.replace(ctrl=ctrl)
        # MJX updates data.act during mjx.step(); we mirror it afterward
        # in the pipeline so the env can read current activations.
        return data, state
