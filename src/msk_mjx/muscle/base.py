"""Base types and protocols for muscle-tendon unit solvers."""

from __future__ import annotations

from typing import NamedTuple, Protocol

import jax.numpy as jnp
from mujoco import mjx


class MTUState(NamedTuple):
    """State tracked by muscle-tendon unit solvers.

    For rigid tendons only ``activation`` is meaningful.
    For elastic tendons all three fields are integrated each substep.
    """

    activation: jnp.ndarray      # (n_muscles,) in [0, 1]
    fiber_length: jnp.ndarray    # (n_muscles,) normalized by optimal length
    fiber_velocity: jnp.ndarray  # (n_muscles,) normalized by v_max


class MTUSolver(Protocol):
    """Interface that both rigid and elastic tendon solvers implement.

    All methods must be pure functions compatible with jax.jit / jax.vmap.
    """

    @property
    def n_muscles(self) -> int: ...

    def init_state(self, sys: mjx.Model, data: mjx.Data) -> MTUState:
        """Return a zeroed-out MTU state matching the model."""
        ...

    def apply(
        self,
        sys: mjx.Model,
        data: mjx.Data,
        state: MTUState,
        excitation: jnp.ndarray,
        dt: float,
    ) -> tuple[mjx.Data, MTUState]:
        """Compute muscle forces and return (updated data, new state).

        * Rigid solver: writes ``data.ctrl`` and delegates to MJX.
        * Elastic solver: computes forces via Hill model, writes
          ``data.qfrc_applied``, and integrates fiber states.
        """
        ...
