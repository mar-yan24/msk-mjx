"""Elastic tendon MTU solver — Millard 2012 with De Groote 2016 curves.

Replaces MJX's built-in rigid-tendon actuation with a custom Hill-type
muscle model that tracks fiber length and velocity as explicit state.

Pipeline integration point:
    MJX fwd_position → fwd_velocity → **this solver** → fwd_acceleration → constraint → euler

The solver uses semi-implicit Euler with sub-cycling (multiple MTU
substeps per physics step) for stability.

References:
    Millard et al. (2013). Flexing Computational Muscle: Modeling and
    Simulation of Musculotendon Dynamics. ASME J Biomech Eng, 135(2).

    De Groote et al. (2016). Evaluation of Direct Collocation Optimal
    Control Problem Formulations for Solving the Muscle Redundancy Problem.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from mujoco import mjx

from msk_mjx.muscle.activation import activation_step
from msk_mjx.muscle.base import MTUState


class ElasticTendonSolver:
    """Elastic tendon MTU solver.

    Tracks three states per muscle:
        - activation  (a):   first-order excitation→activation dynamics
        - fiber_length (l_m): integrated from force-balance ODE
        - fiber_velocity (v_m): from implicit solve of Hill equilibrium

    Force computation:
        F_tendon = f_t(l_t_norm) * F_max
        F_muscle = [a * f_act(l_m_norm) * f_v(v_m_norm) + f_pass(l_m_norm)] * F_max
        Equilibrium: F_tendon = F_muscle * cos(pennation_angle)

    Satisfies the :class:`MTUSolver` protocol.
    """

    def __init__(
        self,
        n_muscles: int,
        n_substeps: int = 4,
        tau_act: float = 0.015,
        tau_deact: float = 0.060,
    ):
        self._n_muscles = n_muscles
        self._n_substeps = n_substeps
        self._tau_act = tau_act
        self._tau_deact = tau_deact

    @property
    def n_muscles(self) -> int:
        return self._n_muscles

    def init_state(self, sys: mjx.Model, data: mjx.Data) -> MTUState:
        n = self._n_muscles
        return MTUState(
            activation=jnp.full(n, 0.05),
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
        """Compute muscle forces with elastic tendon dynamics.

        Implementation steps (all TODO):
            1. Read current MTU lengths / moment arms from ``data``
               (via ``data.ten_length`` or actuator length sensors).
            2. Sub-cycle with ``jax.lax.scan``:
               a. ``activation_step(excitation, a, dt_sub)``
               b. Tendon length = MTU length − fiber_length * cos(pennation)
               c. Tendon force  = ``tendon_force_length(l_t / l_t_slack) * F_max``
               d. Solve fiber velocity from Hill force balance
               e. Integrate fiber length (semi-implicit Euler)
            3. Compute final muscle forces.
            4. Map to generalized forces via moment arms and write to
               ``data.qfrc_applied``.

        The sub-cycling loop should look like::

            def substep(carry, _):
                a, l_m, v_m = carry
                ...
                return (a, l_m, v_m), None

            (a, l_m, v_m), _ = jax.lax.scan(
                substep, (state.activation, state.fiber_length,
                          state.fiber_velocity),
                xs=None, length=self._n_substeps,
            )
        """
        raise NotImplementedError(
            "ElasticTendonSolver.apply() not yet implemented. "
            "See docstring for the implementation outline."
        )
