"""First-order muscle activation dynamics (Winters 1995 / Thelen 2003)."""

import jax.numpy as jnp


def activation_step(
    excitation: jnp.ndarray,
    activation: jnp.ndarray,
    dt: float,
    tau_act: float = 0.015,
    tau_deact: float = 0.060,
) -> jnp.ndarray:
    """Update muscle activation using first-order ODE.

    da/dt = (u - a) / tau(u, a)

    where tau = tau_act when excitation > activation (activating),
    tau_deact otherwise (deactivating). tau_act < tau_deact reflects
    the physiological asymmetry: muscles activate faster than they relax.

    Args:
        excitation: Neural excitation signal, shape (n_muscles,), in [0, 1].
        activation: Current activation, shape (n_muscles,), in [0, 1].
        dt: Time step in seconds.
        tau_act: Activation time constant (default 15 ms).
        tau_deact: Deactivation time constant (default 60 ms).

    Returns:
        Updated activation, shape (n_muscles,), clamped to [0, 1].
    """
    tau = jnp.where(excitation > activation, tau_act, tau_deact)
    a_next = activation + (excitation - activation) * (dt / tau)
    return jnp.clip(a_next, 0.0, 1.0)
