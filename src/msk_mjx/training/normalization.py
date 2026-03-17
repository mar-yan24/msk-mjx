"""Online observation normalization (Welford's algorithm).

All state is stored as plain JAX arrays inside a NamedTuple so that the
normalizer is fully compatible with ``jax.jit`` and ``jax.vmap``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class NormalizerState(NamedTuple):
    """Running statistics for observation normalization."""

    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray  # scalar, float for numerical stability


def init_normalizer(obs_size: int) -> NormalizerState:
    """Create zeroed normalizer state."""
    return NormalizerState(
        mean=jnp.zeros(obs_size),
        var=jnp.ones(obs_size),
        count=jnp.float32(1e-4),
    )


def update_normalizer(
    state: NormalizerState,
    batch: jnp.ndarray,
) -> NormalizerState:
    """Update running stats with a batch of observations.

    Uses the parallel form of Welford's algorithm for combining
    the existing statistics with a new batch.

    Args:
        state: Current normalizer state.
        batch: Observation batch, shape ``(B, obs_size)``.
    """
    batch_mean = jnp.mean(batch, axis=0)
    batch_var = jnp.var(batch, axis=0)
    batch_count = jnp.float32(batch.shape[0])

    delta = batch_mean - state.mean
    total_count = state.count + batch_count

    new_mean = state.mean + delta * (batch_count / total_count)
    m_a = state.var * state.count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / total_count
    new_var = m2 / total_count

    return NormalizerState(mean=new_mean, var=new_var, count=total_count)


def normalize(
    obs: jnp.ndarray,
    state: NormalizerState,
    epsilon: float = 1e-8,
) -> jnp.ndarray:
    """Normalize observations using running statistics."""
    return (obs - state.mean) / jnp.sqrt(state.var + epsilon)
