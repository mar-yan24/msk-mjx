"""De Groote 2016 analytical muscle curves.

Smooth, differentiable approximations of the Hill-type muscle curves.
All functions operate on normalized quantities and are compatible
with ``jax.grad``.

Reference:
    De Groote et al. (2016). Evaluation of Direct Collocation Optimal
    Control Problem Formulations for Solving the Muscle Redundancy
    Problem. Annals of Biomedical Engineering, 44(10), 2922-2936.
"""

import jax.numpy as jnp


def active_force_length(l_m_norm: jnp.ndarray) -> jnp.ndarray:
    """Active force-length relationship.

    Bell-shaped curve centered at ``l_m_norm = 1.0`` (optimal fiber length).

    Args:
        l_m_norm: Normalized muscle fiber length (l_m / l_m_opt).

    Returns:
        Active force multiplier in [0, 1].
    """
    raise NotImplementedError


def passive_force_length(l_m_norm: jnp.ndarray) -> jnp.ndarray:
    """Passive force-length relationship.

    Exponential curve producing force at long fiber lengths.

    Args:
        l_m_norm: Normalized muscle fiber length.

    Returns:
        Passive force multiplier (>= 0).
    """
    raise NotImplementedError


def force_velocity(v_m_norm: jnp.ndarray) -> jnp.ndarray:
    """Force-velocity relationship.

    Hyperbolic curve: force drops during shortening (concentric)
    and rises during lengthening (eccentric).

    Args:
        v_m_norm: Normalized fiber velocity (v_m / v_max).
            Negative = shortening, positive = lengthening.

    Returns:
        Force-velocity multiplier.
    """
    raise NotImplementedError


def tendon_force_length(l_t_norm: jnp.ndarray) -> jnp.ndarray:
    """Tendon force-length relationship.

    Exponential-linear curve: zero force below slack length,
    stiffening above.

    Args:
        l_t_norm: Normalized tendon length (l_t / l_t_slack).

    Returns:
        Tendon force multiplier (>= 0).
    """
    raise NotImplementedError
