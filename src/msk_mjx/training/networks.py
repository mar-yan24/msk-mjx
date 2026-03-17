"""Policy and value network definitions (Flax nn.Module).

Diagonal-Gaussian actor with learned log-std and a separate value head.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn


class PolicyNetwork(nn.Module):
    """MLP → action mean, with a separate learned log_std parameter."""

    action_size: int
    hidden_sizes: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        for h in self.hidden_sizes:
            x = nn.Dense(h)(x)
            x = nn.tanh(x)
        mean = nn.Dense(self.action_size)(x)
        log_std = self.param(
            "log_std",
            nn.initializers.zeros,
            (self.action_size,),
        )
        return mean, log_std


class ValueNetwork(nn.Module):
    """MLP → scalar value estimate."""

    hidden_sizes: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for h in self.hidden_sizes:
            x = nn.Dense(h)(x)
            x = nn.tanh(x)
        return nn.Dense(1)(x).squeeze(-1)


class ActorCritic(nn.Module):
    """Combined policy + value network (shared input, separate heads)."""

    action_size: int
    policy_hidden: Sequence[int] = (256, 256)
    value_hidden: Sequence[int] = (256, 256)

    def setup(self):
        self.policy = PolicyNetwork(self.action_size, self.policy_hidden)
        self.value = ValueNetwork(self.value_hidden)

    def __call__(self, x: jnp.ndarray):
        return self.policy(x), self.value(x)


def make_networks(
    obs_size: int,
    action_size: int,
    policy_layers: Sequence[int] = (256, 256),
    value_layers: Sequence[int] = (256, 256),
    rng: jax.Array | None = None,
) -> tuple[ActorCritic, dict]:
    """Create an ActorCritic and initialise its parameters.

    Returns:
        (network, params) tuple.
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)
    net = ActorCritic(
        action_size=action_size,
        policy_hidden=policy_layers,
        value_hidden=value_layers,
    )
    dummy = jnp.zeros(obs_size)
    params = net.init(rng, dummy)
    return net, params
