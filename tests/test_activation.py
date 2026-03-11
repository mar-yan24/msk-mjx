"""Tests for first-order activation dynamics."""

import jax.numpy as jnp

from msk_mjx.muscle.activation import activation_step


def test_activation_rises_with_high_excitation():
    a = jnp.array([0.1])
    u = jnp.array([1.0])
    assert activation_step(u, a, dt=0.01)[0] > a[0]


def test_activation_falls_with_low_excitation():
    a = jnp.array([0.9])
    u = jnp.array([0.0])
    assert activation_step(u, a, dt=0.01)[0] < a[0]


def test_clamped_to_unit_interval():
    a = jnp.array([0.99])
    u = jnp.array([1.0])
    a_next = activation_step(u, a, dt=1.0)  # large dt forces overshoot
    assert 0.0 <= float(a_next[0]) <= 1.0


def test_equilibrium_unchanged():
    a = jnp.array([0.5])
    u = jnp.array([0.5])
    assert jnp.allclose(activation_step(u, a, dt=0.01), a, atol=1e-6)


def test_activation_faster_than_deactivation():
    a = jnp.array([0.5, 0.5])
    u = jnp.array([1.0, 0.0])
    a_next = activation_step(u, a, dt=0.01)
    rise = float(a_next[0] - 0.5)
    fall = float(0.5 - a_next[1])
    assert rise > fall  # tau_act (15 ms) < tau_deact (60 ms)
