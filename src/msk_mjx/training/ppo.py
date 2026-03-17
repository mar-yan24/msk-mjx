"""Custom PPO implementation in JAX.

Designed for GPU-batched musculoskeletal RL with MJX environments.
Uses ``jax.lax.scan`` for efficient rollout collection and Flax/optax
for network definition and optimisation.
"""

from __future__ import annotations

import functools
import pickle
import time
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from brax.envs.base import PipelineEnv, State

from msk_mjx.training.networks import ActorCritic, make_networks
from msk_mjx.training.normalization import (
    NormalizerState,
    init_normalizer,
    normalize,
    update_normalizer,
)


# ------------------------------------------------------------------
# Data containers
# ------------------------------------------------------------------

class Transition(NamedTuple):
    """Single-step transition collected during rollout."""

    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray


class TrainConfig(NamedTuple):
    """Hyperparameters for PPO training."""

    n_envs: int = 2048
    total_timesteps: int = 50_000_000
    episode_length: int = 1000
    unroll_length: int = 10
    learning_rate: float = 3e-4
    n_minibatches: int = 32
    n_epochs: int = 4
    discount: float = 0.97
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_cost: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_observations: bool = True
    reward_scaling: float = 1.0
    seed: int = 0
    num_evals: int = 10
    policy_hidden_layers: tuple[int, ...] = (256, 256)
    value_hidden_layers: tuple[int, ...] = (256, 256)
    checkpoint_dir: str = "logs/checkpoints"
    log_dir: str = "logs"


# ------------------------------------------------------------------
# Gaussian policy helpers
# ------------------------------------------------------------------

def _log_prob_gaussian(action: jnp.ndarray, mean: jnp.ndarray, log_std: jnp.ndarray):
    """Log probability under diagonal Gaussian."""
    var = jnp.exp(2.0 * log_std)
    log_p = -0.5 * (jnp.square(action - mean) / var + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
    return jnp.sum(log_p, axis=-1)


def _entropy_gaussian(log_std: jnp.ndarray):
    """Entropy of diagonal Gaussian."""
    return jnp.sum(0.5 * jnp.log(2.0 * jnp.pi * jnp.e) + log_std, axis=-1)


def _sample_action(rng, mean, log_std):
    """Sample from diagonal Gaussian."""
    std = jnp.exp(log_std)
    return mean + std * jax.random.normal(rng, mean.shape)


# ------------------------------------------------------------------
# GAE computation
# ------------------------------------------------------------------

def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    last_value: jnp.ndarray,
    discount: float,
    gae_lambda: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Generalised Advantage Estimation.

    Args:
        rewards: ``(T, B)`` rewards.
        values: ``(T, B)`` value estimates.
        dones: ``(T, B)`` episode termination flags.
        last_value: ``(B,)`` bootstrap value for final step.
        discount: Discount factor γ.
        gae_lambda: GAE-λ parameter.

    Returns:
        ``(advantages, returns)`` each of shape ``(T, B)``.
    """
    T = rewards.shape[0]

    def _scan_fn(carry, t):
        gae = carry
        not_done = 1.0 - dones[t]
        # For the last step, next value is last_value; otherwise values[t+1].
        next_val = jnp.where(t == T - 1, last_value, values[t + 1])
        delta = rewards[t] + discount * next_val * not_done - values[t]
        gae = delta + discount * gae_lambda * not_done * gae
        return gae, gae

    # Scan backwards through time.
    _, advantages = jax.lax.scan(
        _scan_fn,
        jnp.zeros_like(values[0]),
        jnp.arange(T - 1, -1, -1),
    )
    # Reverse back to chronological order.
    advantages = advantages[::-1]
    returns = advantages + values
    return advantages, returns


# ------------------------------------------------------------------
# PPO loss
# ------------------------------------------------------------------

def ppo_loss(
    params: Any,
    network: ActorCritic,
    batch: dict,
    clip_epsilon: float,
    entropy_cost: float,
    value_loss_coef: float,
):
    """Clipped PPO surrogate loss + value loss + entropy bonus.

    Returns (total_loss, metrics_dict).
    """
    (mean, log_std), value = network.apply(params, batch["obs"])

    # Policy loss.
    new_log_prob = _log_prob_gaussian(batch["action"], mean, log_std)
    ratio = jnp.exp(new_log_prob - batch["log_prob"])
    adv = batch["advantage"]
    # Normalise advantages.
    adv = (adv - jnp.mean(adv)) / (jnp.std(adv) + 1e-8)
    clipped_ratio = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    policy_loss = -jnp.mean(jnp.minimum(ratio * adv, clipped_ratio * adv))

    # Value loss.
    value_loss = 0.5 * jnp.mean(jnp.square(value - batch["return"]))

    # Entropy bonus.
    entropy = jnp.mean(_entropy_gaussian(log_std))

    total = policy_loss + value_loss_coef * value_loss - entropy_cost * entropy

    metrics = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approx_kl": jnp.mean(jnp.square(new_log_prob - batch["log_prob"])) * 0.5,
    }
    return total, metrics


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train(env: PipelineEnv, config: TrainConfig) -> dict:
    """Run full PPO training.

    Args:
        env: Brax ``PipelineEnv`` (will be vmapped across ``n_envs``).
        config: Hyperparameters.

    Returns:
        Dictionary with ``params``, ``normalizer``, and ``metrics_history``.
    """
    rng = jax.random.PRNGKey(config.seed)

    # Determine obs/action sizes from a trial reset.
    rng, rng_init = jax.random.split(rng)
    init_state = jax.jit(env.reset)(rng_init)
    obs_size = init_state.obs.shape[-1]
    action_size = env.action_size

    # Build networks.
    rng, rng_net = jax.random.split(rng)
    network, params = make_networks(
        obs_size, action_size,
        policy_layers=config.policy_hidden_layers,
        value_layers=config.value_hidden_layers,
        rng=rng_net,
    )

    # Optimiser.
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate),
    )
    opt_state = optimizer.init(params)

    # Observation normalizer.
    norm_state = init_normalizer(obs_size)

    # Batched env reset.
    rng, rng_reset = jax.random.split(rng)
    reset_keys = jax.random.split(rng_reset, config.n_envs)
    env_state = jax.jit(jax.vmap(env.reset))(reset_keys)

    # ---- Rollout collection (vmapped over envs, scanned over time) ----
    @jax.jit
    def collect_rollout(params, norm_state, env_state, rng):
        """Collect ``unroll_length`` steps across all envs."""

        def _step(carry, _):
            env_st, rng = carry
            rng, rng_act = jax.random.split(rng)

            obs = env_st.obs
            obs_norm = normalize(obs, norm_state) if config.normalize_observations else obs

            (mean, log_std), value = jax.vmap(
                functools.partial(network.apply, params)
            )(obs_norm)

            action = jax.vmap(_sample_action)(
                jax.random.split(rng_act, config.n_envs), mean, log_std
            )
            log_prob = jax.vmap(_log_prob_gaussian)(action, mean, log_std)

            # Clip actions to [0, 1] for muscle excitations.
            action_clipped = jnp.clip(action, 0.0, 1.0)
            next_env_st = jax.vmap(env.step)(env_st, action_clipped)

            transition = Transition(
                obs=obs_norm,
                action=action,
                reward=next_env_st.reward * config.reward_scaling,
                done=next_env_st.done,
                log_prob=log_prob,
                value=value,
            )
            return (next_env_st, rng), transition

        (env_state_new, _), trajectory = jax.lax.scan(
            _step, (env_state, rng), None, length=config.unroll_length
        )
        return env_state_new, trajectory

    # ---- PPO update ----
    @jax.jit
    def update_step(params, opt_state, norm_state, env_state, rng):
        """One PPO iteration: collect → GAE → minibatch updates."""
        rng, rng_rollout = jax.random.split(rng)
        env_state, trajectory = collect_rollout(
            params, norm_state, env_state, rng_rollout
        )

        # Update normalizer with new observations.
        all_obs = trajectory.obs.reshape(-1, obs_size)
        new_norm_state = update_normalizer(norm_state, all_obs) if config.normalize_observations else norm_state

        # Bootstrap value for GAE.
        last_obs = env_state.obs
        last_obs_norm = normalize(last_obs, new_norm_state) if config.normalize_observations else last_obs
        _, last_value = jax.vmap(
            functools.partial(network.apply, params)
        )(last_obs_norm)

        advantages, returns = compute_gae(
            trajectory.reward, trajectory.value, trajectory.done,
            last_value, config.discount, config.gae_lambda,
        )

        # Flatten (T, B) → (T*B,).
        T, B = trajectory.obs.shape[:2]
        flat = {
            "obs": trajectory.obs.reshape(T * B, -1),
            "action": trajectory.action.reshape(T * B, -1),
            "log_prob": trajectory.log_prob.reshape(T * B),
            "advantage": advantages.reshape(T * B),
            "return": returns.reshape(T * B),
        }

        # Minibatch SGD for n_epochs.
        batch_size = T * B
        minibatch_size = batch_size // config.n_minibatches

        def _epoch(carry, rng_epoch):
            params, opt_state = carry
            perm = jax.random.permutation(rng_epoch, batch_size)
            shuffled = jax.tree.map(lambda x: x[perm], flat)

            def _minibatch(carry, start):
                params, opt_state = carry
                mb = jax.tree.map(lambda x: jax.lax.dynamic_slice_in_dim(x, start, minibatch_size), shuffled)
                grads, metrics = jax.grad(ppo_loss, has_aux=True)(
                    params, network, mb,
                    config.clip_epsilon, config.entropy_cost, config.value_loss_coef,
                )
                updates, opt_state_new = optimizer.update(grads, opt_state, params)
                params_new = optax.apply_updates(params, updates)
                return (params_new, opt_state_new), metrics

            starts = jnp.arange(0, batch_size, minibatch_size)[:config.n_minibatches]
            (params, opt_state), _ = jax.lax.scan(_minibatch, (params, opt_state), starts)
            return (params, opt_state), None

        rng, rng_epochs = jax.random.split(rng)
        epoch_keys = jax.random.split(rng_epochs, config.n_epochs)
        (params, opt_state), _ = jax.lax.scan(_epoch, (params, opt_state), epoch_keys)

        avg_reward = jnp.mean(trajectory.reward)
        return params, opt_state, new_norm_state, env_state, rng, avg_reward

    # ---- Main training loop ----
    total_steps = config.total_timesteps
    steps_per_iter = config.unroll_length * config.n_envs
    n_iters = total_steps // steps_per_iter
    eval_every = max(1, n_iters // config.num_evals)

    metrics_history: list[dict] = []
    print(f"PPO training: {n_iters} iterations, {steps_per_iter} steps/iter, "
          f"{total_steps} total steps")
    print(f"  obs_size={obs_size}, action_size={action_size}, n_envs={config.n_envs}")

    t_start = time.time()
    for i in range(n_iters):
        params, opt_state, norm_state, env_state, rng, avg_reward = update_step(
            params, opt_state, norm_state, env_state, rng
        )

        if (i + 1) % eval_every == 0 or i == 0:
            elapsed = time.time() - t_start
            steps_done = (i + 1) * steps_per_iter
            sps = steps_done / elapsed if elapsed > 0 else 0
            avg_r = float(avg_reward)
            metrics_history.append({
                "iteration": i + 1,
                "steps": steps_done,
                "avg_reward": avg_r,
                "sps": sps,
            })
            print(f"  iter {i+1:>6d} | steps {steps_done:>10d} | "
                  f"reward {avg_r:>8.3f} | SPS {sps:>8.0f}")

    elapsed = time.time() - t_start
    print(f"Training complete in {elapsed:.1f}s")

    # Save checkpoint.
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "ppo_final.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump({
            "params": jax.device_get(params),
            "normalizer": jax.device_get(norm_state),
            "config": config._asdict(),
            "obs_size": obs_size,
            "action_size": action_size,
        }, f)
    print(f"Checkpoint saved to {ckpt_path}")

    return {
        "params": params,
        "normalizer": norm_state,
        "metrics_history": metrics_history,
    }
