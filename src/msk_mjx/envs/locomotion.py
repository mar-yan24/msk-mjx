"""Musculoskeletal locomotion environment.

Wraps a MuJoCo model with muscle actuators in a Brax-compatible
``PipelineEnv`` for GPU-batched RL training via MJX.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import mujoco
from brax.envs.base import PipelineEnv, State
from mujoco import mjx

from msk_mjx.envs import observations as obs_fn
from msk_mjx.envs import rewards as rew_fn
from msk_mjx.muscle.base import MTUSolver
from msk_mjx.muscle.rigid import RigidTendonSolver


class LocomotionEnv(PipelineEnv):
    """Walk forward using muscle excitations as actions.

    Works out-of-the-box with rigid tendons (MJX built-in muscle model).
    Pass a custom ``MTUSolver`` to switch to elastic tendons once
    the solver is implemented.

    Actions:
        Muscle excitation signals, shape ``(n_muscles,)``, in [0, 1].

    Observations (default):
        Joint positions (minus root x/y), joint velocities, muscle
        activations — concatenated into a single vector.
    """

    def __init__(
        self,
        model_path: str,
        *,
        solver: MTUSolver | None = None,
        n_frames: int = 5,
        forward_reward_weight: float = 1.25,
        healthy_reward: float = 5.0,
        ctrl_cost_weight: float = 0.1,
        min_z: float = 0.8,
        max_z: float = 2.0,
        reset_noise_scale: float = 0.01,
    ):
        mj_model = mujoco.MjModel.from_xml_path(model_path)
        sys = mjx.put_model(mj_model)
        super().__init__(sys, backend="mjx", n_frames=n_frames)

        self._mj_model = mj_model
        self._solver = solver or RigidTendonSolver(n_muscles=mj_model.nu)
        self._forward_reward_weight = forward_reward_weight
        self._healthy_reward = healthy_reward
        self._ctrl_cost_weight = ctrl_cost_weight
        self._min_z = min_z
        self._max_z = max_z
        self._reset_noise_scale = reset_noise_scale

    # ------------------------------------------------------------------
    # PipelineEnv interface
    # ------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> State:
        rng, rng_q, rng_qd = jax.random.split(rng, 3)

        qpos = self.sys.qpos0 + self._reset_noise_scale * jax.random.normal(
            rng_q, (self.sys.nq,)
        )
        qvel = self._reset_noise_scale * jax.random.normal(
            rng_qd, (self.sys.nv,)
        )

        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = obs_fn.get_observations(pipeline_state)

        reward, done = jnp.zeros(2)
        metrics = {
            "forward_reward": jnp.float32(0),
            "healthy_reward": jnp.float32(0),
            "ctrl_cost": jnp.float32(0),
        }

        return State(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
        )

    def step(self, state: State, action: jax.Array) -> State:
        action = jnp.clip(action, 0.0, 1.0)

        prev_com = state.pipeline_state.subtree_com[0]
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        curr_com = pipeline_state.subtree_com[0]

        # --- rewards ---
        fwd = self._forward_reward_weight * rew_fn.forward_velocity(
            prev_com, curr_com, self.dt
        )
        cost = -self._ctrl_cost_weight * rew_fn.ctrl_cost(action)

        com_z = pipeline_state.subtree_com[0, 2]
        is_healthy = rew_fn.healthy(com_z, self._min_z, self._max_z)
        hlth = self._healthy_reward * is_healthy

        reward = fwd + hlth + cost
        done = 1.0 - is_healthy

        obs = obs_fn.get_observations(pipeline_state)
        metrics = {
            "forward_reward": fwd,
            "healthy_reward": hlth,
            "ctrl_cost": cost,
        }

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
        )
