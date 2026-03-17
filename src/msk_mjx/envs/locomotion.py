"""Musculoskeletal locomotion environment.

Wraps a MuJoCo model with muscle actuators in a Brax-compatible
``PipelineEnv`` for GPU-batched RL training via MJX.

Supports diverse task configurations (walking, running, standing, etc.)
through ``TaskConfig`` — all reward terms are always computed and gated
by their weights so that JIT compilation sees no data-dependent branches.
"""

from __future__ import annotations

import re
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
from brax.envs.base import PipelineEnv, State
from mujoco import mjx

from msk_mjx.envs import observations as obs_fn
from msk_mjx.envs import rewards as rew_fn
from msk_mjx.envs.task_config import TaskConfig
from msk_mjx.muscle.base import MTUSolver
from msk_mjx.muscle.rigid import RigidTendonSolver

# Foot geom names whose meshes get converted to spheres for ground contact.
_FOOT_GEOMS = {
    "calcn_r_geom_1", "calcn_l_geom_1",
    "toes_r_geom_1", "toes_l_geom_1",
}


def _load_xml_mjx_safe(xml_path: str) -> mujoco.MjModel:
    """Load a MuJoCo XML, stripping MJX-unsupported sensors.

    Compiles with the correct working directory so that ``<include>``
    and mesh file references resolve properly.
    """
    xml_path = Path(xml_path).resolve()
    xml = xml_path.read_text()
    # Remove sensors MJX can't handle.
    xml = re.sub(r'<jointlimitfrc\b[^/]*/>', '', xml)
    xml = re.sub(r'<touch\b[^/]*/>', '', xml)

    # Compile with VFS so includes/assets resolve relative to the XML dir.
    import os
    old_cwd = os.getcwd()
    try:
        os.chdir(xml_path.parent)
        mj_model = mujoco.MjModel.from_xml_string(xml)
    finally:
        os.chdir(old_cwd)
    return mj_model


def _patch_geoms(mj_model: mujoco.MjModel) -> None:
    """In-place geom fixes for MJX compatibility.

    * Disables hfield geoms (no hfield collision in MJX).
    * Converts foot mesh geoms to spheres (plane-mesh not supported).
    * Disables collision on all other mesh geoms (visual only).
    """
    for i in range(mj_model.ngeom):
        gname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if mj_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_HFIELD:
            mj_model.geom_contype[i] = 0
            mj_model.geom_conaffinity[i] = 0
        elif mj_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            if gname in _FOOT_GEOMS:
                mj_model.geom_type[i] = mujoco.mjtGeom.mjGEOM_SPHERE
                mj_model.geom_size[i, 0] = 0.03
            else:
                mj_model.geom_contype[i] = 0
                mj_model.geom_conaffinity[i] = 0


class LocomotionEnv(PipelineEnv):
    """Locomotion with muscle excitations and configurable reward shaping.

    Works out-of-the-box with rigid tendons (MJX built-in muscle model).
    Pass a custom ``MTUSolver`` to switch to elastic tendons once
    the solver is implemented.

    Actions:
        Muscle excitation signals, shape ``(n_muscles,)``, in [0, 1].

    Observations:
        Joint positions (minus root x/y), joint velocities, muscle
        activations, and target velocity — concatenated into a single vector.
    """

    def __init__(
        self,
        model_path: str,
        *,
        task_config: TaskConfig | None = None,
        solver: MTUSolver | None = None,
        n_frames: int = 5,
        # Legacy kwargs for backward compatibility — ignored when task_config is set.
        forward_reward_weight: float = 1.25,
        healthy_reward: float = 5.0,
        ctrl_cost_weight: float = 0.1,
        min_z: float = 0.8,
        max_z: float = 2.0,
        reset_noise_scale: float = 0.01,
    ):
        mj_model = _load_xml_mjx_safe(model_path)
        _patch_geoms(mj_model)

        sys = mjx.put_model(mj_model)
        super().__init__(sys, backend="mjx", n_frames=n_frames)

        self._mj_model = mj_model
        self._solver = solver or RigidTendonSolver(n_muscles=mj_model.nu)

        # Use TaskConfig if provided, otherwise build one from legacy kwargs.
        if task_config is not None:
            self._cfg = task_config
        else:
            self._cfg = TaskConfig(
                forward_reward_weight=forward_reward_weight,
                healthy_reward=healthy_reward,
                ctrl_cost_weight=ctrl_cost_weight,
                min_z=min_z,
                max_z=max_z,
                reset_noise_scale=reset_noise_scale,
            )

        # Pre-compute joint limit arrays for joint-limit penalty.
        self._jnt_range = jnp.array(mj_model.jnt_range)
        self._jnt_limited = jnp.array(mj_model.jnt_limited, dtype=jnp.bool_)

        # Left/right joint index pairs for symmetry reward.
        # In the 22-muscle 2D model the generalized coords (after root x,y,θ)
        # are ordered: [hip_r, knee_r, ankle_r, hip_l, knee_l, ankle_l, ...].
        # We compare right (indices 0,1,2 relative to qpos[3:]) vs left (3,4,5).
        self._left_q_ids = jnp.array([3, 4, 5])
        self._right_q_ids = jnp.array([0, 1, 2])

    @property
    def task_config(self) -> TaskConfig:
        return self._cfg

    @property
    def action_size(self) -> int:
        return self._mj_model.nu

    # ------------------------------------------------------------------
    # PipelineEnv interface
    # ------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> State:
        rng, rng_q, rng_qd = jax.random.split(rng, 3)

        qpos = self.sys.qpos0 + self._cfg.reset_noise_scale * jax.random.normal(
            rng_q, (self.sys.nq,)
        )
        qvel = self._cfg.reset_noise_scale * jax.random.normal(
            rng_qd, (self.sys.nv,)
        )

        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = obs_fn.get_full_observations(pipeline_state, self._cfg.target_velocity)

        reward, done = jnp.zeros(2)
        metrics = {
            "forward_reward": jnp.float32(0),
            "velocity_tracking": jnp.float32(0),
            "healthy_reward": jnp.float32(0),
            "ctrl_cost": jnp.float32(0),
            "energy_cost": jnp.float32(0),
            "symmetry_reward": jnp.float32(0),
            "joint_limit_penalty": jnp.float32(0),
            "posture_reward": jnp.float32(0),
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
        cfg = self._cfg

        prev_com = state.pipeline_state.subtree_com[0]
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        curr_com = pipeline_state.subtree_com[0]

        # --- always compute every reward term ---
        fwd = cfg.forward_reward_weight * rew_fn.forward_velocity(
            prev_com, curr_com, self.dt
        )

        vel_track = cfg.velocity_tracking_weight * rew_fn.velocity_tracking(
            prev_com, curr_com, self.dt, cfg.target_velocity
        )

        cost = -cfg.ctrl_cost_weight * rew_fn.ctrl_cost(action)

        com_z = pipeline_state.subtree_com[0, 2]
        is_healthy = rew_fn.healthy(com_z, cfg.min_z, cfg.max_z)
        hlth = cfg.healthy_reward * is_healthy

        # Energy cost: use actuator force * ctrl as proxy when force data is
        # available; otherwise fall back to ctrl^2 (same as ctrl_cost).
        energy = -cfg.energy_cost_weight * rew_fn.mechanical_work(
            pipeline_state.actuator_force, action
        )

        # Symmetry: compare left/right joint positions and velocities.
        qpos_nroot = pipeline_state.qpos[3:]  # skip root x, y, θ
        sym = cfg.symmetry_reward_weight * rew_fn.symmetry_reward(
            qpos_nroot, pipeline_state.qvel[3:],
            self._left_q_ids, self._right_q_ids,
        )

        # Joint limit penalty from qpos vs joint ranges.
        jlim = cfg.joint_limit_penalty_weight * rew_fn.joint_limit_penalty(
            pipeline_state.qpos, self._jnt_range, self._jnt_limited,
        )

        # Posture: pelvis pitch is the 3rd element of qpos (root θ in 2D).
        pelvis_tilt = pipeline_state.qpos[2]
        post = cfg.posture_reward_weight * rew_fn.posture_reward(pelvis_tilt)

        reward = fwd + vel_track + hlth + cost + energy + sym + jlim + post
        done = 1.0 - is_healthy

        obs = obs_fn.get_full_observations(pipeline_state, cfg.target_velocity)

        metrics = {
            "forward_reward": fwd,
            "velocity_tracking": vel_track,
            "healthy_reward": hlth,
            "ctrl_cost": cost,
            "energy_cost": energy,
            "symmetry_reward": sym,
            "joint_limit_penalty": jlim,
            "posture_reward": post,
        }

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
        )
