# msk-mjx

**GPU-accelerated musculoskeletal reinforcement learning with differentiable elastic tendons**

msk-mjx aims to be a JAX-based framework for training RL locomotion policies on a full human musculoskeletal model with physiologically accurate elastic tendon dynamics, all running on GPU via [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html).

No existing system combines GPU-accelerated physics, differentiable elastic tendons, and a validated human musculoskeletal model. msk-mjx is designed to fill that gap.

## Motivation

Tendons store and release elastic energy during locomotion. The Achilles tendon alone recovers roughly a third of metabolic energy during running. Every GPU-accelerated musculoskeletal project today (MuscleMimic, KinTwin, MIMIC-MJX) uses rigid tendons, which cannot capture this mechanism. The only simulators with proper elastic tendons (Hyfydy/SCONE, OpenSim) run on CPU and lack differentiability.

msk-mjx aims to close this gap by implementing a custom Millard2012-based elastic tendon solver in pure JAX, integrated directly into the MJX simulation pipeline, targeting significant throughput improvements over CPU baselines while maintaining biomechanical fidelity comparable to OpenSim.

## Goals

- **Differentiable elastic tendon solver** in JAX using the Millard2012 acceleration formulation with De Groote 2016 analytical curves, compatible with `jax.grad`
- **GPU-batched musculoskeletal simulation** through MJX, supporting thousands of parallel environments on a single GPU
- **Rajagopal 2016 model integration** (37 DOF, 80 Hill-type muscle-tendon units), building on MyoSuite's validated myoLeg conversion
- **Gymnasium RL interface** compatible with Brax's `PipelineEnv`, with configurable observations, rewards, and support for muscle synergy action representations
- **Biomechanical validation** against OpenSim reference curves, normative ground reaction force data, and EMG activation timing

## Planned architecture

The core idea is to replace MJX's built-in rigid-tendon actuation stage with a custom elastic MTU solver while leaving kinematics, constraint solving, and integration untouched:

```
MJX fwd_position  →  MJX fwd_velocity  →  elastic_mtu_actuation (custom)  →  MJX fwd_acceleration  →  MJX fwd_constraint  →  euler
```

The MTU solver will track three states per muscle (activation, fiber length, fiber velocity) using semi-implicit Euler integration with sub-cycling (multiple MTU substeps per physics step). Parallelism will be composed as:

```
jit( vmap_envs( vmap_muscles( scan_substeps( mtu_step ) ) ) )
```

## Planned project structure

```
msk-mjx/
├── src/msk_mjx/
│   ├── muscle/              # Elastic tendon solver
│   │   ├── curves.py        # De Groote 2016 analytical curves
│   │   ├── activation.py    # Activation dynamics ODE
│   │   ├── elastic_tendon.py
│   │   ├── metabolic.py     # Umberger/Bhargava metabolic cost
│   │   └── parameters.py    # Load params from OpenSim XML
│   ├── physics/             # MJX pipeline integration
│   │   ├── custom_step.py
│   │   ├── pipeline.py      # Brax PipelineEnv
│   │   └── domain_randomize.py
│   ├── envs/                # Gymnasium environments
│   │   ├── locomotion.py
│   │   ├── observations.py
│   │   └── rewards.py
│   ├── models/rajagopal/    # Model XML + meshes
│   ├── training/            # RL algorithms
│   │   ├── ppo.py
│   │   └── sar.py           # Synergistic action representations
│   └── validation/          # Biomechanical validation
│       ├── opensim_compare.py
│       ├── grf_analysis.py
│       └── emg_compare.py
├── tests/
├── notebooks/
└── configs/
```

## Landscape

|  | Rigid tendons | Elastic tendons |
|--|---------------|-----------------|
| **GPU-accelerated** | MuscleMimic, KinTwin, MIMIC-MJX | **msk-mjx** (planned) |
| **CPU** | MyoSuite, DEP-RL, KINESIS | Hyfydy/SCONE, OpenSim |

msk-mjx is designed to be the first system combining GPU acceleration, differentiability, and elastic tendon dynamics for human musculoskeletal RL.

## Status

This project is in early development. See the [architecture document](docs/architecture.md) for the full technical design.

## License

Apache License 2.0