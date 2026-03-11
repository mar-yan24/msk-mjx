# msk-mjx

**GPU-accelerated musculoskeletal reinforcement learning with differentiable elastic tendons**

msk-mjx is a JAX-based framework for training RL locomotion policies on a full human musculoskeletal model (80 muscles, 37 DOF) with physiologically accurate elastic tendon dynamics, all running on GPU via [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html).

No existing system combines GPU-accelerated physics, differentiable elastic tendons, and a validated human musculoskeletal model. msk-mjx fills that gap by implementing a custom Millard2012-based elastic tendon solver in pure JAX, integrated directly into the MJX simulation pipeline.

## Why elastic tendons matter

Tendons store and release elastic energy during locomotion. The Achilles tendon alone recovers ~35% of metabolic energy during running. Every GPU-accelerated musculoskeletal project today (MuscleMimic, KinTwin, MIMIC-MJX) uses rigid tendons, which cannot capture this mechanism. The only simulators with proper elastic tendons (Hyfydy/SCONE, OpenSim) run on CPU and lack differentiability.

msk-mjx delivers **50-100x throughput** over CPU baselines while maintaining biomechanical fidelity comparable to OpenSim.

## Key features

- **Differentiable elastic tendon solver**: Millard2012 acceleration formulation implemented in JAX with De Groote 2016 analytical curves, fully compatible with `jax.grad`
- **GPU-batched simulation**: 2048-8192 parallel environments on a single A100, targeting 500K-1.2M steps/second
- **Rajagopal 2016 musculoskeletal model**: 37 DOF, 80 Hill-type muscle-tendon units, validated skeletal geometry from MyoSuite's myoLeg conversion
- **Gymnasium RL interface**: Brax-compatible `PipelineEnv` with configurable observation spaces, reward functions, and muscle synergy action representations
- **Biomechanical validation suite**: automated comparison against OpenSim reference curves, normative GRF data, and EMG activation timing

## Architecture overview

msk-mjx replaces MJX's built-in rigid-tendon actuation stage with a custom elastic MTU solver while leaving kinematics, constraint solving, and integration untouched:

```
MJX fwd_position  →  MJX fwd_velocity  →  elastic_mtu_actuation (custom)  →  MJX fwd_acceleration  →  MJX fwd_constraint  →  euler
```

The MTU solver tracks three states per muscle (activation, fiber length, fiber velocity) using semi-implicit Euler integration with sub-cycling (~5-20 substeps per physics step). Parallelism is composed as:

```
jit( vmap_envs( vmap_muscles( scan_substeps( mtu_step ) ) ) )
```

## Installation

```bash
# Requires Python 3.10+ and CUDA 12
pip install msk-mjx

# Or from source
git clone https://github.com/<org>/msk-mjx.git
cd msk-mjx
pip install -e ".[dev]"
```

### Dependencies

- `jax[cuda12]` >= 0.4.25
- `mujoco` == 3.3.2
- `mujoco-mjx`
- `brax`
- `gymnasium`
- `flax`, `optax`

> **Note:** Pin MuJoCo to 3.3.2. Version 3.3.3 introduced a breaking tendon armature bug ([#2681](https://github.com/google-deepmind/mujoco/issues/2681)).

## Quick start

### Simulate a single step

```python
import mujoco
from mujoco import mjx
from msk_mjx.physics.custom_step import custom_step

# Load the Rajagopal model
model = mujoco.MjModel.from_xml_path("src/msk_mjx/models/rajagopal/rajagopal.xml")
mjx_model = mjx.put_model(model)
mjx_data = mjx.put_data(model, mujoco.MjData(model))

# Step with elastic tendons
mjx_data = custom_step(mjx_model, mjx_data)
```

### Train a walking policy

```python
from msk_mjx.envs.locomotion import WalkingEnv
from msk_mjx.training.ppo import train

env = WalkingEnv(
    n_envs=4096,
    reward_config="configs/walk.yaml",
)

policy = train(
    env,
    total_steps=1_000_000_000,
    use_sar=True,         # muscle synergy action space
    n_synergies=15,
)
```

### Run on Google Colab

See [`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb) for a single-notebook training demo that runs on a free Colab T4 GPU.

## Project structure

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

## Validation

Validation proceeds at three levels:

1. **Curve-level**: JAX force-length, force-velocity, and tendon curves match OpenSim reference implementations within float32 tolerance (~1e-6)
2. **Single-muscle**: Isometric and isokinetic contraction forces match OpenSim's `Millard2012EquilibriumMuscle` (peak force error <2%, F-L R² > 0.99)
3. **Full-gait**: Trained policies produce physiologically plausible locomotion validated against normative GRF profiles, EMG activation timing (cross-correlation r > 0.7), joint kinematics, and metabolic cost of transport (~3.0-3.5 J/kg/m at 1.3 m/s)

```bash
# Run the validation suite
pytest tests/ -v
python -m msk_mjx.validation.opensim_compare
```

## Performance

Expected throughput on a single A100 (80 GB):

| Metric | msk-mjx (GPU) | MyoSuite (CPU) |
|--------|---------------|----------------|
| Steps/second | 500K-1.2M | 5K-20K |
| Time to 1B steps | ~15 min | ~14 hours |
| Max parallel envs | 2048-8192 | ~128 |

The elastic tendon sub-cycling adds an estimated 2-4x overhead compared to rigid-tendon MJX simulation.

## How this compares to existing work

|  | Rigid tendons | Elastic tendons |
|--|---------------|-----------------|
| **GPU-accelerated** | MuscleMimic, KinTwin, MIMIC-MJX | **msk-mjx** |
| **CPU** | MyoSuite, DEP-RL, KINESIS | Hyfydy/SCONE, OpenSim |

msk-mjx is the first system to combine GPU acceleration, differentiability, and elastic tendon dynamics for human musculoskeletal RL.

## Citation

```bibtex
@software{msk_mjx,
  title  = {msk-mjx: GPU-accelerated musculoskeletal RL with differentiable elastic tendons},
  year   = {2026},
  url    = {https://github.com/<org>/msk-mjx}
}
```

## License

MIT