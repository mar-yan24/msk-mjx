"""Interactive MuJoCo viewer for the MSK model.

Usage:
    python scripts/play.py                       # stand keyframe + viewer
    python scripts/play.py --keyframe walk_left   # different pose
    python scripts/play.py --no-viewer --random-actions  # headless sanity check
"""

import argparse
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "22muscle_2D" / "myoLeg22_2D_BASELINE.xml"

KEYFRAMES = ["stand", "walk_left", "walk_right", "squat", "lunge"]


def load_model(model_path: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    return model, data


def set_keyframe(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    if key_id < 0:
        available = [
            model.key(i).name for i in range(model.nkey) if model.key(i).name
        ]
        raise ValueError(
            f"Keyframe '{name}' not found. Available: {available}"
        )
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)


def print_summary(model: mujoco.MjModel) -> None:
    print(f"Model loaded: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print(f"  bodies:    {model.nbody}")
    print(f"  joints:    {model.njnt}")
    print(f"  actuators: {model.nu}")
    names = [model.actuator(i).name for i in range(model.nu)]
    print(f"  actuator names: {names}")
    print(f"  keyframes: {model.nkey}")
    for i in range(model.nkey):
        print(f"    [{i}] {model.key(i).name}")


def run_headless(
    model: mujoco.MjModel, data: mujoco.MjData, random_actions: bool
) -> None:
    n_steps = 1000
    print(f"Running {n_steps} steps headless (random_actions={random_actions})...")
    for _ in range(n_steps):
        if random_actions:
            data.ctrl[:] = np.random.uniform(0, 1, size=model.nu)
        mujoco.mj_step(model, data)
    print(f"Done. Final time={data.time:.3f}s, qpos[:3]={data.qpos[:3]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive MSK model viewer")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to MuJoCo XML model",
    )
    parser.add_argument(
        "--keyframe",
        type=str,
        default="stand",
        choices=KEYFRAMES,
        help="Initial keyframe (default: stand)",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run headless without the interactive viewer",
    )
    parser.add_argument(
        "--random-actions",
        action="store_true",
        help="Apply random muscle excitations (headless mode)",
    )
    args = parser.parse_args()

    model, data = load_model(args.model)
    print_summary(model)
    set_keyframe(model, data, args.keyframe)
    print(f"Keyframe '{args.keyframe}' applied.")

    if args.no_viewer:
        run_headless(model, data, args.random_actions)
    else:
        print("Launching viewer... (close window to exit)")
        mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
