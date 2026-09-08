"""Capture the initial robot view and straight side-arm raise diagnostics.

Usage (on the machine with Isaac Sim):

    $ISAACSIM_ROOT/python.sh capture_arm_test.py --headless \
        --env_config '{"robot_physics":true}'

Outputs:
    <prefix>_initial.png
    <prefix>_raise_right.png
    <prefix>_raise_left.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture initial view and raised-arm test frames."
    )
    parser.add_argument("--prefix", type=str, default="arm_test")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--env_config", type=str, default="{}")
    return parser.parse_args()


def _save_robot_view(env, name: str) -> None:
    from PIL import Image

    try:
        rgb = env.get_camera_image()
        Image.fromarray(rgb).save(name)
        print(f"saved robot view: {os.path.abspath(name)}")
    except Exception as exc:
        print(f"failed to save robot view {name}: {exc}")


def _save_third_view(env, name: str) -> None:
    from PIL import Image
    from isaacsim.core.utils.viewports import set_camera_view

    try:
        try:
            set_camera_view(
                eye=[2.0, -4.5, 2.6],
                target=[1.96, 0.0, 1.2],
                up=[0.0, 0.0, 1.0],
                camera_prim_path="/World/Camera",
            )
        except TypeError:
            set_camera_view(
                eye=[2.0, -4.5, 2.6],
                target=[1.96, 0.0, 1.2],
                camera_prim_path="/World/Camera",
            )
        for _ in range(5):
            env.world.step(render=True)
        rgb = env.camera.get_rgb()
        Image.fromarray(rgb).save(name)
        print(f"saved third view: {os.path.abspath(name)}")
    except Exception as exc:
        print(f"failed to save third view {name}: {exc}")


def _print_shoulder_elbow(env, label: str) -> None:
    if not env._articulation_ok or env._articulation is None:
        print(f"{label}: articulation unavailable")
        return
    try:
        names = env._articulation_dof_names()
        values = env._articulation_joint_positions()
        pairs = []
        for name, value in zip(names, values):
            lower = name.lower()
            if "shoulder" in lower or "elbow" in lower:
                pairs.append((name, round(float(value), 3)))
        print(f"{label}: {pairs}")
    except Exception as exc:
        print(f"{label}: joint read failed: {exc}")


def main() -> int:
    args = parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    simulation_app = SimulationApp({"headless": args.headless})
    env = None
    try:
        import numpy as np

        import environment

        task_dict = json.loads(args.env_config)
        task_dict["headless"] = args.headless
        env = environment.setup_scene(simulation_app, task_dict=task_dict)
        env.reset_scene()
        _save_robot_view(env, f"{args.prefix}_initial.png")
        _save_third_view(env, f"{args.prefix}_initial_third.png")

        env._set_robot_pose(np.array([1.96, 0.0, 0.0]), 90.0)
        env.execute_action("raise_right_arm", n_steps=120)
        _print_shoulder_elbow(env, "right_arm joints")
        print(
            "right_arm side distance:",
            round(float(env.get_distance_to_target()), 4),
            "success:",
            env.check_success(),
        )
        _save_robot_view(env, f"{args.prefix}_raise_right.png")
        _save_third_view(env, f"{args.prefix}_raise_right_third.png")

        env.reset_scene()
        env._articulation = None
        env._articulation_ok = False
        env._reach_joint_indices = None
        env.reset_scene()
        env._set_robot_pose(np.array([1.96, 0.0, 0.0]), -90.0)
        env.execute_action("raise_left_arm", n_steps=120)
        _print_shoulder_elbow(env, "left_arm joints")
        print(
            "left_arm side distance:",
            round(float(env.get_distance_to_target()), 4),
            "success:",
            env.check_success(),
        )
        _save_robot_view(env, f"{args.prefix}_raise_left.png")
        _save_third_view(env, f"{args.prefix}_raise_left_third.png")
        return 0
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
