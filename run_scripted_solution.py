"""Run a scripted Level 0 route to validate environment solvability.

Usage (one line, on the machine with Isaac Sim):

    /home/ybh/isaacsim/python.sh run_scripted_solution.py --headless

Sequence:
    forward to wall -> reach right -> retreat
    -> backward away -> turn left 90 -> side-step right through opening
    -> raise right arm straight out and touch the ball
"""

from __future__ import annotations

import argparse
import sys

from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scripted Level 0 solution test.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--env_config", type=str, default="{}")
    return parser.parse_args()


def _step(env, action: str, count: int = 1, frames: int = 30) -> None:
    for _ in range(count):
        result = env.execute_action(action, n_steps=frames)
        print(
            f"{action}: legal={result.legal} distance={result.distance:.4f} "
            f"success={result.success}"
        )


def main() -> int:
    args = parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    simulation_app = SimulationApp({"headless": args.headless})
    env = None
    try:
        import json

        import environment

        task_dict = json.loads(args.env_config)
        task_dict["headless"] = args.headless
        env = environment.setup_scene(simulation_app, task_dict=task_dict)
        env.reset_scene()

        _step(env, "forward", count=7, frames=10)
        _step(env, "reach_right_arm", count=1, frames=60)
        _step(env, "retreat_right_arm", count=1, frames=60)
        _step(env, "backward", count=4, frames=10)
        _step(env, "turn_left", count=6, frames=5)

        reached = False
        for side_step in range(12):
            _step(env, "raise_right_arm", count=1, frames=60)
            if env.check_success():
                reached = True
                break
            _step(env, "retreat_right_arm", count=1, frames=40)
            _step(env, "right", count=1, frames=10)

        print("scripted_success:", reached)
        print("final_distance:", round(float(env.get_distance_to_target()), 6))
        print("final_position:", [round(float(v), 4) for v in env._root_position()])
        print("final_yaw:", round(float(env.get_torso_rotation()), 2))
        return 0 if reached else 1
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
