"""Capture the initial robot head-camera view and save it as a PNG.

Usage (on the machine with Isaac Sim):

    $ISAACSIM_ROOT/python.sh capture_scene.py --output scene_initial.png --headless

Then open the saved image with any image viewer, e.g.:

    xdg-open scene_initial.png
"""

from __future__ import annotations

import argparse
import json
import os

from isaacsim import SimulationApp


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture the initial scene view.")
    parser.add_argument("--output", type=str, default="scene_initial.png")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--env_config", type=str, default="{}")
    args = parser.parse_args()

    simulation_app = SimulationApp({"headless": args.headless})
    try:
        from PIL import Image

        import environment

        task_dict = json.loads(args.env_config)
        task_dict["headless"] = args.headless
        env = environment.setup_scene(simulation_app, task_dict=task_dict)
        rgb, _ = env.reset_scene()
        Image.fromarray(rgb).save(args.output)
        print(f"saved: {os.path.abspath(args.output)}")
    finally:
        if "env" in locals():
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
