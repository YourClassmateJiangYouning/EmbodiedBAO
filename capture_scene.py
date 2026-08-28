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
        from isaacsim.core.utils.viewports import set_camera_view

        task_dict = json.loads(args.env_config)
        task_dict["headless"] = args.headless
        env = environment.setup_scene(simulation_app, task_dict=task_dict)
        rgb, _ = env.reset_scene()
        print("stage_up_axis:", env.world.stage.GetMetadata("upAxis"))
        print("robot_root_pose:", env.robot_root.get_world_poses())
        from pxr import Usd, UsdGeom

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        for link_name in ("torso_link", "right_ankle_link"):
            prim = env.world.stage.GetPrimAtPath(f"/World/H1/{link_name}")
            if prim and prim.IsValid():
                print(
                    f"{link_name}_world_bbox:",
                    bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange(),
                )
        Image.fromarray(rgb).save(args.output)
        print(f"saved: {os.path.abspath(args.output)} brightness={float(rgb.mean()):.1f}")

        views = {
            "scene_third.png": ([0.5, -6.0, 3.5], [2.0, 0.0, 1.2], [0.0, 0.0, 1.0]),
            "scene_top.png": ([2.0, 0.0, 4.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        }
        for name, (eye, target, up) in views.items():
            set_camera_view(
                eye=eye,
                target=target,
                up=up,
                camera_prim_path="/World/Camera",
            )
            for _ in range(5):
                env.world.step(render=True)
            view_rgb = env.camera.get_rgb()
            Image.fromarray(view_rgb).save(name)
            print(f"saved: {os.path.abspath(name)} brightness={float(view_rgb.mean()):.1f}")
    finally:
        if "env" in locals():
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
