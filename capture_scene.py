"""Capture the initial robot head-camera view and save it as a PNG.

Usage (on the machine with Isaac Sim):

    $ISAACSIM_ROOT/python.sh capture_scene.py --output scene_initial.png --headless

Then open the saved image with any image viewer, e.g.:

    xdg-open scene_initial.png
"""

from __future__ import annotations

import argparse
import json
import math
import os

from isaacsim import SimulationApp


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture the initial scene view.")
    parser.add_argument("--output", type=str, default="scene_initial.png")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--env_config", type=str, default="{}")
    parser.add_argument("--hide_robot", action="store_true", help="Hide the robot mesh")
    args = parser.parse_args()

    simulation_app = SimulationApp({"headless": args.headless})
    try:
        from PIL import Image
        from isaacsim.core.utils.viewports import set_camera_view

        import environment
        task_dict = json.loads(args.env_config)
        task_dict["headless"] = args.headless
        if args.hide_robot:
            task_dict["hide_robot"] = True
        env = environment.setup_scene(simulation_app, task_dict=task_dict)
        rgb, _ = env.reset_scene()
        from pxr import Usd, UsdGeom

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
        )
        print("articulation_ok:", env._articulation_ok)
        if getattr(env, "_articulation_error", ""):
            print("articulation_error:", env._articulation_error)
        print("hide_robot_config:", task_dict.get("hide_robot", False))
        h1_prim = env.world.stage.GetPrimAtPath("/World/H1")
        print("H1_active:", h1_prim.IsActive() if h1_prim and h1_prim.IsValid() else "missing")
        print("head_camera_position:", env._head_camera_position().tolist())
        print("head_candidates:")
        for prim in env.world.stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith("/World/H1"):
                continue
            name = prim.GetName().lower()
            if not any(key in name for key in ("head", "d435", "mid360", "torso")):
                continue
            try:
                rng = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
                lo = rng.GetMin()
                hi = rng.GetMax()
                if all(
                    math.isfinite(v)
                    for v in (lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])
                ):
                    center = [(lo[i] + hi[i]) / 2.0 for i in range(3)]
                    center_user = [center[0], center[2], center[1]]
                    print(
                        path,
                        "center_user=",
                        [round(float(v), 3) for v in center_user],
                    )
            except Exception:
                pass
        print("stage_up_axis:", env.world.stage.GetMetadata("upAxis"))
        print("robot_root_pose:", env.robot_root.get_world_poses())
        for link_name in ("torso_link", "right_ankle_link"):
            prim = env.world.stage.GetPrimAtPath(f"/World/H1/{link_name}")
            if prim and prim.IsValid():
                print(
                    f"{link_name}_world_bbox:",
                    bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange(),
                )
        cam_prim = env.world.stage.GetPrimAtPath("/World/Camera")
        print(
            "camera_matrix:",
            UsdGeom.Xformable(cam_prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            ),
        )
        eye_prim = env.world.stage.GetPrimAtPath("/World/RobotEyeCamera")
        if eye_prim and eye_prim.IsValid():
            print(
                "eye_camera_matrix:",
                UsdGeom.Xformable(eye_prim).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                ),
            )
        Image.fromarray(rgb).save(args.output)
        print(
            f"saved: {os.path.abspath(args.output)} "
            f"brightness={float(rgb.mean()):.1f} std={float(rgb.std()):.1f} "
            f"center={rgb[rgb.shape[0] // 2, rgb.shape[1] // 2].tolist()}"
        )

        views = {
            "scene_third.png": ([1.0, -4.5, 2.2], [2.4, 0.0, 1.2], [0.0, 0.0, 1.0]),
            "scene_top.png": ([2.0, 0.0, 4.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        }
        for name, (eye, target, up) in views.items():
            try:
                try:
                    set_camera_view(
                        eye=eye,
                        target=target,
                        up=up,
                        camera_prim_path="/World/Camera",
                    )
                except TypeError:
                    set_camera_view(
                        eye=eye,
                        target=target,
                        camera_prim_path="/World/Camera",
                    )
                for _ in range(5):
                    env.world.step(render=True)
                view_rgb = env.camera.get_rgb()
                Image.fromarray(view_rgb).save(name)
                print(
                    "camera_matrix:",
                    UsdGeom.Xformable(cam_prim).ComputeLocalToWorldTransform(
                        Usd.TimeCode.Default()
                    ),
                )
                print(
                    f"saved: {os.path.abspath(name)} "
                    f"brightness={float(view_rgb.mean()):.1f} "
                    f"std={float(view_rgb.std()):.1f} "
                    f"center={view_rgb[view_rgb.shape[0] // 2, view_rgb.shape[1] // 2].tolist()}"
                )
            except Exception as exc:
                print(f"failed to save {name}: {exc}")
        print("all views done")
    finally:
        if "env" in locals():
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
