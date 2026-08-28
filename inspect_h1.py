"""Inspect a Unitree H1 USD and print its structure and joint data.

Usage (on the machine with Isaac Sim python / usd-core):

    $ISAACSIM_ROOT/python.sh inspect_h1.py /path/to/h1.usd

Prints the stage up axis, articulation root, links, joints with their types,
axes, limits and drive settings, plus the overall bounding box.
"""

from __future__ import annotations

import argparse
import json
import sys

from pxr import Sdf, Usd, UsdGeom, UsdPhysics


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect an H1 USD file.")
    parser.add_argument("usd_path", type=str)
    parser.add_argument("--json", type=str, default="", help="Optional JSON output path")
    args = parser.parse_args()

    stage = Usd.Stage.Open(args.usd_path)
    if stage is None:
        print(f"Failed to open: {args.usd_path}")
        return 1

    info: dict = {
        "usd_path": args.usd_path,
        "up_axis": stage.GetMetadata("upAxis"),
        "meters_per_unit": stage.GetMetadata("metersPerUnit"),
        "default_prim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else "",
        "links": [],
        "joints": [],
        "bounding_box": None,
    }

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    min_corner = None
    max_corner = None

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        type_name = prim.GetTypeName()
        if prim.IsA(UsdGeom.Xformable) and not prim.IsA(UsdPhysics.Joint):
            info["links"].append({"path": path, "type": type_name})
            try:
                bound = cache.ComputeWorldBound(prim)
                rng = bound.ComputeAlignedRange()
                if not rng.IsEmpty:
                    lo = rng.GetMin()
                    hi = rng.GetMax()
                    min_corner = lo if min_corner is None else [min(a, b) for a, b in zip(min_corner, lo)]
                    max_corner = hi if max_corner is None else [max(a, b) for a, b in zip(max_corner, hi)]
            except Exception:
                pass

        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint_info: dict = {"path": path, "type": type_name, "attrs": {}}
        for attr in prim.GetAttributes():
            name = attr.GetName()
            if any(key in name.lower() for key in ("axis", "limit", "drive", "target", "stiffness", "damping", "effort", "break", "joint")):
                try:
                    joint_info["attrs"][name] = str(attr.Get())
                except Exception:
                    pass
        for api_name in ("linear", "angular"):
            drive = UsdPhysics.DriveAPI.Get(prim, api_name)
            if drive:
                joint_info["drive_" + api_name] = {
                    "target_position": str(drive.GetTargetPositionAttr().Get()),
                    "target_velocity": str(drive.GetTargetVelocityAttr().Get()),
                    "stiffness": str(drive.GetStiffnessAttr().Get()),
                    "damping": str(drive.GetDampingAttr().Get()),
                }
        info["joints"].append(joint_info)

    if min_corner is not None and max_corner is not None:
        info["bounding_box"] = {"min": min_corner, "max": max_corner}
        info["size"] = [round(b - a, 4) for a, b in zip(min_corner, max_corner)]

    print(json.dumps(info, indent=2, ensure_ascii=False))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(info, handle, indent=2, ensure_ascii=False)
        print(f"saved: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
