"""Convert a Unitree H1 URDF into a USD file with Isaac Sim.

Usage (on the machine with Isaac Sim):

    $ISAACSIM_ROOT/python.sh convert_h1_urdf.py \
        /path/to/h1_with_hand.urdf /path/to/output/h1.usd

The URDF importer is loaded together with the SimulationApp, so this script
must be executed with the Isaac Sim python and not a system python.
"""

from __future__ import annotations

import argparse
import os

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.kit.commands
import omni.usd


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a URDF to USD.")
    parser.add_argument("urdf_path", type=str, help="Input URDF file")
    parser.add_argument("output_usd", type=str, help="Output USD file")
    args = parser.parse_args()

    urdf_path = os.path.abspath(args.urdf_path)
    output_usd = os.path.abspath(args.output_usd)
    if not os.path.isfile(urdf_path):
        print(f"URDF file not found: {urdf_path}")
        return 1
    os.makedirs(os.path.dirname(output_usd), exist_ok=True)

    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.fix_base = True
    import_config.distance_scale = 1.0

    status, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=import_config,
        get_articulation_root=True,
    )
    if not status:
        print("URDF import failed.")
        return 1

    stage = omni.usd.get_context().get_stage()
    stage.GetRootLayer().Export(output_usd)
    print(f"Articulation root prim: {prim_path}")
    print(f"Saved USD to: {output_usd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
