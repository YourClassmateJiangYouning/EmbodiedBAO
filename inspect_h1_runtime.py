"""Runtime-safe wrapper for inspect_h1.py under Isaac Sim.

Use when plain Isaac python cannot import pxr before SimulationApp starts.
"""

from __future__ import annotations

import argparse
import sys

from isaacsim import SimulationApp


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect H1 after SimulationApp starts.")
    parser.add_argument("usd_path", type=str)
    parser.add_argument("--json", type=str, default="")
    args = parser.parse_args()

    simulation_app = SimulationApp({"headless": True})
    try:
        import inspect_h1

        sys.argv = ["inspect_h1_runtime.py", args.usd_path]
        if args.json:
            sys.argv += ["--json", args.json]
        return inspect_h1.main()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
