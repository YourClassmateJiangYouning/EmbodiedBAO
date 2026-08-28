"""Download the Unitree H1 USD asset from HuggingFace (MirrorBench-style).

Create a public HuggingFace dataset named e.g. "EmbodiedBAOAssets", upload
assets/H1/h1.usd into it, then run:

    export EMBODIEDBAO_ASSETS_REPO=YourUserName/EmbodiedBAOAssets
    python download.py

The file is saved to assets/H1/h1.usd, which environment.py discovers
automatically.
"""

from __future__ import annotations

import argparse
import os

from huggingface_hub import hf_hub_download


DEFAULT_REPO = "YourClassmateJiangYouning/EmbodiedBAOAssets"


def target_path() -> str:
    root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root, "assets", "H1", "h1.usd")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the H1 USD asset.")
    parser.add_argument(
        "--repo",
        type=str,
        default=os.environ.get("EMBODIEDBAO_ASSETS_REPO", DEFAULT_REPO),
        help="HuggingFace dataset repo id",
    )
    parser.add_argument("--force", action="store_true", help="Redownload even if present")
    args = parser.parse_args()

    target = target_path()
    if os.path.exists(target) and not args.force:
        print(f"Asset already exists: {target}")
        return 0

    os.makedirs(os.path.dirname(target), exist_ok=True)
    print(f"Downloading h1.usd from {args.repo} ...")
    hf_hub_download(
        repo_id=args.repo,
        filename="h1.usd",
        local_dir=os.path.dirname(target),
        repo_type="dataset",
    )
    print(f"Saved: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
