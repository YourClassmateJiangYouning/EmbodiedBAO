"""Run one quick Level 0 episode per supported MLLM in a single Isaac session.

Usage (on the machine with Isaac Sim):

    /home/ybh/isaacsim/python.sh run_models_smoke.py --headless \
        --env_config '{"robot_physics":true}'

The script reuses one BAOEnv instance so model smoke tests do not pay the
Isaac Sim startup cost once per model.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

from isaacsim import SimulationApp

from ai_agent import SUPPORTED_MODELS


DEFAULT_MODELS: List[str] = SUPPORTED_MODELS + ["random"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test every supported model with one episode."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model names; default is every supported model plus random.",
    )
    parser.add_argument("--level", type=int, choices=[0, 1, 2, 3], default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--env_config", type=str, default="{}")
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


def _write_progress(message: str) -> None:
    path = os.path.join(os.getcwd(), "run_progress.txt")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def _run_models(args: argparse.Namespace) -> int:
    import environment
    from experiments import BAOExperimentRunner
    from main import save_episodes_csv

    models = list(args.models) if args.models else DEFAULT_MODELS
    task_dict = json.loads(args.env_config)
    task_dict["headless"] = args.headless
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    simulation_app = SimulationApp({"headless": args.headless})
    env = None
    try:
        env = environment.setup_scene(simulation_app, task_dict=task_dict)
        _write_progress(
            f"model smoke start: level={args.level} models={models} "
            f"episodes={args.episodes} rounds={args.rounds}"
        )
        summaries: Dict[str, Dict[str, Any]] = {}
        for model in models:
            safe_model = str(model).replace("/", "-").replace("\\", "-")
            try:
                log_tag = args.tag or f"{timestamp}-{safe_model}"
                runner = BAOExperimentRunner(
                    env=env,
                    model=safe_model,
                    max_steps=args.max_steps,
                    tag=log_tag,
                )
                runner.save_args(args)
                episodes = runner.run_level(
                    level=args.level,
                    episodes=args.episodes,
                    rounds=args.rounds,
                )
                csv_path = save_episodes_csv(
                    episodes,
                    model=safe_model,
                    level=args.level,
                    timestamp=timestamp,
                )
                summary = BAOExperimentRunner._summarize(args.level, episodes)
                summaries[safe_model] = summary
                print(
                    f"[models] {safe_model}: "
                    f"success_rate={summary['success_rate']:.3f} "
                    f"end_reasons={summary['end_reason_counts']}"
                )
                print(f"[models] saved {csv_path}")
                _write_progress(
                    f"model {safe_model}: level={args.level} "
                    f"success_rate={summary['success_rate']:.3f} "
                    f"end_reasons={summary['end_reason_counts']}"
                )
            except Exception as exc:
                summaries[safe_model] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"[models] {safe_model}: ERROR {exc}")
                _write_progress(f"model {safe_model}: ERROR {type(exc).__name__}: {exc}")

        summary_path = os.path.join(
            "results", f"models_level{args.level}_{timestamp}.json"
        )
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summaries, handle, indent=2, ensure_ascii=False)
        print(f"[models] summary saved: {summary_path}")
        return 0
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(_run_models(parse_args()))
