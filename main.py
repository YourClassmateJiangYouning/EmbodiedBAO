"""EmbodiedBAO program entry point.

Usage:
    python main.py --model gpt-4o --level 0 --episodes 50
    python main.py --model claude-3.5-sonnet --level 3 --episodes 50
    python main.py --model gemini-2.5-pro --all-levels

Flow: initialize the Isaac Sim scene (environment.setup_scene), create the AI
agent (ai_agent.AIAgent), run the requested level(s) with experiments.py,
save one CSV per (model, level) under results/, and close the environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import traceback
from typing import Any, Dict, List


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EmbodiedBAO experiment entry point.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name")
    parser.add_argument(
        "--level", type=int, choices=[0, 1, 2, 3], default=0, help="Level to run"
    )
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per level")
    parser.add_argument(
        "--all-levels", action="store_true", help="Run levels 0, 1, 2, 3"
    )
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--collision_timeout", type=int, default=3)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--tag", type=str, default="", help="Optional run tag")
    parser.add_argument(
        "--env_config",
        type=str,
        default="{}",
        help='JSON dict passed to BAOEnv, e.g. \'{"rendermode":"RaytracedLighting","spp":4}\'',
    )
    return parser.parse_args(argv)


def save_episodes_csv(
    episodes: List[Dict[str, Any]],
    model: str,
    level: int,
    results_root: str = "results",
    timestamp: str = "",
) -> str:
    """Flatten episodes into one CSV row per step."""
    os.makedirs(results_root, exist_ok=True)
    timestamp = timestamp or time.strftime("%Y%m%d-%H%M%S")
    safe_model = model.replace("/", "-").replace("\\", "-")
    path = os.path.join(results_root, f"{safe_model}_{level}_{timestamp}.csv")

    fields = [
        "episode_id",
        "level",
        "model_name",
        "step_number",
        "action_taken",
        "hand_x",
        "hand_y",
        "hand_z",
        "roll",
        "pitch",
        "yaw",
        "distance_to_target",
        "collision_with_wall",
        "wall_collision_x",
        "wall_collision_y",
        "wall_collision_z",
        "success",
        "llm_response_time_ms",
        "episode_success",
        "end_reason",
    ]

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for episode in episodes:
            for step in episode.get("steps", []):
                hand = step.get("hand_position") or [0.0, 0.0, 0.0]
                orientation = step.get("torso_orientation") or {}
                collision_point = step.get("wall_collision_point") or [None, None, None]
                writer.writerow(
                    {
                        "episode_id": step.get("episode_id", episode.get("episode_id")),
                        "level": step.get("level", episode.get("level")),
                        "model_name": step.get("model_name", episode.get("model_name")),
                        "step_number": step.get("step_number"),
                        "action_taken": step.get("action_taken"),
                        "hand_x": hand[0] if len(hand) > 0 else None,
                        "hand_y": hand[1] if len(hand) > 1 else None,
                        "hand_z": hand[2] if len(hand) > 2 else None,
                        "roll": orientation.get("roll"),
                        "pitch": orientation.get("pitch"),
                        "yaw": orientation.get("yaw"),
                        "distance_to_target": step.get("distance_to_target"),
                        "collision_with_wall": step.get("collision_with_wall"),
                        "wall_collision_x": collision_point[0] if collision_point else None,
                        "wall_collision_y": collision_point[1] if collision_point else None,
                        "wall_collision_z": collision_point[2] if collision_point else None,
                        "success": step.get("success"),
                        "llm_response_time_ms": step.get("llm_response_time_ms"),
                        "episode_success": episode.get("success"),
                        "end_reason": episode.get("end_reason"),
                    }
                )
    return path


def _progress_callback(
    level: int, completed: int, total: int, episodes_done: List[Dict[str, Any]]
) -> None:
    if completed % 10 == 0 or completed == total:
        success_count = sum(1 for ep in episodes_done if ep.get("success"))
        rate = success_count / completed if completed else 0.0
        print(
            f"[main] level {level}: episode {completed}/{total}, "
            f"current success rate = {rate:.3f}"
        )
        _write_progress(
            f"level {level}: episode {completed}/{total} success_rate={rate:.3f}"
        )


def _write_progress(message: str) -> None:
    """Append a timestamped line to run_progress.txt (stdout is swallowed)."""
    path = os.path.join(os.getcwd(), "run_progress.txt")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def main() -> None:
    args = parse_args()
    levels = [0, 1, 2, 3] if args.all_levels else [args.level]
    _write_progress(
        f"main start: model={args.model} levels={levels} episodes={args.episodes}"
    )
    env = None
    try:
        from isaacsim import SimulationApp

        simulation_app = SimulationApp({"headless": args.headless})
        _write_progress("SimulationApp started")

        import ai_agent
        import environment
        from experiments import BAOExperimentRunner

        task_dict = json.loads(args.env_config)
        task_dict["headless"] = args.headless
        env = environment.setup_scene(simulation_app, task_dict=task_dict)
        _write_progress("environment created")

        # Fail fast if the API key/model configuration is invalid; also
        # supports the "random" baseline via the create_agent factory.
        agent = ai_agent.create_agent(model=args.model)
        _write_progress(f"agent created: {args.model}")

        runner = BAOExperimentRunner(
            env=env,
            model=args.model,
            max_steps=args.max_steps,
            collision_timeout=args.collision_timeout,
            tag=args.tag,
        )
        runner.save_args(args)
        _write_progress("runner created")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        for level in levels:
            _write_progress(f"level {level} start")
            episodes = runner.run_level(
                level=level,
                episodes=args.episodes,
                progress_callback=lambda completed, total, episode_result, episodes_done, lvl=level: _progress_callback(
                    lvl, completed, total, episodes_done
                ),
            )
            csv_path = save_episodes_csv(
                episodes, model=args.model, level=level, timestamp=timestamp
            )
            print(f"[main] saved {csv_path}")
            _write_progress(f"csv saved: {csv_path}")
        _write_progress("all levels done")
    except Exception as exc:
        _write_progress(f"ERROR: {type(exc).__name__}: {exc}")
        _write_progress(traceback.format_exc())
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
