"""EmbodiedBAO program entry point.

Usage:
    python main.py --model gpt-4o --level 0
    python main.py --model gpt-4o --level 4
    python main.py --model gpt-4o --level 5
    python main.py --model claude-3.5-sonnet --level 3
    python main.py --model gemini-2.5-pro --all-levels

Flow: initialize the Isaac Sim scene (environment.setup_scene), create the AI
agent (ai_agent.AIAgent), run the requested level(s) with experiments.py,
save one CSV per (model, level) under results/, and close the environment.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import time
import traceback
from typing import Any, Dict, List


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EmbodiedBAO experiment entry point.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name")
    parser.add_argument(
        "--level", type=int, choices=[0, 1, 2, 3, 4, 5], default=0, help="Level to run"
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        choices=[0, 1, 2, 3, 4, 5],
        default=None,
        help="Run multiple levels in one process",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per round")
    parser.add_argument(
        "--rounds", type=int, default=3, help="Repeated rounds per model"
    )
    parser.add_argument(
        "--all-levels", action="store_true", help="Run levels 0, 1, 2, 3, 4, 5"
    )
    parser.add_argument(
        "--cold-only",
        action="store_true",
        help="Run only cold cells for the selected Level 1-4 target",
    )
    parser.add_argument(
        "--primed-only",
        action="store_true",
        help="Run only Level0-primed cells using saved Level0 episodes",
    )
    parser.add_argument(
        "--use-saved-level0",
        action="store_true",
        help="Run paired Level1-4 cells with saved Level0 episodes, without rerunning Level0",
    )
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--tag", type=str, default="", help="Optional run tag")
    parser.add_argument(
        "--env_config",
        type=str,
        default="{}",
        help='JSON dict passed to BAOEnv, e.g. \'{"rendermode":"RaytracedLighting","spp":4}\'',
    )
    return parser.parse_args(argv)


def load_saved_level0_episodes(
    model: str,
    rounds: int,
    results_root: str = "results",
) -> List[Dict[str, Any]]:
    """Load saved Level0 episodes so primed cells can resume without rerunning."""
    model_dir = os.path.join(results_root, "level0", model)
    paths = glob.glob(os.path.join(model_dir, "round*", "episode_000.json"))
    paths += glob.glob(os.path.join(model_dir, "episode_000.json"))
    episodes: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(paths):
        if path in seen:
            continue
        seen.add(path)
        with open(path, "r", encoding="utf-8") as handle:
            episodes.append(json.load(handle))
    episodes.sort(key=lambda ep: int(ep.get("round", 0)))
    episodes = episodes[: int(rounds)]
    if len(episodes) < int(rounds):
        raise RuntimeError(
            f"Need {rounds} saved Level0 episodes under {model_dir}; "
            f"found {len(episodes)}. Finish Level0 first or run without --primed-only."
        )
    return episodes


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
        "round",
        "episode_id",
        "level",
        "condition",
        "model_name",
        "phase",
        "channel_width",
        "sideways_rate",
        "step",
        "action",
        "hand_x",
        "hand_y",
        "hand_z",
        "hand_distance_to_ball",
        "torso_rotation",
        "collision_with_wall",
        "collision_position_x",
        "collision_position_y",
        "collision_position_z",
        "step_success",
        "llm_response_time_ms",
        "episode_success",
        "end_reason",
        "total_steps",
        "final_distance",
        "action_sequence",
    ]

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for episode in episodes:
            for step in episode.get("steps", []):
                writer.writerow(
                    {
                        "round": step.get("round", episode.get("round", 0)),
                        "episode_id": step.get("episode_id", episode.get("episode_id")),
                        "level": step.get("level", episode.get("level")),
                        "condition": step.get(
                            "condition", episode.get("condition")
                        ),
                        "model_name": step.get("model_name", episode.get("model_name")),
                        "phase": step.get("phase", episode.get("phase")),
                        "channel_width": step.get(
                            "channel_width", episode.get("channel_width")
                        ),
                        "sideways_rate": episode.get("sideways_rate"),
                        "step": step.get("step"),
                        "action": step.get("action"),
                        "hand_x": step.get("hand_x"),
                        "hand_y": step.get("hand_y"),
                        "hand_z": step.get("hand_z"),
                        "hand_distance_to_ball": step.get("hand_distance_to_ball"),
                        "torso_rotation": step.get("torso_rotation"),
                        "collision_with_wall": step.get("collision_with_wall"),
                        "collision_position_x": step.get("collision_position_x"),
                        "collision_position_y": step.get("collision_position_y"),
                        "collision_position_z": step.get("collision_position_z"),
                        "step_success": step.get("step_success"),
                        "llm_response_time_ms": step.get("llm_response_time_ms"),
                        "episode_success": episode.get("success"),
                        "end_reason": episode.get("end_reason"),
                        "total_steps": episode.get("total_steps"),
                        "final_distance": episode.get("final_distance"),
                        "action_sequence": episode.get("action_sequence"),
                    }
                )
    return path


def _progress_callback(
    level: int, completed: int, total: int, episodes_done: List[Dict[str, Any]]
) -> None:
    if completed % 10 == 0 or completed == total:
        success_count = sum(1 for ep in episodes_done if ep.get("success"))
        rate = success_count / completed if completed else 0.0
        round_id = int(episodes_done[-1].get("round", 0)) if episodes_done else 0
        print(
            f"[main] level {level} round {round_id}: episode {completed}/{total}, "
            f"current success rate = {rate:.3f}"
        )
        _write_progress(
            f"level {level} round {round_id}: episode {completed}/{total} "
            f"success_rate={rate:.3f}"
        )


def _write_progress(message: str) -> None:
    """Append a timestamped line to run_progress.txt (stdout is swallowed)."""
    path = os.path.join(os.getcwd(), "run_progress.txt")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def main() -> None:
    args = parse_args()
    requested_levels = list(args.levels or [args.level])
    if args.all_levels and args.levels:
        raise ValueError("--all-levels and --levels cannot be used together")
    if args.cold_only and args.primed_only:
        raise ValueError("--cold-only and --primed-only cannot be used together")
    if args.all_levels and (
        args.cold_only or args.primed_only or args.use_saved_level0
    ):
        raise ValueError(
            "--cold-only/--primed-only/--use-saved-level0 cannot be combined "
            "with --all-levels"
        )
    pair_flags = args.cold_only or args.primed_only or args.use_saved_level0
    if pair_flags and any(level not in (1, 2, 3, 4) for level in requested_levels):
        raise ValueError(
            "pair-only flags require every selected level to be 1, 2, 3, or 4"
        )

    cells: List[str] = ["cold", "primed"]
    if args.cold_only:
        cells = ["cold"]
    if args.primed_only:
        cells = ["primed"]

    if args.all_levels:
        levels = [0, 1, 2, 3, 4, 5]
    elif args.levels:
        levels = requested_levels
        if (
            0 not in levels
            and any(level in (1, 2, 3, 4) for level in levels)
            and not args.use_saved_level0
            and not args.primed_only
            and not args.cold_only
        ):
            levels.insert(0, 0)
    elif any(level in (1, 2, 3, 4) for level in requested_levels) and not pair_flags:
        levels = [0, args.level]
    else:
        levels = [args.level]
    _write_progress(
        f"main start: model={args.model} levels={levels} "
        f"rounds={args.rounds} episodes={args.episodes}"
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
            tag=args.tag,
        )
        runner.save_args(args)
        _write_progress("runner created")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        level0_episodes: List[Dict[str, Any]] = []
        if (
            0 not in levels
            and any(level in (1, 2, 3, 4) for level in levels)
            and "primed" in cells
        ):
            level0_episodes = load_saved_level0_episodes(
                model=args.model,
                rounds=args.rounds,
            )
        for level in levels:
            _write_progress(f"level {level} start")
            if level == 0:
                episodes = runner.run_level(
                    level=level,
                    episodes=args.episodes,
                    rounds=args.rounds,
                    progress_callback=lambda completed, total, episode_result, episodes_done, lvl=level: _progress_callback(
                        lvl, completed, total, episodes_done
                    ),
                )
                level0_episodes = episodes
                csv_path = save_episodes_csv(
                    episodes, model=args.model, level=level, timestamp=timestamp
                )
                print(f"[main] saved {csv_path}")
                _write_progress(f"csv saved: {csv_path}")
            elif level in (1, 2, 3, 4):
                episodes = runner.run_level_ablation(
                    level=level,
                    rounds=args.rounds,
                    channel_width=0.60 if level == 4 else 0.38,
                    level0_episodes=level0_episodes,
                    cells=cells,
                )
                csv_path = save_episodes_csv(
                    episodes,
                    model=args.model,
                    level=level,
                    timestamp=timestamp,
                )
                print(f"[main] saved {csv_path}")
                _write_progress(f"csv saved: {csv_path}")
                cold_episodes = [
                    ep for ep in episodes if ep.get("condition") == "cold"
                ]
                primed_episodes = [
                    ep for ep in episodes if ep.get("condition") == "primed"
                ]
                cold_rate = (
                    sum(ep["success"] for ep in cold_episodes)
                    / len(cold_episodes)
                    if cold_episodes
                    else 0.0
                )
                primed_rate = (
                    sum(ep["success"] for ep in primed_episodes)
                    / len(primed_episodes)
                    if primed_episodes
                    else 0.0
                )
                summary_parts: List[str] = []
                if cold_episodes:
                    summary_parts.append(f"cold={cold_rate:.3f}")
                if primed_episodes:
                    summary_parts.append(f"primed={primed_rate:.3f}")
                summary_text = " ".join(summary_parts)
                print(f"[main] level {level}: {summary_text}")
                _write_progress(f"level {level}: {summary_text}")
            elif level == 5:
                episodes = runner.run_level_5(
                    rounds=args.rounds,
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
