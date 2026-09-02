"""BAO four-level evaluation protocol.

Implements the Body-as-Obstacle benchmark protocol for an MLLM-powered
Unitree H1 agent:

    Level 0  Guided Solution          (wall/channel/solution + full CoT)
    Level 1  Autonomous Reasoning     (wall/channel known, no solution)
    Level 2  Implicit Obstacle        (generic task only)
    Level 3  Self-Referential Adjust  (generic task only, minimal priors)

Every episode resets the Isaac Sim scene, runs at most ``max_steps``
(default 30), and terminates only on task success or step exhaustion. Wall
collisions are recorded but never terminate the episode.  Each step follows
the canonical loop:

    image  = env.get_camera_image()
    state  = env.get_robot_state()
    prompt = build_prompt(level, state, ...)
    action = ai_agent.get_action(prompt, image, state, history)
    result = env.execute_action(action)
    success = env.check_success()
    collision = env.check_collision_with_wall()

The module only imports Isaac Sim inside ``main()``, so prompt building and
action parsing can be unit-tested with plain Python.

Expected ``ai_agent`` interface (one of):
    ai_agent.get_action(prompt, image, state, history, options) -> action
    ai_agent.create_agent(model, log_file) -> object with get_action(...)
    ai_agent.get_agent(model, log_file)   -> object with get_action(...)

An action may be a string like "Choice: [3]", or an object exposing ``name``,
``action`` or ``action_choice``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from environment import ACTIONS


ACTION_DESCRIPTIONS: Dict[str, str] = {
    "forward": "translate 5 cm along the world +x axis",
    "backward": "translate 5 cm along the world -x axis",
    "left": "translate 5 cm along the world -z axis",
    "right": "translate 5 cm along the world +z axis",
    "turn_left": "rotate the torso 15 degrees counterclockwise around +y",
    "turn_right": "rotate the torso 15 degrees clockwise around +y",
    "reach": "extend the right arm forward",
    "retreat": "pull the right arm back",
}

ACTION_OPTIONS_STRING: str = "\n".join(
    f"{i + 1}. {name} - {ACTION_DESCRIPTIONS[name]}" for i, name in enumerate(ACTIONS)
)


LEVEL0_FULL_PROMPT = """You are a Unitree H1 humanoid robot in a simulation environment.

【Scene Description】
A transparent wall spans the entire 4m × 4m scene at x=2.0m, dividing it into front and back halves.
- The wall is 2.0m tall and extends from z=-2.0m to z=2.0m
- In the center of the wall at z=0, there is a vertical opening (passage) that goes from the floor to the top of the wall
- The opening is 0.38m wide

【Your Body Parameters】
- Shoulder width: 0.57m
- Body thickness (sideways): 0.22m
- Arm length: 0.338m

【Critical Understanding】
- Shoulder width (0.57m) > opening width (0.38m) → You CANNOT pass through facing forward
- Body thickness (0.22m) < opening width (0.38m) → You CAN pass through if you turn sideways
- Arm length (0.338m) < distance from wall to ball (0.40m) → You CANNOT reach the ball from the front

【Task】
A green ball is located behind the wall at x=2.4m, z=0, height 1.2m. Reach the green ball.

【Complete Solution - Follow These Steps】
Step 1: Move forward to the opening at x=2.0m, z=0
Step 2: Turn your body sideways (rotate 90 degrees left or right)
Step 3: Move forward through the opening while staying sideways
Step 4: Once through the wall (x > 2.0m), turn back to face the ball
Step 5: Reach your arm forward to touch the green ball

【Available Actions】
You must respond with a JSON object containing one action:
{"action": "forward"}  - move forward 5cm
{"action": "backward"} - move backward 5cm
{"action": "left"}     - move left 5cm
{"action": "right"}    - move right 5cm
{"action": "turn_left"} - rotate body 15 degrees counterclockwise
{"action": "turn_right"} - rotate body 15 degrees clockwise
{"action": "reach"}    - extend arm forward
{"action": "retreat"}  - retract arm back"""

LEVEL_PROMPTS: Dict[int, str] = {
    0: LEVEL0_FULL_PROMPT,
    1: (
        "There is a transparent wall across the entire scene with a narrow vertical opening "
        "in the center. You need to get the green ball behind the wall."
    ),
    2: "Your task is to reach the green ball in front of you.",
    3: "Your task is to reach the green ball in front of you.",
}

def build_prompt(
    level: int,
    state: Dict[str, Any],
    history: Optional[Sequence[Dict[str, str]]] = None,
    options: str = ACTION_OPTIONS_STRING,
    max_steps: int = 30,
) -> str:
    """Build the per-level prompt text for one decision step."""
    if level == 0:
        parts = [LEVEL_PROMPTS[0]]
        if history:
            lines = ["Action history (most recent first):"]
            for item in list(history)[-6:][::-1]:
                lines.append(f"- {item.get('action')} -> {item.get('feedback')}")
            parts.append("\n".join(lines))
        position = state.get("position", [0.0, 0.0, 0.0])
        yaw = float(
            state.get(
                "torso_rotation",
                state.get("orientation", {}).get("yaw", 0.0),
            )
        )
        distance = float(state.get("distance_to_target", float("nan")))
        parts.append(
            "\n".join(
                [
                    "Current state:",
                    f"- position (x, y, z): [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]",
                    f"- torso yaw (degrees): {yaw:.1f}",
                    f"- distance to green ball surface: {distance:.3f} m",
                    f"You have at most {max_steps} steps in this episode.",
                ]
            )
        )
        return "\n\n".join(parts)

    lines = [
        "You are a Unitree H1 humanoid robot inside a 4m x 4m room.",
        "You perceive the scene through your head camera and control your body "
        "with discrete actions.",
        LEVEL_PROMPTS[level],
    ]

    if history:
        lines.append("Action history (most recent first):")
        for item in list(history)[-6:][::-1]:
            lines.append(f"- {item.get('action')} -> {item.get('feedback')}")

    lines.append("Current state:")
    position = state.get("position", [0.0, 0.0, 0.0])
    yaw = float(state.get("orientation", {}).get("yaw", 0.0))
    distance = float(state.get("distance_to_target", float("nan")))
    lines.append(f"- position (x, y, z): [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
    lines.append(f"- torso yaw (degrees): {yaw:.1f}")
    lines.append(f"- distance to green ball surface: {distance:.3f} m")
    lines.append("")
    lines.append("Available actions:")
    lines.append(options)
    lines.append("")
    lines.append(
        'Reply with exactly one JSON object: '
        '{"action": "<action>", "confidence": 0.0-1.0, '
        '"reasoning": "<short text>"}.'
    )
    lines.append(f"You have at most {max_steps} steps in this episode.")
    return "\n".join(lines)


def parse_action_text(text: Any) -> Optional[str]:
    """Extract an action name from a model response like 'Choice: [3]'."""
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    match = re.search(r"Choice:?[\n\s]*\[?(\d+)\]?", text, re.IGNORECASE)
    if match:
        index = int(match.group(1))
    else:
        digits = re.findall(r"\d+", text)
        if not digits:
            return None
        index = int(digits[-1])
    if 1 <= index <= len(ACTIONS):
        return ACTIONS[index - 1]
    return None


class AgentAdapter:
    """Thin adapter over the future ``ai_agent`` module."""

    def __init__(self, model: str, log_file: Optional[str] = None) -> None:
        self.model = model
        self.log_file = log_file
        self.history: List[Dict[str, str]] = []
        self.image_history: List[np.ndarray] = []
        self._callable = None
        self._agent = None

        import ai_agent

        if callable(getattr(ai_agent, "get_action", None)):
            self._callable = ai_agent.get_action
        elif callable(getattr(ai_agent, "create_agent", None)):
            self._agent = ai_agent.create_agent(model=model, log_file=log_file)
        elif callable(getattr(ai_agent, "get_agent", None)):
            self._agent = ai_agent.get_agent(model=model, log_file=log_file)
        else:
            raise RuntimeError(
                "ai_agent must expose get_action(prompt, image, state, history, options) "
                "or a create_agent/get_agent factory."
            )

    def query(
        self, prompt: str, image: np.ndarray, state: Dict[str, Any]
    ) -> Tuple[Optional[str], str, float]:
        """Call the model and return (action_name, raw_response, latency_ms)."""
        t0 = time.perf_counter()
        if self._callable is not None:
            try:
                raw = self._callable(
                    prompt=prompt,
                    image=image,
                    state=state,
                    history=self.history,
                    options=ACTION_OPTIONS_STRING,
                    model_name=self.model,
                    log_file=self.log_file,
                )
            except TypeError:
                try:
                    raw = self._callable(
                        prompt=prompt,
                        image=image,
                        state=state,
                        history=self.history,
                        options=ACTION_OPTIONS_STRING,
                    )
                except TypeError:
                    raw = self._callable(prompt, image, self.history)
        else:
            raw = self._agent.get_action(
                prompt=prompt,
                image=image,
                state=state,
                history=self.history,
                options=ACTION_OPTIONS_STRING,
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        action_name, raw_text = self._normalize(raw)
        self._append_log(prompt, raw_text)
        return action_name, raw_text, latency_ms

    def record(self, action_name: str, feedback: str) -> None:
        self.history.append({"action": action_name, "feedback": feedback})

    def _normalize(self, raw: Any) -> Tuple[Optional[str], str]:
        if isinstance(raw, str):
            return parse_action_text(raw), raw
        if raw is None:
            return None, ""
        if isinstance(raw, dict):
            name = raw.get("action")
            if name not in ACTIONS:
                name = None
            return name, json.dumps(raw, ensure_ascii=False)
        raw_text = str(getattr(raw, "text", "") or raw)
        if hasattr(raw, "action_choice"):
            index = int(getattr(raw, "action_choice"))
            name = ACTIONS[index - 1] if 1 <= index <= len(ACTIONS) else None
            return name, raw_text
        name = getattr(raw, "action", None) or getattr(raw, "name", None)
        if name is None:
            name = str(raw).strip().lower()
        return (name if name in ACTIONS else None), raw_text

    def _append_log(self, prompt: str, raw_text: str) -> None:
        if not self.log_file:
            return
        with open(self.log_file, "a", encoding="utf-8") as handle:
            handle.write("-" * 40 + "\n")
            handle.write(prompt + "\n")
            handle.write("MODEL RESPONSE:\n" + raw_text + "\n")
            handle.write("-" * 40 + "\n")


class BAOExperimentRunner:
    """Runs the four-level protocol against an Isaac Sim BAOEnv instance."""

    def __init__(
        self,
        env: Any,
        model: str,
        max_steps: int = 30,
        max_image_history: int = 1,
        tag: Optional[str] = None,
        save_obs: bool = False,
        seed: int = 0,
        results_root: str = "results",
        logs_root: str = "logs",
    ) -> None:
        self.env = env
        self.model = model.replace("/", "-")
        self.max_steps = int(max_steps)
        self.max_image_history = int(max_image_history)
        self.save_obs = bool(save_obs)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.tag = tag or time.strftime("%Y%m%d-%H%M%S")
        self.results_root = results_root
        self.logs_root = logs_root
        self.log_dir = os.path.join(logs_root, self.tag)
        self.obs_dir = os.path.join(self.log_dir, "obs")
        os.makedirs(self.log_dir, exist_ok=True)
        if self.save_obs:
            os.makedirs(self.obs_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run_all(
        self,
        levels: Sequence[int] = (0, 1, 2, 3),
        episodes_per_level: int = 50,
        rounds: int = 3,
    ) -> Dict[int, List[Dict[str, Any]]]:
        results: Dict[int, List[Dict[str, Any]]] = {}
        for level in levels:
            results[int(level)] = self.run_level(
                level=int(level),
                episodes=episodes_per_level,
                rounds=rounds,
            )
        return results

    def run_level(
        self,
        level: int,
        episodes: int = 50,
        rounds: int = 3,
        progress_callback: Optional[
            Callable[[int, int, Dict[str, Any], List[Dict[str, Any]]], None]
        ] = None,
    ) -> List[Dict[str, Any]]:
        all_episodes: List[Dict[str, Any]] = []
        episodes_per_round = int(episodes)
        total = episodes_per_round * int(rounds)
        completed = 0
        for round_id in range(int(rounds)):
            round_episodes: List[Dict[str, Any]] = []
            for episode_id in range(episodes_per_round):
                episode_result = self._run_episode(
                    level=level,
                    episode_id=episode_id,
                    round_id=round_id,
                )
                round_episodes.append(episode_result)
                all_episodes.append(episode_result)
                self._save_episode(episode_result)
                completed += 1
                print(
                    f"[level {level} round {round_id}] episode {episode_id + 1}/"
                    f"{episodes_per_round} success={episode_result['success']} "
                    f"end={episode_result['end_reason']} "
                    f"steps={len(episode_result['steps'])}"
                )
                if progress_callback is not None:
                    progress_callback(
                        completed,
                        total,
                        episode_result,
                        all_episodes,
                    )
            self._save_summary(level, round_id, round_episodes)
        return all_episodes

    # ------------------------------------------------------------------
    # Single episode
    # ------------------------------------------------------------------

    def _run_episode(self, level: int, episode_id: int, round_id: int = 0) -> Dict[str, Any]:
        agent_log = os.path.join(
            self.log_dir,
            f"round{round_id}_level{level}_episode{episode_id:03d}_agent.txt",
        )
        agent = AgentAdapter(model=self.model, log_file=agent_log)

        _, distance = self.env.reset_scene()
        history: List[Dict[str, str]] = []
        image_history: List[np.ndarray] = []
        steps: List[Dict[str, Any]] = []
        action_sequence: List[str] = []
        wall_collision_count = 0
        total_llm_time_ms = 0.0
        success = False
        end_reason = "max_steps"

        for step in range(self.max_steps):
            rgb = self.env.get_camera_image()
            state = self.env.get_robot_state()
            state["distance_to_target"] = distance

            prompt = build_prompt(
                level=level,
                state=state,
                history=history,
                options=ACTION_OPTIONS_STRING,
                max_steps=self.max_steps,
            )
            action_name, raw_response, latency_ms = agent.query(prompt, rgb, state)
            total_llm_time_ms += latency_ms

            if action_name is None:
                action_taken = "invalid"
                feedback = "invalid action response"
                collision_info: Optional[Dict[str, Any]] = None
                collided = False
                distance = self.env.get_distance_to_target()
                step_success = self.env.check_success()
            else:
                result = self.env.execute_action(action_name)
                action_taken = action_name
                feedback = result.feedback
                collision_info = result.collision
                distance = result.distance
                step_success = result.success
                explicit_collision = self.env.check_collision_with_wall()
                collided = (not result.legal) or explicit_collision
                if explicit_collision and collision_info is None:
                    point = self.env.get_collision_position()
                    if point is not None:
                        collision_info = {"point": point}
            if collided:
                wall_collision_count += 1

            new_state = self.env.get_robot_state()
            hand_pos = new_state.get(
                "hand_position", new_state.get("end_effector_position", [0.0, 0.0, 0.0])
            )
            collision_point = (
                collision_info.get("point") if collision_info else None
            )
            step_record = {
                "episode_id": episode_id,
                "level": level,
                "model_name": self.model,
                "round": round_id,
                "step": step,
                "action": action_taken,
                "hand_x": float(hand_pos[0]) if len(hand_pos) > 0 else None,
                "hand_y": float(hand_pos[1]) if len(hand_pos) > 1 else None,
                "hand_z": float(hand_pos[2]) if len(hand_pos) > 2 else None,
                "hand_distance_to_ball": distance,
                "torso_rotation": self.env.get_torso_rotation(),
                "collision_with_wall": bool(collided),
                "collision_position_x": (
                    float(collision_point[0]) if collision_point else None
                ),
                "collision_position_y": (
                    float(collision_point[1]) if collision_point else None
                ),
                "collision_position_z": (
                    float(collision_point[2]) if collision_point else None
                ),
                "step_success": bool(step_success),
                "llm_response_time_ms": round(latency_ms, 3),
            }
            steps.append(step_record)
            action_sequence.append(action_taken)
            history.append({"action": action_taken, "feedback": feedback})
            agent.record(action_taken, feedback)
            image_history.append(rgb)
            if self.max_image_history > 0:
                image_history = image_history[-self.max_image_history :]

            if self.save_obs:
                self._save_observation(round_id, episode_id, step, action_taken, rgb)

            if step_success:
                success = True
                end_reason = "success"
                break

        return {
            "episode_id": episode_id,
            "level": level,
            "model_name": self.model,
            "round": round_id,
            "success": success,
            "total_steps": len(steps),
            "action_sequence": ",".join(action_sequence),
            "final_distance": self.env.get_distance_to_target(),
            "max_steps": self.max_steps,
            "end_reason": end_reason,
            "wall_collision_count": wall_collision_count,
            "invalid_response_count": sum(
                1 for record in steps if record["action"] == "invalid"
            ),
            "total_llm_time_ms": round(total_llm_time_ms, 3),
            "steps": steps,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _result_dir(self, level: int, round_id: int = 0) -> str:
        path = os.path.join(
            self.results_root, f"level{level}", self.model, f"round{round_id}"
        )
        os.makedirs(path, exist_ok=True)
        return path

    def _save_episode(self, episode: Dict[str, Any]) -> None:
        path = os.path.join(
            self._result_dir(episode["level"], episode.get("round", 0)),
            f"episode_{episode['episode_id']:03d}.json",
        )
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(episode, handle, indent=2, ensure_ascii=False)

    def _save_summary(
        self, level: int, round_id: int, episodes: List[Dict[str, Any]]
    ) -> None:
        path = os.path.join(
            self._result_dir(level, round_id), f"summary_{self.tag}.json"
        )
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self._summarize(level, episodes), handle, indent=2, ensure_ascii=False)

    def _save_observation(
        self,
        round_id: int,
        episode_id: int,
        step: int,
        action: str,
        rgb: np.ndarray,
    ) -> None:
        from PIL import Image

        path = os.path.join(
            self.obs_dir,
            f"round{round_id}_ep{episode_id:03d}_step{step:02d}_{action}.png",
        )
        Image.fromarray(rgb).save(path)

    def save_args(self, args: Any) -> None:
        path = os.path.join(self.log_dir, "args.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, indent=2, ensure_ascii=False)

    @staticmethod
    def _summarize(level: int, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not episodes:
            return {"level": level, "episodes": 0}
        action_counter: Counter = Counter()
        for episode in episodes:
            for record in episode.get("steps", []):
                action_counter.update([str(record.get("action", ""))])
        return {
            "level": level,
            "model_name": episodes[0]["model_name"],
            "rounds": sorted({int(e.get("round", 0)) for e in episodes}),
            "episodes": len(episodes),
            "success_rate": float(np.mean([e["success"] for e in episodes])),
            "avg_steps": float(np.mean([len(e["steps"]) for e in episodes])),
            "avg_llm_time_ms": float(np.mean([e["total_llm_time_ms"] for e in episodes])),
            "end_reason_counts": dict(
                Counter(e["end_reason"] for e in episodes)
            ),
            "wall_collision_counts": [e["wall_collision_count"] for e in episodes],
            "action_counts": dict(action_counter),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BAO four-level protocol.")
    parser.add_argument("--model", type=str, required=True, help="Model name passed to ai_agent")
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per level")
    parser.add_argument("--rounds", type=int, default=3, help="Repeated rounds per model")
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--max_image_history", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="", help="Log/result tag")
    parser.add_argument("--save_obs", action="store_true", help="Save per-step camera PNGs")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--env_config", type=str, default="{}", help="JSON dict passed to BAOEnv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})
    env = None
    try:
        from environment import BAOEnv

        task_dict = json.loads(args.env_config)
        task_dict["headless"] = args.headless
        env = BAOEnv(simulation_app, task_dict=task_dict)
        runner = BAOExperimentRunner(
            env=env,
            model=args.model,
            max_steps=args.max_steps,
            max_image_history=args.max_image_history,
            tag=args.tag,
            save_obs=args.save_obs,
            seed=args.seed,
        )
        runner.save_args(args)
        for level in args.levels:
            episodes = runner.run_level(
                level=level,
                episodes=args.episodes,
                rounds=args.rounds,
            )
            summary = BAOExperimentRunner._summarize(level, episodes)
            print(
                f"[level {level}] success_rate={summary['success_rate']:.3f} "
                f"avg_steps={summary['avg_steps']:.2f}"
            )
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
