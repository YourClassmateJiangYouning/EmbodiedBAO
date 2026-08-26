"""Quantitative "insight vs gradual learning" analysis for EmbodiedBAO.

The module consumes the per-episode JSON files written by ``experiments.py``
(``results/level{level}/{model}/episode_*.json``) and computes four metrics:

1. Step-ness      : max single-step improvement of the smoothed success-rate
                    curve (sliding window, default 5).  > 0.5 => insight,
                    < 0.2 => gradual.
2. Strategy switch: cosine similarity between adjacent episodes' strategy
                    vectors (torso yaw, arm extension, body displacement
                    direction).  A similarity drop below 0.3 marks a switch.
3. Exploration    : per-episode ratio of exploratory actions (backward, left,
                    right, turn_left, turn_right), before vs after insight.
4. One-shot body adjustment: max single torso rotation / total torso rotation;
                    > 0.7 => one-shot, < 0.3 => gradual.

Outputs: a per-model score table (JSON/CSV/Markdown), a learning-curve plot
with the insight point marked, a strategy-switch similarity plot, and an
exploration-change plot.  Plotting is optional (``--no_plot``) so the analysis
also runs in environments without matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


EXPLORATORY_ACTIONS = {"backward", "left", "right", "turn_left", "turn_right"}

STEP_CLASS_ZH = {
    "insight": "顿悟",
    "gradual": "渐悟",
    "intermediate": "过渡",
}

ONE_SHOT_CLASS_ZH = {
    "one_shot": "一次性调整",
    "gradual": "渐进式调整",
    "intermediate": "过渡",
    "none": "无旋转",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_episodes(results_root: str, level: int, model: str) -> List[Dict[str, Any]]:
    """Load and sort per-episode JSON files for one model at one level."""
    pattern = os.path.join(results_root, f"level{level}", model, "episode_*.json")
    episodes: List[Dict[str, Any]] = []
    for path in glob.glob(pattern):
        with open(path, "r", encoding="utf-8") as handle:
            episodes.append(json.load(handle))
    episodes.sort(key=lambda ep: int(ep.get("episode_id", 0)))
    return episodes


def discover_models(results_root: str, level: int) -> List[str]:
    """List models that already have episode results for a level."""
    level_dir = os.path.join(results_root, f"level{level}")
    if not os.path.isdir(level_dir):
        return []
    models: List[str] = []
    for name in sorted(os.listdir(level_dir)):
        model_dir = os.path.join(level_dir, name)
        if not os.path.isdir(model_dir):
            continue
        has_episodes = any(
            fname.startswith("episode_") and fname.endswith(".json")
            for fname in os.listdir(model_dir)
        )
        if has_episodes:
            models.append(name)
    return models


# ---------------------------------------------------------------------------
# Metric 1: Step-ness of the learning curve
# ---------------------------------------------------------------------------


def sliding_window_rate(values: Sequence[float], window: int = 5) -> np.ndarray:
    """Rolling success-rate sequence with ``window`` and min_periods=1."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return arr
    window = max(1, int(window))
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    out = np.empty(n, dtype=float)
    for t in range(n):
        start = max(0, t - window + 1)
        out[t] = (cumsum[t + 1] - cumsum[start]) / (t - start + 1)
    return out


def compute_step_ness(
    successes: Sequence[bool],
    window: int = 5,
    insight_threshold: float = 0.5,
    gradual_threshold: float = 0.2,
) -> Dict[str, Any]:
    """Step-ness of a binary success sequence.

    The smoothed success-rate sequence uses a sliding window (default 5) with
    partial windows at the start, and the pre-episode baseline is 0.  The
    step-ness score is the maximum single-step improvement of that sequence.
    """
    successes = [bool(s) for s in successes]
    rates = sliding_window_rate([float(s) for s in successes], window=window)
    previous = np.concatenate([[0.0], rates[:-1]])
    improvements = rates - previous
    step_ness = float(np.max(improvements)) if len(improvements) else 0.0
    raw = np.diff([0.0] + [float(s) for s in successes])
    max_raw_improvement = float(np.max(raw)) if len(raw) else 0.0
    first_success = next((i + 1 for i, s in enumerate(successes) if s), None)

    if step_ness > insight_threshold:
        step_class = "insight"
    elif step_ness < gradual_threshold:
        step_class = "gradual"
    else:
        step_class = "intermediate"

    return {
        "step_ness": round(step_ness, 4),
        "step_ness_class": step_class,
        "step_ness_class_zh": STEP_CLASS_ZH[step_class],
        "max_raw_step_improvement": round(max_raw_improvement, 4),
        "first_success_episode": first_success,
        "smoothed_success_rate": [round(float(v), 4) for v in rates],
        "successes": successes,
    }


# ---------------------------------------------------------------------------
# Metric 2: Strategy-switch detection
# ---------------------------------------------------------------------------


def _episode_actions(episode: Dict[str, Any]) -> List[str]:
    return [str(step.get("action_taken", "")) for step in episode.get("steps", [])]


def _episode_yaws(episode: Dict[str, Any]) -> np.ndarray:
    yaws = []
    for step in episode.get("steps", []):
        yaw = step.get("torso_orientation", {}).get("yaw", 0.0)
        yaws.append(float(yaw))
    return np.asarray(yaws, dtype=float)


def _episode_hands(episode: Dict[str, Any]) -> List[np.ndarray]:
    hands = []
    for step in episode.get("steps", []):
        hand = step.get("hand_position", [0.0, 0.0, 0.0])
        hands.append(np.asarray(hand, dtype=float))
    return hands


def strategy_vector(episode: Dict[str, Any]) -> Dict[str, Any]:
    """Strategy vector = [torso yaw (circular), arm extension, displacement dir].

    The raw components are exposed as fields; the numeric vector uses
    (cos yaw, sin yaw, arm extension, normalized dx, normalized dz) so that
    cosine similarity is stable against 0/360 degree wrap-around.
    """
    actions = _episode_actions(episode)
    yaws = _episode_yaws(episode)
    hands = _episode_hands(episode)

    mean_yaw = float(np.mean(yaws)) if len(yaws) else 0.0
    yaw_rad = math.radians(mean_yaw)

    n = max(1, len(actions))
    reach = sum(a == "reach" for a in actions) / n
    retreat = sum(a == "retreat" for a in actions) / n
    extension = float(np.clip(reach - retreat, 0.0, 1.0))

    displacement = np.zeros(2, dtype=float)
    if len(hands) >= 2:
        displacement = (
            hands[-1][[0, 2]] - hands[0][[0, 2]]
        )
    norm = float(np.linalg.norm(displacement))
    disp_unit = displacement / norm if norm > 1e-6 else np.zeros(2, dtype=float)
    disp_deg = (
        math.degrees(math.atan2(displacement[1], displacement[0]))
        if norm > 1e-6
        else 0.0
    )

    vector = np.array(
        [
            math.cos(yaw_rad),
            math.sin(yaw_rad),
            extension,
            disp_unit[0],
            disp_unit[1],
        ],
        dtype=float,
    )
    return {
        "vector": vector,
        "torso_yaw_deg": round(mean_yaw, 3),
        "arm_extension": round(extension, 4),
        "displacement_xz": [round(float(v), 4) for v in displacement],
        "displacement_deg": round(disp_deg, 3),
    }


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a < 1e-9 and norm_b < 1e-9:
        return 1.0
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def detect_strategy_switches(
    episodes: Sequence[Dict[str, Any]], threshold: float = 0.3
) -> Dict[str, Any]:
    vectors = [strategy_vector(ep) for ep in episodes]
    similarities = [
        cosine_similarity(vectors[i]["vector"], vectors[i + 1]["vector"])
        for i in range(len(vectors) - 1)
    ]
    switch_episodes = [
        i + 2 for i, sim in enumerate(similarities) if sim < threshold
    ]
    return {
        "strategy_vectors": vectors,
        "similarities": [round(float(s), 4) for s in similarities],
        "switch_episodes": switch_episodes,
        "switch_count": len(switch_episodes),
        "first_switch_episode": switch_episodes[0] if switch_episodes else None,
    }


# ---------------------------------------------------------------------------
# Metric 3: Exploratory behavior
# ---------------------------------------------------------------------------


def compute_exploration(episodes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ratios: List[float] = []
    for ep in episodes:
        actions = _episode_actions(ep)
        n = len(actions)
        ratios.append(
            sum(a in EXPLORATORY_ACTIONS for a in actions) / n if n else 0.0
        )

    first_success = next(
        (i + 1 for i, ep in enumerate(episodes) if ep.get("success")), None
    )
    if first_success is None:
        before_insight = ratios
        after_insight: List[float] = []
    else:
        before_insight = ratios[: first_success - 1]
        after_insight = ratios[first_success - 1 :]

    failed = [r for r, ep in zip(ratios, episodes) if not ep.get("success")]
    succeeded = [r for r, ep in zip(ratios, episodes) if ep.get("success")]

    def mean(values: List[float]) -> Optional[float]:
        return float(np.mean(values)) if values else None

    before_avg = mean(before_insight)
    after_avg = mean(after_insight)
    delta = (
        after_avg - before_avg
        if before_avg is not None and after_avg is not None
        else None
    )
    return {
        "ratios": [round(r, 4) for r in ratios],
        "failed_episodes_avg": round(mean(failed), 4) if failed else None,
        "successful_episodes_avg": round(mean(succeeded), 4) if succeeded else None,
        "before_insight_avg": round(before_avg, 4) if before_avg is not None else None,
        "after_insight_avg": round(after_avg, 4) if after_avg is not None else None,
        "delta_after_minus_before": round(delta, 4) if delta is not None else None,
        "first_success_episode": first_success,
    }


def compute_mirrorbench_metrics(
    episodes: Sequence[Dict[str, Any]], dist_threshold: float = 0.03
) -> Dict[str, Any]:
    """Supplementary MirrorBench metrics (TSR/SIR/FCR/PCR) and invalid count.

    MirrorBench defines these from the distance trajectory:
        TSR = I(d_T <= d_th)
        SIR = mean over steps of I(d_{i+1} < d_i)
        FCR = 1 - (d_T - d_th) / (d_0 - d_th)
        PCR = 1 - (min_i d_i - d_th) / (d_0 - d_th)
    """
    tsr_list: List[float] = []
    sir_list: List[float] = []
    fcr_list: List[float] = []
    pcr_list: List[float] = []
    invalid_count = 0
    for episode in episodes:
        tsr_list.append(1.0 if episode.get("success") else 0.0)
        dists = [
            float(step.get("distance_to_target", float("nan")))
            for step in episode.get("steps", [])
        ]
        dists = [d for d in dists if d == d]
        if len(dists) >= 2:
            improvements = sum(1 for a, b in zip(dists, dists[1:]) if b < a - 1e-5)
            sir_list.append(improvements / (len(dists) - 1))
        invalid_count += sum(
            1
            for step in episode.get("steps", [])
            if step.get("action_taken") == "invalid"
        )
        if dists:
            denominator = dists[0] - dist_threshold
            if abs(denominator) > 1e-9:
                fcr_list.append(1.0 - (dists[-1] - dist_threshold) / denominator)
                pcr_list.append(1.0 - (min(dists) - dist_threshold) / denominator)
    return {
        "tsr": round(float(np.mean(tsr_list)), 4) if tsr_list else None,
        "sir": round(float(np.mean(sir_list)), 4) if sir_list else None,
        "fcr": round(float(np.mean(fcr_list)), 4) if fcr_list else None,
        "pcr": round(float(np.mean(pcr_list)), 4) if pcr_list else None,
        "invalid_response_count": invalid_count,
    }


# ---------------------------------------------------------------------------
# Metric 4: One-shot body adjustment
# ---------------------------------------------------------------------------


def _rotation_runs(yaw_deg: np.ndarray) -> Tuple[List[float], float]:
    """Contiguous same-direction rotation runs; returns (run sums, max run)."""
    if len(yaw_deg) < 2:
        return [], 0.0
    diffs = np.diff(yaw_deg)
    runs: List[float] = []
    current = 0.0
    current_sign = 0.0
    for value in diffs:
        if abs(value) < 1e-9:
            if abs(current) > 1e-9:
                runs.append(abs(current))
            current = 0.0
            current_sign = 0.0
            continue
        sign = 1.0 if value > 0 else -1.0
        if current_sign == 0.0:
            current_sign = sign
        if sign != current_sign:
            runs.append(abs(current))
            current = 0.0
            current_sign = sign
        current += value
    if abs(current) > 1e-9:
        runs.append(abs(current))
    return runs, float(max(runs)) if runs else 0.0


def compute_one_shot_adjustment(
    episode: Dict[str, Any],
    one_shot_threshold: float = 0.7,
    gradual_threshold: float = 0.3,
) -> Dict[str, Any]:
    yaws = _episode_yaws(episode)
    if len(yaws) < 2 or float(np.max(np.abs(yaws - yaws[0]))) < 1e-9:
        return {
            "total_rotation_deg": 0.0,
            "max_single_rotation_deg": 0.0,
            "one_shot_index": 0.0,
            "one_shot_class": "none",
            "one_shot_class_zh": ONE_SHOT_CLASS_ZH["none"],
            "max_rotation_run_deg": 0.0,
        }

    unwrapped = np.degrees(np.unwrap(np.radians(yaws)))
    diffs = np.abs(np.diff(unwrapped))
    total_rotation = float(np.sum(diffs))
    max_single = float(np.max(diffs))
    index = max_single / total_rotation if total_rotation > 1e-9 else 0.0
    _, max_run = _rotation_runs(unwrapped)

    if index > one_shot_threshold:
        one_class = "one_shot"
    elif index < gradual_threshold:
        one_class = "gradual"
    else:
        one_class = "intermediate"

    return {
        "total_rotation_deg": round(total_rotation, 3),
        "max_single_rotation_deg": round(max_single, 3),
        "one_shot_index": round(index, 4),
        "one_shot_class": one_class,
        "one_shot_class_zh": ONE_SHOT_CLASS_ZH[one_class],
        "max_rotation_run_deg": round(max_run, 3),
    }


# ---------------------------------------------------------------------------
# Aggregate analysis
# ---------------------------------------------------------------------------


def analyze_episodes(
    episodes: Sequence[Dict[str, Any]],
    model: str,
    level: int,
    window: int = 5,
    insight_threshold: float = 0.5,
    gradual_threshold: float = 0.2,
    switch_threshold: float = 0.3,
    one_shot_threshold: float = 0.7,
    gradual_adjust_threshold: float = 0.3,
) -> Dict[str, Any]:
    successes = [bool(ep.get("success", False)) for ep in episodes]
    step_ness = compute_step_ness(
        successes,
        window=window,
        insight_threshold=insight_threshold,
        gradual_threshold=gradual_threshold,
    )
    switches = detect_strategy_switches(episodes, threshold=switch_threshold)
    exploration = compute_exploration(episodes)
    mirrorbench_metrics = compute_mirrorbench_metrics(episodes)

    one_shot_rows: List[Dict[str, Any]] = []
    for ep in episodes:
        row = compute_one_shot_adjustment(
            ep,
            one_shot_threshold=one_shot_threshold,
            gradual_threshold=gradual_adjust_threshold,
        )
        row["episode_id"] = ep.get("episode_id", 0)
        one_shot_rows.append(row)

    indexes = [row["one_shot_index"] for row in one_shot_rows]
    mean_index = float(np.mean(indexes)) if indexes else 0.0
    one_shot_ratio = (
        sum(row["one_shot_class"] == "one_shot" for row in one_shot_rows)
        / len(one_shot_rows)
        if one_shot_rows
        else 0.0
    )
    if mean_index > one_shot_threshold:
        mean_class = "one_shot"
    elif mean_index < gradual_adjust_threshold:
        mean_class = "gradual"
    else:
        mean_class = "intermediate"

    return {
        "model": model,
        "level": level,
        "episodes": len(episodes),
        "step_ness": step_ness,
        "strategy_switches": switches,
        "exploration": exploration,
        "mirrorbench_metrics": mirrorbench_metrics,
        "one_shot_adjustment": {
            "per_episode": one_shot_rows,
            "mean_index": round(mean_index, 4),
            "one_shot_ratio": round(one_shot_ratio, 4),
            "class": mean_class,
            "class_zh": ONE_SHOT_CLASS_ZH[mean_class],
        },
    }


def table_row(analysis: Dict[str, Any]) -> Dict[str, Any]:
    step = analysis["step_ness"]
    switches = analysis["strategy_switches"]
    exploration = analysis["exploration"]
    one_shot = analysis["one_shot_adjustment"]
    return {
        "model": analysis["model"],
        "level": analysis["level"],
        "episodes": analysis["episodes"],
        "step_ness": step["step_ness"],
        "step_ness_class": step["step_ness_class"],
        "step_ness_class_zh": step["step_ness_class_zh"],
        "first_success_episode": step["first_success_episode"],
        "strategy_switch_count": switches["switch_count"],
        "first_strategy_switch_episode": switches["first_switch_episode"],
        "exploration_failed_avg": exploration["failed_episodes_avg"],
        "exploration_success_avg": exploration["successful_episodes_avg"],
        "exploration_before_insight_avg": exploration["before_insight_avg"],
        "exploration_after_insight_avg": exploration["after_insight_avg"],
        "exploration_delta": exploration["delta_after_minus_before"],
        "one_shot_index_mean": one_shot["mean_index"],
        "one_shot_ratio": one_shot["one_shot_ratio"],
        "one_shot_class": one_shot["class"],
        "one_shot_class_zh": one_shot["class_zh"],
    }


def format_markdown_table(rows: Sequence[Dict[str, Any]]) -> str:
    header = [
        "Model",
        "Level",
        "Step-ness",
        "Step Class",
        "First Success",
        "Switches",
        "Explor. Fail",
        "Explor. Succ",
        "Explor. Delta",
        "One-shot Idx",
        "One-shot Class",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        values = [
            str(row["model"]),
            str(row["level"]),
            f"{row['step_ness']:.4f}",
            row["step_ness_class_zh"],
            "-" if row["first_success_episode"] is None else str(row["first_success_episode"]),
            str(row["strategy_switch_count"]),
            "-" if row["exploration_failed_avg"] is None else f"{row['exploration_failed_avg']:.4f}",
            "-" if row["exploration_success_avg"] is None else f"{row['exploration_success_avg']:.4f}",
            "-" if row["exploration_delta"] is None else f"{row['exploration_delta']:.4f}",
            f"{row['one_shot_index_mean']:.4f}",
            row["one_shot_class_zh"],
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_learning_curve(analysis: Dict[str, Any], out_path: str) -> None:
    plt = _import_pyplot()
    step = analysis["step_ness"]
    rates = step["smoothed_success_rate"]
    x = list(range(1, len(rates) + 1))
    plt.figure(figsize=(8, 4))
    plt.plot(x, rates, marker="o", markersize=3, label="Success rate (window=5)")
    first_success = step["first_success_episode"]
    if first_success is not None and first_success <= len(rates):
        plt.scatter(
            [first_success],
            [rates[first_success - 1]],
            color="red",
            s=90,
            marker="*",
            label="Insight point",
        )
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.title(f"Learning Curve - {analysis['model']} (Level {analysis['level']})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_strategy_switch(analysis: Dict[str, Any], out_path: str, threshold: float = 0.3) -> None:
    plt = _import_pyplot()
    switches = analysis["strategy_switches"]
    similarities = switches["similarities"]
    if not similarities:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No adjacent episodes to compare", ha="center", va="center")
        plt.title(f"Strategy Switch - {analysis['model']} (Level {analysis['level']})")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return
    x = list(range(2, len(similarities) + 2))
    switch_x = switches["switch_episodes"]
    switch_y = [
        similarities[i - 2] for i in switch_x if 2 <= i <= len(similarities) + 1
    ]
    plt.figure(figsize=(8, 4))
    plt.plot(x, similarities, marker="o", markersize=3, label="Cosine similarity")
    plt.axhline(threshold, color="red", linestyle="--", linewidth=0.8, label="Switch threshold")
    if switch_x:
        plt.scatter(switch_x, switch_y, color="red", s=60, marker="x", label="Strategy switch")
    plt.xlabel("Episode")
    plt.ylabel("Strategy Cosine Similarity")
    plt.title(f"Strategy Switch Detection - {analysis['model']} (Level {analysis['level']})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_exploration(analysis: Dict[str, Any], out_path: str) -> None:
    plt = _import_pyplot()
    exploration = analysis["exploration"]
    ratios = exploration["ratios"]
    x = list(range(1, len(ratios) + 1))
    plt.figure(figsize=(8, 4))
    plt.bar(x, ratios, color="#4c72b0", alpha=0.75, label="Exploration ratio")
    first_success = exploration["first_success_episode"]
    if first_success is not None:
        plt.axvline(first_success, color="red", linestyle="--", linewidth=1.0, label="Insight point")
    before = exploration["before_insight_avg"]
    after = exploration["after_insight_avg"]
    if before is not None:
        plt.axhline(before, color="orange", linestyle=":", linewidth=1.0, label="Before insight avg")
    if after is not None:
        plt.axhline(after, color="green", linestyle=":", linewidth=1.0, label="After insight avg")
    plt.xlabel("Episode")
    plt.ylabel("Exploration Ratio")
    plt.ylim(0.0, 1.0)
    plt.title(f"Exploratory Behavior - {analysis['model']} (Level {analysis['level']})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantify insight vs gradual learning in BAO experiments."
    )
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--level", type=int, default=2, help="Level to analyze (usually 2 or 3)")
    parser.add_argument("--models", type=str, nargs="*", default=None, help="Model names; default: all")
    parser.add_argument("--out_dir", type=str, default="analysis")
    parser.add_argument("--window", type=int, default=5, help="Sliding window for success rate")
    parser.add_argument("--insight_threshold", type=float, default=0.5)
    parser.add_argument("--gradual_threshold", type=float, default=0.2)
    parser.add_argument("--switch_threshold", type=float, default=0.3)
    parser.add_argument("--one_shot_threshold", type=float, default=0.7)
    parser.add_argument("--gradual_adjust_threshold", type=float, default=0.3)
    parser.add_argument("--no_plot", action="store_true", help="Skip matplotlib plots")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = args.models or discover_models(args.results_root, args.level)
    if not models:
        print(
            f"No episode results found under {os.path.join(args.results_root, 'level' + str(args.level))}."
        )
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for model in models:
        episodes = load_episodes(args.results_root, args.level, model)
        if not episodes:
            print(f"[analysis] skip {model}: no episodes")
            continue
        analysis = analyze_episodes(
            episodes,
            model=model,
            level=args.level,
            window=args.window,
            insight_threshold=args.insight_threshold,
            gradual_threshold=args.gradual_threshold,
            switch_threshold=args.switch_threshold,
            one_shot_threshold=args.one_shot_threshold,
            gradual_adjust_threshold=args.gradual_adjust_threshold,
        )
        rows.append(table_row(analysis))

        safe_model = model.replace("/", "-").replace("\\", "-")
        json_path = os.path.join(args.out_dir, f"analysis_level{args.level}_{safe_model}.json")
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(analysis, handle, indent=2, ensure_ascii=False)
        print(f"[analysis] {model}: step_ness={analysis['step_ness']['step_ness']:.4f} "
              f"({analysis['step_ness']['step_ness_class_zh']}), "
              f"switches={analysis['strategy_switches']['switch_count']}, "
              f"one_shot_index={analysis['one_shot_adjustment']['mean_index']:.4f}, "
              f"TSR={analysis['mirrorbench_metrics']['tsr']}, "
              f"SIR={analysis['mirrorbench_metrics']['sir']}")

        if not args.no_plot:
            try:
                plot_learning_curve(
                    analysis, os.path.join(args.out_dir, f"learning_curve_{safe_model}.png")
                )
                plot_strategy_switch(
                    analysis,
                    os.path.join(args.out_dir, f"strategy_switch_{safe_model}.png"),
                    threshold=args.switch_threshold,
                )
                plot_exploration(
                    analysis, os.path.join(args.out_dir, f"exploration_{safe_model}.png")
                )
            except Exception as exc:
                print(f"[analysis] plotting failed for {model}: {exc}")

    if not rows:
        print("[analysis] no analyzable episodes.")
        return 1

    table_path = os.path.join(args.out_dir, f"score_table_level{args.level}.csv")
    with open(table_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    markdown = format_markdown_table(rows)
    md_path = os.path.join(args.out_dir, f"score_table_level{args.level}.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(markdown + "\n")

    print("\n" + markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
