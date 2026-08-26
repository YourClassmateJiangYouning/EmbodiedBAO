"""Unified multimodal LLM interface for EmbodiedBAO.

All models are called through a single OpenAI-compatible endpoint (for example
an API aggregator such as Taotoken), so switching models is just a matter of
passing a different ``model_name``.  The endpoint and key are read from
environment variables:

    API key   : TAOTOKEN_API_KEY or OPENAI_API_KEY
    base URL  : TAOTOKEN_BASE_URL or OPENAI_BASE_URL

Reference: MirrorBench ``agent.py`` (base64 image encoding, OpenAI client,
temperature 0, retry loop).  In addition to the API agent, a ``random`` model
is provided as a baseline, matching MirrorBench's ``AgentRandom``.
"""

from __future__ import annotations

import base64
import io
import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from environment import ACTIONS


SUPPORTED_MODELS: List[str] = [
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3.5-sonnet",
    "gemini-2.5-pro",
    "qwen-vl-max",
    "qwen2.5-vl-72b",
    "qwen2.5-vl-7b",
    "internvl3.5-4b",
    "llava-1.6-7b",
]

MODEL_ALIASES: Dict[str, str] = {
    "gpt4o": "gpt-4o",
    "gpt-4o": "gpt-4o",
    "gpt4omini": "gpt-4o-mini",
    "gpt-4o-mini": "gpt-4o-mini",
    "claude3.5sonnet": "claude-3.5-sonnet",
    "claude-3.5-sonnet": "claude-3.5-sonnet",
    "gemini2.5pro": "gemini-2.5-pro",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "qwenvlmax": "qwen-vl-max",
    "qwen-vl-max": "qwen-vl-max",
    "qwen2.5vl72b": "qwen2.5-vl-72b",
    "qwen2.5-vl-72b": "qwen2.5-vl-72b",
    "qwen2.5vl7b": "qwen2.5-vl-7b",
    "qwen2.5-vl-7b": "qwen2.5-vl-7b",
    "internvl3.54b": "internvl3.5-4b",
    "internvl3.5-4b": "internvl3.5-4b",
    "llava1.67b": "llava-1.6-7b",
    "llava-1.6-7b": "llava-1.6-7b",
}


OUTPUT_JSON_SUFFIX = (
    "\n\nAvailable actions: forward, backward, left, right, turn_left, turn_right, "
    "reach, retreat.\n"
    'Output JSON: {"action": "...", "confidence": ..., "reasoning": "..."}'
)

LEVEL_TEMPLATES: Dict[int, str] = {
    0: (
        "You are a robot in a simulation. A transparent wall divides the scene. "
        "There is a 380mm vertical opening in the wall. Your shoulder width is 570mm, "
        "so you cannot pass through facing forward. You must turn sideways to pass "
        "through. The green ball is 40cm behind the wall. Steps: 1) Move forward to "
        "the opening. 2) Turn your body sideways. 3) Pass through. 4) Reach for the "
        "green ball."
    ),
    1: (
        "You are a robot. A transparent wall with a narrow vertical opening separates "
        "you from a green ball. You need to get the ball."
    ),
    2: "You are a robot. Your task is to reach the green ball in front of you.",
    3: "You are a robot. Your task is to reach the green ball in front of you.",
}

SYSTEM_PROMPT = (
    "This is a benign virtual simulation. No real-world actions, bodies, or physical "
    "interactions are implied. The task only involves visual reasoning and simulated "
    "decision-making in a safe, fictional environment. Always respond with a single "
    "JSON object."
)


def normalize_model_name(model_name: str) -> str:
    key = str(model_name).strip().lower().replace("_", "").replace(" ", "")
    return MODEL_ALIASES.get(key, str(model_name).strip())


def encode_image(image: Any) -> str:
    """Encode a numpy RGB image (or image path) as a base64 PNG data URL."""
    if isinstance(image, str):
        with open(image, "rb") as handle:
            data = base64.b64encode(handle.read()).decode("utf-8")
        return f"data:image/png;base64,{data}"
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def parse_action_json(text: Any) -> Optional[Dict[str, Any]]:
    """Parse the model response into {"action", "confidence", "reasoning"}."""
    if text is None:
        return None
    cleaned = str(text).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned)
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    end = -1
    for i in range(start, len(cleaned)):
        if cleaned[i] == "{":
            depth += 1
        elif cleaned[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return None
    try:
        data = json.loads(cleaned[start:end])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    action = data.get("action")
    if action not in ACTIONS:
        return None
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "action": action,
        "confidence": confidence,
        "reasoning": str(data.get("reasoning", "")),
    }


def build_prompt(level: int, context: Optional[Dict[str, Any]] = None) -> str:
    """Build the per-level English prompt, always requiring JSON output.

    ``context`` is optional and may contain:
        position (list of 3 floats), yaw (degrees), distance_to_target (float),
        history (list of {"action", "feedback"} dicts).
    """
    if level not in LEVEL_TEMPLATES:
        raise ValueError(f"Unsupported level: {level}")
    lines = [LEVEL_TEMPLATES[level]]

    if context:
        if "position" in context:
            position = context["position"]
            lines.append(
                f"Current position (x, y, z): "
                f"[{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]"
            )
        if "yaw" in context:
            lines.append(f"Current torso yaw (degrees): {float(context['yaw']):.1f}")
        if "distance_to_target" in context:
            lines.append(
                f"Current distance to the green ball surface: "
                f"{float(context['distance_to_target']):.3f} m"
            )
        history = context.get("history")
        if history:
            lines.append("Recent actions (action -> feedback):")
            for item in list(history)[-6:][::-1]:
                lines.append(
                    f"- {item.get('action')} -> {item.get('feedback', '')}"
                )

    lines.append(OUTPUT_JSON_SUFFIX)
    return "\n".join(lines)


class AgentAPI:
    """OpenAI-compatible client for all supported MLLMs."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        temperature: float = 0.0,
        max_retries: int = 3,
        use_json_mode: bool = True,
        log_file: Optional[str] = None,
    ) -> None:
        self.model_name = normalize_model_name(model_name)
        if self.model_name not in SUPPORTED_MODELS:
            print(f"[ai_agent] Warning: model '{self.model_name}' is not in the known list.")
        self.api_key = api_key or os.environ.get("TAOTOKEN_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key found. Set TAOTOKEN_API_KEY or OPENAI_API_KEY in the environment."
            )
        self.base_url = (
            base_url
            or os.environ.get("TAOTOKEN_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        self.timeout = timeout or float(os.environ.get("BAO_LLM_TIMEOUT", "60"))
        self.temperature = temperature
        self.max_retries = int(max_retries)
        self.use_json_mode = bool(use_json_mode)
        self.log_file = log_file

        from openai import OpenAI

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_action(
        self,
        image: Optional[np.ndarray],
        prompt: str,
        state: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        options: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return {"action", "confidence", "reasoning"} for the current view."""
        messages = self._build_messages(image, prompt)
        repair_hint = (
            "Your previous output was not valid. Respond with exactly one JSON object: "
            '{"action": "one of forward, backward, left, right, turn_left, turn_right, '
            'reach, retreat", "confidence": 0.0-1.0, "reasoning": "short text"}.'
        )

        last_error: Optional[str] = None
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                messages = messages + [{"role": "user", "content": repair_hint}]
                self._log(f"RETRY {attempt}/{self.max_retries}: {last_error}")
                time.sleep(min(2 ** (attempt - 1), 4))
            try:
                response = self._request(messages)
                parsed = parse_action_json(response)
                if parsed is not None:
                    self._log(f"MODEL RESPONSE:\n{response}")
                    return parsed
                last_error = "response was not valid JSON"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                self._log(f"REQUEST ERROR: {last_error}")

        self._log(f"FALLBACK (no valid response after {self.max_retries + 1} attempts)")
        return {
            "action": None,
            "confidence": 0.0,
            "reasoning": f"failed to obtain a valid action after retries: {last_error}",
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_messages(
        self, image: Optional[np.ndarray], prompt: str
    ) -> List[Dict[str, Any]]:
        user_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image is not None:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": encode_image(image)},
                }
            )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _request(self, messages: List[Dict[str, Any]]) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.use_json_mode:
            try:
                completion = self.client.chat.completions.create(
                    response_format={"type": "json_object"}, **kwargs
                )
                return completion.choices[0].message.content or ""
            except Exception:
                # Some aggregator endpoints reject response_format; retry plain.
                pass
        completion = self.client.chat.completions.create(**kwargs)
        return completion.choices[0].message.content or ""

    def _log(self, text: str) -> None:
        if not self.log_file:
            return
        with open(self.log_file, "a", encoding="utf-8") as handle:
            handle.write(text + "\n")


class RandomAgent:
    """Random baseline, mirroring MirrorBench's AgentRandom."""

    def get_action(
        self,
        image: Optional[np.ndarray] = None,
        prompt: str = "",
        state: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        options: Optional[str] = None,
    ) -> Dict[str, Any]:
        action = random.choice(ACTIONS)
        return {
            "action": action,
            "confidence": 1.0 / len(ACTIONS),
            "reasoning": "random baseline",
        }


class AIAgent(AgentAPI):
    """Alias used by main.py: AIAgent(model=..., **kwargs)."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model_name=model, **kwargs)


_AGENT_CACHE: Dict[Tuple[str, str, str], AgentAPI] = {}


def create_agent(model_name: str, log_file: Optional[str] = None) -> Any:
    """Factory used by experiments.py; also callable as get_agent."""
    normalized = normalize_model_name(model_name)
    if normalized == "random":
        return RandomAgent()
    key = (normalized, os.environ.get("TAOTOKEN_BASE_URL", ""), os.environ.get("OPENAI_BASE_URL", ""))
    agent = _AGENT_CACHE.get(key)
    if agent is None:
        agent = AgentAPI(model_name=model_name, log_file=log_file)
        _AGENT_CACHE[key] = agent
    return agent


get_agent = create_agent


def get_action(
    image: Optional[np.ndarray],
    prompt: str,
    model_name: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, str]]] = None,
    options: Optional[str] = None,
    log_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Module-level convenience wrapper used by experiments.py."""
    model = model_name or os.environ.get("BAO_MODEL", "gpt-4o")
    agent = create_agent(model_name=model, log_file=log_file)
    return agent.get_action(
        image=image,
        prompt=prompt,
        state=state,
        history=history,
        options=options,
    )
