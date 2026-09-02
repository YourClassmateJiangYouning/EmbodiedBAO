# EmbodiedBAO

Embodied intelligence experiment implementing the psychological
"Body-as-Obstacle" (BAO) task in NVIDIA Isaac Sim.

## Experimental Scene

- Scene size: 4m x 4m. World axes: x forward (0 to 4), y up (ground at
  y = 0), z lateral (-2 to 2).
- Transparent acrylic wall at x = 2m: 4m wide (spans z), 2m high, with
  physics collision. It splits the scene into a front area (x < 2) and a
  rear area (x > 2).
- Vertical channel at the center of the wall (z = 0), 380mm wide, running
  from the ground to the top of the wall. It is the only passage between
  the two areas.
- Unitree H1 robot starts in the front area, 50cm in front of the channel,
  at (1.5, 0, 0), facing +x.
- Green target ball (8cm diameter) is 40cm behind the channel at
  (2.4, 1.2, 0), at height 1.2m.
- Channel width 380mm < robot shoulder width 570mm: the robot cannot pass
  through facing forward.
- Channel width 380mm > robot torso thickness 220mm: the robot can pass
  through sideways.
- Target distance 40cm > arm length 34cm: the ball cannot be reached from
  the front side, but it can be reached after passing through the channel.

## Core Research Question

Can an embodied agent demonstrate insight-like body self-awareness in the
BAO task?

## Modules

1. `main.py` - entry point, command line parsing, experiment scheduling.
2. `environment.py` - builds the BAO scene in Isaac Sim.
3. `experiments.py` - implements the Level 0-3 tiered protocol.
4. `ai_agent.py` - unified MLLM interface.
5. `analysis.py` - Step-ness, strategy switching, exploratory behavior, and
   one-shot adjustment metrics.
6. `run_models_smoke.py` - one quick episode per supported MLLM using a
   single Isaac Sim session.

## Installation

- Tested with NVIDIA Isaac Sim 4.5. Set the `ISAACSIM_ROOT` environment
  variable.
- Install dependencies:
  - Windows: `%ISAACSIM_ROOT%\python.bat -m pip install -r requirements.txt`
  - Linux: `$ISAACSIM_ROOT/python.sh -m pip install -r requirements.txt`
- API configuration (OpenAI-compatible endpoint), in priority order:
  - `BOYUE_API_KEY` / `BOYUE_BASE_URL` (lab server; when only
    `BOYUE_API_KEY` is set, `http://35.220.164.252:3888/v1/` is used by
    default)
  - `TAOTOKEN_API_KEY` / `TAOTOKEN_BASE_URL`
  - `OPENAI_API_KEY` / `OPENAI_BASE_URL`
- If the lab machine uses proxies that block the API server, set
  `BAO_DISABLE_PROXY=1`.
- H1 robot asset: auto-detected from Isaac Sim or Isaac Lab built-in
  assets. Override with the `EMBODIEDBAO_H1_USD` environment variable.
- If the H1 asset is not found, locate it on the machine with
  `find ~ -iname "h1*.usd*" 2>/dev/null`, then run
  `export EMBODIEDBAO_H1_USD=/full/path/to/h1.usd` before starting the
  experiment.
- HuggingFace download (MirrorBench-style, keeps the git repo light):
  1. Create a public HuggingFace dataset, for example
     `YourUserName/EmbodiedBAOAssets`, and upload `h1.usd` into it.
  2. Run:
     `export EMBODIEDBAO_ASSETS_REPO=YourUserName/EmbodiedBAOAssets`
     `python download.py`
  3. The file lands at `assets/H1/h1.usd` and is found automatically.
- If no H1 USD is available, download the official Unitree H1 URDF (for
  example from `github.com/unitreerobotics/unitree_ros`, file
  `robots/h1_description/urdf/h1_with_hand.urdf`) and convert it:
  `$ISAACSIM_ROOT/python.sh convert_h1_urdf.py /path/to/h1_with_hand.urdf
  /path/to/output/h1.usd`. Then point `EMBODIEDBAO_H1_USD` at the output
  file or place it at `assets/H1/h1.usd`.

## H1 Robot Data

To inspect the loaded H1 model (stage up axis, links, joints, joint axes,
limits, drive settings, bounding box), run:

```text
$ISAACSIM_ROOT/python.sh inspect_h1.py /path/to/h1.usd --json h1_info.json
```

The output is a JSON report that can be used to verify the robot's actual
kinematics against the task assumptions (shoulder width, torso thickness,
reachable joints).

## Usage

Single level:

```text
%ISAACSIM_ROOT%\python.bat main.py --model gpt-4o --level 0 --episodes 50 --headless
```

All levels:

```text
%ISAACSIM_ROOT%\python.bat main.py --model gemini-2.5-pro --all-levels --headless
```

Level 4 channel-widening experiment:

```text
%ISAACSIM_ROOT%\python.bat main.py --model gpt-4o --level 4 --headless
```

Batch evaluation:

- Windows: `evaluate.bat MODEL_NAME [EPISODES]`
- Linux: `./evaluate.sh MODEL_NAME [EPISODES]`

All-model smoke test:

```text
$ISAACSIM_ROOT/python.sh run_models_smoke.py --headless \
    --env_config '{"robot_physics":true}'
```

Analysis:

```text
%ISAACSIM_ROOT%\python.bat analysis.py --results_root results --level 2 --models gpt-4o
```

Outputs:

- `logs/` - run logs and optional observation images
- `results/` - per-episode JSON, summaries, and CSV files

## Reference Works

### MirrorBench (Guo et al., 2026)

- Adapts the psychological Mirror Self-Recognition (MSR) test to embodied
  MLLMs.
- Uses a Level 0-3 tiered protocol through prompt ablation while keeping
  the physical environment fixed.
  - Level 0: mirror disclosed + full CoT template (guided mirror
    perception).
  - Level 1: mirror disclosed, no CoT (autonomous mirror reasoning).
  - Level 2: mirror not disclosed (implicit mirror discovery).
  - Level 3: neither mirror nor target-on-body disclosed (self-referential
    mirror recognition).
- Metrics: TSR, SIR, FCR, PCR, all derived from the hand-target distance
  trajectory.
- Failure mode: mirror-self confusion, which makes some models perform
  worse than random.

### HumanCLAW (2026)

- Decouples VLM action decisions from low-level motion execution. Atomic
  skills are realized by a half-physics simulator.
- Progressive success: Find -> Nav -> Interact, requiring both geometric
  and model-acknowledged completion.
- Automatic root-cause attribution of failures: body trajectory loss
  (unaware arrived/jammed, stop while far, sit on air, sit wrong).
- Conclusion: the bottleneck of current VLMs is embodied self-awareness,
  not perception.

## Design Notes

- The transparent wall is a collision obstacle and the channel is the only
  passage. Body dimensions are the hidden knowledge of the task.
- The tiered protocol follows MirrorBench prompt ablation:
  - Level 0: full guidance (wall, channel, body size, sideways strategy)
    plus the full 5-step solution.
  - Level 1: wall and narrow opening disclosed; the agent must find the
    sideways solution itself.
  - Level 2: generic task only ("reach the green ball in front of you").
  - Level 3: generic task only, with no target-behind-wall prior.
- Analysis plan: Step-ness (insight vs gradual), strategy switching
  (front-side-front), exploratory behavior, and one-shot body adjustment.

## environment.py

- Coordinate system: x forward, y up, z lateral. The robot faces +x.
- Transparent wall: two OmniGlass panels on the x = 2m plane with a 0.38m
  vertical channel centered at z = 0. Set `use_omni_glass=False` to use a
  semi-transparent PreviewSurface fallback.
- Robot: Unitree H1 loaded from Isaac Sim/Isaac Lab built-in assets or a
  local path. Root movement is kinematic; actions are gated by analytic
  collision checks that return the colliding part and contact point.
- Actions: `forward/backward/left/right` are fixed world-axis translations
  (+x/-x/-z/+z, 5cm), `turn_left/turn_right` rotate 15 degrees, and
  `reach/retreat` extend/retract the arm. Reach prefers articulation joint
  control and falls back to a kinematic hand model.
- Main interface: `reset_scene()`, `get_camera_image()` (1024x1024 RGB),
  `get_robot_state()`, `get_hand_position()`, `get_distance_to_target()`,
  `get_torso_rotation()`, `check_success()` (< 3cm),
  `execute_action(action)`, `check_collision_with_wall()` (bool),
  `get_collision_position()`, and `set_channel_width(width)` for Level 4
  dynamic channel experiments.
- `execute_action` returns `StepResult(rgb, legal, feedback, distance,
  success, collision, state)`. `legal=False` means the action was blocked
  by the transparent wall.
- Optional speed knobs via `--env_config` JSON (defaults match the spec:
  1024x1024 camera, 30 render steps per action, 30 reset steps):
  `camera_resolution`, `action_steps`, `reset_steps`, `rendermode`, `spp`.
  Example: `{"rendermode":"RaytracedLighting","spp":4,"camera_resolution":[512,512],
  "action_steps":5,"reset_steps":5}`.

## experiments.py

- Level 0 (guided solution): wall, 380mm opening, 570mm shoulder width,
  90-degree sideways strategy, and the full 5-step solution are disclosed.
- Level 1 (autonomous reasoning): wall and narrow opening disclosed, no
  solution and no CoT.
- Level 2 (implicit obstacle discovery): generic task only, no wall or
  channel information.
- Level 3 (self-referential body adjustment): generic task only, no wall,
  channel, or target-behind-wall prior.
- Level 4 (channel-widening memory): phase A uses the full Level 0 guidance
  with a 0.38m channel, then phase B silently widens the channel to 0.60m
  and only asks the agent to reach the ball. The session agent and history
  are not reset between phases.
- Each level runs `--rounds x --episodes` episodes (default 3 rounds x 50).
  Each episode has at most 30 steps and ends only on success or step
  exhaustion; wall collisions are recorded but do not terminate.
- Per-step loop: `get_camera_image -> get_robot_state -> build prompt ->
  ai_agent.get_action -> execute_action -> check_success ->
  check_collision_with_wall`. All specified fields are recorded, including
  `llm_response_time_ms` and `action_sequence`.
- Outputs:
  `results/level{level}/{model}/round*/episode_*.json`,
  `results/level{level}/{model}/round*/summary_{tag}.json`, and
  `logs/{tag}/` (agent logs and optional observation PNGs).
- Expected `ai_agent` interface:
  `ai_agent.get_action(prompt, image, state, history, options)`, or
  `create_agent(model, log_file)` / `get_agent(...)` returning an object
  with `get_action(...)`.

## ai_agent.py

- Supported models: GPT-4o, GPT-4o-mini, Claude-3.5-Sonnet,
  Gemini-2.5-Pro, Qwen-VL-Max, Qwen2.5-VL-72B/7B, InternVL3.5-4B,
  LLaVA-1.6-7B, plus a `random` baseline.
- Single OpenAI-compatible endpoint for all models. Switch models with
  `model_name`. API key and base URL come from environment variables.
- `get_action(image, prompt)` returns `{"action", "confidence",
  "reasoning"}`. The response is parsed as JSON (code fences and
  surrounding text tolerated), validated against the action list, and
  retried up to 3 times on timeout/errors.
- `build_prompt(level, context)` builds English prompts for Level 0-3 and
  always requires JSON output.
- Also supports `create_agent(model, log_file)`, `get_agent(...)`, and the
  module-level `get_action(...)` used by experiments.py.

## analysis.py

- Input: `results/level{level}/{model}/round*/episode_*.json` written by
  experiments.py.
- Step-ness: sliding-window (size 5) success rate sequence; max
  single-step improvement > 0.5 is insight, < 0.2 is gradual.
- Strategy switching: strategy vector
  `[cos(yaw), sin(yaw), arm_extension, displacement dx, dz]`; cosine
  similarity < 0.3 between adjacent episodes marks a strategy switch.
- Exploratory behavior: exploratory action set
  `{backward, left, right, turn_left, turn_right}`; per-episode ratios and
  before/after insight averages are reported.
- One-shot body adjustment: `max single rotation / total rotation`;
  > 0.7 is one-shot, < 0.3 is gradual. The largest continuous
  same-direction rotation run is also reported as an auxiliary diagnostic.
- Outputs: per-model JSON, `score_table_level{level}.csv/.md`, a learning
  curve with the insight point marked, a strategy-switch similarity plot,
  and an exploration-change plot. Use `--no_plot` to skip plotting.

## main.py

- Arguments: `--model` (default gpt-4o), `--level 0-3` (default 0),
  `--episodes` (default 50), `--rounds` (default 3), `--all-levels`, plus
  `--max_steps/--headless/--tag`.
- Flow: `environment.setup_scene()` initializes the scene,
  `ai_agent.create_agent(model=...)` creates the agent,
  `BAOExperimentRunner` runs the requested levels, step data is saved as
  `results/{model}_{level}_{timestamp}.csv`, and the environment is
  closed.
- Progress is logged every 10 episodes together with the current success
  rate.
