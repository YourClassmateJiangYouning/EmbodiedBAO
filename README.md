# EmbodiedBAO

具身智能实验：**身体作为障碍（Body-as-Obstacle, BAO）任务**，运行环境为 NVIDIA Isaac Sim。

## 实验场景

- 场景大小：4m x 4m
- 场景中央有一块横跨整个场景的透明板（宽 4m、高 2m），将场景完全分割为前后两个区域
- 透明板中央有一条从上到下完全贯通的垂直通道，宽度 380mm
- 机器人（Unitree H1）站在前半区，起始位置在通道正前方 50cm 处
- 绿色目标球（直径 8cm）位于后半区，通道正后方 40cm 处，高度 1.2m
- 通道宽度 380mm < 机器人肩宽 570mm（正面过不去），但 > 机器人躯干厚度 220mm（侧身能通过）
- 目标球距板 40cm > 机器人臂长 34cm（正面站在板前够不到），但侧身通过通道进入后半区后能够到

## 核心研究问题

具身智能体能否在 BAO 任务中表现出“顿悟式”的身体自我意识？

## 计划模块

1. `main.py`：主入口，解析命令行参数，调度实验
2. `environment.py`：在 Isaac Sim 中搭建实验场景
3. `experiments.py`：实现 Level 0-3 的分级实验逻辑
4. `ai_agent.py`：统一接口调用各种 MLLM
5. `analysis.py`：计算 Step-ness、策略切换、探索性行为、一次性调整四个指标

## 安装与运行

- 环境：NVIDIA Isaac Sim 4.5（与 MirrorBench 相同的测试环境），设置 `ISAACSIM_ROOT` 环境变量。
- 依赖：`%ISAACSIM_ROOT%\python.bat -m pip install -r requirements.txt`。
- API：聚合平台（如 Taotoken）的 OpenAI 兼容接口，配置 `TAOTOKEN_API_KEY`/`TAOTOKEN_BASE_URL`（或 `OPENAI_API_KEY`/`OPENAI_BASE_URL`）。
- 机器人资产：Isaac Sim/Isaac Lab 内置 Unitree H1 自动探测；也可用 `EMBODIEDBAO_H1_USD` 指定本地 USD 路径。
- 运行：
  - 单级实验：`%ISAACSIM_ROOT%\python.bat main.py --model gpt-4o --level 0 --episodes 50 --headless`
  - 全级实验：`%ISAACSIM_ROOT%\python.bat main.py --model gemini-2.5-pro --all-levels --headless`
  - 批量评估：`evaluate.bat MODEL_NAME [EPISODES]`（Windows）或 `./evaluate.sh MODEL_NAME [EPISODES]`（Linux）
  - 分析：`%ISAACSIM_ROOT%\python.bat analysis.py --results_root results --level 2 --models gpt-4o`

## 参考工作

### MirrorBench（Guo et al., 2026）

- 论文与代码均已学习。其核心是把心理学的镜像自我识别（MSR）测试迁移到具身 MLLM 评估。
- 分级协议：Level 0-3 通过**提示词消融**控制先验知识量，物理环境保持不变，难度单调递增。
  - Level 0：明确告知镜子存在 + 完整 CoT 推理模板（引导镜像感知）
  - Level 1：告知镜子存在，无 CoT（自主镜像推理）
  - Level 2：不告知镜子（隐式镜像发现）
  - Level 3：不告知镜子、不告知目标在自己身体上（自我参照镜像识别）
- 评估指标：TSR（任务成功率）、SIR（逐步改进比）、FCR（最终完成率）、PCR（峰值完成率），全部由手-目标距离轨迹推导。
- 动作空间：6 方向平移；最大步数 = 理论最小步数 + 10；反馈为“无碰撞执行成功 / 因碰撞执行失败”。
- 代码框架：`env.py`（Isaac Sim 场景）、`agent.py`（OpenAI 统一调用 + random 基线）、`prompts.py`（L0-L3 提示）、`inference.py`（单场景调度）、`action.py`（动作空间）、`preprocess.py` + `download.py`（资产池）、`evaluate.bat/.sh`（全量评估）。
- 失败模式：Mirror-Self Confusion（镜像-自我混淆）导致持续朝镜像方向移动，性能低于随机策略。

### HumanCLAW（2026）

- 框架核心：把 VLM 的**动作决策**与低层运动执行解耦，VLM 输出原子技能，半物理仿真执行并返回真实物理后果。
- 分级/渐进成功率：Find -> Nav -> Interact，要求客观几何判据 + 模型主观确认同时满足。
- 失败根因自动归因：身体轨迹丢失相关错误（unaware arrived / unaware jammed、stop while far、sit on air、sit wrong 等）是主要瓶颈。
- 结论：当前 VLM 的瓶颈不在感知（目标一旦出现在视野中几乎都能识别），而在**具身自我意识**——不知道自己的身体在哪里、是否到达、是否碰撞。

## BAO 任务设计草案（待逐模块细化确认）

- 透明板 = 保留碰撞的障碍，通道 = 唯一可通过路径；身体几何尺寸是任务的“隐藏知识”。
- 分级思路（沿用 MirrorBench 提示词消融）：
  - Level 0：完整引导（告知通道、身体尺寸与侧身策略）
  - Level 1：告知通道与尺寸，要求自主发现侧身动作
  - Level 2：告知通道存在，要求自主估计身体是否可通过
  - Level 3：只给目标与动作空间，一切由模型自行推断
- 动作空间与反馈需要覆盖“正面被板挡住、通道内卡住、侧身通过、到达后可够到目标”等状态。
- 分析模块计划覆盖 Step-ness（顿悟性）、策略切换（前-侧-前）、探索性行为、一次性调整，具体定义待用户确认。

## 目录约定（沿用 MirrorBench）

- `logs/`：单次运行日志与观测图像
- `results/`：结构化评估结果 JSON

## environment.py 接口（已实现）

- 坐标系：`x` 向前（场景 0~4m）、`y` 向上（地面 y=0）、`z` 横向（-2~2m）；机器人面向 `+x`。
- 透明板：x=2m 平面上的两块亚克力面板，中央保留 0.38m 贯通通道；默认绑定 `OmniGlass.mdl` 材质（`use_omni_glass=False` 可切换为半透明 PreviewSurface）。
- 机器人：从内置/Isaac Lab/本地资产加载 Unitree H1，可用 `EMBODIEDBAO_H1_USD` 环境变量覆盖；root 采用姿态遥操作，动作合法性由解析几何碰撞门控，碰撞返回部位（torso/shoulder/hand）与接触点。
- 动作空间：`forward/backward/left/right`（5cm）、`turn_left/turn_right`（15°）、`reach/retreat`（手臂伸展/缩回，优先使用 Articulation 关节控制，失败时退回运动学手部模型）。
- 主要接口：`reset_scene()`、`get_camera_image()`（1024×1024 RGB）、`get_robot_state()`、`get_hand_position()`、`get_distance_to_target()`（到球面距离）、`get_body_obstacle_status()`、`check_success()`（<3cm）、`execute_action(action)`、`check_collision_with_wall()`。
- `execute_action` 返回 `StepResult(rgb, legal, feedback, distance, success, collision, state)`，`legal=False` 表示动作被透明板阻挡。
- 本机未安装 Isaac Sim，几何与碰撞逻辑已通过纯 Python 自检；完整运行请用 `%ISAACSIM_ROOT%\python.bat environment.py`（含无头 smoke test）。

## experiments.py 四级协议（已实现）

- Level 0 引导式解决：完整告知透明板/380mm 通道/570mm 肩宽/90° 侧身方案，并附 4 步 CoT。
- Level 1 自主推理：告知透明板与窄通道，不提供解决方案。
- Level 2 隐式障碍发现：只给“reach the green ball in front of you”，不提墙与通道。
- Level 3 自我指认的身体调整：提示词与 L2 相同（按你的规格），仅通过不披露任何先验区分难度。
- 每级 50 个 Episode（`--episodes` 可调），每 Episode 最多 30 步，成功、碰撞累计 3 次或步数耗尽即结束。
- 每步执行：`get_camera_image -> get_robot_state -> 构建 Prompt -> ai_agent.get_action -> execute_action -> check_success -> check_collision_with_wall`，记录全部指定字段（含 `llm_response_time_ms`、`action_sequence`）。
- 输出：`results/level{level}/{model}/episode_*.json`、`summary_{tag}.json`、`logs/{tag}/`（agent 日志与可选观测 PNG）。
- `ai_agent` 预期接口：`ai_agent.get_action(prompt, image, state, history, options)`，或 `create_agent(model, log_file)`/`get_agent(...)` 返回带 `get_action(...)` 的对象。
- 运行：`%ISAACSIM_ROOT%\python.bat experiments.py --model <model> --levels 0 1 2 3 --episodes 50 --headless`。

## ai_agent.py 统一 MLLM 接口（已实现）

- 支持模型：GPT-4o、GPT-4o-mini、Claude-3.5-Sonnet、Gemini-2.5-Pro、Qwen-VL-Max、Qwen2.5-VL-72B/7B、InternVL3.5-4B、LLaVA-1.6-7B。
- 统一走 OpenAI 兼容聚合平台（如 Taotoken）：`TAOTOKEN_API_KEY`/`OPENAI_API_KEY`、`TAOTOKEN_BASE_URL`/`OPENAI_BASE_URL`，`model_name` 切换模型。
- `get_action(image, prompt)` 返回 `{"action", "confidence", "reasoning"}`，强制 JSON 解析（容错代码块/前后缀文本），动作合法性校验，超时/异常自动重试 3 次；`random` 模型为随机基线。
- `build_prompt(level, context)` 输出英文四级 Prompt，始终要求 JSON 输出。
- 兼容 `create_agent(model, log_file)` / `get_agent(...)` / 模块级 `get_action(...)` 三种调用方式，experiments.py 可直接消费返回字典。

## analysis.py 顿悟/渐悟量化分析（已实现）

- 输入：`results/level{level}/{model}/episode_*.json`（experiments.py 的输出）。
- Step-ness：成功率序列滑动窗口 5，最大单步改进 > 0.5 判顿悟、< 0.2 判渐悟。
- 策略切换：策略向量 `[cos(yaw), sin(yaw), 手臂伸展度, 位移方向 dx, dz]`，相邻 Episode 余弦相似度 < 0.3 视为策略切换。
- 探索性行为：探索动作集 `{backward, left, right, turn_left, turn_right}`，输出每 Episode 占比及顿悟前/后均值与差值。
- 一次性身体调整：`单次最大旋转 / 总旋转`，> 0.7 判一次性调整、< 0.3 判渐进式调整；另附最大连续同向旋转角度作为辅助诊断。
- 输出：每模型 JSON、`score_table_level{level}.csv/.md`，以及学习曲线（标注顿悟点）、策略切换相似度、探索性行为三张图；`--no_plot` 可跳过绘图。
- 运行：`python analysis.py --results_root results --level 2 --models gpt-4o qwen2.5-vl-7b`（`--models` 省略时自动扫描全部模型）。

## main.py 程序主入口（已实现）

- 参数：`--model`（默认 gpt-4o）、`--level 0-3`（默认 0）、`--episodes`（默认 50）、`--all-levels`，另支持 `--max_steps/--collision_timeout/--headless/--tag`。
- 流程：`environment.setup_scene()` 初始化场景 -> `ai_agent.AIAgent(model=...)` 创建代理 -> `BAOExperimentRunner` 运行对应 Level -> 每步数据落盘为 `results/{model}_{level}_{timestamp}.csv` -> `env.close()`。
- 每 10 个 Episode 输出进度与当前成功率；`--all-levels` 依次运行 Level 0-3。
- 运行示例：`python main.py --model gpt-4o --level 0 --episodes 50`、`python main.py --model gemini-2.5-pro --all-levels`。
