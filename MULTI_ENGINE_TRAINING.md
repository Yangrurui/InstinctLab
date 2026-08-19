# 跨引擎训练机器人任务：Agent 简报

> **架构立场已更新。** 本文第 2 节（既定立场、分层、硬约束、关键代码）与第 7 节描述的是「统一张量 Adapter + 自研 `UnifiedManagerBasedRLEnv`」方案，**已被取代**。当前权威设计是 [`CROSS_ENGINE_DESIGN.md`](CROSS_ENGINE_DESIGN.md)：统一层改为声明式 `TaskSpec`，由每个引擎的 adapter 编译成该引擎的**原生** env 配置。
>
> **这些文件已删除**，下文提到它们的地方按史料读，不要照着敲：`envs/unified_manager_based_rl_env.py`、`managers/unified.py`、`tasks/locomotion/mdp/unified.py`、`tasks/locomotion/unified_flat_env_cfg.py`、`rl/` 包、`scripts/instinct_rl/{train,play}_unified.py`、`scripts/profile_backend.py`。训练入口现在是 `scripts/train.py --engine isaacsim|mjlab --task <id>`。`backends/` 与 `sim/` 契约层保留，只服务 `tests/simulators/` 的 sim2sim 断言，场景描述见 `verify/scene.py`。
>
> 本文第 3–6、8 节（路径分类、公开框架对照、单引擎产品、与 HumanoidVerse 的差异）仍然有效，作为选型背景保留。

> 给后续任务 agent 用。选型背景以本文为准；实现契约以 [`CROSS_ENGINE_DESIGN.md`](CROSS_ENGINE_DESIGN.md) 为准。
>
> 资料截至 2026-08。核心问题不是“有哪些仿真器”，而是 **任务 / MDP / 算法换物理引擎后能否保持同一套代码**。

## 1. 何时读本文

在下列任务开始前先读完，再改代码：

- 评估或接入新仿真引擎（Genesis、Newton、IsaacGym、SAPIEN 等）
- 对比 HumanoidVerse / ProtoMotions / RoboVerse / Isaac Lab 3 / RoboRenForce
- 扩展 `SimulatorBackend`、Capability、统一训练入口
- 把已有单引擎任务改成可换后端
- 做 sim2sim 或跨引擎 checkpoint 兼容

不要用本文替代实现计划。接口语义、reset 顺序、G1 canonical 关节/body 列表见 `UNIFIED_ENGINE_PLAN.md`。

## 2. InstinctLab 既定立场（必须遵守）

本仓库走 **路径 B：Simulator Adapter**。Isaac Sim 与 MJLab 只实现 backend，不复制 Task / Manager / MDP。

### 2.1 已注册后端

| 名字 | Provider | 用途 |
|---|---|---|
| `isaacsim` | `instinctlab.backends.isaacsim:IsaacSimBackendProvider` | PhysX / Isaac Sim 训练 |
| `mjlab` | `instinctlab.backends.mjlab:MjlabBackendProvider` | MuJoCo Warp 训练 |
| `mock` | `instinctlab.backends.mock:MockBackendProvider` | 无物理的契约测试 |

注册表：`source/instinctlab/instinctlab/sim/backend.py` 中的 `BACKENDS`。

### 2.2 统一训练入口

```bash
python scripts/instinct_rl/train_unified.py --backend isaacsim --task Instinct-Locomotion-Flat-G1-v0 --headless
python scripts/instinct_rl/train_unified.py --backend mjlab --task Instinct-Locomotion-Flat-G1-v0
```

回放：`scripts/instinct_rl/play_unified.py`。旧 `scripts/instinct_rl/train.py` 仍是 Isaac-only 遗留路径，不要往里面加新的统一任务。

首个统一任务：`Instinct-Locomotion-Flat-G1-v0`。

### 2.3 分层（依赖只能向下）

```
train_unified / play_unified
        │  先解析 --backend，再 bootstrap，再 import 引擎 SDK
        ▼
TASKS：一份 EnvCfg / Schema / AgentCfg
        ▼
UnifiedManagerBasedRLEnv + 统一 Managers / MDP
        ▼
SceneView（只读 canonical 张量）
        ▼
SimulatorBackend 契约
        ├── IsaacSimBackend
        ├── MjlabBackend
        └── Mock
```

硬约束：

1. Env、Reward、Obs、Algo、Manager **禁止** import `isaaclab` 仿真 SDK、`mujoco`、`mjlab`、`omni.*`、PhysX 类型。
2. 公共层 **禁止** `if backend == "isaacsim"` / `"mjlab"` 分支。引擎差异只允许出现在 adapter、backend-scoped options、Capability 校验。
3. 关节/body 顺序以 `RobotSpec` 里冻结的 DFS 名为唯一真值，禁止依赖 URDF / USD / MJCF 隐式遍历。
4. 不支持的 Capability **启动即失败**，禁止静默降级。
5. 训练契约（`SimulatorBackend`）和真机 / 单环境部署（`instinct_onboard`）分开，不要塞进同一个基类。
6. 不要为了多一个引擎复制一份 Task。先冻 canonical 顺序和 Capability，再写 adapter。

### 2.4 关键代码

| 职责 | 路径 |
|---|---|
| 后端契约与注册 | `source/instinctlab/instinctlab/sim/backend.py` |
| Capability | `source/instinctlab/instinctlab/sim/capabilities.py` |
| 机器人规范顺序 | `source/instinctlab/instinctlab/sim/robot_spec.py` |
| 统一环境 | `source/instinctlab/instinctlab/envs/unified_manager_based_rl_env.py` |
| Isaac 适配 | `source/instinctlab/instinctlab/backends/isaacsim/` |
| MJLab 适配 | `source/instinctlab/instinctlab/backends/mjlab/` |
| 训练入口 | `scripts/instinct_rl/train_unified.py` |
| 真引擎三格 | `tests/simulators/`：`reset-root-vel` / `air-time-advance` / `material-write-scope` |
| 实现计划 | `UNIFIED_ENGINE_PLAN.md` |

默认 `pytest tests/` 不启真引擎（`pytest.ini` 排除 `mjlab` / `isaacsim`）。Isaac 与 MJLab 不能同进程。Isaac 必须先 `AppLauncher` 再 import `torch`，且 Kit 会读 `sys.argv`，所以只能指定该文件跑：

```bash
pytest -o addopts= -m mjlab tests/simulators/test_mjlab_behavior.py
pytest -o addopts= -m isaacsim tests/simulators/test_isaacsim_behavior.py
```

### 2.5 吞吐：先量再砍

`scripts/profile_backend.py` 拆 `policy_step`。零动作会摔倒并 reset，数字会被 reset 污染；稳态用 `--no-reset`。aten profiler 用 `--aten-ops`，默认关掉（会灌高 `policy_step`）。

4096 env、`cuda:1`、20 step、`--no-reset`（2026-08-17）：

| 项 | ms / policy step | 能否动 |
|---|---|---|
| `policy_step` | 29.0 | |
| `write_data_to_sim`（mjlab 隐式 PD，4 子步） | 11.5 | 否，控制律 |
| `sim_step`（MuJoCo Warp，4 子步） | 9.9 | 否，求解器 |
| `scene_update`（含 air-time） | 1.5 | 否 |
| obs / reward | 各约 2.2 | 否，公式 |
| `synchronize` + contact/cvel/effort | 0.40 + 0.23 | 否，占比太小 |
| `apply_action` / `process_action` | 0.13 / 0.05 | 已有 control cache |

零动作、允许 reset 时，`reset` 约 16 ms、`event` 约 10 ms（reset 事件套在 `_reset_idx` 里，有重叠）。这是摔倒重开，不是 bridge。走路后 collection 变慢应先看 `sim_step` 接触变多，不在 synchronize。

结论：**adapter 热路径和 MDP 公式都不要为吞吐去改。** Isaac adapter 继续零行。再抠拷贝或重写 reward 都违反 2.3，也补不回求解器/PD 的时间。

复现：

```bash
python scripts/profile_backend.py --backend mjlab --device cuda:1 --num-envs 4096 --warmup 8 --steps 20 --no-reset
```

## 3. 先分清两件事

“换引擎训练同一任务”有两条完全不同的路：

| | 产品内换求解器 | 产品外 Adapter |
|---|---|---|
| 典型 | Isaac Lab 3：`physics=physx\|newton_mjwarp\|ovphysx` | InstinctLab / HumanoidVerse / ProtoMotions |
| 任务代码 | 仍写 Isaac Lab API | 不写任何引擎 SDK |
| 换的是什么 | 同一框架里的物理后端 | 整个仿真产品 |
| 何时用 | 任务已是 Isaac Lab 原生，只要 PhysX ↔ MJWarp | 要同时用 Isaac Sim 和 MJLab（或更多独立产品） |

InstinctLab **不走** Isaac Lab 3 工厂来统一 MJLab。MJLab 是独立产品，用 Adapter 接入。

## 4. 四种实现路径

越往上，引擎覆盖越广、语义越薄；越往下，训练吞吐和状态对齐越好，产品绑定越紧。

### A. Gymnasium 环境接口（最薄）

只统一 `reset / step / obs / action / reward`。任意引擎都能包一层。

- **能做**：算法对照、接标准 RL 库
- **不能做**：sim2sim。关节顺序、接触力、域随机化、传感器相位全部不保证
- **代表**：Gymnasium；RoboRenForce 的 `train_gym.py`
- **本仓库**：不要用 Gym 当跨引擎对齐层

### B. Simulator Adapter（训练最常用，本仓库采用）

Env / Reward / Obs / Algo 禁止 import 引擎 SDK。Adapter 把 native 状态映射到冻结的 canonical 张量。换引擎只改 CLI 或配置。

- **代表**：InstinctLab、HumanoidVerse、ProtoMotions
- **InstinctLab 比 HumanoidVerse 更高**：统一边界在 Manager-Based Env，不只在 Simulator
- **改造顺序**（来自 PBHC / HumanoidVerse 指南，不要反过来）：
  1. 选定规范引擎，冻结张量约定
  2. 从能跑的 Env 抽出接口
  3. 把现有引擎改成第一个 adapter，确认训练不回退
  4. 加配置 / CLI 注入
  5. 处理入口脚本的 SDK 引导（import 顺序、AppLauncher）
  6. 加第二个 adapter
  7. 写对齐测试，再扩第三个

### C. 声明式场景层（覆盖最广）

用统一 Scenario / Handler 描述机器人、物体、初态和查询，再交给各仿真器。覆盖 RL、IL、数据集和评测，不只是向量化训练。

- **代表**：RoboVerse / MetaSim
- **后端最多**：IsaacSim、IsaacGym、MuJoCo、SAPIEN、Genesis、PyBullet、Newton（另有实验性 MJX）
- **代价**：抽象比训练 adapter 更宽，吞吐和 MDP 控制更弱
- **本仓库**：不把 MetaSim 引进训练热路径；评测/数据集需求可另议

### D. 框架内物理后端工厂（产品内）

任务仍写 Isaac Lab API。`Articulation` 等类型在运行时按 `SimulationCfg.physics` 分发到 PhysX / Newton / OvPhysX。

- **代表**：Isaac Lab 3.0 Multi-Backend Architecture
- **切换**：`physics=newton_mjwarp` 等 preset
- **状态**：2026-08 仍为 beta；Newton 集成以经典 RL 和平坦地形 locomotion 为主
- **本仓库**：不要为了“官方 Newton”把统一任务改回继承 Isaac Lab Env

## 5. 公开框架对照

只收录“同一任务实现能换仿真后端”的框架。MJLab 本体、ManiSkill、MuJoCo Playground 是单引擎产品，见第 6 节。

| 方案 | 路径 | 可切换后端 | 任务是否真复用 | 更适合 |
|---|---|---|---|---|
| **InstinctLab** | B | `isaacsim` / `mjlab` / `mock` | 同一 EnvCfg / MDP / Managers | 本仓库人形全身控制，Isaac 与 MJLab 对齐训练 |
| **HumanoidVerse** | B | IsaacGym / IsaacSim / Genesis | Hydra `+simulator=` 换 adapter | 人形 locomotion；训练与部署接口分开 |
| **ProtoMotions** | B | IsaacGym / IsaacLab / Newton / Genesis / MuJoCo | `--simulator` 换后端 | 动作模仿、数字人与人形 tracking |
| **RoboVerse / MetaSim** | C | 再加上 SAPIEN、PyBullet 等 | 同一 scenario，handler 换后端 | 多仿真评测、IL 数据集、跨引擎 benchmark |
| **Isaac Lab 3.0** | D | PhysX / Newton+MJWarp / OvPhysX | 同一任务 + `physics=` 预设 | 留在 Isaac 生态内换求解器，官方策略互迁 |
| **RoboRenForce** | A + 训练栈 | Isaac Lab / MJLab / Gymnasium | 统一 runner，任务按平台分包 | 一套 RL/VLA 管线接多种环境包，**不是**同一任务实现 |
| **PBHC / HumanoidVerse 指南** | B 方法论 | 按项目接入 | 先冻张量约定，再抽 `BaseSimulator` | 已有单引擎代码要改造成可切换 |

### 5.1 各方案怎么换引擎

| 方案 | 切换方式 | 状态约定 | 能力差异 |
|---|---|---|---|
| InstinctLab | `train_unified.py --backend isaacsim\|mjlab` | 冻结 DFS 关节/body 名；四元数 WXYZ；`SceneView` 只读 canonical 张量 | Capability 启动校验 |
| HumanoidVerse | `+simulator=isaacgym\|isaacsim\|genesis` | 以规范引擎张量布局为准 | 文档要求任务无 SDK import；其仓库仍有少量 `simulator.name` 分支 |
| ProtoMotions | `--simulator isaaclab\|newton\|genesis\|mujoco\|isaacgym` | `base_simulator` API | 建议每引擎独立环境；MuJoCo 仅 CPU 调试 |
| RoboVerse | `--sim isaacgym\|isaacsim\|mujoco` 等 | MetaSim typed state + query | 后端分 Active / Experimental |
| Isaac Lab 3.0 | `physics=newton_mjwarp\|physx\|ovphysx` | 仍是 Isaac Articulation API；数据以 Warp array 为主 | `PresetCfg` 为每个后端准备物理参数 |

### 5.2 链接

- HumanoidVerse: https://github.com/LeCAR-Lab/HumanoidVerse
- ProtoMotions: https://github.com/NVlabs/ProtoMotions
- RoboVerse / MetaSim: https://github.com/RoboVerseOrg/MetaSim
- Isaac Lab 3 多后端: https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/overview/core-concepts/multi_backend_architecture.html
- RoboRenForce: https://github.com/Renforce-Dynamics/RoboRenForce
- InstinctLab 计划附录 A/B 收录了 PBHC HumanoidVerse 实现指南（`UNIFIED_ENGINE_PLAN.md`）

## 6. 单引擎产品（常被接到 Adapter 里）

这些 **不是** 跨引擎方案。把它们当后端候选，不要当成“已经能换引擎”。

| 产品 | 实际是什么 | 和本仓库的关系 |
|---|---|---|
| **MJLab** | Isaac Lab 风格 Manager API + MuJoCo Warp | 已作为 `mjlab` backend |
| **ManiSkill 3** | SAPIEN GPU 并行，操作任务强 | 通常作为 MetaSim 的 sapien 后端；本仓库不接 |
| **Newton** | NVIDIA / DeepMind / Disney 的 GPU 物理层，求解器含 MJWarp | Isaac Lab 3 与 ProtoMotions 的后端，不是训练框架 |
| **MuJoCo Playground** | DeepMind 任务套件，单引擎 | 可做部署/验证参考，不是训练多后端 |
| **Genesis** | 刚体/软体/流体统一 API | HumanoidVerse / ProtoMotions / MetaSim 已接；本仓库未接 |

## 7. 给后续任务的决策树

```
要在本仓库训同一套人形任务？
    是 → 用现有 --backend isaacsim|mjlab
         新引擎 = 新 SimulatorBackend + Capability，不复制 Task
    否 ↓

任务已是 Isaac Lab 原生，只要官方 PhysX ↔ MuJoCo-Warp？
    是 → 评估 Isaac Lab 3 `physics=newton_mjwarp`
         不要把 Instinct 统一任务改回继承 Isaac Env
    否 ↓

要 SAPIEN / PyBullet / IL 数据集 / 跨引擎 benchmark？
    是 → RoboVerse MetaSim（独立评测栈，不替代本仓库训练契约）
    否 ↓

要动作捕捉、数字人、多形态 tracking？
    是 → 参考 ProtoMotions 的 simulator 接口与任务模型
    否 ↓

已有单引擎代码要改造成可切换？
    是 → 按第 4 节 B 的改造顺序；细节见 UNIFIED_ENGINE_PLAN.md 附录 A
```

### 7.1 若任务是“接入新引擎”

按这个清单做，缺一项就不要宣称任务可跨引擎：

1. 在 `BACKENDS.register` 增加 provider，lazy import，bootstrap 不得污染其他引擎进程
2. 实现完整 `SimulatorBackend`，对外只暴露 canonical 张量
3. 用 `CanonicalIndexMap` 按名称映射，禁止假定 identity
4. 声明 `CapabilitySet`；任务 `RuntimeRequirements` 校验失败则启动报错
5. 为该引擎补 `RobotSpec.asset_for("<name>")`
6. 统一任务注册表只增加 backend 名，不增加第二份 EnvCfg
7. 补 contract / smoke / 短训练回归后再谈性能

### 7.2 明确不要做的事

- 不要用 Gymnasium 当 sim2sim 对齐层
- 不要把 GPU 训练仿真和真机 / 单环境 MuJoCo 部署塞进同一个基类
- 不要在 Manager / MDP 里写引擎名字分支
- 不要为 Isaac 和 MJLab 各写一份 reward / observation
- 不要把 PhysX 接触力与 MuJoCo 接触力当逐值等价；需要绝对力值的任务必须声明 `CONTACT_FORCE_VECTOR` 与允许误差
- 不要把 Isaac Lab 3 的 factory 模式抄进本仓库公共层；本仓库的分发点是 `BACKENDS.load(name)`，不是 `Articulation` 工厂

## 8. 和 HumanoidVerse 的差异（避免抄错）

`UNIFIED_ENGINE_PLAN.md` 明确写了：模仿 PBHC 的依赖方向，但把统一边界提升到 Manager-Based 环境。

| | HumanoidVerse | InstinctLab |
|---|---|---|
| 配置 | Hydra `+simulator=` | argparse `--backend` |
| 统一边界 | `BaseSimulator` | `SimulatorBackend` + `UnifiedManagerBasedRLEnv` |
| 部署 | 另有 `URCIRobot` | `instinct_onboard`，不在本契约里 |
| 已接引擎 | IsaacGym / IsaacSim / Genesis | IsaacSim / MJLab / mock |
| 状态顺序 | 以规范引擎为准 | 显式冻结 DFS 名 + `CanonicalIndexMap` |
| 能力差异 | 任务里仍有少量引擎分支 | Capability 启动失败，禁止任务内分支 |

抄 HumanoidVerse 时只抄“任务不 import SDK、adapter 对齐张量”这两条。不要抄它的 Hydra 结构，也不要抄它在任务里按 `simulator.name` 分支的现状。

## 9. 交接时建议带上的上下文

把后续任务交给其他 agent 时，最少附上：

1. 本文：`MULTI_ENGINE_TRAINING.md`
2. 实现契约：`UNIFIED_ENGINE_PLAN.md`（尤其 SimulatorBackend、Capability、reset 顺序）
3. 任务目标一句话，例如：“为 Genesis 加 backend，不改 Flat G1 的 MDP”
4. 禁止项：第 2.3 节硬约束 + 第 7.2 节

可选：聊天旁的对照画布 `.cursor/projects/root-InstinctLab/canvases/multi-engine-robot-training.canvas.tsx`（可视化摘要，不是源文件）。
