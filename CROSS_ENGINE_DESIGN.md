# 跨引擎训练架构设计（权威版）

> 本文取代 [`UNIFIED_ENGINE_PLAN.md`](UNIFIED_ENGINE_PLAN.md) 中「SimulatorBackend 契约」「统一 Manager-Based 环境」两部分的架构立场。该文件的附录 A/B/C（PBHC、HumanoidVerse、MJLab Entity API 摘录）仍然有效，作为参考资料保留。
>
> 选型背景与公开框架对照见 [`MULTI_ENGINE_TRAINING.md`](MULTI_ENGINE_TRAINING.md)。

## 1. 结论

| 目标 | 是否可行 | 说明 |
|---|---|---|
| 统一接口，用户不感知引擎 | 可行 | 一份 `TaskSpec` + `--engine`，观测/动作同序同维同语义 |
| 两引擎数值逐位一致 | **不可行** | 求解器、执行器模型、接触力语义不可对齐，见 §3 |
| 各引擎逐位复现自身参考实现 | 可行 | 但与上一条统一接口冲突，本仓库选择统一接口优先 |
| 引擎不具备的功能自动跳过 | 可行 | 三级 Requirement 模型，见 §7 |

核心思路：**统一层从「统一张量」下移到「统一声明 + 一份可移植 MDP 库」。** 统一层由每个引擎的 adapter 编译成该引擎的**原生**环境配置；env 循环与 manager 全部使用引擎原生实现，不自研。MDP 术语只写一份，两个引擎的原生 manager 共用。

### 1.1 项目的主要价值是迁移

本项目的目标场景是：**把任意引擎的开源项目接进来，不改任务逻辑就能换引擎跑。** 不限于 Isaac Lab 项目——基于 mjlab 的项目同样要能反向迁移到 Isaac，新引擎与新特性要能持续接入。这决定了设计重心，见 §12 迁移架构、§13 迁移工作流、§14 总框架路线。

关键实测结论：两个框架的数据属性显式名**已经收敛**，term 函数签名形状相同，env 级访问器同名同义。因此一个只读 `asset.data.*` + torch 数学的 term 函数，在两个引擎的原生 manager 下都能**直接跑**。迁移成本不在 env / manager 层，而集中在 4 个可枚举的族，且大部分是 per-robot 一次性工作。

§2–§13 描述 isaacsim + mjlab 双引擎的完整设计，结论在该范围内成立。但其中有五处把「Isaac Lab 是源头」硬编码进了结构，扩展到 N 引擎与双向迁移前必须先改，见 §14。

## 2. 决策记录

### D1：关节/body 顺序以 DFS 为主

`RobotSpec.joint_names` / `body_names` 的 DFS 名序是唯一真值。两个引擎的 term 配置都发出 `preserve_order=True`，由引擎自己按名解析出索引。

已验证两侧都支持这一机制：

- Isaac Lab：`SceneEntityCfg.preserve_order`（`managers/scene_entity_cfg.py:102`）、`JointAction` 的 `find_joints(..., preserve_order=cfg.preserve_order)`（`envs/mdp/actions/joint_actions.py:65`）
- MJLab：`SceneEntityCfg.preserve_order`（`managers/scene_entity_config.py:126`）、`envs/mdp/actions/actions.py:113`

因此 DFS 顺序可以在**原生 env 内部**实现，不需要自研 env 做张量置换。这是本设计成立的前提。

已知偏差：main 分支使用 PhysX BFS 顺序（`left_shoulder_pitch, right_shoulder_pitch, waist_pitch, ...`），本设计下 isaacsim 路径改为 DFS（`waist_pitch, waist_roll, waist_yaw, left_hip_pitch, ...`）。这是一处**被接受并记录**的偏差，进入差异白名单（§8）。已有 main checkpoint 迁移需要一次按名置换。

### D2：退役自研 env 与 managers

以下组件退役，改用引擎原生 env + `TaskSpec` 编译：

| 文件 | 行数 |
|---|---|
| `envs/unified_manager_based_rl_env.py` | 246 |
| `managers/unified.py` | 643 |
| `tasks/locomotion/mdp/unified.py` | 593 |
| `tasks/locomotion/unified_flat_env_cfg.py` | 439（改写为 TaskSpec，预计 <150 行） |

`SimulatorBackend` 双实现（`backends/isaacsim/backend.py` 1119 行 + `backends/mjlab/simulator.py` 1464 行）**不作废**，降级为 `instinctlab/verify/` 的状态读写层，用于 sim2sim 断言与 `instinct_onboard` 对齐，退出训练热路径。`scripts/profile_backend.py` 与 `tests/simulators/` 继续服务该层。

### D3：main 分支是唯一 golden，允许声明的轻微差异

不引入 InstinctMJ 作为依赖。main 的 `tasks/locomotion/config/g1/flat_env_cfg.py::G1FlatEnvCfg` 是任务定义的唯一真值。InstinctMJ 只作为「如何在 mjlab 上表达同一任务」的参考实现——其 env 子类、`MultiRewardManager`、`ForceThresholdContactSensor` 按需移植进 `engines/mjlab/`。

验收标准从「diff 必须为空」放宽为「diff 落在差异白名单内」。

## 3. 可对齐 / 不可对齐边界

分界线非常清晰。**MDP 语义层可以完全统一；引擎物理层不能。**

### 3.1 统一（进入 `TaskSpec`）

观测组成与顺序（6 项，policy 96 维 / critic 99 维）、观测噪声、奖励项与权重（16 项）、终止项、命令参数、动作 scale/offset、时间参数（dt 0.005 / decimation 4 / episode 20 s）、奖励 × dt、质量与 reset/push 随机化的意图与范围、PPO 超参。

这些在 main 与 InstinctMJ 之间**本来就是一致的**，统一没有代价。

### 3.2 不统一（进入 `EngineProfile`，默认值 = 该引擎参考实现）

| 项 | isaacsim | mjlab |
|---|---|---|
| 求解器 | PhysX TGS，8 pos / 4 vel iter | MuJoCo Newton，10 iter / 20 ls / 500 ccd |
| 积分器 | PhysX 内部 | implicitfast |
| 执行器 | `ImplicitActuator`（PhysX 内隐式 PD） | `BuiltinPdActuator`（MuJoCo position+velocity） |
| 摩擦随机化 | static (0.25,0.8) + dynamic (0.2,0.6) + restitution (0,0.8)，64 bucket，per-shape | 单一 slide friction (0.2,0.8)，per-env 共享，restitution 无效 |
| 接触力语义 | physx net normal resultant | MuJoCo net resultant world |
| 关节加速度 | Isaac Lab 惰性有限差分 | MuJoCo `qacc` |
| 派生量时序 | 每 substep `scene.update` | reward/termination 滞后 1 substep |
| 资产格式 | URDF → USD | MJCF |

其中「接触力语义」「关节加速度」「派生量时序」是**语义差异**，不可通过参数消除，必须以差异清单形式声明，不得当作逐值等价。

## 4. 分层与依赖方向

依赖只能向下。

```
scripts/train.py  --engine isaacsim|mjlab  --task Instinct-Locomotion-Flat-G1-v0
        │  先按 --engine 选 adapter，adapter.bootstrap()（Isaac 的 AppLauncher 早于 import torch）
        ▼
instinctlab/spec/            引擎无关声明层（禁止任何引擎 import）
instinctlab/mdp/             可移植 MDP term 库（一份实现，两引擎共用）
instinctlab/compat/          薄兼容层：署名词汇表 / denylist / 纯 torch math / SensorRef
        ▼
instinctlab/engines/<name>/  只承载真正需要两份实现的族：
                             actions / events(DR) / scene / assets / sim
        ▼
引擎原生栈（不改、不包、不重写）
   isaacsim → isaaclab.ManagerBasedRLEnv + instinctlab.envs.InstinctRlEnv
   mjlab    → mjlab.ManagerBasedRlEnv    + engines/mjlab 的同名子类
        ▼
instinct_rl OnPolicyRunner（两侧已是同一 VecEnv 契约，无需改动）

instinctlab/migrate/         迁移工具：analyze（AST 分类报告）+ codemod（确定性改写）
instinctlab/verify/          可选，非热路径：跨引擎状态导出与 sim2sim 断言
```

## 5. `spec/` 层 API

```python
# instinctlab/spec/capability.py
class Requirement(str, Enum):
    REQUIRED = "required"   # 缺失 -> 启动报错
    OPTIONAL = "optional"   # 缺失 -> 警告并跳过，记入 manifest
    EMULATE  = "emulate"    # 缺失 -> 用 adapter 注册的替代实现；无替代则降级为 OPTIONAL
```

```python
# instinctlab/spec/entity.py
@dataclass(frozen=True)
class EntityRef:
    """按 canonical 名引用实体子集。adapter 负责翻译成 native 名。"""
    entity: str = "robot"
    joints: tuple[str, ...] | None = None   # canonical 名或正则
    bodies: tuple[str, ...] | None = None
    preserve_order: bool = False
```

```python
# instinctlab/spec/mdp.py
@dataclass(frozen=True)
class NoiseSpec:
    kind: Literal["uniform", "gaussian"]
    lo: float
    hi: float

@dataclass(frozen=True)
class TermSpec:
    """所有 MDP 术语的基类。

    两种取值方式，二者必居其一：

    - `func`：**可移植族**（观测 / 奖励 / 终止 / 命令）直接携带 `instinctlab.mdp` 的函数引用。
      这些函数在两个引擎的原生 manager 下都能跑，不需要每引擎一份实现。写法与 Isaac Lab
      原生 cfg 完全一致，这是迁移成本低的关键。
    - `kind`：**per-engine 族**（动作 / 事件与 DR / 场景 / 资产 / sim）用语义名，由每引擎
      `engines/<name>/` 的注册表映射到该引擎实现。
    """
    func: Callable[..., Any] | None = None
    kind: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    target: EntityRef | None = None
    level: Requirement = Requirement.OPTIONAL
    engine_params: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.func is None) == (self.kind is None):
            raise ValueError("TermSpec must set exactly one of func / kind")

    def resolved_params(self, engine: str) -> dict[str, Any]:
        merged = dict(self.params)
        merged.update(self.engine_params.get(engine, {}))
        return merged

@dataclass(frozen=True)
class ObsTermSpec(TermSpec):
    noise: NoiseSpec | None = None
    scale: float | None = None
    clip: tuple[float, float] | None = None
    level: Requirement = Requirement.REQUIRED   # 观测缺失会改变网络输入

@dataclass(frozen=True)
class RewardTermSpec(TermSpec):
    weight: float = 0.0                         # level 默认 OPTIONAL

@dataclass(frozen=True)
class DoneTermSpec(TermSpec):
    time_out: bool = False
    level: Requirement = Requirement.REQUIRED

@dataclass(frozen=True)
class EventTermSpec(TermSpec):
    mode: Literal["startup", "reset", "interval"] = "reset"
    interval_range_s: tuple[float, float] | None = None

@dataclass(frozen=True)
class ObsGroupSpec:
    terms: Mapping[str, ObsTermSpec]            # 插入顺序 = flatten 顺序
    enable_corruption: bool = True
    concatenate_terms: bool = False
    history_length: int = 0
```

```python
# instinctlab/spec/task.py
@dataclass(frozen=True)
class SimSpec:
    physics_dt: float
    decimation: int
    episode_length_s: float
    is_finite_horizon: bool = False
    scale_rewards_by_dt: bool = True
    profiles: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    # profiles 只放**覆盖**。默认值由 adapter 提供，且默认值就是该引擎参考实现的值。

@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    robot: RobotSpec                    # 复用现有 sim/robot_spec.py
    scene: SceneSpec
    sim: SimSpec
    mdp: MdpSpec
    agent: AgentSpec
    engines: tuple[str, ...]
    engine_extras: Mapping[str, Any] = field(default_factory=dict)
    # engine_extras 是逃生舱：承载单引擎独有特性（Isaac 的 tiled camera、USD props 等）。
    # 用了它的任务不可移植，编译时记入 manifest。
```

**硬约束**：`spec/` 禁止 import 任何引擎 SDK，禁止 `if engine == ...`。引擎差异只能是**按引擎名分区的数据**（`profiles` / `engine_params` / `engine_extras`），不能是逻辑分支。

## 6. `engines/` 层 API

```python
# instinctlab/engines/base.py
@dataclass
class Resolution:
    """一次编译的完整可审计结果。"""
    resolved: dict[str, str]        # "reward/feet_air_time" -> 原生函数 qualname
    skipped: dict[str, str]         # key -> 原因
    emulated: dict[str, str]
    profile: dict[str, Any]
    engine_extras_used: tuple[str, ...]

    def summary_table(self) -> str: ...   # 启动时一次性打印

@dataclass
class CompiledTask:
    env_cls: type
    env_cfg: Any                    # 该引擎的原生 EnvCfg
    agent_cfg: Any
    resolution: Resolution

class EngineAdapter(Protocol):
    name: str
    SUPPORTED_VERSIONS: str         # 例如 ">=1.5,<1.7"，启动校验

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None: ...
    @staticmethod
    def bootstrap(args: argparse.Namespace) -> object: ...
    def capabilities(self) -> CapabilitySet: ...
    def compile(self, spec: TaskSpec, *, num_envs: int, device: str) -> CompiledTask: ...
    def contract_report(self, spec: TaskSpec) -> dict[str, Any]: ...
```

### 6.1 术语表：能力矩阵的唯一来源

每个引擎的 `terms.py` 就是该引擎的能力矩阵。查得到映射即支持，查不到即不支持。

```python
# instinctlab/engines/isaacsim/terms.py
TERMS = TermRegistry("isaacsim")

@TERMS.observation("base_ang_vel")
def _base_ang_vel(spec: ObsTermSpec, ctx: CompileCtx) -> ObsTerm:
    return ObsTerm(func=instinct_mdp.base_ang_vel, noise=ctx.noise(spec.noise))

@TERMS.reward("track_lin_vel_xy_exp")
def _track_lin_vel(spec: RewardTermSpec, ctx: CompileCtx) -> RewTerm:
    p = spec.resolved_params(ctx.engine)
    return RewTerm(
        func=locomotion_mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=spec.weight,
        params={"command_name": p["command"], "std": p["std"]},
    )

@TERMS.event("randomize_friction", provides=(Capability.DR_SLIDING_FRICTION, Capability.DR_RESTITUTION))
def _friction(spec: EventTermSpec, ctx: CompileCtx) -> EventTerm:
    p = ctx.profile["friction_dr"]      # 默认值 = main 的 64-bucket per-shape 方案
    return EventTerm(
        func=mdp.randomize_rigid_body_material, mode="startup",
        params={"asset_cfg": ctx.entity(spec.target), **p},
    )
```

`provides=` 让 adapter 的 `CapabilitySet` 从术语表自动推导，消除「实现了但没 advertise」这类声明脱节（现状 `MjlabBackend` 实现了 restitution 写入但未声明 `DR_RESTITUTION`）。

### 6.2 编译上下文：canonical → native 的唯一收口

```python
@dataclass
class CompileCtx:
    engine: str
    spec: TaskSpec
    profile: Mapping[str, Any]
    resolution: Resolution

    def entity(self, ref: EntityRef | None) -> Any:
        """EntityRef -> 该引擎的 SceneEntityCfg。D1 的 DFS 决策在此落地。"""

    def noise(self, noise: NoiseSpec | None) -> Any: ...
```

`ctx.entity()` 是 canonical 名翻译为 native 名、以及 `preserve_order=True` 的唯一施加点。整个架构里只有这一处需要知道 DFS 决策。

### 6.3 跳过机制

「引擎不具备的功能就跳过」的全部实现：

```python
def compile_family(family: str, specs: Mapping[str, TermSpec], ctx: CompileCtx, registry: TermRegistry):
    out = {}
    for name, spec in specs.items():
        key = f"{family}/{name}"
        builder = registry.lookup(family, spec.kind)
        if builder is None:
            if spec.level is Requirement.REQUIRED:
                raise UnsupportedTerm(key, ctx.engine, spec.kind)
            ctx.resolution.skipped[key] = f"engine {ctx.engine!r} has no term kind {spec.kind!r}"
            continue
        out[name] = builder(spec, ctx)
        ctx.resolution.resolved[key] = qualname_of(out[name])
    return out
```

配套三件事，缺一不可：

1. **启动汇总表**：一次性打印全部 skipped / emulated，而不是散落的 warning。
2. **manifest 落盘**：checkpoint 旁记录 engine、版本、DFS 关节顺序、skipped 集合、engine_extras 使用情况。否则「为什么这个 policy 不如那个」无法追查。
3. **`--strict-capabilities`**：把所有 OPTIONAL 提升为 REQUIRED，用于 CI 与正式训练。

## 7. Requirement 三级默认值

| 术语族 | 默认 level | 理由 |
|---|---|---|
| 观测项 | REQUIRED | 缺失改变网络输入维度与语义 |
| 动作项 | REQUIRED | 缺失则无法控制 |
| 终止项 | REQUIRED | 缺失改变 episode 结构 |
| 奖励项 | OPTIONAL | 正则化项缺失可接受，但必须记录——静默丢奖励项等于换优化目标 |
| 事件 / 域随机化 | OPTIONAL | 引擎能力差异集中在这里 |
| 命令 | REQUIRED | 观测里有 command 项 |
| Curriculum | OPTIONAL | — |
| 传感器 / 可视化 | OPTIONAL | — |

任务可逐项覆盖：`RewardTermSpec(..., level=Requirement.REQUIRED)`。

## 8. 差异白名单

`tests/parity/isaacsim.locomotion_flat.allow.yaml`：

```yaml
- path: actions.joint_pos.joint_names
  reason: D1 统一为 DFS 顺序；main 使用 [".*"]（PhysX BFS）
- path: actions.joint_pos.preserve_order
  reason: D1 同上
- path: observations.policy.joint_pos.params.asset_cfg.joint_names
  reason: D1 同上
```

L0 测试编译 `TaskSpec` 后与 golden 逐字段比对，diff 必须为空或全部命中白名单。**新增白名单条目必须写 reason 并在 review 中被看到**——这是防止偏差悄悄累积的唯一闸门。

## 9. Locomotion Flat G1 的 TaskSpec（示意）

注意可移植族用 **`func=` 函数引用**（写法与 Isaac Lab 原生 cfg 一致），per-engine 族用 **`kind=` 语义名**。

```python
# instinctlab/tasks/locomotion/flat_g1.py — 无任何引擎 import
from instinctlab import mdp                     # 可移植 term 库

LOCOMOTION_FLAT_G1 = TaskSpec(
    task_id="Instinct-Locomotion-Flat-G1-v0",
    robot=ASSETS.make("unitree_g1_29dof"),
    engines=("isaacsim", "mjlab"),
    sim=SimSpec(physics_dt=0.005, decimation=4, episode_length_s=20.0),
    mdp=MdpSpec(
        # per-engine：joint_names vs actuator_names、PD 施加方式不同
        actions={"joint_pos": ActionTermSpec(
            kind="joint_position",
            target=EntityRef(joints=G1_29DOF_DFS_JOINT_NAMES, preserve_order=True),
            params={"scale": "beyondmimic", "use_default_offset": True},
        )},
        # 可移植：直接给函数
        observations={
            "policy": ObsGroupSpec(enable_corruption=True, terms={
                "base_ang_vel":      ObsTermSpec(mdp.base_ang_vel,      noise=NoiseSpec("uniform", -0.2, 0.2)),
                "projected_gravity": ObsTermSpec(mdp.projected_gravity, noise=NoiseSpec("uniform", -0.05, 0.05)),
                "velocity_commands": ObsTermSpec(mdp.generated_commands, params={"command_name": "base_velocity"}),
                "joint_pos":         ObsTermSpec(mdp.joint_pos_rel,     noise=NoiseSpec("uniform", -0.01, 0.01)),
                "joint_vel":         ObsTermSpec(mdp.joint_vel,         noise=NoiseSpec("uniform", -1.5, 1.5)),
                "actions":           ObsTermSpec(mdp.last_action),
            }),
            "critic": ObsGroupSpec(enable_corruption=False, terms={"base_lin_vel": ObsTermSpec(mdp.base_lin_vel), ...}),
        },
        rewards={"rewards": {
            "termination_penalty":  RewardTermSpec(mdp.is_terminated, weight=-200.0),
            "track_lin_vel_xy_exp": RewardTermSpec(mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0,
                                                   params={"command_name": "base_velocity", "std": 0.5}),
            "feet_air_time":        RewardTermSpec(mdp.feet_air_time_positive_biped, weight=1.0,
                                                   params={"threshold": 0.5, "sensor": FEET_CONTACT}),
            # ... 其余 13 项
        }},
        terminations={
            "time_out":     DoneTermSpec(mdp.time_out, time_out=True),
            "base_contact": DoneTermSpec(mdp.illegal_contact,
                                         params={"threshold": 1.0, "sensor": BASE_CONTACT}),
        },
        # per-engine：DR 范式完全不同
        events={
            "physics_material": EventTermSpec(kind="randomize_friction", mode="startup"),
            "add_base_mass":    EventTermSpec(kind="randomize_body_mass", mode="startup",
                                              params={"add_range": (-5.0, 5.0)},
                                              target=EntityRef(bodies=("torso_link",))),
            "push_robot":       EventTermSpec(kind="push_by_setting_velocity", mode="interval",
                                              interval_range_s=(10.0, 15.0),
                                              params={"lin_vel_xy": (-0.5, 0.5)}),
            # ... reset_base / reset_robot_joints
        },
        commands={"base_velocity": CommandTermSpec(mdp.UniformVelocityCommand, params={
            "resampling_time_range": (10.0, 10.0), "rel_standing_envs": 0.2,
            "rel_heading_envs": 0.5, "heading_command": True, "heading_control_stiffness": 0.5,
            "lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.5, 1.5),
        })},
    ),
    agent=AgentSpec(...),   # PPO 超参，两引擎共用
)
```

注意 `physics_material` 事件**不带任何分布参数**——分布来自各引擎 profile 的默认值（isaacsim 是 64-bucket per-shape，mjlab 是 per-env 共享单摩擦）。这正是「各自的特性不变」的落地方式。

对比同一份配置在 main 分支的写法，差异只有三处：`RewTerm` → `RewardTermSpec`、`mdp` 的 import 来源、事件块改为 `kind=` 声明。**这就是迁移成本的形状。**

## 10. 实施阶段与验收

| 阶段 | 内容 | 完成判据 |
|---|---|---|
| P0 | 把 main 的 `G1FlatEnvCfg` 结构化 dump 成唯一 golden；建立差异白名单文件 | golden 与白名单入库，dump 可复现 |
| P1 | `compat/`：署名词汇表 + denylist + 纯 torch math（**已完成**）→ `SensorRef` + `env` 访问器；`EntityView` 暂缓，触发条件见 §12.3.2 | 同一 term 函数在两引擎下读到语义一致的数据；denylist 误用报错；词汇表与 math 的每条断言由测试对着已安装引擎复核 |
| P2 | `spec/` + `EngineAdapter` + `TermRegistry` + 三级 Requirement；spec 层 import 隔离测试 | 纯 python 环境可 import spec；mock adapter 可编译 |
| P3 | `mdp/`：移植 locomotion 所需的可移植 term（先做 flat G1 的 16 reward + 6 obs + 2 done） | 每个 term 有两引擎下的数值一致性测试 |
| P4 | isaacsim adapter：`TaskSpec` → `ManagerBasedRLEnvCfg`，term 配置带 `preserve_order=True` | 编译产物与 golden 的 diff 落在白名单内 |
| P5 | mjlab adapter：移植 env 子类 / `MultiRewardManager` / 力阈值接触传感器 | 同一 TaskSpec 编译通过，差异全部落在白名单内 |
| P6 | `train.py --engine` 收敛；退役 unified 栈；任务 id 单一注册；manifest 落盘 | isaacsim 跑 200 iter 曲线与 main 重合；mjlab 达到同等策略质量 |
| P7 | `migrate/`：analyze + codemod；`mdp/` 补齐到 Isaac 83 核心 term | 拿一个真实开源 Isaac Lab 项目走完 §13 五步 |
| P8 | `verify/`：旧 `SimulatorBackend` 改作状态导出；sim2sim 断言与容差声明 | 同 policy 在两引擎的落地/接触指标在声明容差内 |
| P9 | 按能力矩阵推进 terrain / raycaster / camera / motion_reference | 能力矩阵文档与 `terms.py` 自动一致 |

P4 之前不动任何现有训练路径。P1 在 P2 之前是有意的：`compat/` 是 `mdp/` 与 adapter 共同的地基，先把 6 个语义陷阱固定下来，后面所有 term 才有确定语义。

与 §14.5 里程碑的对应：P0–P6 = M1，P7 = M2，P8–P9 与 §14 的轨 D／E 并行。P1 的 `compat/` 须按 §14.2 的 S1（署名中枢词汇表 `vocab.py`）与 S2（开放选择器种类）来写，不要先做双引擎形状再返工。

### Parity 测试四级

| 级别 | 方法 | 成本 |
|---|---|---|
| L0 配置结构 diff | 编译 TaskSpec 与 golden 逐字段比对；空 diff 或命中白名单 | 秒级，无 GPU，进 CI |
| L1 定动作 rollout | 固定 seed + 固定动作序列跑 N 步，isaacsim 的 obs/reward/done 与 main 按名对齐后逐位比对 | 分钟级，需 GPU，单引擎单进程 |
| L2 短训练曲线 | 200 iter，isaacsim 与 main 的 reward 曲线比对 | 小时级 |
| L3 跨引擎行为 | 同一 TaskSpec 两引擎各训到收敛，比对成功率/步态指标，**不比对张量** | 天级 |

L0 是本设计的最大收益：因为编译目标就是 Isaac Lab 原生 cfg，而 main 的 `G1FlatEnvCfg` 本身就是 Isaac Lab 原生 cfg，「是否和原来一致」退化成一个纯静态字段 diff。

## 11. 引擎升级与新引擎接入

### 应对上游升级

- **版本区间**：adapter 声明 `SUPPORTED_VERSIONS`，启动校验，超出区间报错而非猜测。
- **契约报告 golden**：`adapter.contract_report(spec)` dump 出「语义名 → 原生函数」「能力集」「profile 实际值」。CI 与 golden 比对，升级引擎时差异变成可读 diff，而不是静默数值漂移。
- **失败面前移**：adapter 只构造 cfg，上游改 API 会在编译期抛 import/attr 错误，在训练开始前暴露。

### 接入新引擎

通过 entry point `instinctlab.engines` 注册，新引擎可以是独立 pip 包，不必改核心仓库。清单（缺一项不得宣称支持该任务）：

1. `bootstrap()` 不污染其他引擎进程
2. `compile()` 产出该引擎**原生** env
3. `terms.py` 覆盖任务所需语义名
4. `assets.py` 提供该引擎资产格式（`RobotSpec.asset_for(engine)` + checksum）
5. `caps.py` 声明能力集与版本区间
6. contract report golden 入库
7. L0 / L1 测试通过

新引擎**不需要**先补齐全部术语。缺的走 OPTIONAL 跳过并记录，能跑通的子集立即可用。

## 12. 迁移架构：`compat/` 与 `mdp/`

### 12.1 为什么「0% 签名一致」不决定迁移成本

实测：Isaac Lab `envs/mdp` 有 83 个核心 term，mjlab `envs/mdp` 中同名的只有 28 个（34%），**零个**函数签名完全一致。

但这个数字比较的是**两个独立编写的库**。迁移不需要在它们之间对齐——InstinctLab 拥有**一份**可移植 term 库即可，两个引擎的原生 manager 都能调用它。能这样做的三个前提均已逐行验证成立：

1. **数据属性显式名已收敛**。`root_link_lin_vel_b` / `root_link_ang_vel_b` / `root_link_pos_w` / `root_link_quat_w` / `projected_gravity_b` / `joint_pos` / `joint_vel` 在两侧同名同义（`IsaacLab/.../articulation_data.py:804,813,844,852,788,754,763` ↔ `mjlab/.../entity/data.py:595,600,448,453,584,364,374`）。
2. **term 签名同形**：`(env, ..., asset_cfg)`，`asset_cfg` 只用到 `.name` / `.joint_ids` / `.body_ids`，两侧都有。
3. **env 级访问器同名同义**：`num_envs`、`device`、`step_dt`、`episode_length_buf`、`max_episode_length`、`common_step_counter`、`scene[...]`、`scene.env_origins`、`command_manager.get_command`、`action_manager.action` / `prev_action`。

### 12.2 实证样本

`InstinctLab-main/.../tasks/locomotion/mdp/rewards.py` 全部 4 个自定义 reward 的判定：

| 函数 | 判定 | 所需处理 |
|---|---|---|
| `stand_still` | 完全可移植 | 无 |
| `track_lin_vel_xy_yaw_frame_exp` | 机械改写后可移植 | legacy 别名 → 显式名；math 工具改纯 torch |
| `track_ang_vel_z_world_exp` | 机械改写后可移植 | 同上 |
| `feet_air_time_positive_biped` | 需传感器 facade | sensor 引用与 air_time 索引方式不同；数学相同 |

**没有一个需要重写数学或重新推导语义。**

### 12.3 `instinctlab/compat/`

```
compat/vocab.py     署名的中枢词汇表：每个量的参考系 / 原点 / 单位 / 旋转约定 + 各引擎 spoke 映射（S1）  ✅ 已实现
compat/denylist.py  同名不同义陷阱 + Isaac legacy 别名改写表                                    ✅ 已实现
compat/math.py      纯 torch 数学工具，四元数一律 wxyz（D8）                                     ✅ 已实现
compat/sensors.py   SensorRef：统一接触/射线传感器引用与索引
compat/env.py       统一 env 访问器（get_command 空值、physics_dt 路径、类型名）
compat/entity.py    EntityView：统一 asset.data 词汇，每引擎一实现 —— 触发条件见 §12.3.2
```

`vocab.py` / `denylist.py` 的每条断言都由 `tests/test_compat_vocab.py` 对着**已安装的引擎**复核，不依赖任何引擎运行时：mjlab 的 `EntityData` 可独立 import，Isaac 的 `ArticulationData` 用 `ast` 读源码（`import isaaclab.assets` 会拉起 `omni`）。写这些测试时纠正了本节此前的三处说法，见下。

**denylist：5 个同名不同义的语义陷阱**，误用必须报错，不得默认放行：

| 陷阱 | isaacsim | mjlab |
|---|---|---|
| `joint_acc` | 有限差分（`_previous_joint_vel`） | MuJoCo `qacc` 解析值 |
| `applied_torque` | 关节空间 (nv) | **无同名属性**。关节空间等价物是 `qfrc_actuator` (nv)；`actuator_force` 是 nu 维执行器空间，是假朋友 |
| `default_root_state` 速度行 | COM 系 | link 系 |
| `body_link_lin_vel_w`（非根 body） | per-body COM 偏移换算 | 用 root 的 `subtree_com` |
| 重力向量 | `GRAVITY_VEC_W`（**大写**），从 live sim 重力归一化，跟随任务改重力 | `gravity_vec_w`（小写），entity 构建期硬编码 `[0,0,-1]` |

三处修正（均由测试实证）：

1. **legacy 别名是 19 个，不是 2 个**，且分两组：13 个指向 COM 量，6 个指向 link 量。最危险的是 **`root_lin_vel_b`**——它读起来像 link 量，实际是 `root_com_lin_vel_b`。codemod 若按直觉改写成 `root_link_lin_vel_b` 就换了一个物理量，而 mjlab 没有这些别名，下游没有任何东西会报错。完整改写表在 `denylist.LEGACY_COM_ALIASES` / `LEGACY_LINK_ALIASES`，测试双向断言它等于 Isaac 自己 docstring 里的 `Same as :attr:` 声明。
2. **重力向量不是同名陷阱，是拼写 + 语义双重差异**。Isaac Lab 的 `ArticulationData` 上**没有** `gravity_vec_w`；它叫 `GRAVITY_VEC_W`。可移植 term 一律改用 `projected_gravity_b`（两侧推导一致）；随机化重力的任务必须按 per-engine 处理。
3. **`body_link_lin_vel_w` 不能既可移植又被 denylist**。中枢因此**不提供任何 per-body 速度**；根部速度用 `root_link_lin_vel_w`，per-body 速度走 per-engine term 并声明容差。

另有一个探测方法上的坑值得记录：`hasattr(EntityData, name)` 会漏掉 mjlab 用 dataclass 注解声明、无类级默认值的字段（`gravity_vec_w`、`default_root_state`、`soft_joint_pos_limits` 都是这种）。判定属性是否存在必须同时查 `__annotations__`。

#### 12.3.1 `compat/math.py`：第三份拷贝，但是被钉住的那份

Isaac Lab 拥有 `utils/math.py` 原本，mjlab 以 `utils/lab_api/math.py` 整份 vendor 了它。两边**共有 59 个函数，其中 54 个逐字符相同**；余下 5 个（`_sqrt_positive_part`、`quat_from_matrix`、`apply_delta_pose`、`convert_camera_frame_orientation_convention`、`rigid_body_twist_transform`）只是格式与等价改写，float64 下数值差 `0.0`。

可移植 term 不能 import 其中任何一份：Isaac 那份连独立 import 都做不到（`isaaclab.utils.__init__` 拉 `pxr`）。所以 `compat/` 存第三份拷贝，收录范围由**本仓库真实调用点**决定（`isaaclab.utils.math` 在 40 处被引入），共 23 个函数加内部闭包，保留 Isaac 的 BSD-3 署名。

拷贝的风险由 `tests/test_compat_math.py` 承担：对**两个引擎**逐函数比对，float64 与 float32 各一轮，断言 `torch.equal` 而非容差——目前全部恰好相等，所以任何上游改动都藏不住。不能比源码文本，因为本仓库 black 会重排。测试输入除随机批外含恒等四元数、接近 π 的旋转、π 的奇数倍角，正是 `axis_angle_from_quat` 的 Taylor 分支与 `wrap_to_pi` 边界所在；注入两处变异验证过这套断言确实会失败。

两处**因缺席而生效的迁移规则**：

- `convert_quat` 不 vendor。两引擎签名都是 `convert_quat(quat, to="xyzw")`，默认离开中枢约定。代之以 `quat_wxyz_to_xyzw` / `quat_xyzw_to_wxyz`，方向写在名字里，没有默认值。本仓库现有 6 处调用全部显式传了 `to=` 且全部贴着引擎边界（PhysX body pose、`ArticulationView` 读回、warp mesh transform），D8 的判断得到实证。
- `quat_rotate` / `quat_rotate_inverse` 不 vendor。它们是 Isaac Lab **v2.1.0 起废弃**、mjlab 直接删除的别名，用了就跑不了 mjlab。与 `quat_apply` / `quat_apply_inverse` 逐位相等（测试断言），所以 codemod 可机械改写。

`instinctlab/utils/math.py` 已改用它，成为本仓库**第一个脱离引擎的模块**：8 个函数对 main 逐位相同，6 个 `torch.jit.script` 函数照常编译，在 `isaaclab` / `omni` / `pxr` 全被屏蔽的环境下可导入——这条性质写成了回归测试。

#### 12.3.2 `EntityView` 暂缓，及其触发条件

原计划里 `compat/entity.py` 提供统一的 `asset.data` 视图。落地时发现它与本设计的核心前提冲突：§12.1 的结论是可移植 term **直接读 `asset.data.<attr>`**，迁移才便宜（Isaac Lab 的 term 原样可用）。若要求 term 改走 `EntityView`，那些 term 就都得重写，迁移成本回到原点。

而它当前能挡的问题，别处已经挡住了：同名不同义走 denylist（访问即报错），Isaac legacy 别名走 codemod 改写，per-body 速度已从中枢移除。两个引擎在中枢量上拼写一致，视图层此刻只是一次多余的属性解引用，还落在每步每环境的热路径上。

因此暂不实现，触发条件写死：**接入第三个引擎（D6）时，若其数据属性拼写偏离中枢**，`EntityView` 就是把偏离吸收在一处、不让它渗进 term 的地方。在那之前 `vocab.py` 的 spoke 表已经把映射记下来了，届时是消费它而不是重新发现它。

### 12.4 `instinctlab/mdp/`

一份可移植 term 库，覆盖 Isaac 83 个核心 term 的等价物，**由移植 main 与 Isaac Lab 的现有实现种子化，不重新推导数学**。多数函数 3–10 行。

按族的可移植性：

| 族 | 可移植性 | 落点 |
|---|---|---|
| 观测（本体感知） | 直接可移植 | `instinctlab/mdp/observations.py` |
| 奖励（正则化 / 跟踪 / 姿态） | 直接可移植 | `instinctlab/mdp/rewards.py` |
| 终止（非接触类） | 直接可移植 | `instinctlab/mdp/terminations.py` |
| 接触 / 传感器相关 term | facade 后可移植 | 同上，经 `compat/sensors.py` |
| 命令 | 可移植（采样逻辑共享，基类各写薄壳） | `instinctlab/mdp/commands/` |
| 动作 | **per-engine** | `engines/<name>/actions.py` |
| 事件 / 域随机化 | **per-engine** | `engines/<name>/events.py` |
| 场景 / 地形 / 资产 / sim | **per-engine** | `engines/<name>/{scene,assets,sim}.py` |

per-engine 的四个族就是迁移的全部不可消除面。其中场景 / 资产 / sim 主要是 **per-robot** 的：G1 补完一次，后续所有 G1 任务几乎零成本。

## 13. 迁移工作流

| 步骤 | 动作 | 要点 |
|---|---|---|
| 1 | **先在 Isaac 上跑通，不改行为** | 按原样接进来，仍用 isaaclab 原生 env；L0 配置 diff 必须为空。不先固定基线，后面无法区分「迁移引入的差异」与「本来就有的差异」 |
| 2 | **跑迁移分析器** `instinct-migrate analyze` | AST 遍历每个 term 的函数体，提取 `.data.<attr>` 访问与 import，按 `compat/vocab.py` 与 `compat/denylist.py` 分类为 portable / needs-rewrite / needs-facade / per-engine / blocker，输出逐项报告 |
| 3 | **codemod 自动改写机械部分** | `isaaclab.envs.mdp` → `instinctlab.mdp`；legacy 别名 → 显式帧名（19 条）；`isaaclab.utils.math` → `compat/math.py`；`quat_rotate` / `quat_rotate_inverse` → `quat_apply` / `quat_apply_inverse`（逐位相等，见 §12.3.1）；`sensor_cfg` → `SensorRef`。确定性变换，生成 diff 供 review。**不带 `to=` 的 `convert_quat` 不属于此列**，须标为人工确认 |
| 4 | **补 per-engine 条目** | 只剩机器人资产（MJCF + 执行器 profile）、sim profile、动作映射（`actuator_names`）、DR 事件（`dr.*` 组合）。不可消除的人工工作，主要 per-robot |
| 5 | **验证** | L0 Isaac 侧 diff 仍为空 → L1 定动作 rollout → mjlab smoke + 短训练；差异全部落进白名单 |

第 2 步是「基本不需要考虑引擎差异」的实现方式：**不是假装没有差异，而是让工具把需要考虑的范围从整个项目收缩成一张明确清单。**

### 13.1 真正的 blocker（约占 Isaac 核心 term 的 10%）

分析器必须在第 2 步就报出来，让用户在投入前知道。

| Isaac 能力 | mjlab 现状 | 处置 |
|---|---|---|
| TiledCamera / `image` / `image_features` | 有 `CameraSensor`，无 tiled 批渲染与预训练 encoder 管线 | 视觉任务暂不跨引擎 |
| RTX 纹理 / 视觉材质 DR | 仅 `mat_rgba` / `geom_rgba` 等基础项 | 降级为可用子集 |
| Deformable / nodal state | MuJoCo 无等价 | REQUIRED 报错 |
| OSC / RMPFlow / Pink IK / binary gripper 动作 | 仅 joint / tendon / site / DLS IK | 操作类任务需逐个新实现 |
| `randomize_rigid_body_collider_offsets` | PhysX 专有 | 不支持 |
| `randomize_physics_scene_gravity` | 无 event 层入口 | 需在 sim cfg 层实现，中等难度 |
| `randomize_rigid_body_scale` | `geom_size` 部分替代 | 部分支持，差异入白名单 |
| `modify_env_param` / `modify_term_cfg` | 仅 reward / termination curriculum | 易实现，补进 `instinctlab/mdp` |

### 13.2 一次性投入

`instinctlab/mdp/` 覆盖 83 个核心 term 是真实的一次性工作量（多数为移植现有实现）。`compat/` 层比原估计更小：词汇表 + denylist + 纯 torch math 已经落地，传感器桥与 env 访问器待做，`EntityView` 按 §12.3.2 暂缓。换来的是此后每个 Isaac Lab 项目的迁移只剩第 4 步的 per-robot 工作。

## 14. 总框架路线：从双引擎到 N 引擎

### 14.1 结构缺口：只有目标端，没有来源端

§2–§13 的设计有 `engines/<name>/`（把我们的 `TaskSpec` 编译到引擎），但没有对称的组件把**别人的项目读成 `TaskSpec`**。`migrate/` 目前的定位是一次性迁移脚本，不是可复用、有明确 IR 输出的组件。

补上来源端之后，整件事是标准的编译器架构：

```
开源项目（任意引擎 idiom）
   │  frontend / 导入器      instinctlab/frontends/<idiom>/
   ▼
IR   spec/ TaskSpec + mdp/ 可移植 term + compat/ 中枢词汇表
   │  backend / 适配器       instinctlab/engines/<name>/
   ▼
引擎原生栈
```

「Isaac Lab 项目跑在 mjlab 上」= frontend(isaaclab) + backend(mjlab)；「mjlab 项目跑在 Isaac 上」= frontend(mjlab) + backend(isaacsim)。成本是 **O(N+M)** 而不是逐对转换器的 O(N×M)。

frontend 与 backend 相互独立：一个引擎可以只有 backend（能作为训练目标）而没有 frontend（不需要导入该 idiom 的项目）。

### 14.2 必须改的五处结构

| 编号 | 现状 | 需改为 | 触发原因 |
|---|---|---|---|
| S1 | 可移植 term 直接读 `root_link_lin_vel_b`，能跑通是因为两引擎**碰巧**同名。中枢事实上是 Isaac 的命名，但无处声明 | `compat/vocab.py` 定义**署名的**中枢词汇表：每个量给出参考系、原点、单位、旋转约定（四元数固定 wxyz，见 D8），各引擎提供 spoke 映射。中枢可以沿用 Isaac 拼写，但须写成「我们选它」 | 第三个引擎不会遵守 Isaac 拼写。没有署名中枢就会退化成双边映射，N×M 回归。四元数是活样本：两引擎都是 wxyz，但 mjlab 侧**未文档化** |
| S2 | `EntityRef` 只有 `joints` / `bodies` | 带**可注册选择器种类**的开放结构，引擎包注册自己的种类；IR 保留目标引擎不认识的引用，由 capability 检查在编译期报错或显式降级 | mjlab `SceneEntityCfg` 有 10 种选择器（joint / body / geom / site / actuator / tendon / camera / light / material / pair），Isaac 只有 4 种，仅 joint / body 重合。**这是 mjlab → Isaac 方向的硬门槛** |
| S3 | `Capability` 是封闭 enum | 带命名空间的字符串 ID（`contact.air_time`、`dr.friction.per_geom`、`sensor.tiled_camera`、`physics.differentiable`），引擎包导入时注册；未注册 ID 启动期报错以防拼写错误 | 新引擎会带来现有引擎都没有的能力（可微物理、软体、触觉）。封闭枚举意味着每次都要改核心包 |
| S4 | D3：`main` 是唯一 golden | golden 定义为**「该项目跑在它原本的引擎上」**。D3 成为特例（我们的 locomotion 项目原生引擎是 Isaac）；每个导入项目在导入时自动获得自己的基线 | 「main 是 golden」对第三方项目没有意义，它们的参照物是自己发表的结果 |
| S5 | parity L0–L3 绑定在具体任务上 | 增加 **conformance suite**：与任务无关的行为探针（自由落体、重力下静态保持、关节 PD 阶跃响应、接触冲量、摩擦滑移），任意 `(机器人, 引擎)` 组合都能跑，产出签名向量 | 逐项目 golden 的成本随项目数线性增长；行为探针是每个「机器人 × 引擎」一次性的，之后所有用该机器人的项目共享 |

### 14.3 组件矩阵

| 引擎 | backend（作目标） | frontend（作来源） | 资产管线 | DR profile |
|---|---|---|---|---|
| isaacsim | 已有 | — | 已有 | 部分 |
| mjlab | 已有 | — | 已有 | 部分 |
| 第三个引擎（待触发，见 D6） | — | — | — | — |
| metasim（通用后端，备选） | — | — | 部分 | — |

两个 frontend 都还不存在，是当前最大的结构缺口。

### 14.4 与既有多引擎框架的关系

容器内两个参考实现各解决了**不同的一半**：

| 项目 | 引擎数 | 抽象层位置 | MDP term 组成 | 结论 |
|---|---|---|---|---|
| MetaSim | 11（blender / genesis / isaacgym / isaacsim / mjx / mujoco / newton / pybullet / pyrep / sapien + hybrid / parallel） | `BaseSimHandler` 统一 sim 句柄；`scenario/` 做场景 IR | **无**。无 `RewardManager` / `ObservationManager`；reward 写成 task 方法 | 互补而非竞争。场景／资产 IR 与 11 个 handler 是可借资产；MDP 组成层是它的空白 |
| holosoma | 3（isaacgym / isaacsim / mujoco） | `base_simulator` 抽象 + **自行重写的** `managers/` | 有，但是自建的一套，与两个引擎的原生 manager 都不兼容 | 即 D2 已否决的路径。自建 manager 意味着任何开源项目都要手工重写 term，迁移成本没有下降 |
| 本项目 | 2 → N | IR 在 **cfg 层**，编译成引擎**原生** manager cfg；不抽象 sim 句柄 | 可移植 term 库 `instinctlab.mdp`，一份实现两引擎共用 | term 函数免重写是迁移成本能降下来的唯一原因 |

**战略选项**：若某个新引擎只需「能跑通、不要求原生 manager 保真」，可写单个 `engines/metasim/` backend 把 MetaSim 当通用后端，一次拿到 6 个以上引擎。代价是拿不到原生 manager 语义，仅适合 sim2sim 验证与快速试跑，不作为主训练引擎。列为 M5 的备选路径。

### 14.5 工作轨与里程碑

轨（A 阻塞其余全部；D 独立，可最早启动）：

| 轨 | 范围 | 关键交付物 |
|---|---|---|
| A · IR 与中枢 | `spec/` `compat/` `mdp/`；S1 / S2 / S3 | 带语义定义的 `vocab.py`、6 项 denylist、可移植 term 库 |
| B · Backend | `engines/<name>/` | `TaskSpec` → 原生 manager cfg 编译器 + 每引擎术语注册表 |
| C · Frontend | `frontends/<idiom>/` | 项目源码 → `TaskSpec` + 未转换清单 + 置信度报告 |
| D · 资产管线 | `assets/pipeline/`：URDF / MJCF / USD 互转 + 数值校验（D5） | converters 封装 + validators 对照报告 + manifest provenance |
| E · 验证 | S4 泛化 golden、S5 conformance 探针 | 行为探针套件 + 每 `(机器人, 引擎)` 签名基线 + CI |
| F · RL 边界 | agent cfg 翻译器（D4），范围很小 | `AgentSpec` + rsl_rl / rl_games / skrl → instinct_rl 的翻译 + 不可翻译项清单 |

里程碑刻意各证伪一个假设，而不是按组件顺序堆功能：

| 里程碑 | 要证伪的假设 | 退出条件 |
|---|---|---|
| M1 一份 TaskSpec 两引擎原生跑 | IR 能编译成原生 manager cfg 且不牺牲保真度 | 即 §10 的 P0–P6 |
| M2 第一个第三方 Isaac Lab 项目导入 | frontend 能机械化把外部项目读成 IR，剩余人工量可枚举 | 自动转换覆盖率有量化数字；未转换项全部归入已知 per-engine 族，**无意料之外的类别**；在 mjlab 上训练到可比性能（D4） |
| M3 反向：mjlab 项目跑在 Isaac 上 | IR 真的引擎中立，而非伪装的 Isaac 配置 | 至少一个使用 geom / site 选择器的 mjlab 项目在 Isaac 上跑起来，**无法映射处明确报错而非静默降级** |
| M4 迁移能力产品化 | 外部用户能自助完成迁移 | 不了解本框架内部的人按文档独立完成一次迁移，能读懂报告里每一条「需要人工处理」 |
| M5 第三个引擎接入（**待触发**，D6） | 接新引擎的成本就是写一个 backend | 接入过程中**对 `spec/` 与 `compat/` 的改动为零**。任何不得不改核心包处都说明 S1／S2／S3 做得不够 |

M3 是引擎中立性的真正检验，位置在 M4 之前：如果 IR 只是伪装的 Isaac 配置，M3 会先失败，不必等到接第三个引擎。这也是 M5 可以安全推迟的原因。

### 14.6 成本模型

| 成本项 | 触发条件 | 一次性 | 量级 |
|---|---|---|---|
| IR + 中枢词汇表 | 全局一次 | 是 | 大 |
| 可移植 term 库 | 全局一次，可增量 | 是 | 大 |
| backend | 每引擎 | 是 | 中 |
| frontend | 每 idiom | 是 | 中 |
| **机器人资产** | **每机器人 × 引擎** | 是 | 小～大 |
| DR profile | 每引擎 | 是 | 中 |
| **迁移一个项目** | 每项目 | 否 | **小** |

只有最后一行随项目数增长，且其量级**以机器人资产已存在为前提**。如果每个新项目都带一个新机器人，最后一行就退化成「机器人 × 引擎」的量级——这是整个价值命题的关键风险。term 函数可移植性已逐行核对、结论乐观；**真正的单点风险是资产而不是代码**，因此轨 D 的优先级应高于其直觉重要性。

### 14.7 决策记录（续）

#### D4：RL 边界统一到 `instinct_rl`，多 runner 留作后续插件

frontend 把外部 agent cfg（rsl_rl / rl_games / skrl）翻译成 `AgentSpec`，**无法翻译的超参必须进未转换清单**，不得静默丢弃或取默认值。

`AgentSpec` 保留 `runner` 字段，默认 `"instinct_rl"`。现在只实现这一个值，但接口不堵死——将来加多 runner 适配是新增枚举值，不是改结构。

由此确定导入项目的验收标准是**可比性能**，不是「复现原论文数字」。不同 RL 库的 PPO 实现细节（优势归一化时机、学习率调度、梯度裁剪口径）不同，翻译天然有损；把复现原数字写进验收标准会引入一个我们控制不了的失败面。

#### D5：做资产自动转换 + 数值校验管线

`instinctlab/assets/pipeline/`：

```
converters/    URDF ↔ MJCF ↔ USD，封装既有工具（mujoco URDF importer、Isaac URDF importer 等）
validators/    转换前后数值对照：质量 / 惯量张量 / COM / 关节限位 / 关节名集合 / DFS 序 / 碰撞几何数
manifest.py    provenance + checksum：记录源文件、转换器版本、校验报告
```

**validators 比 converters 重要。** 转换工具本身是有损的，让跨引擎资产可信的是那份数值对照报告，而不是转换动作本身。校验不通过的资产禁止进入训练路径。

依据 §14.6 成本模型：逐项目迁移成本收敛到「小」的唯一前提是机器人资产已存在。没有可信的自动管线，每个带新机器人的项目都会退回「机器人 × 引擎」的量级。因此轨 D 的优先级高于其直觉重要性。

#### D6：第三个引擎暂不选定，先把 isaacsim ↔ mjlab 双向打通

M4 转为**待触发**里程碑。双向打通（M3）本身就是 IR 引擎中立性的检验——如果 IR 只是伪装的 Isaac 配置，M3 会先失败，不必等到接第三个引擎才发现。

S1／S2／S3 仍然照做。它们的价值不依赖第三个引擎何时到来：S2 是 M3 的**前置**（mjlab 的 geom / site / actuator 选择器），S1／S3 是避免在双引擎期把结构锁死的低成本保险。

#### D7：IR 只支持 manager 架构的项目

`TaskSpec` 显式假设 manager 语义（term 字典 + 分组观测 + 加权奖励）。这是「编译成原生 manager cfg」得以成立的前提，放弃它会让 IR 退化成又一个 sim 句柄抽象——即 §14.4 中 holosoma 的路径。

frontend 遇到单体 env 项目（HumanoidVerse / PBHC / IsaacGymEnvs 风格）必须**明确报错并说明需要先手工重构成 term 结构**，禁止尝试自动拆解单体 `compute_reward()`。自动拆解只能产出看起来能跑、语义已经漂移的结果，这比直接失败更糟。

这条同时约束了引擎选择：非 manager 架构的引擎（Genesis / IsaacGym / Newton）若要接入，走的是 backend 路径（我们的 TaskSpec 编译过去），而不是 frontend 路径（导入它们生态的项目）。

#### D8：四元数一律 WXYZ，xyzw 只允许出现在引擎 API 边界

中枢约定固定为 **`(w, x, y, z)`**，写入 `compat/vocab.py`。`spec/`、`mdp/`、`compat/` 中任何四元数的入参、返回值、中间量都是 wxyz，不接受「调用方自己知道是哪种」。

两个引擎的 `data.*` 层本来就都是 wxyz，但**证据强度不同**：

| 来源 | 约定 | 是否显式文档化 |
|---|---|---|
| Isaac Lab `articulation_data` | wxyz | **是**。`root_link_quat_w` / `root_com_quat_w` / `body_link_quat_w` 及惯性主轴四元数的 docstring 均写明 `(w, x, y, z)`（`articulation_data.py:853,885,917,949`） |
| mjlab `entity/data.py` | wxyz | **否**。`root_link_quat_w:454` / `root_com_quat_w:474` / `body_link_quat_w:495` / `geom_quat_w:546` / `site_quat_w:567` 的 docstring 只写 "quaternion"；约定实际继承自 MuJoCo `qpos` / `xquat` 的 w-first 惯例 |

mjlab 侧是**未文档化的隐式依赖**。`vocab.py` 该条目必须注明依据是 MuJoCo 的 `qpos` / `xquat` 约定而非 mjlab 文档，并列入引擎升级复查项（§11 contract report）。这是 S1 要求中枢署名的现实理由：不写下来，就没有地方能在升级时发现约定漂移。

**xyzw 真实存在，但只在边界。** 仓库现有的 xyzw 出现点全部贴着引擎底层 API：

| 位置 | 边界 |
|---|---|
| `backends/isaacsim/backend.py:501` | `root_physx_view.set_root_transforms` 要 xyzw（PhysX / USD 约定） |
| `utils/warp/raycast.py:59,64` | warp mesh transform 要 xyzw |
| `sensors/volume_points/volume_points.py:167` | 从 `ArticulationView` 读回，xyzw → wxyz |
| `motion_reference/*` | 内部 `base_quat_w` 是 wxyz，仅在写 `ArticulationView` 时转 xyzw |

规则：**转换只能贴着引擎调用发生，转换结果不得跨函数边界传播。** 进入 `mdp/` 或 `compat/` 的一律已经是 wxyz。

**`convert_quat` 的默认参数是陷阱。** Isaac Lab（`utils/math.py:198`）与 mjlab（`utils/lab_api/math.py:201`，一份拷贝）签名都是 `convert_quat(quat, to="xyzw")`——**默认把 wxyz 转成 xyzw**。因此：

- `compat/math.py` 的 quat 工具**不设默认转换方向**，`to=` 必须显式传；更倾向于根本不导出 xyzw 转换，把它下放到 `engines/<name>/`。
- `migrate/analyze.py` 必须把不带 `to=` 的 `convert_quat(x)` 调用标为**需人工确认**，不得当作机械可改写项。

外部数据源（motion 文件、SMPL / poselib、ROS / IMU 部署栈）常用 xyzw。frontend 与 motion 导入必须在入口处归一到 wxyz，并在 manifest 记录原始约定。

## 15. 硬约束（禁止项）

1. `spec/` 层禁止 import 任何引擎 SDK（`isaaclab`、`mujoco`、`mjlab`、`omni.*`、PhysX 类型）。
2. 公共层禁止 `if engine == "isaacsim"` / `"mjlab"` 分支。引擎差异只能是按引擎名分区的**数据**，或 adapter 内部实现。
3. 禁止再写第三套 env 循环或 manager。引擎原生实现是唯一实现。
4. 禁止为一个引擎复制第二份 `TaskSpec`。
5. 关节/body 顺序以 `RobotSpec` 的 DFS 名为唯一真值；禁止依赖 URDF / USD / MJCF 隐式遍历顺序。
6. REQUIRED 能力缺失必须启动即失败；OPTIONAL 跳过必须打印汇总并写入 manifest，禁止无记录的静默降级。
7. 差异白名单新增条目必须写 reason。
8. 禁止把 PhysX 与 MuJoCo 的接触力当逐值等价；需要绝对力值的任务必须声明容差。
9. 训练契约与 `instinct_onboard` 部署分开，不塞进同一基类。
10. 不用 Gymnasium 当 sim2sim 对齐层；不把 Isaac Lab 3 的 Articulation 工厂抄进公共层。
11. 禁止写成对的「引擎 A → 引擎 B」转换器。跨引擎转换只能经过 IR：frontend 读进来、backend 编出去。
12. 中枢词汇表是 `compat/vocab.py` 中署名的那一份。禁止在 term 里直接假定某个引擎的属性拼写即中枢，即使当前两引擎恰好同名。
13. frontend 遇到 IR 表达不了的构造必须报错并计入未转换清单，禁止丢弃信息后静默产出一个「能跑」的 `TaskSpec`。
14. 四元数一律 `(w, x, y, z)`。`spec/` / `mdp/` / `compat/` 禁止出现 xyzw；转换只能贴着引擎 API 调用发生，且不得跨函数边界传播。禁止调用不带 `to=` 的 `convert_quat`——两个引擎的默认值都是转成 xyzw；改用 `compat.math` 的 `quat_wxyz_to_xyzw` / `quat_xyzw_to_wxyz`。
15. 可移植代码禁止 import `isaaclab.utils.math` 或 `mjlab.utils.lab_api.math`，一律走 `compat/math.py`。vendor 进来的函数不得就地修改：改了就不再是「两引擎共有的那份」，钉住它的测试也就失去意义。
