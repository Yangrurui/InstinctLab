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

`RobotSpec.joint_names` / `body_names` 的 DFS 名序是唯一真值。两个引擎的 term 配置都**逐个列出关节名**并发出 `preserve_order=True`，由引擎自己按名解析出索引。

**只发 `preserve_order=True` 而选择器写 `.*` 是无效的**，这一点代价不小才弄清楚：`resolve_matching_names` 按**模式**的先后排列选择结果，单个 `.*` 意味着所有关节都落在同一个模式里，模式内部仍按实体自身顺序排列——于是 `preserve_order` 真假结果完全相同，Isaac 侧留在 PhysX 的 BFS 序、mjlab 侧留在模型文件的顺序。落点有两处而非一处：动作项和 `joint_pos`/`joint_vel` 两个观测项都要带这份名单，只钉一侧会让策略的输入与输出在两个引擎上索引方式不同。`last_action` 读动作管理器的整条向量，自动跟随动作项。

验收上这件事**没有任何既有检查能看见**：`test_asset_parity.py` 比的是关节名集合，`probe_terms.py` 当时会先把两边重排到 canonical 序再比。现在 `test_asset_parity.py` 断言目录序等于 URDF 的 DFS 前序、`test_parity_static.py` 正向断言三个选择器都持有完整名单，而 `probe_terms.py` 不再重排——两引擎的 `joint_pos` / `joint_vel` / `actions` 实测差为**精确的 0**。

已验证两侧都支持这一机制：

- Isaac Lab：`SceneEntityCfg.preserve_order`（`managers/scene_entity_cfg.py:102`）、`JointAction` 的 `find_joints(..., preserve_order=cfg.preserve_order)`（`envs/mdp/actions/joint_actions.py:65`）
- MJLab：`SceneEntityCfg.preserve_order`（`managers/scene_entity_config.py:126`）、`envs/mdp/actions/actions.py:113`

因此 DFS 顺序可以在**原生 env 内部**实现，不需要自研 env 做张量置换。这是本设计成立的前提。

已知偏差：main 分支使用 PhysX BFS 顺序（`left_shoulder_pitch, right_shoulder_pitch, waist_pitch, ...`），本设计下 isaacsim 路径改为 DFS（`waist_pitch, waist_roll, waist_yaw, left_hip_pitch, ...`）。这是一处**被接受并记录**的偏差，进入差异白名单（§8）。已有 main checkpoint 迁移需要一次按名置换。

### D2：退役自研 env 与 managers

以下组件退役，改用引擎原生 env + `TaskSpec` 编译。**已于 2026-08-19 删除**：

| 文件 | 行数 | 去向 |
|---|---|---|
| `envs/unified_manager_based_rl_env.py` | 246 | 删除，改用引擎原生 env |
| `managers/unified.py` | 643 | 删除 |
| `tasks/locomotion/mdp/unified.py` | 593 | 删除，可移植项在 `instinctlab/mdp/` |
| `tasks/locomotion/unified_flat_env_cfg.py` | 439 | 删除，任务声明是 `tasks/locomotion/config/flat_g1.py` |
| `tasks/locomotion/commands.py` | 123 | 删除，仅服务上者 |
| `rl/`（3 文件） | 204 | 删除，训练走 `utils/wrappers/instinct_rl/` |
| `scripts/instinct_rl/{train,play}_unified.py` | 590 | 删除，入口是 `scripts/train.py` |
| `scripts/profile_backend.py` | 322 | 删除：它测的是 unified env 的 step，对象已不存在 |

`SimulatorBackend` 双实现（`backends/isaacsim/backend.py` + `backends/mjlab/simulator.py`）**不作废**，作为 sim2sim 状态断言层保留，退出训练热路径，由 `tests/simulators/` 服务。它们需要的场景描述从上述 env cfg 里抽出为 `verify/scene.py::locomotion_flat_scene`——那些行为断言写根状态、step、读传感器，从不经过 MDP，所以拿掉 MDP 之后它们一行没改。

删除过程暴露了这套栈留下的两处**静默断裂**，都是「没有异常、没有测试会响」的类型，记在 §12.11。

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
scripts/train.py  --engine isaacsim|mjlab  --task Instinct-Velocity-Flat-G1
        │  先按 --engine 选 adapter，adapter.bootstrap()（Isaac 的 AppLauncher 早于 import torch）
        ▼
instinctlab/spec/            引擎无关声明层（禁止任何引擎 import）
instinctlab/mdp/             可移植 MDP term 库（一份实现，两引擎共用）
instinctlab/compat/          薄兼容层：署名词汇表 / denylist / 纯 torch math / EntityRef / SensorRef / env
        ▼
instinctlab/engines/         base / registry / compile 三个机件同样引擎无关，
                             让启动器能在 bootstrap 之前枚举 adapter
instinctlab/engines/<name>/  只承载真正需要两份实现的族：
                             actions / events(DR) / scene / assets / sim
        ▼
引擎原生栈（不改、不包、不重写）
   isaacsim → isaaclab.ManagerBasedRLEnv + instinctlab.envs.InstinctRlEnv
   mjlab    → mjlab.ManagerBasedRlEnv    + engines/mjlab 的同名子类
        ▼
instinct_rl OnPolicyRunner（两侧已是同一 VecEnv 契约，无需改动）

instinctlab/migrate/         【尚未存在，P7】迁移工具：analyze（AST 分类报告）+ codemod（确定性改写）
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

### 6.4 实现时定下的三处细节

**注册表同时支持「按 kind」与「按族」两种入口。** §6.1 的例子写的是 `@TERMS.observation("base_ang_vel")`（按 kind），而 §5 说观测是可移植族、携带 `func=`——两者不一致。实现取两者的并集：每个族有一个 `@TERMS.portable(family)` 包装器，负责把 term 自带的函数塞进该引擎的原生 term 类；同时任何族都仍可按 kind 注册。于是「某个观测在一个引擎上恰好需要原生实现」有地方安放，而不必让所有引擎为它开特例。`func=` 与 `kind=` 二选一的不变式因此直接决定走哪条路径，无需额外判断。

**`AgentSpec` 用点路径惰性引用 runner。** 项目现有的 `InstinctRlOnPolicyRunnerCfg` 建在 `isaaclab.utils.configclass` 上，`spec/` 直接持有它会把 Isaac Lab 拖进每一个任务声明，隔离即刻失效。改为 `runner="pkg.module:Class"`，由 backend 在已加载引擎之后 `resolve()`。`engine_overrides` 存在，但文档里写明它只该用于让墙钟时间可比的 rollout 长度之类；按引擎改超参会让两次运行失去可比性，与本项目的目的相反。

**`num_envs` 不进 `SceneSpec`。** 跑多少份是编译的入参，不是任务的属性；放进去等于让任务文件声明你有多大的显卡。旧 `sim/scene.py` 里的同名 `SceneSpec` 带这个字段，它属于将在 P8 降级到 `verify/` 的那套栈，两者不要混用。

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

`tests/parity/isaacsim.locomotion_flat.whitelist.json`，实际落地的是 JSON、按点号路径做键，当前 50 条。形如：

```json
{
  "actions.joint_pos.joint_names": "D1 统一为 DFS 顺序；main 使用 [\".*\"]（PhysX BFS）",
  "actions.joint_pos.preserve_order": "D1 同上",
  "observations.policy.joint_pos.params.asset_cfg.joint_names": "D1 同上",
  "observations.critic.base_lin_vel.func": "可移植 term 读 root_link_lin_vel_b，main 读 root_lin_vel_b（= root_com_lin_vel_b 的 legacy 别名）。二者差 ω × R(−com_pos_b)；中枢取 link 量因为它是两引擎都能表达的那个。critic 专用，不入部署策略。",
  "rewards.track_lin_vel_xy_exp.func": "同上，读 root_link_lin_vel_w 而非 COM 别名。这一项会影响策略，不同于 critic 观测，需要在 review 中被明确看到。"
}
```

L0 测试编译 `TaskSpec` 后与 golden 逐字段比对，diff 必须为空或全部命中白名单。**新增白名单条目必须写 reason 并在 review 中被看到**——这是防止偏差悄悄累积的唯一闸门。

**不需要白名单的一项**，记下来以免后人以为漏了：`base_ang_vel` 与 `track_ang_vel_z_world_exp` 同样从 COM 别名改成了 link 拼写，但**数值完全一致**。Isaac 的 `root_link_vel_w` 是 `root_com_vel_w.clone()` 之后**只对 `[:, :3]` 加 COM 偏移修正**，角速度行原样拷贝。刚体角速度本就与参考点无关，但这里是**读源码实测确认**而非物理直觉推断，由 `test_mdp_terms.py` 钉住；若 Isaac 哪天也修正角速度行，那条测试会失败，届时这两项才需要进白名单。

## 9. Locomotion Flat G1 的 TaskSpec（示意）

注意可移植族用 **`func=` 函数引用**（写法与 Isaac Lab 原生 cfg 一致），per-engine 族用 **`kind=` 语义名**。

```python
# instinctlab/tasks/locomotion/config/flat_g1.py — 无任何引擎 import
from instinctlab import mdp                     # 可移植 term 库

LOCOMOTION_FLAT_G1 = TaskSpec(
    task_id="Instinct-Velocity-Flat-G1",
    robot=make_g1_29dof_robot_spec(),        # instinctlab.assets.unitree_g1_spec
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
| P1 | **已完成**。`compat/`：署名词汇表 + denylist + 纯 torch math + `EntityRef` 下降 + 接触传感器读取 + `env` 访问器；`EntityView` 已撤销，见 §12.3.2 | 同一 term 函数在两引擎下读到语义一致的数据；denylist 误用报错；词汇表 / math / 选择器表 / 传感器轴序的每条断言由测试对着已安装引擎复核 |
| P2 | **已完成**。`spec/`（`capability` / `entity` / `sensor` / `mdp` / `task`）+ `engines/`（`base` / `registry` / `compile`）+ 三级 Requirement；spec 与 engines 机件的 import 隔离测试 | 纯 python 环境可 import spec；mock adapter 端到端编译整个 MdpSpec，跳过 / emulate / strict 三条路径各有测试与变异检验 |
| P3 | **已完成**（flat G1 部分）。`mdp/`：20 个可移植 term（observations / rewards / terminations）；清出 3 个**不可移植**项交给 per-engine 注册表，见 §12.4.1 | 属性可移植性由 AST 扫描对着 denylist / legacy 别名表 / 两引擎数据类静态把关；term 数值由构造输入的桩验证；变异检验覆盖 |
| P4 | **已完成**。`engines/isaacsim/`（`terms` / `scene` / `assets` / `adapter`）+ `tasks/locomotion/config/flat_g1.py` 的引擎无关声明 | 当时 177 处 diff、0 处未解释；编译产物能构造并 step，观测维度与奖励项与 main 一致。当前数字见 §12.5 |
| P5 | **已完成**。mjlab adapter：`assets` / `scene` / `events` / `rewards` / `terms` / `adapter`；从 InstinctMJ 移植 `reset_joints_by_scale` / `randomize_body_mass` / `contact_slide` | 同一 TaskSpec 编译通过并实际构造 step；与 InstinctMJ 的 AST 对拍一致；26 个 term 两引擎数值一致，见 §12.6 |
| P6 | **已完成**。`scripts/train.py --engine` 收敛（`engines.ADAPTERS` + `tasks.registry` 双注册表、manifest 落盘、agent cfg 脱离 Isaac），见 §12.7；unified 栈已退役，删除清单见 §3 | 两引擎从同一入口、同一 task id 起训；isaacsim 4096 env 跑到 56k step/s，mjlab 5000 iter 收敛 |
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
compat/entity.py    EntityRef 下降到原生 SceneEntityCfg + resolve 后归一化（§12.3.3）            ✅ 已实现
compat/sensors.py   接触传感器读取：元素名 / 轴序归一，力语义差异显式化（§12.3.4）                 ✅ 已实现
compat/env.py       统一 env 访问器（get_command 空值、physics_dt 路径、类型名）

spec/entity.py      EntityRef：开放选择器种类（S2）                                             ✅ 已实现
spec/sensor.py      ContactSensorRef：声明「测什么」，由 backend 决定编成什么                     ✅ 已实现
```

`vocab.py` / `denylist.py` 的每条断言都由 `tests/test_compat_vocab.py` 对着**已安装的引擎**复核，不依赖任何引擎运行时：mjlab 的 `EntityData` 可独立 import，Isaac 的 `ArticulationData` 用 `ast` 读源码（`import isaaclab.assets` 会拉起 `omni`）。写这些测试时纠正了本节此前的三处说法，见下。

**denylist：7 个同名不同义的语义陷阱**，误用必须报错，不得默认放行：

| 陷阱 | isaacsim | mjlab |
|---|---|---|
| `joint_acc` | 有限差分（`_previous_joint_vel`） | MuJoCo `qacc` 解析值 |
| `applied_torque` | 关节空间 (nv) | **无同名属性**。关节空间等价物是 `qfrc_actuator` (nv)；`actuator_force` 是 nu 维执行器空间，是假朋友 |
| `default_root_state` 速度行 | COM 系 | link 系 |
| `body_link_lin_vel_w`（非根 body） | per-body COM 偏移换算 | 用 root 的 `subtree_com` |
| 重力向量 | `GRAVITY_VEC_W`（**大写**），从 live sim 重力归一化，跟随任务改重力 | `gravity_vec_w`（小写），entity 构建期硬编码 `[0,0,-1]` |
| 接触力 | `net_forces_w`，世界系，**仅法向** | 无同名属性。最接近的 `force` 是完整三维接触力，默认在**接触系** |
| `write_root_state_to_sim` 速度行 | 写的是 COM 速度 | 写的是 link 速度。同样十三个数写进去，两个机器人落在不同状态（本任务的 G1 实测差到 0.85 m/s），此后每个读速度的 term 都对不上。要用两侧都有的 frame-qualified 写入口 |

三处修正（均由测试实证）：

1. **legacy 别名是 19 个，不是 2 个**，且分两组：13 个指向 COM 量，6 个指向 link 量。最危险的是 **`root_lin_vel_b`**——它读起来像 link 量，实际是 `root_com_lin_vel_b`。codemod 若按直觉改写成 `root_link_lin_vel_b` 就换了一个物理量，而 mjlab 没有这些别名，下游没有任何东西会报错。完整改写表在 `denylist.LEGACY_COM_ALIASES` / `LEGACY_LINK_ALIASES`，测试双向断言它等于 Isaac 自己 docstring 里的 `Same as :attr:` 声明。
2. **重力向量不是同名陷阱，是拼写 + 语义双重差异**。Isaac Lab 的 `ArticulationData` 上**没有** `gravity_vec_w`；它叫 `GRAVITY_VEC_W`。可移植 term 一律改用 `projected_gravity_b`（两侧推导一致）；随机化重力的任务必须按 per-engine 处理。
3. **`body_link_lin_vel_w` 不能既可移植又被 denylist**。中枢因此**不提供任何 per-body 速度**；根部速度用 `root_link_lin_vel_w`，per-body 速度走 per-engine term 并声明容差。

另有一个探测方法上的坑值得记录：`hasattr(EntityData, name)` 会漏掉 mjlab 用 dataclass 注解声明、无类级默认值的字段（`gravity_vec_w`、`default_root_state`、`soft_joint_pos_limits` 都是这种）。判定属性是否存在必须同时查 `__annotations__`。

#### 12.3.1 `compat/math.py`：第三份拷贝，但是被钉住的那份

Isaac Lab 拥有 `utils/math.py` 原本，mjlab 以 `utils/lab_api/math.py` 整份 vendor 了它。两边**共有 59 个函数，其中 55 个逐字符相同**；余下 4 个（`_sqrt_positive_part`、`quat_from_matrix`、`apply_delta_pose`、`rigid_body_twist_transform`）只是格式与等价改写，float64 下数值差 `0.0`。这 4 个名字由 `tests/test_compat_math.py::test_the_two_engines_copies_have_only_the_known_rewrites_between_them` 正向断言——上游新改写一个公式会多出一个名字，把改写收敛回去会少一个名字，两种都会让测试红。早先文档写的是 54/5，多算了 `convert_camera_frame_orientation_convention`（它后来两边一致了），而当时没有任何检查看得见这件事，这条断言就是为此补的。

可移植 term 不能 import 其中任何一份：Isaac 那份连独立 import 都做不到（`isaaclab.utils.__init__` 拉 `pxr`）。所以 `compat/` 存第三份拷贝，收录范围由**本仓库真实调用点**决定（`isaaclab.utils.math` 在 40 处被引入），共 23 个函数加内部闭包，保留 Isaac 的 BSD-3 署名。

拷贝的风险由 `tests/test_compat_math.py` 承担：对**两个引擎**逐函数比对，float64 与 float32 各一轮，断言 `torch.equal` 而非容差——目前全部恰好相等，所以任何上游改动都藏不住。不能比源码文本，因为本仓库 black 会重排。测试输入除随机批外含恒等四元数、接近 π 的旋转、π 的奇数倍角，正是 `axis_angle_from_quat` 的 Taylor 分支与 `wrap_to_pi` 边界所在；注入两处变异验证过这套断言确实会失败。

两处**因缺席而生效的迁移规则**：

- `convert_quat` 不 vendor。两引擎签名都是 `convert_quat(quat, to="xyzw")`，默认离开中枢约定。代之以 `quat_wxyz_to_xyzw` / `quat_xyzw_to_wxyz`，方向写在名字里，没有默认值。本仓库现有 6 处调用全部显式传了 `to=` 且全部贴着引擎边界（PhysX body pose、`ArticulationView` 读回、warp mesh transform），D8 的判断得到实证。
- `quat_rotate` / `quat_rotate_inverse` 不 vendor。它们是 Isaac Lab **v2.1.0 起废弃**、mjlab 直接删除的别名，用了就跑不了 mjlab。与 `quat_apply` / `quat_apply_inverse` 逐位相等（测试断言），所以 codemod 可机械改写。

`instinctlab/utils/math.py` 已改用它，成为本仓库**第一个脱离引擎的模块**：8 个函数对 main 逐位相同，6 个 `torch.jit.script` 函数照常编译，在 `isaaclab` / `omni` / `pxr` 全被屏蔽的环境下可导入——这条性质写成了回归测试。

#### 12.3.2 `EntityView` 撤销，改为引用解析

原计划里 `compat/entity.py` 提供统一的 `asset.data` 视图。落地时发现它与本设计的核心前提冲突：§12.1 的结论是可移植 term **直接读 `asset.data.<attr>`**，迁移才便宜（Isaac Lab 的 term 原样可用）。若要求 term 改走 `EntityView`，那些 term 就都得重写，迁移成本回到原点。

性能不是理由，这点要写清楚以免后人重新捡起这个论据：实测包装式 proxy 每次属性访问 304ns、直接访问 20ns，按每步约 75 次读算是 21μs/step，占典型 step（10–30ms）的 **0.1%**。真正的理由只有前一段那一条。

而它当前能挡的问题，别处已经挡住了：同名不同义走 denylist（访问即报错），Isaac legacy 别名走 codemod 改写，per-body 速度已从中枢移除。两个引擎在中枢量上拼写一致，数据层不需要间接层。

第三个引擎若拼写偏离中枢，代价由**那个引擎的 backend 承担**——它负责让自己的 data 对象满足 `vocab.HUB`，用同一套 conformance 测试验收。实现上不必包装：**继承引擎的 data 类 + `__getattr__` 兜底**即可，命中的名字走原生查找（实测只贵约 10ns），只有缺失的名字才落到 spoke 表。term 完全无感。

`compat/entity.py` 的名字保留，内容换成**引用解析**——分歧真正所在，且发生在编译期，抽象免费。见下。

#### 12.3.3 `compat/entity.py`：`EntityRef` 下降与 resolve 后归一化

两引擎的 `SceneEntityCfg` 连类名都一样，字段结构也一样（`name` + `preserve_order` + 每种选择器一对 `<kind>_names` / `<kind>_ids`），所以下降是无重命名的字段映射。底层的名字解析更是同一份代码：`resolve_matching_names` 除 docstring 措辞外逐字符相同，8 种 pattern 组合 ×`preserve_order` 两态实测结果全同。**这层不需要重新实现，只需要路由。**

真正的分歧有两处。

**选择器种类。** 12 种里只有 2 种重合：

| | 种类 |
|---|---|
| 两者都有 | `joint`、`body` |
| 仅 Isaac | `fixed_tendon`、`object_collection` |
| 仅 mjlab | `actuator`、`camera`、`geom`、`light`、`material`、`pair`、`site`、`tendon` |

所以 `EntityRef` 给这两种命名字段，其余走开放映射 `other`；目标引擎表达不了的种类**报错而非丢弃**——这正是 S2 说的 mjlab → Isaac 硬门槛。注意 Isaac 的 `fixed_tendon` 与 mjlab 的 `tendon` **不合并**：没有证据表明它们是同一种选择器，合并会让引用在对面解析到另一组元素上。

**`resolve()` 之后 `<kind>_names` 装的东西不同。** Isaac 把用户写的**正则留在原处**（匹配结果丢给 `_`），mjlab **覆盖成匹配到的真实名字**。而且有真实消费者：Isaac 自己的 `envs/mdp/events.py:1451` 把这个字段 `"|".join(...)` 拼回正则去匹配 USD prim path，本仓库的 `reference_masked_proprioception.py:67` 也存了它。这是 denylist 那类陷阱在 config 层的翻版。`resolved_names()` 绕过它——走两边都一致的**索引**，因此不需要任何 per-engine 分支。

#### 12.3.4 `compat/sensors.py`：与实体相反，这里确实需要运行时垫片

两引擎的接触传感器是**结构性对立**：Isaac 声明**一个宽传感器**（prim_path 正则覆盖整机），term 用 `SceneEntityCfg.body_ids` 切片；mjlab 声明**多个窄传感器**，每个由 `primary` 模式限定范围，term 读整只。所以 `ContactSensorRef` 只声明「测什么」，由 backend 决定编成宽传感器的一个切片还是一只专用传感器。

**对得上的部分：** 四个时间量 `current_air_time` / `last_air_time` / `current_contact_time` / `last_contact_time` **两边同名且都是二维 `(env, element)`**，力历史也都是 index 0 最新。这比看上去重要——每个引擎是用**自己的**接触判据、自己的求解器力，在自己的传感器内部算出这些秒数的，term 拿到时口径已经统一。**这是可移植的接触信号。**

**对不上的三处：**

1. 元素名列表叫法不同：Isaac `ContactSensor.body_names`，mjlab `ContactSensor.primary_names`。
2. 力历史轴序相反：Isaac `(env, time, element, 3)`，mjlab `(env, element, time, 3)`。这个错法很隐蔽——**两只脚 + 两个子步**时形状同为 `(env,2,2,3)`，形状断言抓不到，只有值能分辨。测试里专门有一条覆盖这个情形。
3. **力根本不是同一个量，转置也救不了。** Isaac 的 `net_forces_w` 是世界系**仅法向力**（docstring 明确警告不含切向）；mjlab 的 `force` 是完整三维接触力，且默认在**接触系**（除非 `reduce="netforce"` 或 `global_frame=True`）。取模长，一个是法向载荷、一个含摩擦，差值就是那一刻摩擦承担的量。**牛顿阈值不可跨引擎搬运**，已列入 denylist。

因此 `in_contact()` 由接触时长导出而非力阈值——让每个引擎用自己的判据判断，这是唯一能一致的做法。`contact_force_history()` 只统一轴序，不假装数值可比。

引擎识别靠元素名属性的鸭子类型，所以本模块不 import 任何引擎，term 也不必知道自己跑在哪个引擎上。

#### 12.3.5 `compat/env.py`：三处分歧，其余已自发收敛

本以为 env 是分歧重灾区，实测相反。两个 env 类独立演化，公共面几乎逐字重合：`num_envs` / `device` / `physics_dt` / `step_dt` / `max_episode_length` / `max_episode_length_s` / `episode_length_buf` / `scene` / `cfg` / `common_step_counter` / `extras` 与全部七个 manager **两边同名同义**，`scene[name]` 与 `scene.sensors[name]` 也都可用。

这条收敛是「term 直接读 `env.*`、不走访问器」的前提，因此 `test_compat_env.py` 把它写成可执行断言而非注释——引擎升级动了 `step_dt`，失败的是一条测试，而不是一百个已移植的 term。**这也是本模块刻意保持极薄的原因：不需要垫片的地方加垫片，就是已经撤销过一次的 `EntityView` 错误。**

真正没收敛的只有三处：

1. **空命令管理器行为不同。** 任务未声明命令时 mjlab 装 `NullCommandManager`，其 `get_command` 对任何名字返回 `None`；Isaac 永远装真管理器，同样调用抛 `KeyError`。mjlab 这侧更危险：`None` 在若干帧后被下标时才炸，一个速度跟踪奖励静默拿到 `None`，报出来的是离成因很远的形状错误。`get_command()` 让两边都在同一处大声失败，并列出该 env 实际有哪些命令。
2. **类名拼写不同**：`ManagerBasedRLEnv` 与 `ManagerBasedRlEnv`，只差一个大写 L——正是能活过 code review 的那种差异。可移植 term 应标注 `compat.env.RlEnv`（结构化 Protocol），`ENV_TYPE_NAMES` 保留原生拼写供 codemod 识别。
3. **物理步长的配置路径不同**：`cfg.sim.dt` 与 `cfg.sim.mujoco.timestep`。这属于编译期、归 adapter，term 侧读 `env.physics_dt` 两边一致；记在 `PHYSICS_DT_CFG_PATH` 里是因为后人会来这里找它。

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

#### 12.4.1 实测：flat G1 里三个「看起来可移植其实不是」的奖励

按 golden 清点，flat G1 用到 20 个可移植函数 + 3 个不可移植项。**这 3 个是本阶段最有价值的产出**——它们全都长得像普通正则化奖励，直接照搬会在两引擎上各自优化不同的东西：

| term | 读了什么 | 为什么不能可移植 |
|---|---|---|
| `dof_acc_l2` | `joint_acc` | Isaac 是**跨步有限差分**（`ArticulationData._previous_joint_vel`），mjlab 是 MuJoCo **解析 `qacc`**。接触附近二者差值超过该奖励自身尺度，而权重是 `-2e-7`——小到差异只表现为步态略有不同，不会有人去查 |
| `dof_torques_l2` | `applied_torque` | mjlab **没有这个属性**。关节空间等价物是 `qfrc_actuator`(nv)；`actuator_force` 是假朋友，它在执行器空间(nu)，一对一驱动时才碰巧同维 |
| `contact_slide` | 接触力牛顿阈值 + **逐 body 线速度** | 两条都不成立：两引擎报的力不是同一个量（仅法向 vs 完整三维）；中枢**刻意不提供任何逐 body 速度**，因为 Isaac 把每个 body 速度偏移到它自己的 COM，mjlab 报的是根的 `subtree_com` |

三者一律用 `kind=` 声明、per-engine 实现并写明容差。**这不是绕开问题，这正是设计在起作用**：另一种做法是让一个 term 在两边都产出看着合理的数值，然后各自优化不同的目标。

另有两处**签名必须改**，无法回避：

- **接触类 term 收 `ContactSensorRef` 而非 `sensor_cfg`。** Isaac 声明一个宽传感器由 term 切片，mjlab 声明窄传感器由 term 整读，不存在一个可以同时传给两边的原生对象。
- **`illegal_contact` 去掉了 `threshold`。** 两边的原版都对力取模长设牛顿阈值，但那个阈值在一边是「法向载荷 > N」、另一边是「含摩擦的总载荷 > N」，差值是那一刻摩擦承担的量——脚踩在斜坡上会一边越过阈值一边不越过。可移植版改问每个引擎自己的传感器「算不算接触」（走接触时长）。代价是失去了忽略轻微擦碰的能力；真需要力阈值的任务应当 per-engine 声明并写下容差，那才是共享阈值原本在做的事情的诚实版本。

### 12.5 P4 实测：flat G1 编译产物 vs main

`tasks/locomotion/config/flat_g1.py` 把 main 的 `G1FlatEnvCfg` 重述成一份不含任何引擎 import 的 `TaskSpec`；`scripts/check_parity.py` 用 `engines/isaacsim/` 编译它，与 golden 逐字段比对。结果：**314 处差异，0 处未解释**（`tests/parity/isaacsim.locomotion_flat.whitelist.json`，50 条），且编译产物能真实构造并 step——观测维度、16 个奖励项、命令与事件管理器全部正常。

> 两次数字变动都记在这里，因为「差异变多」本身需要有人解释。P5 期间三个奖励从「刻意缺席」改为「按引擎注册」，差异由 177 降到 134、白名单由 57 条降到 46 条（见 §12.6）。P6 修 D1 时差异反而涨到 314：把 5 个选择器（动作项 + policy/critic 各两个关节观测）从一个 `.*` 改成逐个列出 29 个关节名，逐字段比对自然多出 173 条**逐元素**差异。条数涨了，未解释数仍是 0，而这 173 条正是 D1 生效的证据。

差异只有六类，每类都是一个决定：

| 类别 | 条数 | 是什么 |
|---|---|---|
| `.func` 指向 `instinctlab.mdp` 而非引擎自带 | ~56 | 整个设计要做的那次替换，也是同一份声明能编到 mjlab 的原因 |
| action scale 逐关节 vs 按执行器组正则 | ~43 | 同样 29 个数，改从 `RobotSpec.joint_properties` 取，任务因此不需要引擎 import；逐关节验证相等 |
| 接触 term 收 `ContactSensorRef` 而非 `SceneEntityCfg` | ~26 | §12.4.1 的签名改动 |
| 选择器 `str` → 单元素 tuple、`preserve_order=True` | ~5 | `EntityRef` 归一化；D1 |
| `run_name` / `viewer` / 空容器 | ~6 | 日志标签与相机，不进 MDP |

**场景与 sim 子树零差异**：地形、机器人 spawn、接触传感器、PhysX 求解器设置逐字段相同。这一段本来就该相同——它是把 main 的常量按 profile 重新装配，而不是重新推导。

比对过程本身暴露了三个陷阱，已进 §15 硬约束：

- **golden 不能按字段名排序。** 原先 `dump()` 与 `json.dumps` 都排了序，于是「观测项顺序」这个决定策略能否加载 checkpoint 的性质，在逐字段比对里完全不可见——路径按名字索引，重排后 diff 为空。现在 dump 保留声明顺序，`tests/test_parity_static.py` 直接钉住顺序。
- **Isaac Sim 的 `app.close()` 会把进程退出码改写成 0。** 退出码放在它之后，这个检查永远不会失败。
- **term 容器不能是 dict。** `CommandManager` 会往传进去的容器上写 `debug_vis`，dict 没地方放。

第四个陷阱是 P5 期间补的：**白名单条目必须会过期**。`verify.structure.unused()` 报告「解释了零处差异」的条目，`check_parity.py` 视其为失败。加这个检查时它当场抓出 11 条死条目——其中 `rewards.feet_slide` 是一条项级前缀条目，理由早已过时却仍覆盖着整个奖励项的所有子路径。白名单只有在每条都还在挣它的位置时才可信。

### 12.6 P5 实测：同一份声明编到 mjlab

`engines/mjlab/` 编译同一份 `flat_g1()`，构造真实环境并 step 通过（`scripts/check_mjlab.py`）。观测分组形状与 Isaac 侧逐项相同，动作维度 29，**动作顺序等于目录声明的 DFS 顺序**（D1）。

mjlab 侧的参考实现是 InstinctMJ 的 `G1LocomotionFlatEnvCfg`。它没有安装（D3），所以 `tests/test_mjlab_reference.py` 直接读它的语法树比对。同一份 `TaskSpec` 对上了两份互不知情的参考：

| 对比项 | 结果 |
|---|---|
| policy 观测项与顺序 | 完全一致 |
| 观测噪声区间 | 完全一致 |
| 16 个奖励项名称与权重 | 完全一致 |
| 终止项 | 完全一致 |
| 事件名 / mode / interval | 完全一致 |
| 时序（0.005 / 4 / 20.0） | 与两份参考都一致 |
| 求解器设置 | 不在任务里，在 mjlab profile 里，取值等于参考 |

**三个「不可移植」奖励的处理改了。** P3 判定 `dof_acc_l2`、`dof_torques_l2`、`feet_slide` 不可移植是对的，但由此让它们缺席是错的——「不可移植」说的是**写法**，不是任务该不该有这一项。两个引擎其实都有这三项，只是实现不同。现在它们用 `kind=` 按名声明、由各引擎注册各自的参考实现（Isaac 用 main 自己的，mjlab 用移植自 InstinctMJ 的），`level=REQUIRED` 保证没有哪个引擎能悄悄少掉它们。这正是 `kind` 机制存在的理由，也是「保留各引擎特性」与「任务完整」两件事的交点。

#### 12.6.1 mjlab 侧的四处真实分歧

都不是拼写问题，都得由 backend 承担：

- **域随机化是另一套机制。** Isaac 用带分布参数的事件函数；mjlab 用声明式 `dr` 模块，原语按它改写的模型字段命名。摩擦最尖锐：PhysX 有静/动两个系数加恢复系数、从 bucket 池里抽；MuJoCo 只有一个滑动系数、且没有逐 geom 的恢复系数。mjlab profile 取两个区间的并（`(0.2, 0.8)`），与 InstinctMJ 的折叠一致；恢复系数直接丢弃而非近似，因此 mjlab 的能力矩阵不宣称 `DR_RESTITUTION`——真需要它的任务会在启动时报出能力名而失败。
- **mjlab 缺三个事件函数，其中两个有「邻居」。** `reset_joints_by_offset` 是加偏移、Isaac 的 `reset_joints_by_scale` 是乘缩放；`dr.body_mass` 只改质量、Isaac 的 `randomize_rigid_body_mass` 默认按比例同步缩放惯量。两个替代品都能编译通过并随机化出**另一个东西**，所以在 `engines/mjlab/events.py` 里各自移植了一份。
- **动作选的是 actuator 而非 joint。** 选择器 kind 不同，D1 的落点也在这里：显式传 DFS 关节名并置 `preserve_order`，而不是靠 `.*` 展开——后者跟的是模型文件自己的顺序。
- **摩擦作用在 geom 上而非 body 上。** 「机器人的所有表面」这个引用在 Isaac 下降成 body 选择、在 mjlab 下降成 geom 选择，由 builder 完成——builder 知道自己那个函数要什么，任务不必知道。

另外两处是 mjlab 侧的形状约定：`soft_joint_pos_limits` 是模型常量（首维为 1），Isaac 是逐环境的，按 `env_ids` 索引的移植代码会在第二个环境上越界；以及 mjlab 从**事件函数本身**读它要写哪些模型字段，包装函数必须把这个声明转发出去，否则字段不会被逐环境展开，写入同样在第二个环境上失败。

### 12.7 两引擎逐值比对：26 项一致，5 项按设计不一致

`scripts/compare_terms.py --run` 在两个进程里各构造一次环境（Isaac Sim 必须先启动 app，两个引擎不能同进程），把机器人**写**进同一个状态，逐项求值后比对。写而不是 step：term 是状态的函数，在两边都被**放进**同一状态下的一致性说的是 term；step 之后的一致性还掺进了积分器，是另一件弱得多的事。

结果：**26 项在 float32 精度内一致**（最大 4.8e-07，多数精确为 0），5 项按设计不比。一致的包括全部关节量、投影重力、基座线角速度、速度跟踪与全部姿态类奖励。不比的 5 项各自记了理由：气空时间是跨步累积量、`feet_slide` / `dof_acc_l2` / `dof_torques_l2` 读的正是两引擎测法不同的量、接触终止依赖同一个力读数。

**这个比对本身抓出了一个陷阱，而且是它最该抓的那种。** 第一次跑时所有速度类 term 全部不一致（差到 0.83）。原因不在 term：`write_root_state_to_sim` 在 Isaac 下把速度当**质心**速度、在 mjlab 下当**连杆**速度，同样十三个数写进去，两个机器人根本不在同一个物理状态。改用两个引擎都有的 `write_root_link_pose_to_sim` / `write_root_link_velocity_to_sim` 后，状态回读一致到 1e-8，全部速度类 term 随即一致。已作为 `write_root_state_to_sim` 进 denylist，并有测试钉住那两个限定写入接口的存在——否则这条建议会随上游改名而悄悄失效。

`compat/denylist.py` 里原本就有 `default_root_state` 的读侧条目，说的是同一件事的另一半。写侧这条是它的对偶，而且是实测出来的。

### 12.8 P6：一个入口，以及它逼出来的两件事

`scripts/train.py --engine <name>` 是两引擎唯一的训练入口。它的结构由一个排序约束决定：引擎必须在**任何东西被 import 之前**选定，因为 Isaac Sim 的 `AppLauncher` 要先于 `isaaclab`、也先于 torch 运行。所以先用一个只认 `--engine` 的 parser 选中适配器，由适配器补自己的命令行开关（`--device` 也归适配器，因为 `AppLauncher` 坚持自己声明它、并拒绝已有同名参数的 parser），再 `bootstrap`，之后程序的其余部分才存在。

入口之后没有任何一处知道自己拿到的是哪个引擎：适配器负责编译、负责构造（`CompiledTask.make_env()`，因为 Isaac 从 `cfg.sim.device` 取设备而 mjlab 要求构造参数）、也负责说出自己的 RL wrapper（`wrap_for_rl`）。`tests/test_train_entry.py` 用 AST 读这个文件，禁止其中出现任何引擎名字面量——第三个引擎的成本是否真的被买下来，看的就是这一条。

两个注册表都不 import 被注册的东西：`engines.ADAPTERS` 与 `tasks.registry.TASKS` 存的是点号路径。Gym 注册表做不到这件事——注册 `Instinct-Locomotion-Flat-G1-v0` 就要 import 它指向的 Isaac Lab env cfg，于是「列出有哪些任务」这个动作本身需要 Isaac Sim。

**第一件被逼出来的事：D4 的耦合必须真的断掉，不能只是推迟。** 硬约束 25 原本写的是「`agent_cfg` 惰性解析，耦合待 D4 收尾时移除」。但 mjlab 训练要读 PPO 超参，惰性只是把失败推到第一次访问：`configclass` 住在 `isaaclab.utils`，而这个包的 `__init__` import `mesh` → `pxr`。于是「读一个学习率」需要 Isaac Sim 运行时在路径上。现在 `configclass` 按 `compat/math.py` 的先例整份 vendor 到 `instinctlab/utils/configclass.py`（BSD-3，`tests/test_configclass_vendor.py` 用 AST 逐函数对着上游钉住，另有一条在 Isaac Sim 下比对真实 agent 配置 `to_dict()` 的端到端断言）；agent 配置从 `config/g1/agents/` 移到 `tasks/locomotion/config/flat_g1_ppo.py`，脱离那条注册 Gym id 的 import 链，main 的旧路径改为 re-export，保证是**同一个类对象**而不是一份会漂移的副本。

**第二件：`ContactSensorRef` 的解析必须只做一次。** 首次 4096 env 训练时 Isaac 侧 16.6 秒/迭代、GPU 利用率 1%、CPU 满载——这个形状说明时间花在 Python 里而不是物理里。cProfile 定位到 `omni.physics.tensors` 的 `prim_paths`：21.6 秒里占 18.7 秒，来自 Isaac Lab 的 `ContactSensor.body_names`——它是个**每次访问都从 physics view 重建**的 property，4096 env 下单次约 70 毫秒。三个接触类 term 每次求值都重新解析自己的脚，于是每步付三遍。

这不是疏忽，是架构选择的代价，而且值得写下来：Isaac Lab 自己的 term 不付这笔钱，因为 `SceneEntityCfg` 在 manager 建好时解析一次，之后 term 手里只有下标。而 `ContactSensorRef` 是**由 term 而不是由 manager** 解析的——这正是同一份声明既能对上 Isaac Lab 的一个宽传感器、又能对上 mjlab 的几个窄传感器的原因。manager 白送的缓存，换成这个结构就得自己做。现在 `compat/sensors.py` 按 (sensor, ref) 记忆解析结果，弱引用持有 sensor；成立的前提是传感器的元素表在场景建好后不再变，两个引擎都满足。修完 5,699 → 56,339 step/s，同一场景快十倍。

一般化的教训：**跨引擎间接层的每一次「运行时解析」，都要问它在原生实现里是不是初始化期就做完的。**是的话，这层间接就必须自己把它缓存住，否则语义对了、性能塌了，而且塌得没有任何数值信号——`compare_terms.py` 在修复前后给出完全相同的结果。

### 12.9 第三件：一个字段没写，训练照跑，终止死了

前面所有验收——134 处 diff 命中白名单、mjlab 侧观测顺序与 16 个奖励权重逐项对上 InstinctMJ、26 项 term 逐值一致到 4.8e-07——都通过了，而 mjlab 侧的 `base_contact` 终止**从第一轮起就没有触发过**。

`engines/mjlab/scene.py` 建接触传感器时写了 `fields=("force",)`。理由当时看是充分的：这一层要读的是力历史，`force` 就是那个字段。但 mjlab 的 `fields` 默认值是 `("found", "force")`，而它的空中/接触计时是从 `found` 累加的：

```python
# mjlab/sensor/contact_sensor.py::_update_air_time_tracking
if contact_data.found is None or "found" not in self.cfg.fields:
    return
```

静默 return。传感器构造成功，`track_air_time=True` 被接受，环境正常 step，`current_contact_time` 永远是 0。于是由接触时长导出的一切同时死掉：`illegal_contact` 恒假（回合只能超时结束）、`feet_air_time` 恒付零（16 项奖励里的一项，权重 1.0）。没有异常，没有警告，没有一条数值断言会响——**因为所有断言比的都是配置和单点数值，而这是一个只在时间轴上才存在的量**。

发现它靠的是和 InstinctMJ 已有训练跑对回合长度曲线：

| 迭代 | 0 | 10 | 50 | 100 | 200 | 500 | 900 |
|---|---|---|---|---|---|---|---|
| InstinctMJ（mjlab 参照） | 10.5 | 47.2 | 24.8 | 62.3 | 645.8 | 956.1 | 990.9 |
| main（Isaac 参照） | 11.8 | 40.5 | 17.8 | 65.2 | 758.6 | 908.3 | 964.3 |
| 本项目 isaacsim | 14.6 | 37.2 | 8.4 | 49.2 | 703.0 | 974.3 | 995.5 |
| 本项目 mjlab（修复前） | — | — | **1000.0** | **1000.0** | **1000.0** | **1000.0** | **1000.0** |

Isaac 侧逐点落在两份参照之间，是健康的；mjlab 侧从头到尾满格。**奖励曲线看不出来**——它是 16 项的加权和，少一项仍然平滑上升，看起来一直在学。回合长度直接暴露终止是否活着，这是它不可替代的地方。

第二个信号是日志：那次跑里 `Metrics/*` 和所有 `Episode_Reward/*` 都是**精确的 0.0000**。原因是两引擎的日志寿命不同——Isaac Lab 只在 `_reset_idx` 里清空 `extras["log"]`，两次 reset 之间上一批统计一直可见；mjlab 每步开头就清空，只有恰好发生 reset 的那一步有内容。终止一死，4096 个环境完全同步、每 1000 步才一起超时一次，24 步的 rollout 几乎永远采不到，wrapper 于是补零。**满屏精确的 0.0000 是信号不是噪声**：它当时被我读成「日志没接上」，实际是终止失效的二阶症状。终止修好后回合长度自然离散化，reset 每步都在发生，补零几乎不再触发。

修复是把 `found` 加回去（即 mjlab 的默认值）。防回归写在 `tests/test_mjlab_contact_wiring.py`：一条静态断言说明这个字段是必需的，一条 live 测试把机器人放到平面上 step 20 步、断言脚上真的累积出接触时长。变异检验确认后者能抓住原写法（接触时长恒为 0）。**后者才是真正的防线**——它问的是引擎而不是配置，而这类 bug 的定义就是「配置是对的」。

一般化的教训有两条。**覆盖引擎默认值之前，先查它内部还有谁读这个字段**：默认值往往编码着实现内部的依赖，而这些依赖不会出现在参数文档里（这里其实出现了，`track_air_time` 的 docstring 写了 "Requires `found` in fields"——但那行字在 40 行的参数说明中间，而我们是在照着 InstinctMJ 的 `fields=("force",)` 抄，它能这么写是因为它的传感器子类重写了计时逻辑改用力阈值）。以及 **S5 conformance suite 的必要性在这里被证实**：与任务无关的行为探针（接触冲量、静态保持）本来就该在任何训练启动前跑一遍，成本几秒，而这次的代价是一次 5000 轮的空跑。

### 12.10 覆盖率审计：两个参照到底守住了什么

12.9 那个 bug 是在「所有验收都通过」的状态下跑满 5000 轮的，所以这次不再重跑同一批检查，而是反过来问：这批检查**没比什么**。审计对象是「训练一致」这个完整命题，比配置一致大——它还包含超参、env 类、RL wrapper、训练循环、随机种子。

**mjlab 侧发现四处空白，均已补上，补上后逐项一致。** `tests/reference_mjlab.py` 能从 InstinctMJ 的语法树里取出的东西，比测试实际用到的多：事件只比了 name/mode/period 而没比 `params`，奖励只比了名字和权重而没比 `params`，`reward_functions()` 和 `scene_sensors()` 两个提取器根本没有调用者。也就是说质量随机化范围、推力范围、`feet_air_time` 的 0.5 秒阈值、每个 `joint_deviation_*` 到底罚哪几个关节——这些直接改变训练的量，当时全部无人看守。补了四条测试（事件区间、奖励底层实现、实体选择器、标量参数），变异检验确认能抓住 ±5→±2 的质量范围、0.5→0.3 的阈值、以及把膝关节选择器换成髋。

其中 `feet_slide` 的选择器是唯一一处字面不同：参照写死 `("left_ankle_roll_link", "right_ankle_roll_link")`，声明用模式 `.*_ankle_roll_link`。断言比的是**解析结果**而不是字面量——两者在机器人自己的 body 表里解析出同样的名字、同样的顺序，才算一致。

**Isaac 侧的空白在 env 类而不在配置。** 逐字段比对覆盖 `ManagerBasedRLEnvCfg` 的全部字段（133 处差异，0 处无法解释），但它比的是配置，不是消费配置的那个类：main 注册的 entry_point 是 `instinctlab.envs:InstinctRlEnv`，编译产物用的是朴素 `ManagerBasedRLEnv`。这个子类做三件事——把 `MultiRewardCfg` 路由到多奖励管理器、跑 MonitorManager、包装 step/reset 记日志。对这个任务前者不触发（`G1FlatRewardsCfg` 是普通类）、后者无事可做（`G1FlatMonitorCfg` 是 `pass`），所以两者 step 行为等价。这两个前提现在由 `test_parity_static.py` 从 main 的声明里读出来断言，任一变假都会响——`num_rewards` 从 1 变成向量意味着两边优化的根本不是同一个目标。

**训练循环里找到两处真差异，都已修。**

一是**环境从来没有播种**。两个参照都把 agent 的 seed 交给环境（main 是 `env_cfg.seed = agent_cfg.seed`，InstinctMJ 是 `cfg.env.seed = seed`），两个引擎也都在 `__init__` 里从 `cfg.seed` 播种——而这个字段默认是 `None`，两边都是「不播种」。我们只在入口调了 `torch.manual_seed`：它在环境构造**之前**、且不覆盖 numpy/random。表现是训练完全正常、只是不可复现，随机质量摩擦和推力取决于进程 RNG 当时在哪。这是又一个「没有异常、没有断言会响」的类别。

二是 **torch 后端设置，而这里两个参照本身就不一致**：main 的训练脚本开 TF32 matmul（连同 `cudnn.allow_tf32=True`、`deterministic=False`、`benchmark=False`），InstinctMJ 什么都不设。TF32 改变策略更新中每一次 matmul 的算术。放进共享入口就必然有一个引擎是错的，所以放进各自适配器的 `bootstrap`——「复现一次参照训练」包括复现它跑在什么栈上。

**超参本身两侧都对上了。** Isaac 侧的 agent 配置和 main 逐字节相同（只有 docstring 和两行 import 变了，因为它被移出 Isaac-only 包路径以便脱离 Isaac Sim 解析）；mjlab 侧和 InstinctMJ 那次 5000 轮跑 dump 出的 `agent.yaml` 比，35 个共有字段全部一致，它多出的 37 个字段（MoE / VAE / 蒸馏 / AMP 判别器）在那次跑里全部关闭。顺带发现 `flat_g1_ppo.py` 的 docstring 声称 `tests/test_agent_cfg.py` 钉住了每个字段——**而那个文件不存在**。现在存在了，并且它比对的是参照训练**实际写出的 yaml** 而不是参照的源码。

一般化的教训：**文档里写「由某测试保证」时，要检查那个测试存在**；以及 **对拍脚本能提取的信息比它断言的多，是一种典型的覆盖率假象**——提取器写好了、没人调用，读代码的人会以为比过了。

两侧各跑满 5000 轮后的终值：

| 第 5000 轮 | 回合长度 | 单步回报 | 线速度误差 | 角速度误差 |
|---|---|---|---|---|
| main 参照（Isaac） | 977.14 | 0.012 | 0.311 | 0.695 |
| 本项目 isaacsim | 980.17 | 0.014 | 0.358 | 0.638 |
| InstinctMJ 参照（mjlab） | 977.66 | 0.016 | 0.367 | 0.580 |
| 本项目 mjlab | 970.30 | 0.015 | 0.350 | 0.571 |

最有说服力的一列是角速度误差：两个 mjlab 跑都落在 0.57–0.58，两个 Isaac 跑都落在 0.64–0.70，线速度误差则相反。**引擎留下了可辨认的指纹，而本项目的两次跑各自带着自己所跑引擎的指纹**——这比「数值接近」更能说明对齐的是引擎行为本身，而不是碰巧调出了相似的结果。

### 12.11 第四件：退役的栈没走干净，把 golden 本身弄坏了

删 unified 栈时才发现，**main 的 `G1FlatEnvCfg` 已经构造不起来了**——而它是 D3 指定的唯一 golden，我们所有 Isaac 侧的验收都以它为准。两处断裂互相独立，都由 unified 改造留下：

**一、`mdp/__init__.py` 的 `from .unified import *` 遮蔽了 main 的奖励实现。** 发布版这里导出的是 `.rewards` 和 isaaclab 的 mdp；unified 改造把它换成星号导入自己的重实现。那些重实现在**算术上忠实**、在**签名上不同**：`feet_air_time_positive_biped` 收 `sensor_name` + `body_names`，而 main 的配置传的是 `sensor_cfg=SceneEntityCfg(...)`。Isaac Lab 的 manager 在构造期按签名校验 term 参数，于是 main 的任务直接被拒。没有 import 错误——没有人按名字 import 一个 term——也没有测试会响。

**二、`tasks/__init__.py` 不再在 import 时注册 Gym id，而没有人调用替代品。** 为了让跨引擎入口能在选定引擎之前读任务表，这个包必须保持引擎无关，注册因此改成显式的 `register_legacy_isaac_tasks()`。但 main 的 `scripts/instinct_rl/train.py` 仍写着 `import instinctlab.tasks  # noqa: F401`，靠副作用注册——全仓库没有一处调用那个函数。main 的入口连自己的 task id 都找不到。

最难受的一点：**golden 是从这个坏掉的状态 dump 的**。它把四个奖励的实现记成 `mdp.unified.*`，而 main 真正跑的那次（2026-08-17，5000 轮）dump 出来的是 `mdp.rewards.*`。也就是说「133 处差异、0 处无法解释」这个结论，比对的对象是一份**无法实例化**的配置。结论本身没被推翻——两版实现数值等价，且我们的 Isaac 训练曲线确实落在 main 的参照跑上——但它当时是**碰巧**成立的。

删除后重新 dump，golden 只有那四行 `func` 改变，parity 仍是 133 处差异、0 处无法解释；main 的任务恢复构造并 step，16 个奖励项齐全。

一般化的教训有两条。**退役一层不等于停止使用它**：只要 `__init__.py` 还在 re-export，被退役的实现就仍然是活的，而且是**优先于**正版的活的。星号导入是这件事的特有形状——它急切绑定名字，且静默压过下面的惰性查找。已加静态测试禁止这个包出现星号导入。以及 **参照实现必须有人定期真的跑一遍**：我们对 main 做了逐字段比对、静态不变量、白名单过期检查，唯独没有「构造它一次」。一个连不上电的标尺，量什么都是准的。

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
| S2 | ~~`EntityRef` 只有 `joints` / `bodies`~~ **已实现** | 带**可注册选择器种类**的开放结构：引擎包在自己的 `__init__.py` 里调 `compat.entity.register()` 声明种类、配置类路径与容器类型，共享层不再持有种类表；目标引擎不认识的种类报 `UnsupportedSelector` | mjlab `SceneEntityCfg` 有 10 种选择器（joint / body / geom / site / actuator / tendon / camera / light / material / pair），Isaac 只有 4 种，仅 joint / body 重合。**这是 mjlab → Isaac 方向的硬门槛** |
| S3 | ~~`Capability` 是封闭 enum~~ **已实现** | 带命名空间的字符串 ID（`contact.air_time`、`dr.friction.sliding`…），`capability(id, 说明)` 注册并返回 id，模块常量绑定到该调用；引擎包可注册核心没有的能力。未注册 ID 报 `UnknownCapability` 而非被当作「后端不支持」 | 新引擎会带来现有引擎都没有的能力（可微物理、软体、触觉）。封闭枚举意味着每次都要改核心包。**拼写错误当作「不支持」处理会变成一个被静默跳过的 term**——正是本项目反复吃亏的那种失败 |
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
| A · IR 与中枢 | `spec/` `compat/` `mdp/`；S1 / S2 / S3 | 带语义定义的 `vocab.py`、7 项 denylist、可移植 term 库 |
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
16. `spec/` 禁止 import 任何引擎，含函数体内的延迟 import。由 `tests/test_spec_isolation.py` 静态 + 动态双重把关。
17. `resolve()` 之后禁止直接读 `cfg.<kind>_names`——两引擎装的东西不同（Isaac 是正则，mjlab 是匹配结果）。一律走 `compat.entity.resolved_names()`。
18. 可移植 term 判断接触一律用 `compat.sensors.in_contact()`（由接触时长导出），禁止对接触力取模长设牛顿阈值：两引擎的「接触力」不是同一个物理量。需要力值的 term 必须 per-engine 并声明容差。
19. golden dump 与结构比对禁止按字段名排序。观测组是按属性顺序拼接的，排序后的 golden 对一个观测向量布局已经不同的配置仍然相等——这类错误编不出编译期信号，只会在加载 checkpoint 时炸。
20. 白名单条目的 key 是**路径前缀**，按路径段匹配（`p` 后接 `.` 或 `[`），不要写结尾的 `.`。禁止用 `rewards` / `observations` 这类整族前缀，由 `tests/test_parity_static.py` 把关。
21. 需要非零退出码的 Isaac Sim 脚本必须在 `app.close()` **之前** `os._exit(status)`。Isaac Sim 的关闭流程会把进程退出码改写成 0，放在之后的检查永远不会失败。
22. 禁止调用不带坐标系限定的 `write_root_state_to_sim` / `write_root_velocity_to_sim`——Isaac 收的是质心速度、mjlab 收的是连杆速度，同样的数写进去两个机器人不在同一状态。一律用 `write_root_link_*_to_sim` 或 `write_root_com_*_to_sim` 明说是哪个。
23. 白名单条目必须会过期。`verify.structure.unused()` 报出「解释了零处差异」的条目即视为失败，禁止留着一条理由已过时的宽前缀条目继续覆盖整个 term。
24. 「不可移植」只约束写法，不约束任务是否该有这一项。两引擎都有、但测法不同的 term，用 `kind=` 按名声明 + 各引擎注册各自实现 + `level=REQUIRED`；禁止因为写不出一份共享实现就让任务少掉一项。
25. 构造环境禁止依赖 RL 库可导入，`CompiledTask.agent_cfg` 惰性解析。agent 配置本身也禁止依赖任何引擎：超参是引擎无关的，声明它的装饰器也必须是（用 `instinctlab.utils.configclass`，不要用 `isaaclab.utils.configclass`），且不得放在会 import 引擎的包路径下。
26. 跨引擎间接层里的每一次运行时解析，都要先问它在原生实现里是不是初始化期做完的。是的话这层必须自己缓存——`ContactSensorRef` 的元素解析没缓存时，Isaac 侧每步付三次 `ContactSensor.body_names`（4096 env 下单次 70 毫秒），整个环境慢十倍，而所有数值断言照常通过。语义正确不蕴含性能可用，且这类退化没有数值信号。
27. `train.py` 里禁止出现任何引擎名字面量，禁止按引擎分支。构造方式、命令行开关、RL wrapper 一律由适配器回答（`make_env` / `add_cli_args` / `wrap_for_rl`），由 `tests/test_train_entry.py` 用 AST 把关。
28. 用 `os._exit` 结束的脚本必须先 `sys.stdout.flush()`。`os._exit` 不刷 stdio 缓冲，输出重定向到文件时会**静默丢掉**最后一段——`check_parity.py` 曾因此在报告成功的同时丢掉了整个构造与 step 的结果。
