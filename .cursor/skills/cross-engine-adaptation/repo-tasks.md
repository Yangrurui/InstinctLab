# 存量任务适配地图

本仓库自己的任务大部分**还没有**跨引擎化。这份文件回答：还剩什么、真正的瓶颈在哪、按什么顺序做、以及**哪些东西看起来要适配其实不要**。

## 现状

跨引擎栈（`tasks/registry.py`）里**只有一个任务**：

```22:24:/root/InstinctLab/source/instinctlab/instinctlab/tasks/registry.py
TASKS: dict[str, str] = {
    "Instinct-Velocity-Flat-G1": "instinctlab.tasks.locomotion.config.g1:flat_g1",
}
```

其余全部仍是 Isaac-only 的 `ManagerBasedRLEnv` + `InstinctRlEnv` 任务，经 `register_legacy_isaac_tasks()` 注册 Gym id：

| 任务 | 位置 | 相对 flat G1 的增量 |
|---|---|---|
| **parkour** | `tasks/parkour/` | 10 种程序化子地形 + 深度相机 + 高度扫描 + VolumePoints + 虚拟障碍 + AMP + 自定义 terrain-aware 命令 |
| **shadowing/beyondmimic** | `tasks/shadowing/beyondmimic/` | motion_reference（单 buffer，平面） |
| **shadowing/whole_body** | `tasks/shadowing/whole_body/` | 多帧参考（`num_frames=10`）+ MoE policy |
| **shadowing/perceptive** | `tasks/shadowing/perceptive/` | 上述 + 动作匹配地形 + 深度相机 + 额外 DR。**含一个独立变体** `Instinct-Perceptive-Vae-G1-v0`（`perceptive_vae_cfg` + `instinct_rl_vae_cfg`），适配时别只看主 id |
| **shadowing/perceptive_hoi** | `tasks/shadowing/perceptive_hoi/` | 上述 + 6 个 OMOMO kinematic mesh 物体，按参考驱动 |

## 真正的瓶颈不是任务，是五个共享子系统

逐个任务看会得出「每个任务都很难」的结论，那是错的成本模型。**这些任务共享同一批 Isaac 耦合的子系统**，瓶颈全部在子系统里。这与迁移工作流里「第 4 步主要是 per-robot 而非 per-task」是同一个道理：**成本按子系统摊，不按任务摊**。

| 子系统 | 耦合 | 谁依赖它 | 处置 |
|---|---|---|---|
| `motion_reference/` | **重**：`SensorBase` + `omni.physics.tensors` + warp 采样 kernel | 全部 4 个 shadowing + parkour(AMP) | 数据模型（`MotionReferenceData`）可引擎无关；manager/buffer 需 per-engine |
| `sensors/` | **重**：`isaacsim.core` + PhysX view + warp mesh raycast | parkour、perceptive ×2 | GroupedRayCasterCamera / VolumePoints 在 mjlab 无直接等价，需新建 |
| `terrains/` | **重**：Isaac terrain pipeline + `pxr` + warp | parkour、perceptive | **生成算法（Perlin HF / trimesh）本身引擎无关可复用**；importer 与虚拟障碍需 per-engine |
| `monitors/` + `managers/` | 中：经 `InstinctRlEnv` 挂载 | 全部 legacy 任务 | 见下 |
| `envs/mdp/` | **重**：整套 Isaac 耦合 MDP 库（commands / rewards / events / curriculums） | 全部 legacy 任务 | 按 term 逐个迁入可移植 `mdp/` 或 per-engine terms |

**推荐顺序按子系统排，不按任务排**：`motion_reference` → beyondmimic 打通 → `envs/mdp` 的 imitation term 族 → whole_body 几乎白送 → 再决定要不要碰 `sensors`/`terrains`（perceptive 与 parkour 共用，是最贵的一块）。

**beyondmimic 是最便宜的入口**：平面地形、单 motion buffer、无视觉，只需要 `motion_reference` + imitation term 族两块。拿它做 golden 打通 motion tracking 这条链，另外三个变体的增量就小得多。

## `InstinctRlEnv` 的等价性对 flat G1 成立，对这些任务不成立

flat G1 的跨引擎产物用朴素 `ManagerBasedRLEnv`，成立的前提是它的奖励容器不是 `MultiRewardCfg`、monitor 配置为空。**这个前提对存量任务全部不成立**：

- 全部 shadowing 与 parkour 的奖励都包在 `MultiRewardCfg` 里，`InstinctRlEnv.load_managers()` 会换成 `MultiRewardManager`，`compute()` 返回按 group 的 dict 而非单向量。
- 全部 shadowing 与 parkour 都配了非空 `monitors`，`InstinctRlEnv` 强制构造 `MonitorManager`。

目前这些任务的 `MultiRewardCfg` 都**只含一个 group**，所以 `num_rewards == 1`、行为等同单奖励——但 `whole_body` 有一份设了 `advantage_mixing_weights=[0.7, 0.3]` 的多 critic runner 配置（未注册 Gym id），说明这个机制是要保留的、不能在适配时抹掉。

**适配这些任务时必须重新做一次「消费配置的那个类是否等价」的断言**（规则 33），不能沿用 flat G1 的结论。`num_rewards` 从 1 变成向量意味着两边优化的不是同一个目标。

## 不要适配的东西

适配前先删掉或跳过这些，否则会为不存在的需求付工作量：

| 对象 | 状态 | 依据 |
|---|---|---|
| `actuators/` | **死代码**，全仓库零引用；包内只有注释与空 import，无类定义 | 实测 grep 零匹配 |
| `rl/` | **空目录残留**，源码已随 unified 栈删除，仅剩 `__pycache__` | 实测目录内无 `.py` |
| `tasks/shadowing/mdp/` | **未接线**：8 个文件都被 import 为 `shadowing_mdp`，但零处 `shadowing_mdp.*` 引用；活跃 term 全在 `envs/mdp/` | 实测 grep 零匹配 |
| `tasks/parkour/mdp/` 的 `sub_terrain_out_of_bounds`、`push_by_setting_velocity_without_stand` | 定义了但配置未使用 | 调查报告 |
| `rsl_rl_cfg_entry_point` | **死链，三处**：`whole_body` / `perceptive` / `perceptive_hoi` 的 `config/g1/__init__.py:13,24` 都指向不存在的 `rsl_rl_ppo_cfg`，而 `agents/` 里只有 `instinct_rl_ppo_cfg.py`。因为仓库已统一到 `instinct_rl`、没人解析这个 entry point，所以它从不报错。（`beyondmimic` 的那条指向真实存在的 `beyondmimic_ppo_cfg`，不是死链） | 起 Isaac 后逐个 `import_module` 全部 legacy Gym kwargs，三处 `ModuleNotFoundError` |

「零引用」必须用 `Grep` 工具或 `rg` 实测确认——shell 里 `rg` 不可用，`cmd || echo 零引用` 会把命令不存在也报成零引用，产生假阳性。

## parkour 的星号导入链：已知危害的活体实例

```1:9:/root/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/mdp/__init__.py
from isaaclab.envs.mdp import *  # noqa: F401, F403

from instinctlab.envs.mdp import *  # noqa: F401, F403

from .commands import *  # noqa: F401, F403
from .curriculums import *  # noqa: F401, F403
from .events import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
```

这与遮蔽了 main 奖励实现、导致 golden 从坏配置里 dump 出来的那个模式**完全同形**（[silent-failures.md](silent-failures.md) 第 5 条），而且仍然活着。星号导入急切绑定名字，**后导入的静默胜出**。当初禁止 locomotion 那个包出现星号导入的静态守卫已随 D3 删除，**所以现在仓库里没有任何检查在防这件事**——适配 parkour 时要自己把守卫带回来。

跨三层做 AST 统计（141 + 174 + 14 个顶层名），实测碰撞 **1 处**：

- **`joint_torques_l2`**：`isaaclab.envs.mdp` 与 `instinctlab.envs.mdp` 各有一份，**InstinctLab 那份胜出**。它签名更宽（多 `normalize_by_stiffness` / `normalize_by_num_joints` 两个 kwarg），docstring 自称「默认行为相同」。

今天是良性的（默认行为确实一致），但：**做这次调查的 explore 子 agent 把 parkour 的 `dof_torques_l2` 标成了「来自 Isaac Lab / 星号导入」——它标错了。** 一个专门读这份代码的读者都无法从配置读出实际绑定的是哪一份，这正是危害本身。上游改一次实现、或有人调换两行 import 顺序，行为就会静默改变。

**适配 parkour 时第一件事是把这个 `__init__.py` 改成惰性 `__getattr__` 查找**，并重跑一次碰撞统计。（locomotion 那份修好的样板已随 D3 退役删除，可从 `git show main~..:` 一侧的历史里取；形状是：`__all__` 显式列名 + `__getattr__` 按固定优先级顺序在几个模块里依次查找，任何一个名字在多于一个模块里出现就报错而不是静默取第一个。）碰撞统计脚本很短，值得在动任何 term 之前跑：对每一层的 `.py` 做 AST，收集模块级 `FunctionDef`/`ClassDef` 名，报出现在多于一层的名字。

## 适配一个存量任务的顺序

沿用迁移工作流，但第 1 步已经满足（它本来就跑在 Isaac 上），关键是**不要跳过固定 golden**：

1. **在 Isaac 上按原样真的构造并 step 一次，记录 golden**，含回合长度曲线基线。存量任务最容易在这步翻车——它们已经「在跑」，于是没人验证配置当前是否还建得起来。main 的任务就曾在无人察觉的情况下完全构造不起来。**建 golden 的同时写下它什么时候该拆**（[silent-failures.md](silent-failures.md) 第 15 条）：locomotion 那份在迁移完成后已连同白名单与两个脚本一并退役，parkour 这份也该有同样的退场时间。
2. 先把该任务依赖的**子系统**摸清（对照上面的表），确认这次要抬哪一个、哪些先跳过。
3. 清理星号导入链，重跑名字碰撞统计。
4. `instinct-migrate analyze` 出分类报告；表达不了的构造必须报错并计入未转换清单。
5. 抬子系统 → 抬 term → per-engine 补齐 → L0→L1→短训练。
6. 验收对**回合长度曲线**，不只对奖励曲线；motion tracking 类任务额外对**参考跟踪误差**与 `dataset_exhausted` 的触发频率——它们和接触计时一样落在时间轴上，单点逐值对拍看不见。
