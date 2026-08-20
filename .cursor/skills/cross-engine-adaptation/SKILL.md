---
name: cross-engine-adaptation
description: 在 InstinctLab 做适配工作的操作顺序与验收方法——接入新仿真引擎、把仓库存量任务（parkour / shadowing / beyondmimic / perceptive / HOI）迁入跨引擎栈、导入外部项目、或改动 spec/compat/mdp/engines 共享层。含静默退化故障目录（无异常、无失败测试、训练照常收敛的那一类）、各层检查的盲区、存量任务适配地图与死代码清单。当提到接入/新增引擎、适配/迁移任务、motion tracking 跨引擎、跨引擎对拍、引擎升级、term 移植、parity/golden 时使用。
---

# 跨引擎适配

约束清单在 [`.cursor/rules/multi-engine-training.mdc`](../../rules/multi-engine-training.mdc)（按 glob 自动附加），权威设计在 [`CROSS_ENGINE_DESIGN.md`](../../../CROSS_ENGINE_DESIGN.md)。**本 skill 不重复它们**，只给操作顺序、每步的验收，以及怎么发现那些不会报错的错。

## 先记住这个项目的 bug 长什么样

双引擎打通期间所有严重缺陷都是同一形状：**没有异常、没有失败的测试、训练照常收敛、奖励曲线照常上升**。没有一个是靠「跑起来报错」发现的。

代价最大的那个跑满 5000 轮才被看见：给 mjlab 接触传感器声明 `fields=("force",)`，漏了 `found`，于是 `illegal_contact` 永不触发、`feet_air_time` 恒为零，回合只能超时结束。传感器构造成功，环境正常 step，策略照常收敛。

由此得到本 skill 的第一条操作原则：**适配工作的默认假设是「它已经悄悄错了」，验收的任务是逼它显形，而不是确认它没报错。** 完整目录与每个 bug 的实际发现路径见 [silent-failures.md](silent-failures.md)。

## 工作流 A：接入一个新引擎

不要先写代码。先回答 [new-engine.md](new-engine.md) 的审问清单——那里每一个问题都对应 isaacsim/mjlab 之间一处真实咬过人的分歧（速度参考系、四元数序、接触量纲、选择器种类、字段默认值……）。答案写进 `compat/vocab.py` 的 spoke 映射和 denylist，**再**动手。

顺序：

1. **审问引擎**，产出该引擎的语义档案。未确认的项不许猜，标记为待验证。
2. `engines/<name>/` 建包：`terms.py`（注册表即能力矩阵）/ `scene.py`（含该引擎参考实现的求解器 profile 默认值）/ `assets.py` / `adapter.py`。包**顶层不得 import 引擎**，builder 函数体内才 import。
3. 能力声明从 `terms.py` 的 `provides=` 推导，禁止手写。缺失术语走 OPTIONAL 跳过，不必先补齐全部。
4. `contract_report()` golden 入库——它必须能在没装该引擎的机器上回答。
5. 逐值对拍：`scripts/probe_terms.py --engine <name> --out /tmp/<name>.json`，与已通过的引擎 diff。**写状态而非 step**，理由见脚本 docstring。
6. 行为探针（S5 conformance suite）：自由落体 / 静态保持 / PD 阶跃 / 接触冲量 / 摩擦滑移。单点数值对拍看不见任何时间轴上的量（接触时长、空中时间、curriculum），只有探针能。
7. 短训练，对**回合长度曲线**，不只对奖励曲线。

第 1 步偷的懒会在第 7 步以「训练能跑但学出来的东西不对」的形式回来，而那时已经没有便宜的定位手段了。

## 工作流 B：适配本仓库的存量任务

跨引擎栈里目前**只有 `Instinct-Velocity-Flat-G1` 一个任务**。parkour 与 4 个 shadowing 变体仍是 Isaac-only。

开工前先读 [repo-tasks.md](repo-tasks.md)，它给出还剩什么、每个任务的具体障碍、以及**哪些东西看起来要适配其实是死代码**（已实测：`actuators/` 零引用、`rl/` 是空目录残留、`tasks/shadowing/mdp/` 整包未接线）。

两个决定成败的判断：

- **成本按子系统摊，不按任务摊。** 逐个任务看会得出「每个都很难」的错误结论。这些任务共享同一批 Isaac 耦合子系统（`motion_reference/` / `sensors/` / `terrains/` / `monitors/`+`managers/` / `envs/mdp/`），瓶颈全在子系统里。抬一次 `motion_reference`，4 个 shadowing 变体同时受益。**推荐顺序按子系统排**，入口选 beyondmimic（平面、单 motion buffer、无视觉）。这与迁移工作流里「第 4 步主要是 per-robot 而非 per-task」是同一个成本结构。
- **flat G1 的结论不能沿用。** 「编译产物用朴素 `ManagerBasedRLEnv` 等价于 main 的 `InstinctRlEnv`」成立的前提是奖励容器不是 `MultiRewardCfg` 且 monitor 为空——**这两个前提对存量任务全部不成立**。必须重新做一次规则 33 的断言。

## 工作流 C：迁移一个外部项目

五步，顺序不能变（详见规则文件「迁移一个 Isaac Lab 项目」）：

1. 先在**它原本的引擎**上按原样跑通，diff 为空，固定基线。这就是该项目的 golden——泛化形式是「该项目跑在它原本的引擎上」。**建它的同时写下它什么时候该被拆掉**：golden 是迁移期脚手架，迁完之后它就退化成一份要跟着实现同步维护的副本，而副本失效的方式全是静默的。判据见 [silent-failures.md](silent-failures.md) 第 15 条。本仓库 locomotion 的那份（D3）已按此退役。
2. `instinct-migrate analyze` 出逐项分类报告。
3. codemod 改写机械部分（import 换 `instinctlab.mdp`、legacy 别名换显式帧名、math 换纯 torch、`sensor_cfg` 换 `SensorRef`）。
4. 补 per-engine 条目（资产 / sim profile / 动作映射 / DR）。这步主要是 **per-robot** 而非 per-task，一个机器人补完一次后续任务近乎零成本。
5. L0 → L1 → 目标引擎 smoke 与短训练。

**第 1 步是最常被跳过、代价最高的一步。** 本仓库真实发生过：golden 是从一份根本无法实例化的配置里 dump 的，逐字段比对、静态不变量、白名单过期检查全部照常通过——一把连不上电的尺子，量什么都是准的。所以第 1 步的验收是「真的构造出来并 step 一次」，不是「配置能 import」。

**同一把尺子还被弯折过一次，更难发现。** 编译器与 main 在 `self_collision` 上不一致，而这处差异是靠**把编译器的 spawn 覆盖复制进 `G1FlatEnvCfg`**（golden 的源文件）抹平的。此后对拍永远相等，代价是「≡ main」不再有内容。规则：**对拍报差异时先判断哪边错，修参照必须因为参照本来就写错，不能因为改了参照能变绿**；并且凡声称「这个文件是 main 的」，就要有检查去问 main（`tests/test_main_reference.py`，含删除侧——删掉参照的文件比改它更该有交代）。

**选参照时优先选本仓库改不动的那种**：InstinctMJ 的配置不是依赖也不在本仓库，main 的文件可以用 `git show main:<path>` 现取现执行。改不动的参照弯折不了，而且不必留在工作区。

frontend 遇到 IR 表达不了的构造必须报错并计入未转换清单。**遇到单体 env 项目（HumanoidVerse / PBHC / IsaacGymEnvs 风格）明确报错**，要求先手工重构成 term 结构；禁止自动拆解单体 `compute_reward()`，那只会产出语义已漂移但看起来能跑的结果。

## 工作流 D：改动共享层（`spec/` / `compat/` / `mdp/`）

改一处共享实现 = 同时改所有引擎的行为，所以最小验收固定为三项：

```bash
python -m pytest tests/ -q                        # 静态与隔离：spec 不 import 引擎、参照文件仍等于 main
python scripts/check_mjlab.py                     # mjlab ≡ InstinctMJ（读语法树，不需装引擎）
python scripts/probe_terms.py                     # 两引擎逐 term 比值（每个引擎各跑一次，见 verification.md）
```

加了跨引擎间接层时额外问一句：**这次运行时解析，在原生实现里是不是初始化期就做完了？** 是的话这层必须自己缓存。没缓存的代价是环境慢十倍而所有数值断言照常通过——`ContactSensorRef` 元素解析就这样让 Isaac 侧从 56,339 掉到 5,699 step/s，GPU 全程空转。

各层检查分别能看见什么、对什么天然失明，以及怎么审计自己的检查，见 [verification.md](verification.md)。

## 每完成一步就提交

**一个增量做完、`python -m pytest tests/ -q` 绿了，立刻提交，不要攒。** 这里的「一步」指工作流里一个可独立验收的增量：一类 term 移植完、一个传感器两边接完、一次共享层改动。

攒着的代价在 parkour 那轮全部兑现了：

- **回退变成手术。** 一次 8 个增量、46 个改动文件堆在同一片工作区里，其中 `engines/mjlab/rough.py` 同时承载「对齐 Isaac 的地形常数」和「兑现 `num_cols=20`」两件事。前者要撤、后者要留，于是 `git checkout` 用不了，只能逐个常数手工改回去，还得先去 InstinctMJ 核对原值。每一步单独提交的话，这是一条 `git revert`。
- **脏工作区会藏东西。** 中途有人往 `.gitignore` 加了 `tests/*`，19 个新测试文件从未跟踪列表里静默消失。已跟踪的旧测试因为有改动仍然显示，`git status` 看起来完全正常。差点整轮验证装置都没进提交。按步提交的话，这在加它的那一步就会暴露——那一步该新增的测试文件凭空少了。
- **并行 agent 分不清谁在改。** 多个 agent 共享同一片工作区时，「这个文件是脏的」既可能是已落地的成果，也可能是别人正改到一半。提交是唯一能区分二者的标记；没有它，每个 agent 都得在报告里手写一句「这些脏文件不是我改的」。

几条具体做法：

- **提交前看一眼未跟踪列表里有没有这一步该有的新测试文件。** 少了就是被忽略规则吃掉了。
- **功能与测量脚手架分开提交。** 埋点常常改在训练主路径上，性质和功能不同，值得单独审、也方便单独撤。
- **提交信息记下这一步测出来的数**（维数、圆柱数、误差量级）。日后判断是否退化时，这些数就是基线；写在对话里的数活不过这次会话。

## 对拍断言要钉住参照，不是钉住自己

**一条对拍断言如果只描述我们这一侧的形状，它在参照读错的时候照样绿。** 这类断言最舒服写，第一天也确实是绿的，所以很难被怀疑。一天之内撞了三次：

- **TF32。** `test_train_entry.py` 断言「mjlab adapter 不设任何 torch flag」，注释写「InstinctMJ leaves torch at its defaults」。后半句是假的——它的 `train.py` 调 `configure_torch_backends()`，默认 `allow_tf32=True`。断言只查我们设没设，所以假前提被它稳稳护住，mjlab 一直跑在和两个参照都不同的数值栈上。
- **`gpu_collision_stack_size`。** 漂移行的「参照是什么」来自一个用子串在 `ast.unparse` 输出里找 `2**29` 的探针，而 unparse 写的是 `2 ** 29`，于是解析成 `None`。行内自检只要求两侧不相等，`"None" != "Isaac Lab default"` 当然成立。
- **AMP 镜像。** 断言 `not hasattr(clip, "symmetric_augmentation_joint_mapping")`——那是参照整数表的字段名，我们按关节名解析，从来就不会有这个属性。镜像实现完之后它依然绿，漂移表记的事却已经反了。

三条共用一个结构：**把参照的状态当成已知常量写死或猜出来，把断言挂在我们这一侧**。参照读错、参照改了、我们把差异修好了——三种情况它都不动。

几条具体做法：

- **正面断言我们的声明说了什么，别断言我们缺对方的某个拼写。** 「`task.scene.motion_references[0].symmetric_augmentation` 里左髋 pitch 映射到右髋 pitch」会随实现变化，`not hasattr(..., <对方的字段名>)` 不会。跨引擎适配里我们本来就不照抄对方的拼写，所以任何基于「缺少对方拼写」的断言从一开始就和事实脱钩，只是碰巧同向。
- **参照读取器不许用同一个值同时表示「参照没有」和「我没读懂」。** 读不懂就抛。`tests/test_reference_readers.py` 遍历所有读取器，返回 `None`、空、或全 `None` 而不在白名单里就红；白名单每条都要写清参照那边为什么真的没有。
- **「这仍是一处漂移」的行必须两侧都验。** 只验「两侧不相等」等于没验——两个都写错也能不相等。
- **变异要往「已对齐」的方向改，不只往「改坏」的方向。** 常规变异检验是把实现改坏看会不会红；漂移行要反过来：把我们这侧改成和参照一致，那一行**必须**变红。不变红说明它根本没在测这件事。上面三条里有两条能被这一步当场抓出来。

### 「出现过」不等于「生效」

前三例只让测试变绿。第四例改变了我们实际跑的物理，值得单独记。

main 的 parkour 里写着 `self.scene.robot.actuators = beyondmimic_g1_29dof_delayed_actuators`。审计据此给我们的 Isaac 任务配了 0–2 步驱动器延迟。但注册的类是 `G1ParkourEnvCfg`，它在 `super().__post_init__()` 之后调 `apply_shoe_config()`，那句是 `self.scene.robot = G1_with_shoe_CFG.replace(...)`——**整体替换**，而 `G1_with_shoe_CFG` 是模块导入时就深拷贝好的、带目录默认 ImplicitPD 的副本。那行赋值对注册任务是死代码。读取器当时问的是「这个名字在文件里出现过吗」，出现过，于是答 True。

代价是实测的：我们的 `dof_acc_l2` 一直是 main 的 1.71 倍，去掉延迟后回到 0.98。训练照常收敛，没有任何报错。

对照组说明这不是「上游都这样」——InstinctMJ 的带鞋分支写的是 `copy.deepcopy(cfg.scene.entities["robot"])`，深拷贝的是**已经改过**的对象，再只替换 `spec_fn`，所以它的延迟活着。同一个意图，两个仓库，一个生效一个丢失。

几条：

- **读配置的顺序，不是读配置里出现过什么。** 凡是「先设字段、后整体替换对象」的写法，都要按执行顺序解析：展开 `super().__post_init__()` 和 mixin 调用，记录每次对目标的整体赋值与字段赋值，后者只有发生在最后一次整体赋值之后才算数。展不开的调用要抛，不能跳过——静默跳过的那个调用正是覆盖发生的地方。
- **静态解析要和运行产物互证。** Isaac Lab 会把最终 env cfg dump 到 `logs/**/params/env.yaml`。跑一次参照、读那份 dump，是判定「实际生效了什么」最省事也最不容易自欺的办法；静态读取器写完之后拿它对一次。
- **「四处 override」这类计数写进测试名字，会把错误的那一项一起焊死。** 名字叫 `..._matches_main_on_the_four_task_overrides` 的测试，在其中一项根本不是 override 的时候依然全绿。

## 三条最容易犯的元错误

1. **能提取但没人断言的信息等同于没比。** `tests/reference_mjlab.py` 曾提供事件 `params`、`reward_functions()`、`scene_sensors()` 而无任何调用者——读代码的人会以为比过了。新增提取器必须同时新增断言。
2. **退役一层不等于停止使用它。** 只要 `__init__.py` 还有 `from .retired import *`，被退役的实现仍是活的，而且星号导入急切绑定名字、**优先于**下面的惰性查找，静默压过正版。
3. **文档声称「由 `tests/xxx.py` 保证」时先确认该文件存在。** 已经出现过声称被钉住、而那个测试文件从未被创建的情况。

## 参考文件

- [silent-failures.md](silent-failures.md) —— 静默退化故障目录：症状、为何无信号、实际发现路径、可迁移规则
- [new-engine.md](new-engine.md) —— 新引擎审问清单 + isaacsim/mjlab 已知答案对照表
- [repo-tasks.md](repo-tasks.md) —— 存量任务适配地图：五个共享子系统、推荐顺序、死代码清单、parkour 星号导入链实测
- [verification.md](verification.md) —— 分层验收、各层盲区、覆盖率审计与变异检验
