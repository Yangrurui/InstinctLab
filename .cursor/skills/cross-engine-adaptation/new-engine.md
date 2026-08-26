# 新引擎审问清单

写 `engines/<name>/` 之前先回答完。每一条都对应 isaacsim/mjlab 之间一处真实咬过人的分歧——两个引擎当前大量同名同义**是巧合，第三个引擎不会遵守**。

答案落地到 `compat/vocab.py` 的 spoke 映射；同名不同义的落地到 `compat/denylist.py`（误用必须报错，禁止默认放行）。**未确认的项不许猜，标记为待验证**。

## 一、状态与坐标系

| 问题 | isaacsim | mjlab | 踩过的坑 |
|---|---|---|---|
| 四元数序 | wxyz（**显式文档化**，`articulation_data.py:853,885,917,949`） | wxyz（**未文档化**，继承自 MuJoCo `qpos`/`xquat`） | 中枢固定 wxyz（D8）。mjlab 那份属隐式依赖，**列入引擎升级复查项** |
| 根部速度默认帧 | COM | link | 写入接口不带帧限定时两个机器人不在同一状态，实测差 0.85 m/s |
| 重力向量 | `GRAVITY_VEC_W`（大写），从 live sim 重力归一化 | `gravity_vec_w`，硬编码 `[0,0,-1]` | 可移植 term 一律改用 `projected_gravity_b` |
| 非根 body 线速度 | per-body COM 偏移 | root `subtree_com` | 中枢**不提供任何 per-body 速度** |
| 默认根状态速度行 | COM 系 | link 系 | denylist |

**通用问法**：任何速度量先问「哪个点的速度」，任何姿态量先问「相对哪个系」。同名不代表同义——`root_lin_vel_b` 在 Isaac 读起来像 link 量，实际是 `root_com_lin_vel_b`，而 mjlab 没有这些别名，按直觉改写就换了物理量且下游不报错。兼容名和禁用语义分别以 `compat/vocab.py`、`compat/denylist.py` 为准。

## 二、关节与自由度空间

| 问题 | isaacsim | mjlab |
|---|---|---|
| 关节加速度 | 跨步有限差分 | 解析 `qacc` |
| 施加力矩 | `applied_torque`，关节空间 (nv) | **无同名属性**；等价物 `qfrc_actuator` (nv)，**不是** `actuator_force`(nu) |
| 关节顺序 | BFS | DFS |
| 关节限位形状 | 逐环境 | `soft_joint_pos_limits` 首维为 1（模型常量） |

D1：`RobotSpec.joint_names` / `body_names` 是唯一真值，DFS 为主。canonical→native 翻译只在 `CompileCtx.entity()` 一处发生。**选择器必须逐个列出关节名**，只写 `.*` 配 `preserve_order=True` 是空操作（见 [silent-failures.md](silent-failures.md) 第 10 条），且动作项与 `joint_pos`/`joint_vel` 观测项都要带，只钉一侧等于没钉。

**nv/nu 混淆是最隐蔽的一类**：维度不同才会报错，维度恰好相同时静默算错。按 `env_ids` 索引一个首维为 1 的张量，在第二个环境上才越界。

## 三、接触

分歧最多的一族，逐条问：

- **`force` 是什么物理量？** Isaac `net_forces_w` 是世界系**仅法向**；mjlab `force` 是完整三维力且默认在**接触系**。不是同一个量，**禁止对它取模长设牛顿阈值**。判断接触一律走 `compat.sensors.in_contact()`（由接触时长导出）。
- **力历史的轴序？** Isaac `(env,time,elem,3)`，mjlab `(env,elem,time,3)`。两脚两子步时**形状相同、只有值能分辨**——必须走 `contact_force_history()`。
- **空中/接触计时从哪个字段累加？** mjlab 从 `found` 累加，缺了它计时函数静默 return。见 [silent-failures.md](silent-failures.md) 第 1 条。
- **传感器粒度？** Isaac 一个宽传感器，mjlab 几个窄传感器。`ContactSensorRef` 要能同时对上两种。
- **元素怎么选？** mjlab 接触传感器**没有 `body_names`**，不能用 `SceneEntityCfg` 从外部切片，走 `compat.sensors.element_ids()`。
- **绝对力值可比吗？** 不可。禁止把 PhysX 与 MuJoCo 接触力当逐值等价，需要绝对力值的任务必须声明容差。

## 四、选择器种类（S2）

mjlab `SceneEntityCfg` 有 10 种（joint / body / geom / site / actuator / tendon / camera / light / material / pair），Isaac 只有 4 种，**12 种里仅 joint / body 重合**。Isaac 的 `fixed_tendon` 与 mjlab 的 `tendon` **不合并**。

`EntityRef` 表达不了的必须报 `UnsupportedSelector`，禁止丢弃。这是 **mjlab → Isaac 方向的硬门槛**，也是新引擎最可能超出现有 IR 的地方。

`resolve()` 后禁止直接读 `cfg.<kind>_names`——Isaac 装的是正则、mjlab 装的是匹配结果，两边都有真实消费者。一律走 `compat.entity.resolved_names()`。

## 五、构造与配置

| 问题 | isaacsim | mjlab |
|---|---|---|
| 设备怎么传 | 从 `cfg.sim.device` 读 | 构造函数位置参数 |
| 物理步长配置路径 | `cfg.sim.dt` | `cfg.sim.mujoco.timestep` |
| 命令查找不存在的名字 | 抛 `KeyError` | `NullCommandManager` 返回 `None` |
| `extras["log"]` 清空时机 | 仅 `_reset_idx` | 每步 |
| env 类名 | `ManagerBasedRLEnv` | `ManagerBasedRlEnv`（差一个大写 L） |

构造方式差异由适配器的 `make_env` 回答，**`train.py` 里禁止出现引擎名字面量**（`tests/test_train_entry.py` 用 AST 把关）。term 侧读 `env.physics_dt`，不碰配置路径。类型标注用 `compat.env.RlEnv`。

其余 env 公共面两引擎已自发收敛（`num_envs` / `device` / `step_dt` / `max_episode_length(_s)` / `episode_length_buf` / `scene` / `cfg` / 七个 manager 同名同义），term **直接读 `env.*`**，不要加访问器——那是已撤销的 `EntityView` 错误。收敛由 `tests/test_compat_env.py` 钉住；新引擎不满足时，由**该引擎 backend** 让自己的对象满足中枢（继承 + `__getattr__` 兜底），代价不外摊。

## 六、Isaac 侧特有的容器约定

交给 Isaac manager 的 term 容器**不能是 dict**——管理器会往容器上写回属性（`CommandManager` 写 `debug_vis`），dict 没地方放，用 `SimpleNamespace`。但观测**组**必须是真的 `ObservationGroupCfg`（有 isinstance 检查），组内 term 靠 `__dict__` 顺序遍历。

事件 term 的 `interval_range_s` 必须由 builder 透传，漏掉一切照常构造、只是周期性推力永不触发。

## 七、数值栈与生命周期

- **bootstrap 不得污染其他引擎的进程。** 引擎必须在任何 import 之前选定（`AppLauncher` 要先于 `isaaclab` 和 torch）。
- **torch 后端开关随各自参照**，不在共享入口二选一（见 silent-failures 第 4 条）。
- **`cfg.seed` 默认 `None` 含义是不播种**，入口必须显式赋值。
- Isaac Sim 的关闭流程会把进程**退出码改写成 0**：需要非零退出码必须在 `app.close()` **之前** `os._exit(status)`。
- 用 `os._exit` 的脚本必须先 `sys.stdout.flush()`——`os._exit` 不刷 stdio 缓冲，重定向到文件时静默丢掉最后一段。

## 八、属性探测

`hasattr(EntityData, name)` 会漏掉用 dataclass 注解声明、无类级默认值的字段（mjlab 的 `gravity_vec_w`、`default_root_state`、`soft_joint_pos_limits`）。**必须同时查 `__annotations__`**。

## 九、事件与 DR

per-engine 族，范式本就不同（高层 event class vs 约 40 个 `mdp.dr.*` 函数）。**缺的事件函数不要用「邻居」顶替**：`reset_joints_by_offset` 是加偏移而 `by_scale` 是乘缩放；`dr.body_mass` 只改质量而 Isaac 默认按比例同步缩放惯量。两者都能编译通过并随机化出**另一个东西**。

mjlab 的 `dr` 包装函数必须用 `@requires_model_fields(*被包装函数.model_fields, ...)` 转发声明，否则模型字段不会逐环境展开、写入失败。

---

## 落地步骤

1. `engines/<name>/`：`terms.py` / `scene.py` / `assets.py` / `adapter.py`。**顶层不 import 引擎**，builder 函数体内才 import——`contract_report()` 因此能在没装该引擎的机器上回答。
2. 能力声明从 `terms.py` 的 `provides=` 推导，禁止手写与实现脱节的声明。capability 用带命名空间的字符串 ID（`contact.air_time` / `dr.friction.per_geom`），引擎包导入时注册，未注册 ID 启动期报错（S3，不要用封闭 enum）。
3. **引擎 profile 的默认值就是该引擎参考实现的值**，`TaskSpec` 只放覆盖。
4. 注册表两种入口：可移植族用 `@TERMS.portable(family)` 注册包装器，其余 `@TERMS.<family>(kind)` 按 kind 注册。缺少 portable 包装器属 **adapter 缺陷**，报错措辞要与「任务要求了引擎没有的能力」区分开。
5. 资产：优先从 `RobotSpec` **推导**而非查表（mjlab 侧只需补执行器增益，因为 MJCF 自带几何）。走 D5 管线时 **validators 比 converters 重要**，校验不通过的资产禁止进训练路径。
6. entry point `instinctlab.engines` 注册，可作独立 pip 包。
7. 验收清单：`bootstrap` 不污染 · `compile` 产出原生 env · `terms.py` 覆盖所需语义名 · `assets.py` + checksum · `caps.py` 声明能力与**版本区间** · contract report golden 入库 · L0/L1 通过。缺失术语走 OPTIONAL 跳过，不必先补齐全部。

## 引擎升级复查项

升级已接入的引擎时，重查所有**未文档化的隐式依赖**——它们不在任何 changelog 里：

- mjlab 四元数 wxyz（继承自 MuJoCo，未文档化）
- mjlab 接触传感器 `fields` 默认值及其内部消费者
- mjlab `soft_joint_pos_limits` 首维语义
- 力历史轴序（两边相反，形状可能相同）
- `convert_quat(quat, to=...)` 两边默认方向都是转成 xyzw
- 已废弃/已删除的 math 函数（`quat_rotate` / `quat_rotate_inverse`：Isaac 已废弃、mjlab 已删除）

`caps.py` 的版本区间就是给这件事用的——升级越界时启动期就该拒绝，而不是等训练曲线不对。
