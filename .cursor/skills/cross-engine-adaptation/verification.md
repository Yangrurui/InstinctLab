# 分层验收与盲区

每一层都有结构性盲区。**盲区不重叠是选择这套分层的理由**，不是冗余。跳过任何一层就等于放弃它独有的那一类缺陷。

## 各层能看见什么

| 层 | 手段 | 独有能看见的 | 结构性看不见的 |
|---|---|---|---|
| L0 静态隔离 | AST 扫描 | `spec/` 误 import 引擎、星号导入、`train.py` 出现引擎名 | 任何运行期行为 |
| L1 逐字段对拍 | golden dump + 白名单 | 参数值漂移、漏传字段 | **字段顺序**、**消费配置的那个类**、配置能否实例化 |
| L2 静态不变量 | `tests/test_parity_static.py` | 观测组顺序、结构性等价前提 | 数值 |
| L3 语法树参照 | `tests/reference_mjlab.py` | 未安装的参照（读 AST 不 import） | 参照的运行期语义 |
| L4 逐值 term 对拍 | `probe_terms.py` 两进程 | 同一状态下 term 数值分歧 | **一切只存在于时间轴上的量** |
| L5 行为探针 | conformance suite | 接触冲量、摩擦、PD 响应、积分器差异 | 任务级组合语义 |
| L6 真实构造 + step | 建起来跑一步 | **参照/产物本身是否可实例化** | 长期行为 |
| L7 短训练曲线 | 回合长度 + 奖励 | 终止是否活着、计时类退化 | 需要大量步数，最贵 |

**L4 的盲区是最容易被误判为已覆盖的。** term 是状态的函数，逐值对拍只能陈述「同一状态下两边算得一样」；接触时长、空中时间、事件触发频率、curriculum 推进全部落在时间轴上，L4 结构性地看不见。接触传感器漏字段那个缺陷，L0–L4 全部通过，最后是 L7 抓住的。

**L6 无可替代。** 对 main 做过逐字段比对、静态不变量、白名单过期检查，唯独没有 L6，结果 golden 是从一份无法实例化的配置里 dump 的，所有验收照常通过。

## L4 为什么写状态而不 step

`probe_terms.py` 把机器人**写**进同一状态后逐项求值，而不是 step 之后比。term 是状态的函数，对「两边被**放到**的同一状态」求值是关于 term 的陈述；step 后再比，一致性里掺进了积分器，是另一件事，而且是弱得多的检验。

现状：26 项在 float32 精度内一致（最大 4.8e-07），5 项按设计不比且各有理由。

两个进程是必须的——Isaac Sim 得先启 app 才能 import `isaaclab`，把两个引擎装进同一解释器不是任何一方预期的用法。

**写状态时禁止用不带坐标系限定的 `write_root_state_to_sim` / `write_root_velocity_to_sim`**：Isaac 收质心速度、mjlab 收连杆速度，同样的数写进去两个机器人根本不在同一状态（实测差 0.85 m/s），后面比什么都没意义。一律 `write_root_link_*_to_sim` / `write_root_com_*_to_sim`。

## L1 的两个陷阱

**不许按字段名排序。** golden dump 与 `verify/structure.py` 的 `dump()` 都不许排序：观测组按属性顺序拼接，排序后的 golden 对一个**观测向量布局已经不同**的配置仍然相等。逐字段 diff 天然看不见重排（路径按名字索引），只有顺序断言能钉住它。

**白名单 key 是路径前缀**，按路径段匹配（`p` 后接 `.` 或 `[`）。**不要写结尾的 `.`**——匹配器自己会补，写了反而永不命中。禁止 `rewards` / `observations` 这类整族前缀，那会吞掉整族的未来差异。新增条目必须写 reason，且必须会过期。

**逐字段对拍不覆盖消费配置的那个类。** main 用 `InstinctRlEnv`（多奖励路由 + MonitorManager），编译产物用朴素 `ManagerBasedRLEnv`。两者等价的前提（奖励容器不是 `MultiRewardCfg`、monitor 配置为空）必须**从参照的声明里读出来断言**，不能默认成立——`num_rewards` 从 1 变成向量意味着两边优化的不是同一个目标。

**比较实体选择器时断言解析结果而非字面量。** 参照写死两个 body 名、我们的声明写正则模式，是同一个选择当且仅当**在实体自己的名字表里解析出同样的名字和同样的顺序**。

## 覆盖率审计：怎么检查自己的检查

接触传感器缺陷通过了当时全部验收，之后做了一次审计，程序如下——**新增一层验收后应当重跑一遍**：

1. **列出对拍脚本能提取的全部信息**，逐项找调用者。**能提取但没人断言的等同于没比。** `tests/reference_mjlab.py` 曾提供事件 `params`、`reward_functions()`、`scene_sensors()` 而无任何调用者，读代码的人会以为比过了——质量随机化范围、接触阈值、每个惩罚罚哪几个关节，当时全部无人看守。
2. **列出参照栈的每一层**，逐层问「这层比过吗」：任务配置 / agent 超参 / PPO 设置 / RL wrapper / runner 循环语义 / 播种 / torch 后端开关。播种和 TF32 两个缺陷就是这一步抓出来的。
3. **对每条抑制机制**（白名单 / skip / 容差）问它是否仍解释着真实差异。加过期检测时当场抓出 11 条死条目。
4. **对每处「由 `tests/xxx.py` 保证」的声称**，确认该文件存在。出现过声称被钉住而测试文件从未被创建的情况。

## 变异检验

新增断言后**必须验证它会失败**：手工改坏被断言的那个值，确认测试变红，再改回来。没做过变异检验的断言与不存在的断言无法区分——上面第 1 类缺陷（提取了但没断言）在代码里看起来和真正的断言一模一样。

## 命令速查

```bash
# 不需要引擎
python -m pytest tests/ -q
python scripts/check_mjlab.py                      # mjlab ≡ InstinctMJ，读语法树

# 需要 Isaac Sim
python scripts/check_parity.py                     # isaacsim ≡ main，逐字段 + 构造 + step
# 重新生成 golden（改动 main 的任务后）。--cfg/--out 必填，check_parity 失败时会打出该跑哪条
python scripts/dump_golden.py \
  --cfg instinctlab.tasks.locomotion.config.g1.flat_env_cfg:G1FlatEnvCfg \
  --out tests/parity/isaacsim.locomotion_flat.golden.json

# 逐值对拍（两进程，各跑一次再 diff）
python scripts/probe_terms.py --engine mjlab   --out /tmp/mjlab.json
python scripts/probe_terms.py --engine isaacsim --out /tmp/isaacsim.json
python scripts/compare_terms.py --run

# 训练
python scripts/train.py --engine isaacsim --task Instinct-Velocity-Flat-G1
python scripts/train.py --engine mjlab    --task Instinct-Velocity-Flat-G1
```

`check_parity.py` 报成功的同时曾丢掉整个构造与 step 结果——`os._exit` 不刷 stdio 缓冲。**输出重定向到文件时，先确认最后一段还在**。

## L7 看曲线

- **必须对回合长度曲线，不只对奖励曲线。** 奖励是各项加权和，一项恒零仍会平滑上升、看起来在学；回合长度直接暴露终止是否活着。
- **满屏精确的 `0.0000` 是信号不是噪声。** 只在 reset 那步产出的日志，在终止失效时会被 wrapper 补零。
- 与参照对的是**形状与量级**，不是逐点数值。导入外部项目的验收标准是**可比性能**，不是复现原论文数字（D4）。
- **同名指标在两个引擎里量纲可以不同，不要直接并排读。** `Episode_Termination/base_contact` 在 Isaac 侧是比例（`0.0275`，与 `time_out` 加起来约 1），在 mjlab 侧是计数（`161.75`）——mjlab 的 `termination_manager.py` 记的是 `torch.count_nonzero`。两边都对、都各自忠于参照，但摆在一起看会得出错误结论。判断「终止是否活着」要看**回合长度**，那是同一个量纲。
- **`python scripts/check_episode_length.py <log>` 每十个训练迭代才有一个数据点**（runner 的记录频率），所以 `--min-points 10` 实际要求约一百个迭代。短跑出来的「too short to judge」不是通过。
