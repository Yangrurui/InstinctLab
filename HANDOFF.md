# InstinctLab current handoff

Updated: 2026-08-27 09:32 UTC

This is the authoritative record for the current repository, server, datasets,
live experiments, accepted baselines, and unresolved work. Historical audit
narratives are in Git history rather than duplicated here.

## Repository

- Repository: `/root/InstinctLab`
- Branch: `feat/unified-engine`
- Current task cleanup: `a9e3805`
- Local `origin`: `git@github.com:Yangrurui/InstinctLab.git`
- Export repository: `git@github.com:Yangrurui/XLab.git`; its `main` was synced
  through `348a73d`. Later local audit commits still need an explicit push.

Every active task is registered once in
`source/instinctlab/instinctlab/tasks/registry.py` and compiled by the selected
engine:

```text
task config -> engine-neutral TaskSpec
            -> Isaac Sim adapter -> native Isaac Lab environment
            -> MJLab adapter     -> native MJLab environment
            -> common instinct_rl runner/checkpoint contract
```

Current code organization:

- Locomotion, Parkour, and Shadowing use named `EnvCfg`, reward, and observation
  config classes. Each family owns its shared `*_env_cfg.py`; concrete G1
  datasets, robots, and public factories remain in the corresponding
  `config/g1/*_cfg.py`. There is no central family dispatcher or `make_task`
  function.
- `spec/` defines schemas and validation. It does not contain task values or
  import an engine SDK.
- Task modules do not import engine implementations. Engine packages do not
  import task modules or one another; tests enforce both boundaries.
- G1 canonical names and robot values live in
  `assets/unitree_g1/catalog.py`. Isaac-native articulation construction
  remains in `assets/unitree_g1/isaacsim.py`.
- Viser's MJLab-backed environment selection belongs to `play/viser.py`, not
  to the Isaac adapter.
- `scripts/train.py` and `scripts/play.py` are the only production
  train/play entry points. The obsolete `scripts/instinct_rl/` copies and
  their `sys.path` workarounds were removed.
- `TaskSpec` remains an engine-neutral interface. Concrete rewards, MDP
  values, solver profiles, and runner selection belong to task configuration,
  not to the schema module.

Verification at `27d9767`:

```text
1188 passed, 2 skipped, 30 deselected
python scripts/check_mjlab.py:
  Instinct-Velocity-Flat-G1 resolved all 39 terms
  constructed 16 MJLab environments and stepped 5 times
```

Post-refactor training-equivalence audit against `f7fef17`:

```text
15/15 registered TaskSpec declarations were structurally identical
12/12 compiled MJLab Shadowing/Mimic/VAE env, agent, and resolution configs were identical
137/137 training-entry, Shadowing-contract, joint-order, registry, and isolation tests passed
```

Task-local Shadowing layout audit against `9eb4bc0`:

```text
15/15 registered TaskSpec declarations were structurally identical
12/12 compiled MJLab Shadowing/Mimic/VAE env, agent, and resolution configs were identical
1199 passed, 2 skipped, 30 deselected
python scripts/check_mjlab.py constructed 16 environments and stepped 5 times
```

Unified task-style audit at `efeaa9d`:

```text
15/15 registered TaskSpec contract hashes matched the pre-refactor baseline
1203 passed, 2 skipped, 30 deselected
python scripts/check_mjlab.py resolved all 39 Locomotion terms,
constructed 16 MJLab environments, and stepped 5 times
```

Unused task metadata cleanup at `a9e3805`:

```text
all 15 registered tasks have empty engine_extras
15/15 previous contracts were reproduced by restoring engine_extras only
1204 passed, 2 skipped, 30 deselected
python scripts/check_mjlab.py resolved all 39 Locomotion terms,
constructed 16 MJLab environments, and stepped 5 times
```

The Shadowing contract hashes changed because the unused metadata was removed;
scene, simulation, MDP, reward, observation, and agent declarations did not
change. Task package exports now contain factories only. Parkour sensor
declarations are imported from their owner, `parkour_env_cfg.py`, rather than
forwarded through the G1 file.

Concrete task classes can now replace one reward or observation term directly;
the registry factories only instantiate the class and convert it to `TaskSpec`.
Parkour's shared declarations were restored to
`tasks/parkour/config/parkour_env_cfg.py`, matching the main repository layout.

The active agent configuration values did not change; the latest cleanup only
removed unused imports from those files. The only change in `scripts/train.py`
was removal of the obsolete `scripts/instinct_rl` path workaround after that
shadowing directory was deleted. On Isaac, the Shadowing term builders now read
the same ordered link tuple from the task's motion reference instead of
importing the task-level constant directly.

These checks prove declaration, contract, and construction integrity. They do
not replace fixed-state, temporal, or production convergence evidence.

## External checkouts

These are sibling checkouts, not submodules:

| Checkout | Revision | Purpose |
|---|---|---|
| `/root/InstinctLab-main` | `ba28d3d2655b15a19b729476a630937a19610a3b` | Isaac/main reference |
| `/root/InstinctMJ` | `4ed2b32f8719ff9fc138708341031e935afda0d2` | MJLab reference |
| `/root/IsaacLab` | `f73c33173801f5f8afea4142482e47b7710c2b75` | Isaac Lab dependency |
| `/root/mjlab` | `08090e8a77228e733373f3b5c54f8b5a68d19d9d` | MJLab dependency |
| `/root/instinct_rl` | `64d7e01` (detached HEAD) | RL runner; batched rollout logging transfers |

Uncommitted external changes that cloning upstream will lose:

- `/root/InstinctMJ`: terrain debug visualization is conditional on
  `debug_vis`; play maps the selected CUDA device to EGL and Warp before
  construction.
Commit, export, or reapply those diffs before leaving this server.

## Python and simulator stack

Use `/root/miniconda3/envs/env_isaaclab/bin/python` (Python 3.11).

```text
torch==2.7.0+cu128
isaacsim==5.1.0.0
mjlab==1.5.0
mujoco==3.10.0
mujoco-warp==3.10.0.1
warp-lang==1.14.0
instinct-rl==1.0.2
instinctlab==0.1.0
```

The MuJoCo-Warp and Warp versions are intentional parity pins. The newer
`mujoco-warp==3.10.0.3` / `warp-lang==1.16.0` stack produced materially
different post-contact dynamics.

The pinned MuJoCo-Warp version does not expose `Data.overflow`. Live MJLab
overflow polling is therefore a no-op; monitor Warp warnings and contact
budgets during production runs.

## Datasets and retained baselines

Required compatibility paths:

```text
/root/Datasets/parkour_motion_without_run.yaml
  -> /root/Datasets/parkour_release/parkour_motion_reference/parkour_motion_without_run.yaml
/root/Datasets/parkour_motion_without_run_retargetted.npz
  -> /root/Datasets/parkour_release/parkour_motion_reference/parkour_motion_without_run_retargetted.npz
/root/Datasets/NoKov-Marslab-Motions-instinctnpz/20251016_diveroll4_single
  -> /root/Datasets/deep_whole_body_parkour_g1_release/
     20251116_50cm_kneeClimbStep1/20251106_diveroll4_roadRamp_noWall
```

Released Parkour data checksums:

```text
parkour_motion_without_run.yaml
  f79e5bbc9207976e1610459ab3727a9e1da6d5c0c6cc75793dcec34b81cb7679
parkour_motion_without_run_retargetted.npz
  7cfb7c1dcaa6f2a55a13c4849be9e17b4c960ce4015c500ac0ddfb9d77f4ba5b
```

Active Whole Body clip:
`diveroll4-ziwen-0-retargeted.npz`, 751 frames, 29 joints, SHA-256
`8274d93046811824640ad373bba13ecd46ed347af8cc6d3d7c116df35a1bec59`.

Important retained run directories:

```text
logs/isaacsim/g1_parkour/20260824_174229_gpu0
logs/mjlab/g1_parkour/20260824_174224_gpu2
logs/mjlab/g1_parkour/20260825_125052_oldstack_mw31001_wp114_gpu2
logs/mjlab/g1_parkour/20260825_230944_motor_vlimit_virtual8_gpu1
logs/isaacsim/g1_shadowing/20260826_184409_isaac_actionoffset_fixed_4096_gpu5_20260826
logs/isaacsim/g1_perceptive_shadowing/20260826_192205_perceptive_terrainmatched_4096_gpu6_20260826
```

## Reproduction status

- **Locomotion flat: accepted on both engines. Rough construction and stepping
  are accepted; post-unification convergence is pending.** The archived original
  logs are not present on this replacement server; recover them only if exact
  numerical provenance is needed.
- **Parkour declarations and the pre-terrain-change reproduction are accepted
  on both engines; post-unification terrain convergence is pending.** Task and
  agent declarations, AMP schema, motion loading, depth pipeline, and contact
  behavior were checked against both references. Retained MJLab Parkour runs
  used the old 0.07 m / dense-box recipe and are not evidence for the shared
  terrain below.
- **Whole Body plane Shadowing: short-horizon parity accepted.** At 4096
  environments, unified/main rewards at iterations 0, 10, 20, 40, and 100 were
  `-1.63/-1.65`, `-0.96/-0.98`, `-0.48/-0.49`, `-0.23/-0.19`, and
  `0.06/0.09`. The old GPU 5 long run predates the joint-reference fix and is
  not valid long-horizon convergence evidence. It was stopped at operator
  request and replaced by the fresh run recorded below.
- **Perceptive Shadowing: fresh post-fix Isaac convergence is in progress.**
  Startup summaries agreed with main, but the old GPU 6 run's shared joint-position
  reference command subtracted Isaac-native BFS defaults positionally from DFS
  reference tensors. That run was stopped at operator request and replaced by
  the fresh GPU 6 run recorded below.
- **Perceptive VAE, HOI, and BeyondMimic: declarations exist, but no accepted
  production reproduction on both engines.**
- Play variants use the corresponding train checkpoint; they do not need an
  independent training reproduction.
- Real multi-node distributed training remains an infrastructure validation
  item.

Known silent faults already fixed include canonical DFS/native BFS action
offset mapping, name-ordered joint-reference defaults, current-time AMP
references, link-origin velocity semantics, critic history/width, Perceptive
depth preprocessing, motion-terrain matching, reset sampling order, contact
sensor hot-path caching, and engine-native capacity profiles.

## Terrain module audit (2026-08-27)

Locomotion rough and Parkour now use one engine-neutral recipe from
`tasks/terrain.py`. `TerrainSpec(kind="rough")` must carry that
`TerrainGeneratorSpec`; the kind selects native importer/capacity behavior but
no longer selects a hidden per-engine recipe. The shared training contract
follows `/root/InstinctLab-main`: 8 m patches, 10×20 grid, 0.05 m horizontal
scale, 0.005 m vertical scale, main's stair widths/borders, cumulative-
proportion columns, and one common tile order ending in `mesh_boxes` then
`hf_pyramid_slope_inv`. Parkour command ranges now use the same `mesh_boxes`
name on both engines and no longer carry a terrain-name engine override.

Isaac lowers the semantic tiles to its existing height-field/trimesh configs.
MJLab lowers them to the vendored filed height fields and a native MuJoCo
random-multi-box implementation. Generator-level scale propagation is wired
for both generic and rough MJLab generators. Hfield repair metadata is keyed by
the generated MuJoCo hfield name, so a pure-mesh cell cannot shift the config
associated with later height fields. Virtual edge extraction remains native:
MJLab repairs hfield surfaces and merges short collinear gaps while Isaac reads
the generated mesh, so volume-point penetration is not a raw cross-engine
parity metric.

The complete 0.05 m MJLab grid needs larger native import buffers than the old
recipe: host construction measured 271 contacts and 1,119 constraints. Rough
defaults are therefore `nconmax=512`, `njmax=1536` in the MJLab adapter; they
are not task terrain parameters. A 16-environment full 10×20 locomotion probe
constructed and stepped ten times without Warp/HFIELD warnings; observed step
peaks were `nacon=432` and `nefc=191`. The Parkour full-grid probe resolved all
20 named columns, passed foot-scanner checks, generated and registered 15,216
edge cylinders, then hit the independent existing first-depth-history-all-zero
camera assertion. Do not call that run a full Parkour live pass or weaken the
camera assertion as a terrain workaround.

Evidence:

```text
1236 passed, 2 skipped, 31 deselected (full default suite)
172 passed (focused terrain/spec/Parkour declaration and native bridge suite)
1 passed (shrunk MJLab rough construction plus five steps at 0.05 m)
full MJLab rough 10x20 / 16-env probe: 10 steps, no HFIELD warning,
  peak nacon=432, peak nefc=191
python scripts/check_mjlab.py resolved all 39 Locomotion terms,
  constructed 16 MJLab environments and stepped 5 times
```

## Sensor module audit (2026-08-27)

The active contact, ray-caster/depth, motion-reference, and volume-point paths
were audited separately against `/root/InstinctLab-main` for Isaac and
`/root/InstinctMJ` for MJLab. Two silent Shadowing mismatches were fixed:

- Perceptive and HOI contact clocks now resolve main's explicit 10 N threshold
  on Isaac and InstinctMJ's explicit 1 N threshold on MJLab. The difference is
  declared on `ContactSensorRef`; neither builder contains a task-name branch.
- MJLab Perceptive/VAE/HOI cameras now use InstinctMJ's `(0, 2)` geom groups,
  `exclude_parent_body=False`, 24 min-distance hops, and 1/60 s refresh clock.
  The refresh implementation retains the last frame and reported pose until an
  environment's clock is due. Parkour continues to inherit InstinctMJ's native
  `(0, 1, 2)`, parent exclusion, six hops, and every-sense refresh.

The audit retained the already documented intentional boundaries: normalized
contact timing/selection is portable but raw solver force is not; Isaac camera
targets are explicit main mesh targets while MJLab cameras use InstinctMJ geom
groups; and volume-point velocity is the declared attach-link quantity rather
than reproducing either reference's known subtree/COM-origin bug.

Evidence after the fixes:

```text
1227 passed, 2 skipped, 31 deselected (full default suite)
6 passed (MJLab CUDA synthetic camera ray-kernel comparison, including Perceptive group filtering)
2 passed (MJLab live contact and motion-reference lifecycle)
1 passed (Isaac live motion-reference lifecycle)
1 passed (Isaac native contact-threshold construction)
python scripts/check_mjlab.py resolved all 39 Locomotion terms,
constructed 16 MJLab environments, and stepped 5 times
```

InstinctMJ's min-distance continuation returns a different result on the CPU
Warp ray kernel for the synthetic near-hit case; both implementations agree on
the production CUDA path. The live parity test therefore marks that one case
CUDA-only rather than treating CPU behavior as production evidence.

The AMP and Shadow motion-reference follow-up audit at `7ccb704` also checked
the shared clip runtime, the two native sensor lifecycles, and the active task
consumers against main and InstinctMJ. AMP policy/reference terms retain one
ordered schema and one numerical builder; policy history reads live robot state,
the dedicated reference buffer samples current time `t`, and Parkour command/
termination look-ahead remains separate at `t + dt`. Shadow rewards consume the
current reference, commands and failure terminations consume look-ahead data,
and reset keeps its separately floor-indexed initial sample.

Two silent reset-coordinate differences were fixed:

- InstinctMJ/main gate the zero-ground correction once for the selected reset
  batch. If any selected pose penetrates zero, `-minimum_link_height` is applied
  to every selected pose, including airborne poses. The shared runtime now
  preserves that effective behavior without a host scalar poll. On the released
  Perceptive dataset, 4,844 of 4,896 retained frames have positive minimum link
  height and 52 have negative height, so the old per-environment clamp was not
  equivalent for mixed batches. Isaac Perceptive remains intentionally exempt
  through its engine override.
- Late binding of live environment origins translated robot root/link buffers
  but omitted HOI object positions. All three buffers (look-ahead, reset, and
  current-reference) now translate valid object slots through the same world-
  position path; absent object slots remain neutral.

The object-free multi-clip packed sampler was checked against the retained
per-clip path as an oracle. With identical reset RNG, all fields in the reset,
current-reference, and look-ahead buffers matched, including motion selection,
floor/rounded frame indices, velocities, validity, and exhaustion bookkeeping.

Evidence for this follow-up:

```text
1230 passed, 2 skipped, 31 deselected (full default suite)
155 passed (focused AMP, motion-reference, and Shadow runtime/contract suite)
155 passed, 1 deselected (main/MJ source readers and Parkour/Shadow declarations)
1 passed per engine (native motion-reference lifecycle and AMP write-state probe)
1 passed per engine (full Shadow reset plus four-step rollout)
```

The OMOMO motion files are not installed on this server, so HOI object-origin
coverage is fixed-state rather than a full simulator rollout. Production HOI
reproduction remains open below.

The Perceptive long runs active during this audit on Isaac GPU 6 and MJLab GPU 7
were started before the audit. The MJLab process
therefore still has the old Parkour-default camera settings in memory and must
not be used as post-fix Perceptive camera or reset-height evidence. The Isaac
process also predated the fixes and was stopped at operator request on
2026-08-27. Its retained logs remain diagnostic only; they were not relabeled.

## Perceptive Isaac performance audit (2026-08-27)

The remaining Isaac/main training-time gap was not in PhysX, camera ray casts,
or Warp synchronization. Matched 4096-environment profiles showed two unified
hot-path costs:

- Ten object-free motion clips were sampled independently every step, producing
  1,500 `fill_buffers` and 21,000 field-gather calls over a 30-step profile.
  The runtime now concatenates compatible clips once and samples them with one
  device gather; the same profile now makes 150 and 2,100 calls respectively.
  HOI/object-bearing clips retain the scene-object-aware fallback.
- `base_ang_vel` requested Isaac's link velocity, which fetched COM offsets and
  constructed an unused link linear velocity. Link and COM angular velocity are
  bitwise identical in Isaac Lab; the portable term now prefers the direct COM
  property on Isaac and retains the link fallback on MJLab. `get_coms` fell from
  two calls per environment step to the one required by link-velocity imitation.

The environment-only probe improved from about 15,600 to 18,266 env-step/s;
the matched main probe was 17,671 env-step/s. In short PPO runs, steady samples
at iterations 10/20/30 were:

| Run | Collection seconds | Median collection | Median total FPS |
|---|---|---:|---:|
| unified before | 6.740 / 6.922 / 6.488 | 6.740 | 11,543 |
| main reference | 5.677 / 5.550 / 5.766 | 5.677 | 13,293 |
| unified current | 7.930 / 5.810 / 5.619 | 5.810 | 12,926 |

Iteration 10 is a noisy warm-up point; the current median is about 2.3% slower
than main rather than the previous 18.7%. The first current rollout reproduced
the same reward, episode length, reward-term, and termination summaries before
and after the logging change.

The Tensor-to-Python audit also removed per-step scalar polling from depth miss
handling, delayed depth generation tracking, AMP current-frame refresh, and
adaptive sampling. `/root/instinct_rl` commit `64d7e01` batches the runner's
reward/done/episode-length host logging once per rollout, reducing one-reward
rollouts from 72 GPU-to-CPU transfers to three.

Remaining conversions are either explicit output/debug boundaries or outside
plain Perceptive's hot path. The notable unresolved cases are MJLab camera
min-distance continuation (`still.any()` once per hop; a fused Warp traversal
is needed to remove it without launching every hop for every ray), HOI invalid
object routing, and independent-motion-bin reset sampling. Isaac Lab itself
also calls `Tensor.item()` from observation-history `CircularBuffer.max_length`
about 20 times per Perceptive step; both unified and main use that dependency.

The old GPU 6 Perceptive process loaded code before these fixes, so its timings
are not post-fix performance evidence. The replacement run uses InstinctLab
`1ee8654` and instinct_rl `64d7e01`; at iteration 10 it reported 14,142 steps/s
(5.224 s collection), versus main's 13,293 steps/s (5.677 s collection).

## Perceptive joint-reference audit (2026-08-27)

The active Isaac Perceptive run remained near 64-step episodes at iteration
7,510, with `base_pos_too_far` causing about 56% of resets. Checkpoint
normalizer statistics exposed a shape-compatible joint-order bug in
`JointPositionReference`: motion data is canonical DFS, but the command had
subtracted `default_joint_pos` directly in Isaac's native BFS order. It also
held a live view of defaults modified by the later startup randomization,
whereas main snapshots nominal defaults when the command is constructed.

The command now resolves default columns by joint name, gathers them into the
motion-reference order, and clones the pre-randomization tensor. Applying that
exact order correction to the iteration-6,000 checkpoint reduced the
semantic `joint_pos_ref` normalizer-mean MAE against main from `0.32020` to
`0.00854` (maximum absolute difference `0.04765`). This is direct retained-run
evidence for the fault, not only a static source inference.

Verification:

```text
52 passed (joint/action, Shadow MDP, and training-flow focus)
1231 passed, 2 skipped, 31 deselected (full default suite)
scripts/check_mjlab.py resolved all 39 Locomotion terms and stepped 16 envs 5x
```

Every registered task whose MDP declares `shadow_joint_position_reference`
uses this shared implementation: Whole Body train/play, Perceptive train/play
and OneMotion train/play, Perceptive VAE train/play, Perceptive HOI train/play,
and BeyondMimic train/play (12 task IDs). Isaac is exposed to the BFS/DFS fault;
MJLab's native G1 joint order is already DFS. Parkour has a motion reference but
does not declare this command and is not affected by this fault. The per-task
test now discovers consumers by command kind, including the two VAE IDs (21
focused tests pass at `5c3f294`).

At operator request, the old GPU 6 process was stopped and a fresh seed-42 run
was started from the clean `1ee8654` worktree, with a new log directory. At
iteration 20 its reward/length were `-0.52/9.08`, close to main's
`-0.49/8.63`; this is an early sanity check, not convergence evidence.

## Canonical joint-order audit (2026-08-27)

Isaac's native PhysX articulation remains breadth-first; changing that native
layout would break Isaac Lab's own buffers. The engine-neutral robot, policy,
motion, and checkpoint interface is the G1 catalog's 29-joint depth-first
order. Every Isaac boundary must therefore resolve names and gather/scatter
explicitly rather than writing a DFS tensor positionally into native storage.

The audit covered action targets and scales, policy/critic/AMP observations,
joint rewards and limits, default/randomized poses, reset state writes, Shadow
commands and motion-reference buffers, direct-backend state/control writes,
diagnostic state capture/replay, and checkpoint loading. Four silent gaps were
closed:

- An ordered selector such as `joints=".*"` is now expanded against
  `RobotSpec.joint_names` into exact canonical names before either engine
  resolves it. `preserve_order=True` on one regex alone does not stop Isaac
  from enumerating matches in native BFS order.
- Task validation now rejects unordered or non-canonical policy/action joint
  axes, non-canonical ordered joint references inside other MDP terms, and a
  motion-reference joint list that is not a canonical subsequence.
- Manifest-backed checkpoints now compare the ordered joint names and robot
  schema version. A BFS checkpoint and DFS runtime have the same width and name
  set, so task ID and contract hash drift checks alone cannot protect them.
  Pre-manifest checkpoints remain loadable for compatibility but emit an
  explicit BFS/DFS warning.
- The retained Isaac `ActionOverridenMixin` now maps overridden articulation
  joints by name onto the action term's DFS axis instead of using native BFS
  joint IDs as policy columns. No registered task currently selects this
  legacy action, but leaving the positional path was unsafe.

Existing name bridges were confirmed for Shadow reset, randomized action
offsets, joint-position reference defaults, AMP policy/reference terms, and
motion clip loading. Non-ordered reward selectors were checked individually:
they either reduce corresponding native tensors without an external joint
vector, or, for Parkour velocity limits, use an explicitly ordered DFS selector
with the matching DFS limit vector. The generic Isaac backend's read direction
(native BFS to canonical DFS) and write direction (canonical values paired
with mapped native IDs) have distinct regression tests.

Evidence:

```text
1249 passed, 2 skipped, 31 deselected (full default suite)
210 passed (joint-order/spec/compiler/checkpoint/backend focused suite)
python scripts/check_mjlab.py resolved all 39 Locomotion terms,
  constructed 16 environments, exposed a 29-joint catalog-order action,
  and stepped 5 times
```

A fresh Isaac live probe was attempted twice but Kit stopped before environment
construction while Neuray loaded on this host; its log reports missing
`libGLU.so.1`. The probe never reached the joint assertions, so this audit does
not claim a new Isaac live pass. The existing Parkour live test still contains
the accepted runtime guard that checks native BFS names, DFS observation/action
names, semantic state values, and action-history columns. The three production
training processes were not signaled or restarted during this audit.

## Actuator and control-path audit (2026-08-27)

Isaac's 0--2 physics-step actuator delay is explicitly operator-intended and
was excluded from defect classification. Outside that boundary, the complete
position-control path is aligned: all 15 registered tasks expose one 29-joint
DFS action, use the same per-joint `0.25 * effort / stiffness` scale and
randomized default-position offset, process one action per policy step, and
write the held target on every decimation substep. No task applies an action
clip.

The shared G1 catalog was compared per joint against both reference source
tables. Effort, stiffness, damping, armature, and action scale match main and
InstinctMJ; Isaac's native `velocity_limit_sim` also matches main. MJLab uses
only InstinctMJ's seven `BuiltinPdActuator` groups and has no auxiliary
motor-speed limiter. A stale `KNOWN_DRIFTS` row still claimed that such a
braking actuator existed after it was removed in `611f9be`; the row was
deleted. This was an audit-record defect, not a production behavior change.

Runtime MJLab probes confirmed that compiled `jnt_actfrcrange`, `dof_armature`,
action scale, and action offset match the catalog (maximum scale/offset error
below `3e-8`). A three-environment Whole Body startup probe produced distinct
randomized defaults and exactly matching action offsets. An isolated
Perceptive startup probe confirmed Kp factors in `[0.8, 1.2]`, Kd factors in
`[0.9, 1.1]`, and consistent MuJoCo gain/bias pairs for all seven actuator
groups. The motion-to-terrain matching event was disabled only for that
diagnostic because three environments cannot host every referenced terrain;
production configuration was not changed.

Torque and energy terms retain the documented engine-native sources: Isaac
uses `applied_torque`, while MJLab uses joint-space `qfrc_actuator`. New
fixed-state regressions pin stiffness normalization by actuator target IDs and
the mapping from MuJoCo global `jnt_actfrcrange` rows back to an arbitrary
selected local joint order. Joint acceleration remains an intentional native
difference (Isaac finite difference, MJLab solver `qacc`).

Evidence:

```text
68 passed (asset/action/actuator/reference/reward focus)
1253 passed, 2 skipped, 31 deselected (full default suite)
scripts/check_mjlab.py resolved all 39 Locomotion terms,
  constructed 16 environments in DFS action order, and stepped 5 times
```

The full Parkour live snapshot was stopped before environment construction
completed because MuJoCo Warp's 0.05 m height-field BVH cold build remained in
CPU preprocessing; it produced no rollout evidence. The flat and Shadow probes
cover the same catalog control plant without that terrain cost. HOI remains
blocked by the already-recorded missing OMOMO object assets. Existing training
processes on GPUs 5, 6, and 7 were not signaled or restarted.

## Episode boundary and reset audit (2026-08-27)

The native Isaac Lab and MJLab environment loops agree on the episode
timeline: increment the episode counter after the decimated physics step,
compute termination before reward, auto-reset completed environments, then
return post-reset observations together with the pre-reset terminated and
truncated flags. Both engines use `ceil(episode_length_s / step_dt)` and
classify `episode_length_buf >= max_episode_length` as a timeout. The shared
runner preserves that truncation separately for value bootstrapping while
combining terminated and truncated into PPO's done mask, matching both
references.

Four declaration/lowering differences were fixed:

- Perceptive VAE training no longer silently resamples an exhausted motion
  clip without ending the episode. Only VAE Play retains main/InstinctMJ's
  `reset_without_notice=True` behavior.
- Perceptive, one-motion Perceptive, VAE, and HOI Play remove the three
  tracking-failure terminations removed by their corresponding references.
- BeyondMimic Play now uses the reference 6,000-second horizon instead of the
  ten-second training horizon.
- `illegal_reset_contact` is declared as a timeout in the engine-neutral task
  contract, and both builders now lower the declared `time_out` value instead
  of hard-coding it. Native behavior was already timeout-classified, so this
  last change repairs the contract without changing active Perceptive runs.

A two-environment MJLab temporal probe used a two-step horizon. Step one
returned episode length one with no done flag; step two auto-reset the counter
to zero and returned `truncated=True`, `terminated=False`, runner `done=True`,
and runner `time_outs=True`; step three advanced the new episode to one with
all flags false. This directly checks the boundary and bootstrap plumbing, not
just construction.

Evidence:

```text
4 passed (new termination-set, timeout-lowering, and VAE reset regressions)
88 passed (Shadow contract/runtime/task declaration focus)
109 passed, 4 deselected (MDP, motion, and depth focus)
1257 passed, 2 skipped, 31 deselected (full default suite)
scripts/check_mjlab.py resolved all 39 Locomotion terms,
  constructed 16 environments and stepped 5 times
```

No fresh Isaac rollout is claimed because the host still lacks `libGLU.so.1`.
The live GPU 5--7 training processes were not signaled or restarted. The
configuration fixes affect currently inactive VAE/HOI/BeyondMimic or Play
variants; the active Perceptive timeout behavior was already equivalent in the
native builders.

## Train/play checkpoint contract audit (2026-08-27)

Manifest-backed Shadowing checkpoints trained under a normal task id could
not be loaded through the corresponding `*-Play-v0` task: playback validated
the Play id literally even though the retained checkpoint correctly recorded
the Train id. This was reproduced with the active MJLab Perceptive run before
changing the validator.

The task registry now owns six explicit Play-to-Train checkpoint pairs. The
play entry point validates the checkpoint's recorded identity against that
registered Train id while still checking the Play runtime's canonical DFS
joint order and robot schema. Training resume and every unpaired task omit the
override and remain strict about their own task id. There is no suffix-based
guessing and no engine branch. ONNX export records both the runtime Play
contract and the source checkpoint task id.

All six pairs are guarded for identical asset/schema/joint axes, action
contract, ordered policy observations and history widths, runner architecture,
and experiment directory on both engines. A retained Perceptive training
checkpoint now passes its registered Perceptive Play validation; presenting
that checkpoint as BeyondMimic Play is still rejected before tensor loading.

Evidence:

```text
4 passed (new pairing, strict mismatch, registry, and load-order regressions)
80 passed (checkpoint/registry/train-play/joint/task focus)
1261 passed, 2 skipped, 31 deselected (full default suite)
```

## Live experiments

Snapshot at 2026-08-27 07:35 UTC. The old GPU 5 and GPU 6 runs were stopped and
replaced at explicit operator request. GPU 7 was not signaled.

| GPU | Run | Iteration | Reward | Episode length | Status |
|---:|---|---:|---:|---:|---|
| 5 | unified Isaac Whole Body `jointref_fixed_final_long_4096_gpu5_20260827` | 0 | -1.66 | 20.36 | live; fresh seed 42, InstinctLab `1ee8654`, runner `64d7e01` |
| 6 | unified Isaac Perceptive `jointref_fixed_final_long_4096_gpu6_20260827` | 20 | -0.52 | 9.08 | live; fresh seed 42, InstinctLab `1ee8654`, runner `64d7e01` |
| 7 | unified MJLab Perceptive `stablecaps_final_long_4096_gpu7_20260826` | 9700 | 14.73 | 243.73 | live; natural DFS order avoids the Isaac BFS/DFS fault, but it predates camera reference fixes |

Do not stop or restart these runs without an explicit operator request. Review
reward, episode length, termination mix, and action noise together before
promoting any run.

Current TensorBoard comparison links remain under:

```text
logs/tb_compare/g1_shadowing_diveroll/
logs/tb_compare/g1_perceptive_shadowing/
```

The main Perceptive run's position-monitor summaries can be NaN because of its
known empty-slice monitor behavior; rollout, reward, and optimization remain
finite.

## Ended or failed runs that still matter

- The old Isaac Whole Body `final_long_4096_gpu5_20260826` was stopped at
  operator request at iteration 18,320 (reward 5.67, episode length 74.87).
  It reached roughly 260-step episodes before an adaptive-sampling collapse
  near iteration 14,700, but it contains the faulty Isaac joint-position
  reference command. Retain it for diagnosis only; do not resume or promote it.
- MJLab Whole Body
  `finalaligned_datafixed_4096_gpu0_20260826` reached iteration 17300
  (reward 18.02, episode length 248.04), then ended with CUDA error 719,
  `unspecified launch failure`. The reported `nonzero()` frame may be an
  asynchronous symptom rather than the failing kernel. Keep its checkpoints
  and logs; do not describe it as a completed 50k run.
- The older MJLab Perceptive `perceptive_repro_4096_gpu2` reached iteration
  11750 and ended with `KeyboardInterrupt`. It predates final parity fixes and
  is not a baseline.
- The fixed-capacity MJLab Perceptive run failed after iteration 180 when the
  policy mean became NaN. Commit `6c472bf` restored the stable native capacity
  profile; the GPU 7 replacement above is the relevant run.
- Earlier Whole Body and Perceptive runs predating the final DFS reset,
  current-reference, history, terrain-match, or capacity fixes are diagnostic
  only. Do not resume or promote them.

## Open risks and next work

1. Let the fresh GPU 5 Whole Body and GPU 6 Perceptive runs continue and compare
   matched iterations and termination distributions before declaring
   long-horizon convergence. Do not promote checkpoints from either old run.
2. Diagnose the MJLab Whole Body CUDA 719 failure from the retained checkpoint
   and logs without disturbing live runs.
3. Run production reproductions for Perceptive VAE, HOI, and BeyondMimic on
   both engines.
4. Validate real multi-node distributed training.
5. Recover authoritative Parkour motion segment boundaries. The released NPZ
   concatenates clips without boundary metadata: 55 of 18,981 transitions
   exceed conservative discontinuity thresholds, and up to 2.81% of 10-frame
   AMP windows may cross a jump. Cross-engine parity is verified; source-data
   semantics are not.
6. Continue reducing engine-specific parameter overlays by translating shared
   semantic values in builders. Do not move task policy into an engine package
   to achieve this.
7. Restore the host OpenGL utility dependency providing `libGLU.so.1` before
   rerunning fresh Isaac live joint-order probes; the current failure occurs in
   Kit/Neuray initialization before an environment is constructed.

## Bring-up and verification

Before starting a training run, inspect `pgrep -af scripts/train.py` and
`nvidia-smi`, then choose a new run name.

```bash
python scripts/install.py
PYTHONPATH=source/instinctlab /root/miniconda3/envs/env_isaaclab/bin/python -m pytest -q tests
/root/miniconda3/envs/env_isaaclab/bin/python scripts/check_mjlab.py
PYTHONPATH=source/instinctlab /root/miniconda3/envs/env_isaaclab/bin/python   -m pytest -q tests/test_parkour_g1_declaration.py -k contract
```

For term or physics changes, add fixed-state and temporal probes. A construction
smoke test is not parity evidence. Compare episode length and terminations as
well as reward, and run production-scale contact/constraint checks after
terrain, collision, solver, or actuator changes.
