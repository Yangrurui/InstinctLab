# InstinctLab current handoff

Updated: 2026-08-27 06:22 UTC

This is the authoritative record for the current repository, server, datasets,
live experiments, accepted baselines, and unresolved work. Historical audit
narratives are in Git history rather than duplicated here.

## Repository

- Repository: `/root/InstinctLab`
- Branch: `feat/unified-engine`
- Current task cleanup: `a9e3805`
- Remote: `git@github.com:Yangrurui/InstinctLab.git`
- Remote was at `a1e86b8` before this cleanup; push the local commits before
  decommissioning this server.

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

- **Locomotion flat/rough: accepted on both engines.** The archived original
  logs are not present on this replacement server; recover them only if exact
  numerical provenance is needed.
- **Parkour: accepted on both engines.** Task and agent declarations, AMP
  schema, motion loading, depth pipeline, contact behavior, and production
  training were checked against both references.
- **Whole Body plane Shadowing: short-horizon parity accepted.** At 4096
  environments, unified/main rewards at iterations 0, 10, 20, 40, and 100 were
  `-1.63/-1.65`, `-0.96/-0.98`, `-0.48/-0.49`, `-0.23/-0.19`, and
  `0.06/0.09`. Final-commit long-horizon convergence is still in progress.
- **Perceptive Shadowing: startup distribution aligned.** The terrain-matched
  Isaac/main comparison agreed closely at iteration 0, including termination
  mix. Long-horizon convergence remains open.
- **Perceptive VAE, HOI, and BeyondMimic: declarations exist, but no accepted
  production reproduction on both engines.**
- Play variants use the corresponding train checkpoint; they do not need an
  independent training reproduction.
- Real multi-node distributed training remains an infrastructure validation
  item.

Known silent faults already fixed include canonical DFS/native BFS action
offset mapping, current-time AMP references, link-origin velocity semantics,
critic history/width, Perceptive depth preprocessing, motion-terrain matching,
reset sampling order, contact sensor hot-path caching, and engine-native
capacity profiles.

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

The Perceptive long runs already active on Isaac GPU 6 and MJLab GPU 7 were
started before this audit and were not stopped or restarted. The MJLab process
therefore still has the old Parkour-default camera settings in memory and must
not be used as post-fix Perceptive camera or reset-height evidence. The Isaac
process likewise predates the 10 N main contact-clock resolution, although
current Perceptive MDP terms do not consume that timer. A future post-fix
Perceptive comparison must start new log directories; do not relabel either
active run.

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

The active GPU 6 Perceptive process loaded code before these fixes. Its timings
must not be treated as post-fix performance evidence; start a new log directory
for the next production comparison.

## Live experiments

Snapshot at 2026-08-27 05:16 UTC. These processes were inspected only; none was
stopped, restarted, or signaled.

| GPU | Run | Iteration | Reward | Episode length | Status |
|---:|---|---:|---:|---:|---|
| 5 | unified Isaac Whole Body `final_long_4096_gpu5_20260826` | 15900 | 5.84 | 77.12 | live; reward regressed from its earlier peak, inspect before promotion |
| 6 | unified Isaac Perceptive `final_long_4096_gpu6_20260826` | 6750 | 2.71 | 58.73 | live; predates performance and sensor fixes |
| 7 | unified MJLab Perceptive `stablecaps_final_long_4096_gpu7_20260826` | 8460 | 12.62 | 223.37 | live; predates camera reference fixes |

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

1. Let the live final Whole Body and Perceptive comparisons continue. Compare
   matched iterations and termination distributions before declaring
   long-horizon convergence.
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
