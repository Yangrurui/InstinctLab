# InstinctLab server migration handoff

Updated: 2026-08-26 UTC

This is the authoritative handoff for the unified Isaac Sim/MJLab repository.
It supersedes the old implementation plans and task-specific audit documents.

## Repository state

- Repository: `/root/InstinctLab`
- Branch: `feat/unified-engine`
- Remote: `git@github.com:Yangrurui/InstinctLab.git`
- Baseline before the current AMP/handoff work: `b8c6c8c`
- The branch was one commit ahead of its remote before this work. Push all AMP
  and migration commits before decommissioning this server.

The active registry contains unified locomotion flat/rough, parkour, and twelve
shadowing train/play variants. The source of truth is
`source/instinctlab/instinctlab/tasks/registry.py`. There are no active
Isaac-only parkour, locomotion, or shadowing task paths.

```text
TaskSpec
  -> isaacsim adapter -> native Isaac Lab environment
  -> mjlab adapter    -> native MJLab environment
  -> common instinct_rl runner/checkpoint contract
```

## External checkouts

These are sibling checkouts, not submodules:

| Checkout | Revision | Purpose |
|---|---|---|
| `/root/InstinctLab-main` | `ba28d3d2655b15a19b729476a630937a19610a3b` | Isaac/main reference |
| `/root/InstinctMJ` | `4ed2b32f8719ff9fc138708341031e935afda0d2` | MJLab reference |
| `/root/IsaacLab` | `f73c33173801f5f8afea4142482e47b7710c2b75` | Isaac Lab dependency |
| `/root/mjlab` | `08090e8a77228e733373f3b5c54f8b5a68d19d9d` | MJLab dependency |
| `/root/instinct_rl` | `ba45ed231ebbf0a4099cd31d607e2886814fd165` | RL runner |

`/root/InstinctMJ` has two uncommitted fixes that cloning upstream will lose:

- `manager_based_rl_env.py` calls terrain debug visualization only if the
  terrain implements `debug_vis`.
- `scripts/instinct_rl/play.py` maps the requested CUDA device to
  `MUJOCO_EGL_DEVICE_ID` and Warp before environment/viewer construction.

`/root/instinct_rl` has one uncommitted diagnostic addition in
`algorithms/wasabi.py`: discriminator sign accuracy for actor target `-1` and
reference target `+1`. It does not change optimization. Commit, export, or
reapply these external diffs before leaving this server.

## Python and simulator stack

Use `/root/miniconda3/envs/env_isaaclab` (Python 3.11). Important versions:

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

The MuJoCo-Warp and Warp pins are intentional. The newer
`mujoco-warp==3.10.0.3` / `warp-lang==1.16.0` stack produced materially
different post-contact dynamics. Install with `python scripts/install.py` and
verify versions instead of allowing pip to upgrade them.

The pinned `mujoco-warp==3.10.0.1` predates the public `OverflowType` enum and
does not expose `Data.overflow`. InstinctLab keeps the later bit-name contract
locally so saved diagnostics remain readable, but live MJLab overflow polling
is a no-op when that field is absent. Monitor MuJoCo/Warp warnings and contact
budgets during production runs; restoring live bit polling requires an
upstream backport or a newer physics stack and therefore changes the parity
baseline.

## Data and local artifacts

Copy symlink targets, not only the links:

```text
/root/Datasets/parkour_motion_without_run.yaml
  -> /root/Datasets/parkour_release/parkour_motion_reference/parkour_motion_without_run.yaml
/root/Datasets/parkour_motion_without_run_retargetted.npz
  -> /root/Datasets/parkour_release/parkour_motion_reference/parkour_motion_without_run_retargetted.npz
/root/Datasets/NoKov-Marslab-Motions-instinctnpz/20251016_diveroll4_single
  -> /root/Datasets/deep_whole_body_parkour_g1_release/
     20251116_50cm_kneeClimbStep1/20251106_diveroll4_roadRamp_noWall
```

The released parkour AMP folder was downloaded from the official InstinctMJ
Google Drive link. It contains one 61-byte YAML manifest and one 2.7 MiB NPZ:

```text
parkour_motion_without_run.yaml
  sha256 f79e5bbc9207976e1610459ab3727a9e1da6d5c0c6cc75793dcec34b81cb7679
parkour_motion_without_run_retargetted.npz
  sha256 7cfb7c1dcaa6f2a55a13c4849be9e17b4c960ce4015c500ac0ddfb9d77f4ba5b
```

The shadowing dataset is about 4.4 MiB. `logs/` is
about 11 GiB and is not tracked. Important baselines are:

```text
logs/isaacsim/g1_parkour/20260824_174229_gpu0
logs/mjlab/g1_parkour/20260824_174224_gpu2
logs/mjlab/g1_parkour/20260825_125052_oldstack_mw31001_wp114_gpu2
logs/mjlab/g1_parkour/20260825_230944_motor_vlimit_virtual8_gpu1
logs/mjlab/g1_shadowing/20260825_202243_released_diveroll_gpu0
logs/isaacsim/g1_shadowing/20260825_202305_released_diveroll_gpu1
```

The last Isaac shadowing run predates the canonical DFS-to-native BFS reset fix
and is invalid as convergence evidence. Do not resume it.

## Reproduction status

- **Parkour: complete.** The current unified parkour task is accepted as the
  reproduced Isaac Sim/MJLab implementation. Its task/agent declarations,
  released AMP data path, ordered AMP schema, motion sampling, and production
  training behavior have been checked against the two reference repositories.
  Engine-native physics differences listed below remain intentional. The
  concatenated-release boundary issue is a source-data limitation, not a
  blocker for the reproduction status. A post-shadowing backport audit on
  2026-08-26 confirmed that Parkour automatically benefits from the shared
  contact-sensor engine-detection cache. The other shadowing fixes are not on
  its execution path: Parkour uses additive state reset rather than randomized
  default action offsets, a single clip without terrain metadata or concat-bin
  sampling, the dedicated delayed-depth pipeline, and the generic AMP
  current-time `reference_frame`. Do not copy Perceptive height/depth or
  terrain-origin settings into Parkour. Its focused declaration, AMP, motion,
  depth, contact, and reference-contract suite passed 244 tests (6 deselected).
- **Whole-body plane shadowing: short-horizon parity established.** The final
  Isaac audit reproduces main's early learning curve and runtime at 4096
  environments after correcting the native/canonical action-offset mapping and
  the shadowing runtime differences listed below. A 101-iteration validation is
  accepted as short-horizon evidence; a fresh long-horizon run from the final
  commit is still required before declaring production convergence complete.
- **Other shadowing families: not production-reproduced.** Perceptive,
  perceptive one-motion, perceptive VAE, perceptive HOI, and BeyondMimic have
  unified declarations, but do not yet have accepted production training/play
  evidence on both engines.
- **Locomotion: complete.** Flat (`Instinct-Velocity-Flat-G1`) and rough
  (`Instinct-Velocity-Rough-G1`) locomotion have already been tested on both
  engines and are accepted as reproduced. This status was confirmed on
  2026-08-26 after the handoff had incorrectly left them open. Their original
  run directories are not present in this replacement server's `logs/`, so
  recover the archived logs if exact numerical provenance is needed; do not
  rerun them solely because the earlier handoff status was stale.

Play variants do not require independent training reproduction; validate them
with an accepted checkpoint from their corresponding train task. Real
multi-node distributed training remains an infrastructure validation item, not
a task-reproduction item.

On the replacement server, the released shadowing data was downloaded from the
official InstinctMJ Google Drive folder into:

```text
/root/Datasets/deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1
```

The compatibility path
`/root/Datasets/NoKov-Marslab-Motions-instinctnpz/20251016_diveroll4_single`
points to its `20251106_diveroll4_roadRamp_noWall` subdirectory. The active
clip is `diveroll4-ziwen-0-retargeted.npz` (751 frames, 29 joints, SHA-256
`8274d93046811824640ad373bba13ecd46ed347af8cc6d3d7c116df35a1bec59`).

The unified Perceptive task now binds both engines to the released dataset root
`/root/Datasets/deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1`.
Its top-level `metadata.yaml` pairs ten motions with six terrain meshes. This
replaces main's literal data-directory placeholder and InstinctMJ's unavailable
server-local `~/Xyk/Datasets/her_leveled` path.

## Live process state at handoff

GPU 0 training was explicitly stopped. Two MJLab parkour runs were still live:

- GPU 1: `motor_vlimit_virtual8_gpu1`, around iteration 9390/30000, noise 1.08,
  mean reward 52.90, mean episode length 801.
- GPU 2: `oldstack_mw31001_wp114_gpu2`, around iteration 18960/30000, noise
  0.98, mean reward 73.18, mean episode length 850.

Their stdout logs are:

```text
logs/train_mjlab_parkour_motor_vlimit_virtual8_gpu1.log
logs/train_mjlab_parkour_oldstack_mw31001_wp114_gpu2.log
```

Processes cannot migrate. Let them finish or stop them explicitly, copy their
run directories, and use new `run_name` values on the new server.

Those old-server MJLab processes are not present on the replacement server. A
fresh corrected Isaac shadowing convergence run was started on GPU 0 on
2026-08-26 with 4096 environments, seed 42, and 50000 iterations:

```text
logs/isaacsim/g1_shadowing/20260826_130452_dfsreset_currentref_diveroll_gpu0
```

It uses the DFS reset mapping and current-time reference fixes at `7d8a445`,
does not resume a checkpoint, and reached learning iteration 0 successfully.
Its initial mean reward was -1.80, mean episode length was 13.99, and throughput
was about 12.8k steps/s. These are startup measurements, not convergence
evidence; inspect the long-horizon trend before promoting it to a baseline.

Two fresh Perceptive production-comparison runs were started from scratch at
`6face18` with 4096 environments, seed 42, 50000 iterations, and strict term
resolution. Both completed learning iteration 0 and remain live:

- physical GPU 1, Isaac Sim:
  `logs/isaacsim/g1_perceptive_shadowing/20260826_163829_perceptive_repro_4096_gpu1`;
  initial throughput 5.84k steps/s, collection 12.849 s, learning 3.982 s;
- physical GPU 2, MJLab (isolated as local `cuda:0`):
  `logs/mjlab/g1_perceptive_shadowing/20260826_163818_perceptive_repro_4096_gpu2`;
  initial throughput 11.79k steps/s, collection 4.204 s, learning 4.137 s.

Full stdout is preserved separately and continuously at:

```text
logs/train_isaacsim_perceptive_repro_4096_gpu1.log
logs/train_mjlab_perceptive_repro_4096_gpu2.log
```

Isaac's first simulation start took 126.95 s and emitted about 11.5 MiB of
repeated unresolved visual-reference warnings while cloning 4096 environments;
the warnings did not prevent rollout or learning. These startup measurements
are not convergence evidence.

Both unified Perceptive runs above, and the GPU 0 Whole Body run, predate the
final parity fixes through `530b4b8`. Leave the processes untouched unless an
operator explicitly chooses to stop them, but do not resume them or promote
their curves as final baselines. Their logs remain useful only for diagnosing
the corrected faults.

On 2026-08-26 the operator explicitly requested stopping the obsolete GPU 0
Isaac Whole Body process `dfsreset_currentref_diveroll_gpu0`; PID 26325 exited
cleanly. A final-commit MJLab Whole Body production test was then started on
physical GPU 0 with 4096 environments, seed 42, and 50000 iterations:

```text
logs/mjlab/g1_shadowing/20260826_202114_finalaligned_datafixed_4096_gpu0_20260826
logs/train_mjlab_wholebody_finalaligned_datafixed_4096_gpu0_20260826.log
```

The first launch exposed that the MJLab binding still used InstinctMJ's private
`~/Xyk/Datasets/...` directory. Commit `5f2e0df` binds the same released clip
through the portable `~/Datasets/...` compatibility path; the underlying data
and training semantics are unchanged. The corrected run reached iteration 10:
iteration 0 reported 41.3k steps/s, reward -1.67, and episode length 21.14;
iteration 10 reported 48.4k steps/s, reward -1.05, and episode length 19.03.
The critic width is 889. It is linked into the port-6006 comparison as
`logs/tb_compare/g1_shadowing_diveroll/mjlab_finalaligned_4096_gpu0` and was
still live at this handoff update.

Three additional final-commit long-horizon runs were started on 2026-08-26 at
`07223e2`, all with 4096 environments, seed 42, 50000 iterations, strict term
resolution, and continuously saved stdout:

- physical GPU 5, Isaac Whole Body:
  `logs/isaacsim/g1_shadowing/20260826_203402_final_long_4096_gpu5_20260826` and
  `logs/train_isaacsim_wholebody_final_long_4096_gpu5_20260826.log`; iteration 0
  reward -1.57, episode length 19.24, 15.3k steps/s; iteration 10 reward -0.90,
  episode length 15.50, 26.1k steps/s;
- physical GPU 6, Isaac Perceptive:
  `logs/isaacsim/g1_perceptive_shadowing/20260826_203420_final_long_4096_gpu6_20260826`
  and `logs/train_isaacsim_perceptive_final_long_4096_gpu6_20260826.log`;
  iteration 0 reward -1.52, episode length 19.13, 8.5k steps/s;
- physical GPU 7, MJLab Perceptive (isolated as local `cuda:0`):
  `logs/mjlab/g1_perceptive_shadowing/20260826_203411_final_long_4096_gpu7_20260826`
  and `logs/train_mjlab_perceptive_final_long_4096_gpu7_20260826.log`; iteration
  0 reward -1.76, episode length 20.44, 11.7k steps/s; iteration 10 reward
  -1.26, episode length 18.75, 18.5k steps/s.

Their TensorBoard names are `isaacsim_final_long_4096_gpu5` under the port-6006
Whole Body comparison and `isaacsim_final_long_4096_gpu6` plus
`mjlab_final_long_4096_gpu7` under the port-6007 Perceptive comparison. These
runs were live at this update; do not stop, restart, or promote them before
reviewing long-horizon reward, episode length, and termination trends.

Two matching Isaac reference runs from `/root/InstinctLab-main` were started
from scratch on 2026-08-26 with 4096 environments, seed 42, and 50000
iterations. Both reached learning iteration 0 and remain live:

- physical GPU 3, main Whole Body:
  `/root/InstinctLab-main/logs/main_reference/g1_shadowing/20260826_171625_G1Shadowing_LafanFiltered_pgTermXYalso_independentMotionBins_fixFramerate_diveroll4_main_wholebody_4096_gpu3_retry1`;
  initial throughput 16.15k steps/s, collection 5.512 s, learning 0.576 s;
- physical GPU 4, main Perceptive:
  `/root/InstinctLab-main/logs/main_reference/g1_perceptive_shadowing/20260826_171734_g1Perceptive_concatMotionBins_main_perceptive_4096_gpu4_retry1`;
  initial throughput 9.23k steps/s, collection 6.872 s, learning 3.779 s.

Their continuously written stdout logs are
`/root/InstinctLab-main/logs/main_reference/train_main_wholebody_4096_gpu3_retry1.log`
and
`/root/InstinctLab-main/logs/main_reference/train_main_perceptive_4096_gpu4_retry1.log`.
The main Perceptive literal data placeholder is temporarily satisfied by the
runtime symlink `/root/InstinctLab-main/{AbsolutePathOfYourDataDirectory}` to
the released dataset root; retain it while this run is live. For comparison in
one TensorBoard invocation, the two run directories are linked as
`logs/tb_compare/g1_shadowing_diveroll/main_wholebody_4096_gpu3` and
`logs/tb_compare/g1_perceptive_shadowing/main_perceptive_4096_gpu4`.
Main Perceptive reported NaN values only in the position-monitor summaries at
iteration 0 (along with its existing empty-slice warning); rollout, reward,
loss, and optimization continued. Treat that monitor series cautiously while
comparing the curves.

The completed shadowing audit through `530b4b8` found and corrected the
following silent parity faults:

- Isaac randomized action offsets were written in native BFS articulation order
  into the canonical DFS policy vector. Explicit name mapping (`b986f24`) was
  the primary Whole Body curve correction.
- Shadow rewards used a legacy COM-velocity tensor instead of link-frame origin
  velocity, the critic had an extra projected-gravity term, reset velocities
  used the wrong backend frame, Isaac imported fixed joints differently, and
  support links were included in the illegal-contact penalty.
- A contact-sensor engine probe rebuilt Isaac body names inside every reward
  evaluation. Caching the static result removed the collection hot path;
  PhysX rigid patch guard queries were measured and were not the slowdown.
- The custom shadow velocity observation builders discarded declared history,
  noise, scale, and clipping. Perceptive critic input is now 1667, matching
  main, rather than the incorrect 1646.
- Perceptive motion height preprocessing now follows main on Isaac and
  InstinctMJ on MJLab; reference depth is clamped and normalized before
  crop/resize as in both references.
- Perceptive motion `terrain_id` metadata is now matched to compatible scene
  terrain origins at reset. Previously the independently sampled scene terrain
  was the largest remaining Perceptive semantic mismatch. Concatenated-motion
  reset random draws now also follow the reference batch order.
- Engine-specific simulator capacity profiles are restored. The corrected
  MJLab Perceptive run used about 23.44 GiB instead of 27.68 GiB, saving about
  4.24 GiB without imposing Isaac's PhysX capacities on MuJoCo.

The final 4096-environment Whole Body validation is:

```text
logs/isaacsim/g1_shadowing/20260826_184409_isaac_actionoffset_fixed_4096_gpu5_20260826
logs/train_isaacsim_wholebody_actionoffset_fixed_4096_gpu5_20260826.log
```

Its unified/main mean rewards at iterations 0, 10, 20, 40, and 100 were
respectively -1.63/-1.65, -0.96/-0.98, -0.48/-0.49, -0.23/-0.19, and
0.06/0.09. At iteration 100 it reported 3.314 s collection, 0.452 s learning,
and 26.1k steps/s. A separate 11-iteration run after the final contact fix is
`20260826_185021_isaac_fully_aligned_4096_gpu7_20260826` and retained the same
early behavior. These runs establish short-horizon parity, not long-horizon
convergence.

The strongest final Perceptive Isaac/main comparison is the terrain-matched
4096-environment run:

```text
logs/isaacsim/g1_perceptive_shadowing/20260826_192205_perceptive_terrainmatched_4096_gpu6_20260826
logs/train_isaacsim_perceptive_terrainmatched_4096_gpu6_20260826.log
```

At iteration 0, unified/main mean reward was -1.57/-1.58 and mean episode
length was 19.90/20.48. Illegal-contact, base-position, projected-gravity,
link-position, and dataset-exhaustion termination metrics were respectively
0.1218/0.1217, 0.0040/0.0047, 0.0397/0.0448, 0.0880/0.0875, and
0.0045/0.0044. At iteration 10, reward was -0.87/-0.79 and episode length
13.93/12.59, so the startup distribution is effectively aligned while longer
convergence remains unproven. The final sampling-order run is
`20260826_192749_perceptive_samplingaligned_4096_gpu6_20260826`; its statistical
iteration-10 difference did not improve, although its reset semantics now match
the references.

The corresponding corrected MJLab short validation is
`logs/mjlab/g1_perceptive_shadowing/20260826_192212_perceptive_terrainmatched_4096_gpu5_20260826`.
It has the correct 1667-input critic and completed 11 iterations, but no direct
InstinctMJ production curve was available on this server, so it is construction
and short-rollout evidence rather than convergence evidence.

TensorBoard comparisons are collected under
`logs/tb_compare/g1_shadowing_diveroll` on port 6006 and
`logs/tb_compare/g1_perceptive_shadowing` on port 6007. They include main,
pre-fix diagnostics, the final Isaac validations, and the corrected MJLab
Perceptive validation.

Final verification is 1190 passed, 3 skipped, 30 deselected, with one existing
NumPy warning in the parkour plant probe. `scripts/check_mjlab.py` exits 0 and
constructs and steps the flat locomotion task successfully.

## Current AMP correction

The current migration commit changes AMP timing to match main while preserving
the look-ahead contract:

- motion `data` remains `[t + dt, ...]` for aiming and exhaustion;
- cached `reference_frame` is sampled at current time `t`;
- both engine sensors expose it;
- AMP reference terms read `reference_frame`, not `data[:, 0]`.

The change passed 96 focused tests. With the production parkour clip,
look-ahead offset was 0.02 s, reference offset was 0, and direct current-time
joint sampling had maximum error 0. Existing checkpoints still load because
dimensions are unchanged, but their discriminator learned the old `t + 0.02`
distribution. Start new comparison training from scratch.

`scripts/diagnose_amp_rollout.py` is the deterministic cross-engine AMP probe.
It disables observation noise and records term and discriminator statistics.

## AMP findings

The same Isaac 11k actor and discriminator were replayed on both engines for
32 environments and 500 steps:

| Metric | Isaac | MJLab |
|---|---:|---:|
| Raw AMP reward | 0.7911 | 0.7955 |
| Actor discriminator logit | 0.0881 | 0.0977 |
| Reward after coefficient | 0.1978 | 0.1989 |

There is no fixed MJLab AMP formula/scale deficit. In paired training, action
noise diverged around iteration 210, well before the sustained AMP gap around
iteration 4730. The likely chain is native dynamics/contact/control divergence,
then policy-distribution divergence, then easier discriminator separation.
Independent learned discriminators do not provide a common absolute scale.

With identical explicit RNG generators, URDF and MJCF motion runtimes produced
identical root pose/velocity, joints, motion selection, start time, and mirror
mask; link FK differed by at most `7.2e-7`. Unconstrained engine runs consume
global RNG in different orders, so identical global seeds do not imply matching
per-environment motion or DR assignments.

A controlled action-noise A/B used the paired 11k checkpoints, 128 environments,
1,000 control steps (100 warm-up), seed 42, observation noise off, and an isolated
action generator. It compared policy-mean actions against
`mean + learned_std * epsilon` without consuming the environment RNG:

| Metric | Isaac mean | Isaac sampled | MJLab mean | MJLab sampled |
|---|---:|---:|---:|---:|
| Environment reward / step | 0.1426 | 0.1117 | 0.1232 | 0.0712 |
| Raw AMP reward | 0.7855 | 0.5880 | 0.8262 | 0.5269 |
| Action RMS | 0.4076 | 0.9952 | 0.4448 | 1.5070 |
| Done fraction | 0.001111 | 0.001111 | 0.001120 | 0.001189 |

The deterministic MJLab environment reward was 13.6% below Isaac, while the
sampled reward was 36.3% below. Additively, about 48% of the sampled cross-engine
gap came from the deterministic policy/plant gap and 52% from MJLab's excess
learned-noise damage in this rollout. The raw AMP ordering reversed: MJLab was
higher with mean actions and lower with sampled actions. Independent
discriminators are not an absolute cross-run ruler, but the within-checkpoint A/B
shows that the larger learned MJLab standard deviation is sufficient to explain
the AMP reversal. Reference logits stayed stable between each engine's two arms,
confirming that only actor sampling changed. Full JSON and stdout are under
`logs/diagnostics/amp_noise_ab_20260826/` (untracked).

The released parkour NPZ has 18,982 finite frames at 50 Hz and 29 named joints.
Its joint order is not canonical DFS, but the shared loader remaps it by name
correctly. Isaac's URDF and MJLab's MJCF paths then agree to numerical precision:
joint/root terms are identical, maximum link-position error is `7.63e-6`, and
maximum link-linear-velocity error is `3.82e-4`. The shared half-open resampling
contract produces 18,981 packed frames, matching both reference repositories.

There is nevertheless an unresolved data-semantic risk: the single NPZ appears
to concatenate motion segments without boundary metadata. Forward differencing,
which is also the behavior in InstinctLab-main and InstinctMJ, treats the jumps
as physical motion. Across the 18,981 transitions, 55 (`0.290%`) exceed a
conservative union of root translation, joint, or rotation discontinuity
thresholds. Observed maxima are `620.36 m/s` root linear velocity,
`156.73 rad/s` root angular velocity, and `88.10 rad/s` joint velocity. With a
10-frame AMP history, up to 533 frames (`2.81%`) can include one of these
transitions. Cross-engine parity is therefore verified, but semantic correctness
at internal clip boundaries is not. The preferred repair is to recover or
validate authoritative segment boundaries, split the release into independent
clips, and prevent sampling/history windows from crossing them. Merely clipping
or zeroing boundary velocity would leave pose teleportation in AMP look-ahead
windows.

## Alignment and boundaries

Verified or guarded:

- common task/agent schema and observation/action order;
- canonical DFS policy/motion order and Isaac BFS boundary mapping;
- shared AMP schema, scale, history, mirroring, and current-time reference;
- MJLab reference PD-actuator construction and motor-bus delay grouping;
- motion reset, exhaustion lifecycle, and timestamp refresh;
- sensor/virtual-terrain contracts and generic train/play entry points;
- pinned MJLab physics stack.

Intentional differences remain: PhysX versus MuJoCo solve/contact dynamics;
Isaac/main versus MJLab/InstinctMJ actuator semantics; engine-native contact
forces and joint accelerations; Isaac visual filtering versus MJLab geom groups;
and engine-specific RNG consumption order.

Not yet proven: final-commit long-horizon Whole Body and Perceptive convergence,
production perceptive VAE/HOI/BeyondMimic runs, direct final-commit InstinctMJ
curve comparison, and real multi-node distributed training.
Parkour and flat/rough locomotion are accepted as reproduced; their intentional
engine-native differences and any documented data risks are not treated as open
reproduction work.

Strict Perceptive live smokes on 2026-08-26 now complete one full 192-step
rollout and learning iteration on both engines. MJLab uses its reference-native
terrain-height query (geom group 0, 5 m range) instead of the general ray path;
Isaac selects the whole ray sensor by name instead of applying contact-sensor
body fields. The verified runs are:

```text
logs/mjlab/g1_perceptive_shadowing/20260826_162958_smoke_perceptive_no_vel_limiter_gpu2
logs/isaacsim/g1_perceptive_shadowing/20260826_162701_smoke_perceptive_fixed_gpu1
```

MJLab reported collection/learning times of 1.446/0.249 s; Isaac reported
2.453/0.272 s. The Isaac run ran concurrently on GPU 1 while the corrected
whole-body shadowing run remained live on GPU 0, confirming that waiting for
GPU 0 is unnecessary. The KVDB lock and missing `libGLU.so.1` warnings are
non-fatal for this headless ray-caster task.

MJLab no longer adds a custom joint-velocity-limiter actuator. Its generated G1
now contains only the seven grouped `BuiltinPdActuatorCfg` entries used by
InstinctMJ, and gain randomization again targets the whole robot as in the
reference. A strict Parkour smoke after this plant change completed one rollout
and learning iteration at:

```text
logs/mjlab/g1_parkour/20260826_163025_smoke_parkour_no_vel_limiter_gpu2
```

These are construction/rollout smoke results, not accepted Perceptive
production convergence evidence. Later audit results and the final verification
counts are recorded above.

## New-server bring-up

1. Clone this branch and verify a clean worktree.
2. Recreate the five sibling checkouts at the revisions above and preserve their
   external dirty diffs.
3. Copy datasets and selected logs/checkpoints; recreate symlinks.
4. Activate `env_isaaclab`, run `python scripts/install.py`, and verify versions.
5. Run:

   ```bash
   PYTHONPATH=source/instinctlab /root/miniconda3/envs/env_isaaclab/bin/python -m pytest -q tests
   python scripts/check_mjlab.py
   ```

6. Smoke construct and step one task on each engine.
7. Start new comparisons with equal seed, environment count, commit, dataset,
   and distinct run names.

Before interpreting reward, inspect episode length, action noise, termination
mix, AMP actor/reference logits, contact overflow, and per-terrain breakdown.
