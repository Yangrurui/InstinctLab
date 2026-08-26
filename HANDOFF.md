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
- MJLab actuator motor-bus delay grouping, effort/velocity safeguards;
- motion reset, exhaustion lifecycle, and timestamp refresh;
- sensor/virtual-terrain contracts and generic train/play entry points;
- pinned MJLab physics stack.

Intentional differences remain: PhysX versus MuJoCo solve/contact dynamics;
Isaac/main versus MJLab/InstinctMJ actuator semantics; engine-native contact
forces and joint accelerations; Isaac visual filtering versus MJLab geom groups;
and engine-specific RNG consumption order.

Not yet proven: matched long-horizon training after the AMP timing fix,
corrected long-horizon Isaac shadowing convergence, policy-quality equivalence,
production perceptive/HOI runs, and real multi-node distributed training.

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
