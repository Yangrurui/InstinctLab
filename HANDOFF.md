# InstinctLab current handoff

Updated: 2026-08-27 02:19 UTC

This is the authoritative record for the current repository, server, datasets,
live experiments, accepted baselines, and unresolved work. Historical audit
narratives are in Git history rather than duplicated here.

## Repository

- Repository: `/root/InstinctLab`
- Branch: `feat/unified-engine`
- Current architecture cleanup: `27d9767`
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

- Concrete Shadowing values and public factories are together in
  `tasks/shadowing/config.py`; the old forwarding config files are gone.
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
| `/root/instinct_rl` | `ba45ed231ebbf0a4099cd31d607e2886814fd165` | RL runner |

Uncommitted external changes that cloning upstream will lose:

- `/root/InstinctMJ`: terrain debug visualization is conditional on
  `debug_vis`; play maps the selected CUDA device to EGL and Warp before
  construction.
- `/root/instinct_rl`: WASABI reports discriminator sign accuracy for actor
  target `-1` and reference target `+1`. This is diagnostic only.

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

## Live experiments

Snapshot at 2026-08-27 02:19 UTC. These processes were inspected only; none was
stopped, restarted, or signaled.

| GPU | Run | Iteration | Reward | Episode length | Status |
|---:|---|---:|---:|---:|---|
| 1 | old unified Isaac Perceptive `perceptive_repro_4096_gpu1` | 5040 | 2.53 | 46.08 | live; predates final parity fixes, do not promote |
| 3 | main Isaac Whole Body reference | 16390 | 17.98 | 257.82 | live |
| 4 | main Isaac Perceptive reference | 7670 | 8.62 | 171.42 | live |
| 5 | unified Isaac Whole Body `final_long_4096_gpu5_20260826` | 13010 | 17.91 | 264.61 | live |
| 6 | unified Isaac Perceptive `final_long_4096_gpu6_20260826` | 5520 | 2.53 | 56.29 | live |
| 7 | unified MJLab Perceptive `stablecaps_final_long_4096_gpu7_20260826` | 6600 | 13.23 | 252.07 | live |

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
