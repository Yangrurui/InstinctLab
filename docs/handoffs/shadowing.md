# Shadowing multi-engine handoff

Updated: 2026-08-26 UTC

This document covers only the shadowing task family and the shared infrastructure required by
shadowing. It is not a handoff for parkour, locomotion, generic motion training, virtual terrain,
pose-velocity, or play visualization work.

## Current repository state

- Repository: `/root/InstinctLab`
- Branch: `feat/unified-engine`
- HEAD when this handoff was written: `719710cafe19f888f4a71d3ce55b55639d400e36`
- Remote: `git@github.com:Yangrurui/InstinctLab.git`
- Isaac reference checkout: `/root/InstinctLab-main` at
  `ba28d3d2655b15a19b729476a630937a19610a3b`
- MJLab reference checkout: `/root/InstinctMJ` at
  `4ed2b32f8719ff9fc138708341031e935afda0d2`
- Python environment used during implementation and validation: `env_isaaclab`

At handoff time `scripts/diagnose_amp_rollout.py` is untracked user work. It is not part of the
shadowing commits and must not be discarded, staged with this document, or assumed to migrate via
Git.

## Outcome

Shadowing is now declared once as a shared `TaskSpec` and compiled through the Isaac Sim and MJLab
engine adapters. The old Isaac-only environment classes, task-local MDP copies, Gym-only launch
surface, task-local play script, CLI helpers, and grid-search script were removed rather than kept
as forwarding modules. Generic `scripts/train.py` and `scripts/play.py` consume the same registered
task contract.

The registry contains twelve task IDs:

- `Instinct-Shadowing-WholeBody-Plane-G1-v0`
- `Instinct-Shadowing-WholeBody-Plane-G1-Play-v0`
- `Instinct-Perceptive-Shadowing-G1-v0`
- `Instinct-Perceptive-Shadowing-G1-Play-v0`
- `Instinct-Perceptive-Shadowing-G1-OneMotion-v0`
- `Instinct-Perceptive-Shadowing-G1-OneMotion-Play-v0`
- `Instinct-Perceptive-Vae-G1-v0`
- `Instinct-Perceptive-Vae-G1-Play-v0`
- `Instinct-Perceptive-HOI-Shadowing-G1-v0`
- `Instinct-Perceptive-HOI-Shadowing-G1-Play-v0`
- `Instinct-BeyondMimic-Plane-G1-v0`
- `Instinct-BeyondMimic-Plane-G1-Play-v0`

The OneMotion pair comes from InstinctMJ; main has no corresponding registration.

## Commit ledger

The requested staged rewrite was committed in task-sized changes:

| Phase | Commit | Result |
|---|---|---|
| Reference audit | `8921140` | Audited effective main and InstinctMJ factory/registration output and recorded real reference differences. |
| Task structure | `20896c5` | Replaced Isaac-only shadowing trees with shared declarations and removed legacy entry points/MDP copies. |
| Robot/joint/action | `a343a88` | Established canonical G1 DFS policy order, native order mapping, actuator profiles, and checkpoint schema guards. |
| Motion runtime | `0066bc3` | Added the shared clip loader, named remapping, FK, interpolation, velocity calculation, sampling, exhaustion, mirroring, and runtime buffers. |
| Sensors and MDP | `8f7e587` | Lowered shared commands, observations, rewards, events, terminations, curriculum, contact/ray/camera contracts, and engine-native terms. |
| Training lifecycle | `f85e157` | Exercised registration through runner/checkpoint/play/export and added distributed/resume guards. |
| Lifecycle hardening | `f330a98` | Hardened resource ownership, manifests, diagnostics, live probes, entry points, and failure cleanup. |
| Reset-order correction | `49f6503` | Fixed canonical DFS motion state being written positionally into Isaac's native BFS joints. |
| Checkpoint follow-up | `719710c` | Stopped rejecting same-task checkpoints solely because the serialized declaration hash changed; task ID and contract version remain gates. |

`eaf9a9e` is mainly a later MJLab/parkour actuator-safety change, but it touches shared MJLab
actuator code and is part of the current branch baseline. Do not cherry-pick only the shadowing
commits onto an older actuator stack without re-running the shadowing tests.

## Architectural entry points

- Shared declaration: `source/instinctlab/instinctlab/tasks/shadowing/task_spec.py`
- Task registry: `source/instinctlab/instinctlab/tasks/registry.py`
- Shared motion contract: `source/instinctlab/instinctlab/spec/motion_reference.py`
- Shared motion runtime: `source/instinctlab/instinctlab/engines/motion_reference/`
- Shared reset/events: `source/instinctlab/instinctlab/engines/shadowing_events.py`
- Shared commands: `source/instinctlab/instinctlab/engines/shadowing_commands.py`
- Shared MDP functions: `source/instinctlab/instinctlab/mdp/shadowing.py`
- Isaac lowerings: `source/instinctlab/instinctlab/engines/isaacsim/`
- MJLab lowerings: `source/instinctlab/instinctlab/engines/mjlab/`
- Fixed-input probe: `source/instinctlab/instinctlab/shadowing_probe.py`
- Probe launch/comparison: `scripts/probe_shadowing_rollout.py` and
  `scripts/compare_shadowing_rollouts.py`
- Reference audit: `docs/shadowing_reference_audit.md`
- Training lifecycle notes: `docs/shadowing_training_flow.md`

## Shared task contract

The whole-body baseline uses:

- physics step `0.005 s`, decimation `4`, policy step `0.02 s`;
- episode limit `10 s`, therefore at most 500 policy steps from the time limit;
- canonical `G1_29DOF_DFS_JOINT_NAMES` for policy actions, joint observations, last action, AMP,
  motion buffers, symmetry maps, and checkpoint schema;
- explicit joint selectors with `preserve_order=True`;
- motion quaternions in `wxyz`;
- target clip rate `50 Hz` and `frontbackward` velocity estimation;
- `freeze_last_and_flag` dataset exhaustion;
- five reference commands in stable declaration order;
- shared reward, event, termination, and curriculum declaration order.

Reference population remains engine-profiled: whole-body uses 4096 environments on Isaac and 2048
on MJLab when no CLI override is supplied. A controlled cross-engine experiment should explicitly
set the same `--num_envs` on both sides.

## Production diveroll dataset

The downloaded whole-body binding is:

```text
/root/Datasets/NoKov-Marslab-Motions-instinctnpz/20251016_diveroll4_single
```

It is a symlink to:

```text
/root/Datasets/deep_whole_body_parkour_g1_release/
  20251116_50cm_kneeClimbStep1/20251106_diveroll4_roadRamp_noWall
```

The effective clip is `diveroll4-ziwen-0-retargeted.npz`. Its source inventory is 751 frames,
29 joints, and 90 Hz. The shared runtime remaps the file's named joint columns to canonical DFS and
resamples it to 417 frames at 50 Hz. The file itself does not need to be rewritten. Copy the symlink
target, not only the symlink, when moving servers.

The task still declares different source paths through engine profiles so it matches each reference
layout. On a new machine, either recreate those paths/symlinks or provide an explicit diagnostic
motion override; do not silently add a fallback dataset in library code.

## Critical reset-order bug and fix

The first long comparison exposed a unification bug. The shared motion runtime stores joints in
canonical DFS. MJLab's natural order is also DFS, but Isaac's PhysX articulation is BFS. The reset
event originally wrote the complete canonical vector without `joint_ids`, so Isaac interpreted DFS
values as BFS slots.

At a fixed, non-randomized 0.25 s sample before the fix:

- MJLab motion FK versus simulated links had maximum 3D error `1.23e-7 m`;
- Isaac had `0.348 m` at the left ankle, `0.299 m` at the right ankle, and `0.253 m` at the left
  wrist; maximum height error was `0.176 m` at the left wrist.

Main does not have this inconsistency: its motion loader first reorders clip columns to the Isaac
articulation's native `dof_names`, so its positional reset is internally consistent. The unified
runtime deliberately changed the buffer contract to canonical DFS but initially retained main's
positional write.

Commit `49f6503` fixes the boundary without engine conditionals. Both motion sensors expose their
canonical `joint_names`; the shared reset resolves those names with `preserve_order=True` and passes
the resulting entity-local `joint_ids` to the native write. The fixed-input probe now also gathers
robot state by canonical name and records the schema, rather than comparing raw native arrays.

## Test evidence

After `49f6503`, the focused non-live suite passed:

```text
89 passed
```

It covered joint/reset ordering, shared declaration, engine compilation, MDP contract, motion
runtime, training flow, and MJLab action ordering. Pre-commit checks also passed.

With the production diveroll clip, two-environment, four-step live tests passed on both engines:

```text
tests/test_shadowing_mjlab_live.py       1 passed
tests/test_shadowing_isaacsim_live.py    1 passed
```

The live tests assert that reset robot joint positions, gathered by canonical name, equal the
motion-reference joint positions. They do not merely check tensor shapes.

Important: the checked-in files under `docs/evidence/shadowing_rollout_*_seed2026.*` were generated
before `49f6503`. Their raw joint-array equality was a false positive because the same numbers had
different native joint meanings. They are historical artifacts and must not be cited as post-fix
cross-engine parity evidence. `docs/shadowing_training_flow.md` also still says the production
dataset was unavailable; that statement is stale. Regenerate the evidence on the new server before
updating either document.

## Long-training evidence and interpretation

Runs made with the production diveroll clip are stored locally at:

```text
logs/mjlab/g1_shadowing/20260825_202243_released_diveroll_gpu0
logs/isaacsim/g1_shadowing/20260825_202305_released_diveroll_gpu1
```

MJLab completed 5000 iterations. At iteration 4990 it reported mean reward `13.3575`, mean episode
length `235.71`, median `260.5`, and batch maximum `407`. The highest logged mean episode length was
`267.80` at iteration 4480.

Isaac stopped after iteration 3940 and only has checkpoints through iteration 3000. At iteration
3940 it reported mean reward `1.5388`, mean episode length `34.03`, median `1`, and batch maximum
`200`; its highest logged mean episode length was `63.42` at iteration 3200.

The Isaac run used the broken pre-`49f6503` reset and is not valid evidence of engine convergence or
parity. Its persistent one-step episodes are consistent with the measured reset FK corruption. Do
not resume that Isaac checkpoint for a corrected comparison. The MJLab reset mapping is identity,
so the bug did not change MJLab's physical reset; its run remains useful as an MJLab-only baseline,
but a clean paired experiment is still preferable.

TensorBoard termination scalars have different native aggregation conventions and their absolute
magnitudes must not be directly compared across engines. Episode length, name-aligned state probes,
and per-engine reference tests are safer evidence.

## Verified alignment

The following have static tests and/or live runtime evidence:

- a single shared task declaration compiles strictly on Isaac Sim and MJLab;
- stable train/play task IDs and common generic launchers;
- G1 canonical DFS policy, observation, AMP, motion, symmetry, and checkpoint order;
- Isaac BFS mapping and MJLab natural DFS mapping;
- action targets preserve the declared order;
- reset writes canonical joint values to native joint IDs exactly once;
- named motion-file remapping, `wxyz` quaternions, 50 Hz interpolation, front/backward velocity,
  FK link order, start sampling, update timestamps, mirroring, and exhaustion flags;
- declared observation history, command order, rewards, events, terminations, curriculum, contact
  layout, ray scanner, camera layout, and engine-specific lowerings;
- runner construction, checkpoint/manifest creation, resume state restoration, play contract
  validation, ONNX export, rank/device mapping, and rank-distinct seeds;
- short reset/step lifecycle on both installed simulators with the production clip.

## Intentional engine differences

These differences come from the two effective references and should not be erased by numerical
compatibility patches:

| Area | Isaac/main semantics | MJLab/InstinctMJ semantics |
|---|---|---|
| Native joint enumeration | PhysX breadth-first | MJCF/entity natural depth-first |
| Actuation | Isaac ImplicitPD selected by final G1 override | MuJoCo `BuiltinPd`, MJ stiffness/damping/armature/effort |
| Contacts | PhysX body contact sensor/history | MuJoCo body/geom matching and MJLab sensor layout |
| Actuator effort | Isaac applied torque | MuJoCo `qfrc_actuator` |
| Acceleration | PhysX articulation joint acceleration | MJ native joint acceleration |
| Link velocity | Isaac COM/body representations where configured | MJ root/link-frame representation where configured |
| Physics/contact solve | PhysX solver | MuJoCo/MuJoCo-Warp solver (`iterations=10`, `ls_iterations=20`) |
| Ray/render implementation | Isaac USD/tiled sensors | MJCF/MJLab ray sensors |
| Default whole-body env count | 4096 | 2048 |

Contacts, torque, acceleration, and COM-versus-link quantities must match their own reference
implementation. They are not expected to agree value-for-value between engines.

## What is not yet proven

- Post-fix long-horizon Isaac convergence has not been measured.
- A clean paired post-fix 5000-iteration run with equal environment count and seed has not been run.
- Post-fix rollout evidence files have not been regenerated.
- Short live tests do not prove return distribution or final policy equivalence.
- Multi-node distributed training has contract tests but no real multi-node load run.
- Resume intentionally does not restore simulator episode state, command RNG, or motion-runtime
  buffers because neither reference checkpoint stores them; it restores learning state and starts
  fresh environments.
- Production perceptive terrain/CoACD, camera values on those meshes, HOI object interactions, and
  every task family have not all received long training runs.
- Policy quality still needs playback/export inspection after corrected training.

Do not claim full Isaac/MJLab numerical alignment until these items have runtime evidence.

## New-server bring-up

Install the pinned stack. The repository pins MJLab `1.5.0`, MuJoCo `3.10.0`, MuJoCo-Warp
`3.10.0.1`, Warp `1.14.0`, and IsaacLab commit `f73c331738`. The validated environment also used
Torch `2.7.0`.

Before using a GPU, inspect current jobs:

```bash
nvidia-smi
ps -eo pid,etime,stat,cmd | rg 'train.py|python'
```

Run the focused CPU/static suite:

```bash
conda run -n env_isaaclab python -m pytest -q \
  tests/test_shadowing_joint_action_contract.py \
  tests/test_shadowing_training_flow.py \
  tests/test_shadowing_mdp_contract.py \
  tests/test_shadowing_motion_runtime.py \
  tests/test_engines_compile.py \
  tests/test_mjlab_action_order.py
```

Run live tests only on a confirmed idle GPU:

```bash
export INSTINCTLAB_LIVE_DEVICE=cuda:IDLE_GPU
export INSTINCTLAB_SHADOWING_LIVE_MOTION=/root/Datasets/NoKov-Marslab-Motions-instinctnpz/20251016_diveroll4_single/diveroll4-ziwen-0-retargeted.npz

conda run -n env_isaaclab python -m pytest -o addopts= -m mjlab -q \
  tests/test_shadowing_mjlab_live.py -s
conda run -n env_isaaclab python -m pytest -o addopts= -m isaacsim -q \
  tests/test_shadowing_isaacsim_live.py -s
```

Then regenerate fixed-seed rollout evidence with `scripts/probe_shadowing_rollout.py` for both
engines and compare it with `scripts/compare_shadowing_rollouts.py`. Verify that the new payload
contains `motion_joint_pos` and metadata `joint_names`; their absence identifies a pre-fix artifact.

For the next training comparison, use the same clip, seed, environment count, initial state policy,
and iteration budget on two idle GPUs. Do not reuse the pre-fix Isaac checkpoints. Preserve native
physics semantics and compare learning curves plus name-aligned reset/short-rollout evidence, not
raw native joint arrays or absolute termination aggregates.
