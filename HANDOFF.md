# InstinctLab current handoff

Updated: 2026-08-31 UTC

This is the authoritative current-state record for the repository, server,
datasets, retained experiments, accepted evidence, and unresolved work. Git
history owns closed audit narratives and commit-by-commit provenance.

When a review changes the current verdict, update the relevant section here in
the same turn. Record the finding, severity, evidence, and re-acceptance
condition, but replace superseded conclusions instead of appending another
chronological audit section.

## Repository state

- Repository: `/root/InstinctLab`
- Branch: `feat/unified-engine`
- Latest verified production-code commit: `330d783`
- Local `origin`: `git@github.com:Yangrurui/InstinctLab.git`
- Export repository: `git@github.com:Yangrurui/XLab.git`; `main` was synced
  through `348a73d`. Later local commits still need an explicit push.

The worktree retains the pre-existing user deletion of
`source/instinctlab/instinctlab/tasks/parkour/tools/convert_urdf.py`. Do not
discard, overwrite, or include it in an unrelated commit.

## Current architecture

InstinctLab is a modular monolith with independently packaged engine adapters
and a compiler-style neutral contract:

```text
asset plugin -> selected backend -> engine-core RobotSpec
task config ---------------------> engine-core TaskSpec
                                    -> validation/preflight/compiler
                                    -> selected backend -> native environment
                                    -> common runner and readable run manifest
```

The central design is sound and should be retained:

- Every active task is one `TaskRegistration` in
  `source/instinctlab/instinctlab/tasks/registry.py`. Registry factories only
  instantiate a complete config and convert it to an immutable `TaskSpec`.
- Locomotion and Parkour keep robot-independent base configs outside robot
  directories. Concrete G1 configs inherit once and state complete selectors,
  values, datasets, assets, and runner choices directly.
- Every task family owns its complete `mdp/` implementation. There is no global
  MDP facade, central variant dispatcher, or task-specific policy in an engine.
- `instinctlab-engine-core`, `instinctlab-engine-isaacsim`, and
  `instinctlab-engine-mjlab` are separate wheels. Tasks do not import engines;
  engines do not import tasks, concrete asset packages, or one another.
- `TaskSpec` and `RobotSpec` contain stable engine-neutral contracts, not native
  actuator parameters, reward weights, solver values, or runner settings.
- Canonical G1 joint order is the explicit 29-joint DFS order in `RobotSpec`.
  Isaac's articulation remains native BFS; every boundary resolves names and
  gathers/scatters explicitly. MJLab is naturally DFS.
- G1 native assets and every final actuator parameter remain explicit in
  `assets/unitree_g1/isaacsim.py` and `assets/unitree_g1/mjlab.py`. The neutral
  asset interface routes only `package/variant` ids.
- Backends, assets, actuator models, native terms, native sensors, whole
  terrains, and generated-terrain tiles use transactional lazy registries.
  Discovery is SDK-free until the selected backend bootstraps and provenance is
  recorded for providers actually used by a compilation.
- Production train/play compilation is strict and clean by default. Omissions
  or emulation require an explicit override and remain visible in the manifest.
- Checkpoint loading is deliberately not hash-driven. Manifests contain readable
  task, effective agent, source, package, hardware, and dataset information;
  explicit format versions and the runner's strict tensor loading own
  compatibility.
- `scripts/train.py` and `scripts/play.py` select and bootstrap a backend before
  importing its SDK. Playback handlers are application-owned, not adapter APIs.
- Shared semantic meanings live in `spec/`; motion state lives in
  `motion_reference/`; narrow native-value readers live in `bridge/` and
  `compat/`. Native sensor, terrain, solver, and SDK lifecycle code stays under
  the selected backend package.

## Platform maturity

Overall verdict: stock Locomotion, Parkour, and Shadowing paths have strong
contract and construction coverage, and the external extension wheel proves
real native integration on both engines. The main layering is mature, but the
third-party simulation surface is not yet a stable 1.0 protocol.

| Capability | Current state | Remaining boundary |
|---|---|---|
| Backend | Independent lazy plugins; isolated wheel matrices pass | Publication/release process |
| Task | Immutable application-owned registrations | External task catalogs are optional, not a current goal |
| Robot asset | Versioned native API and conformance command | Broader object/articulation catalogs |
| Actuator | Lazy native model registry, explicit groups, live stiffness bindings, device-native reward ids, composed gain/reward external gate | Broader controller clocks |
| Terrain | Whole-terrain and tile plugins | Region semantics; dynamic/deformable terrain only on demand |
| Sensor | Built-ins plus lazy native builders with timing/reset capabilities | New sensor families need fixed-state and temporal evidence |
| Rigid object | Mesh-backed construction on both engines | Full HOI resources and behavioral evidence |
| Collision relation | Portable entity-pair exclusions lower to both engines | Broader materials and constraints |
| Multiple articulations | One canonical primary `TaskSpec.robot` | Canonical schema and entity-targeted actions/observations for additional articulations |
| Lifecycle | Physics/policy steps and selected sensor periods exist | Named clocks, timestamps, latency, and component reset semantics |
| Record/replay | Diagnostics and structured run manifests exist | Episode traces and native state snapshot/restore |

### Actuator protocol acceptance

The general actuator protocol is accepted through `f180406` and its fail-closed
factory paths through `330d783`. The committed boundary provides:

- selected-joint-aware mixed actuator capability resolution;
- one symbol namespace for robot, terrain, objects, sensors, and backend fields;
- exact Isaac scene-field tracking including `clone_in_fabric`;
- native group-to-model identity and strict adapter return-shape checks;
- a repository-external fixture that constructs real Isaac
  `DelayedPDActuator` and MJLab `BuiltinPdActuator` subclasses and exercises
  action, gain randomization, delay, full reset, and partial reset;
- a class-based motor-power term that resolves actuator ownership, adapter
  identity, selected positions, and device-native ids once during manager-term
  initialization, while reading current native stiffness during each reward
  call using device-side tensor work;
- strict joint-id validation that accepts genuine integral values and integer
  tensors, rejects floats, strings, booleans, and non-integral tensor dtypes,
  and fails closed for duplicate, foreign, missing, or overlapping groups;
- the original public Python-id `joint_stiffness_groups()` diagnostic contract
  plus a separate device-native initialization interface, so external probes do
  not inherit the hot-path representation.

Both manager lifecycles are covered: MJLab instantiates the portable class term
directly; Isaac wraps it in `ManagerTermBase` and falls back to the native no-op
reset when the portable term owns no reset state. Fixed-state tests preserve
selected/unselected mixed-group results plus scalar and per-environment
stiffness broadcasting. The dynamic partial-group regression initializes a term
at stiffness 2, changes the live native value to 4, and observes the reward
change from 4 to 1. The selected-SDK MJLab CUDA version changes a native gain
after initialization and observes 9 to 2.25 with synchronization treated as an
error. The external real task now combines startup gain randomization and the
stiffness-normalized reward in the same environment on both engines, closing
the supported-composition gap rather than relying on the earlier SDK-free
stand-in.

Native factory identity checks also fail closed. A conflicting model id and a
config object that cannot retain `instinctlab_model_id` now raise the public
`ActuatorContractError`; the native assignment cause is preserved where
applicable, and regressions cover both paths. This replaces the prior
undefined-name failure that could hide the actual extension contract violation.

## Accepted verification snapshot

The latest committed platform evidence is:

```text
1415 passed, 3 skipped, 30 deselected, 1 warning (full tests/ suite)
180 passed, 1 deselected (actuator, Parkour, spec, preflight, asset, and
  architecture focus)
Parkour contract subset: 1 passed, 24 deselected
selected-SDK MJLab CUDA sync-debug: partial-group native stiffness changed after
  class-term initialization and motor-power changed from 9 to 2.25 with
  synchronization treated as an error
all registered task/engine pairs passed preflight with absent HOI resources
  supplied by local fixtures
Isaac Parkour: 16 CUDA environments constructed, reset, and stepped; terrain,
  depth, foot scanners, VolumePoints, observation shapes, DFS policy axis,
  AMP, and finite rewards passed
MJLab Parkour: 16 CUDA environments constructed, repeatedly reset, and stepped;
  the first policy depth stack, terrain, foot scanners, VolumePoints,
  observation shapes, DFS policy axis, AMP, and finite rewards passed
MJLab Flat G1: 16 environments constructed, reset, and stepped five times;
  all 39 terms resolved, DFS action order, finite reward, zero terminations
core-only, Isaac-only, MJLab-only, and dual-backend isolated wheels passed
external fixture installed, passed SDK-free probes, constructed two real
  environments per engine, combined startup gain randomization with a live
  stiffness-normalized reward, passed native action/delay/reset checks, then
  uninstalled without breaking built-in backends
G1 asset conformance passed on both engines
Isaac collision exclusions passed cloned-target validation and a
  4,096-environment, five-step capacity probe
```

These results establish declaration, compilation, construction, and targeted
runtime behavior. They do not establish long-run convergence or native-physics
equality. Repository-wide Ruff is not currently a green gate; the last full
run reported 709 existing/current errors. Ruff, compileall, and diff checks pass
for the actuator remediation files.

The MJLab Parkour first-depth P1 is closed by `d15ea14`. MJLab's observation
manager samples class terms during construction, then the first environment
reset senses a valid image without the `scene.update()` call that increments
the native camera epoch. The cleared depth ring therefore saw the same epoch as
the construction sample and previously stayed zero. Parkour and Shadowing now
prime only cleared environments from the current valid image independently of
global epoch advancement, without rolling the ring or changing continuing
environments. Fixed-state partial-reset regressions pin the unchanged-epoch
case. The full 16-environment native Parkour tests passed afterward on MJLab
(`1 passed` in 248.05 s) and Isaac (`1 passed` in 172.40 s), restoring the
MJLab Parkour live path to accepted status. This audit independently repeated
the complete native paths on CUDA 3: MJLab passed in 251.95 s and Isaac passed
in 168.87 s, including reward construction and finite reward steps.
After the live-stiffness change, the complete paths passed again on CUDA 2:
MJLab in 244.28 s and Isaac in 170.76 s.

Isaac's optional Iray loader reports missing `libGLU.so.1` on this server. The
headless physics probes continue successfully, so this is not a current startup
blocker.

## External checkouts and runtime stack

These sibling checkouts are not submodules:

| Checkout | Revision | Purpose |
|---|---|---|
| `/root/InstinctLab-main` | `ba28d3d2655b15a19b729476a630937a19610a3b` | Isaac/main reference |
| `/root/InstinctMJ` | `4ed2b32f8719ff9fc138708341031e935afda0d2` | MJLab reference |
| `/root/IsaacLab` | `f73c33173801f5f8afea4142482e47b7710c2b75` | Isaac Lab dependency |
| `/root/mjlab` | `08090e8a77228e733373f3b5c54f8b5a68d19d9d` | MJLab dependency |
| `/root/instinct_rl` | `64d7e01` (detached HEAD) | RL runner and batched rollout logging |

Uncommitted `/root/InstinctMJ` changes make terrain debug visualization depend
on `debug_vis` and map the selected CUDA device to EGL/Warp before play
construction. Commit, export, or reapply them before leaving this server.

Use `/root/miniconda3/envs/env_isaaclab/bin/python` (Python 3.11):

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

The MuJoCo-Warp/Warp pins are intentional. The newer
`mujoco-warp==3.10.0.3` / `warp-lang==1.16.0` stack changed post-contact
dynamics materially. This pinned MuJoCo-Warp does not expose `Data.overflow`;
monitor Warp warnings and native contact/constraint budgets in production.

## Datasets

### Parkour and Whole Body

Required Parkour compatibility links:

```text
/root/Datasets/parkour_motion_without_run.yaml
  -> /root/Datasets/parkour_release/parkour_motion_reference/parkour_motion_without_run.yaml
/root/Datasets/parkour_motion_without_run_retargetted.npz
  -> /root/Datasets/parkour_release/parkour_motion_reference/parkour_motion_without_run_retargetted.npz
```

Released checksums:

```text
parkour_motion_without_run.yaml
  f79e5bbc9207976e1610459ab3727a9e1da6d5c0c6cc75793dcec34b81cb7679
parkour_motion_without_run_retargetted.npz
  7cfb7c1dcaa6f2a55a13c4849be9e17b4c960ce4015c500ac0ddfb9d77f4ba5b
```

Active Whole Body clip:

```text
/root/Datasets/NoKov-Marslab-Motions-instinctnpz/20251016_diveroll4_single
  -> /root/Datasets/deep_whole_body_parkour_g1_release/
     20251116_50cm_kneeClimbStep1/20251106_diveroll4_roadRamp_noWall
diveroll4-ziwen-0-retargeted.npz
  751 frames, 29 joints
  SHA-256 8274d93046811824640ad373bba13ecd46ed347af8cc6d3d7c116df35a1bec59
```

### BeyondMimic

Official dataset `lvhaidong/LAFAN1_Retargeting_Dataset` is pinned at revision
`ce1572906efe6157840e8474d5a0d7aa87481e74`:

```text
/root/Datasets/LAFAN1_Retargeting_Dataset
  40 G1 CSV clips, 264,705 frames
/root/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz
  40 converted NPZ clips plus conversion_manifest.json
/root/Xyk/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz
  -> the directory above
```

Selected production clip hashes:

```text
sprint1_subject2.csv
  7babbd9d0a3cebf040709cb75fbf4268e925e337a2d44600dcce3d3b2d24a818
sprint1_subject2_retargetted.npz
  f1b1236d13f3f4d695ffb1b6ea8e7faf64363c419f7660336a4bd41da2bb7b55
```

Regenerate with `scripts/lafan1_csv_to_instinct.py`; the BeyondMimic README
owns the exact command and format contract.

### Perceptive VAE and HOI

Canonical VAE dataset:

```text
/root/Datasets/deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1
10 motions, 6 terrain meshes, 8,816 declared frames
```

The diagnostic teacher bundle is
`/root/Datasets/g1_perceptive_vae_teacher/mjlab_gpu7_iter22000`; checkpoint
SHA-256 is
`0c8c6a7a09cbc037e45f6bb2a36867e5400dfb325a30817932df6639a2e197da`.
Its manifest intentionally says `accepted_for_production: false` because it
predates final camera and event-randomization fixes.

HOI remains resource-blocked: both OMOMO motion directories and all six object
meshes are absent. Do not substitute other motions or objects silently.

## Reproduction status

| Workload | Current verdict |
|---|---|
| Locomotion Flat | Accepted on both engines |
| Locomotion Rough | Construction/stepping accepted; post-unification convergence pending |
| Parkour | Declarations and pre-terrain reproduction accepted on both engines; current terrain/sensor live construction passes; post-unification convergence pending |
| Whole Body plane Shadowing | Short-horizon parity accepted; retained long-run final output still needs audit |
| Perceptive Shadowing | No accepted post-fix long convergence; retained run ended and needs final-log audit plus a collision-filter A/B |
| BeyondMimic | Official-data L7, 256 environments, seed 42, 700 iterations accepted on both engines; multi-seed and 4,096-environment baselines not promoted |
| Perceptive VAE | Canonical data and two-iteration training-chain smoke passed on both engines; no accepted production reproduction |
| Perceptive HOI | Declarations/fixed-state coverage only; motions and meshes missing |
| Multi-node training | Not validated |

Play variants consume their paired Train checkpoints and do not need separate
training reproductions. Construction smoke is never convergence evidence.

## Retained runs and diagnostics

Process check at 2026-08-31 01:52 UTC found no `scripts/train.py` process; all
GPUs reported 1 MiB used and zero utilization. The runs below are retained
outputs, not active jobs. Do not restart them without explicit operator
approval.

Accepted BeyondMimic L7 seed-42 runs:

```text
logs/isaacsim/g1_beyondmimic/
  20260827_214841_official_lafan1_256_seed42_700_gpu0_retry1_20260827
logs/mjlab/g1_beyondmimic/
  20260827_214821_official_lafan1_256_seed42_700_gpu1_retry1_20260827
```

Unpromoted production/capacity runs:

```text
logs/isaacsim/g1_beyondmimic/
  20260828_094446_official_lafan1_production_4096_seed42_30000_gpu0_20260828
logs/mjlab/g1_beyondmimic/
  20260828_094428_official_lafan1_production_4096_seed42_30000_gpu1_20260828
```

These processes predate portable termination metric units; do not compare raw
termination tags across engines without normalizing or regenerating them.

Current Shadowing logs needing final audit:

```text
logs/isaacsim/g1_shadowing/
  20260827_153315_jointref_fixed_final_long_4096_gpu5_20260827
logs/isaacsim/g1_perceptive_shadowing/
  20260827_151020_jointref_fixed_final_long_4096_gpu6_20260827
```

VAE smoke runs:

```text
logs/mjlab/g1_perceptive_vae/
  20260828_111846_canonicaldata_mjteacher22000_smoke_retry2_16_seed123_gpu4_20260828
logs/isaacsim/g1_perceptive_vae/
  20260828_113341_canonicaldata_mjteacher22000_smoke_isaac_16_seed123_gpu4_20260828
```

Important historical failure: MJLab Whole Body
`finalaligned_datafixed_4096_gpu0_20260826` ended at iteration 17,300 with CUDA
719 after otherwise finite metrics. The reported Torch `nonzero()` was only a
synchronization point; no Xid/kernel log identifies the failing kernel. Keep
the log and checkpoint, classify it as an unreproduced asynchronous
Warp/MuJoCo/device failure, and do not call it a completed run.

TensorBoard comparisons remain under:

```text
logs/tb_compare/g1_shadowing_diveroll/
logs/tb_compare/g1_perceptive_shadowing/
logs/tb_compare/g1_beyondmimic_production_4096_seed42/
```

### Perceptive reset-contact diagnosis

The retained Isaac `model_6000.pt` fixed-bin probe showed that the stall is not
a DFS/BFS or joint-initialization fault. At bin 12, Isaac's non-support contact
force median was 5,969 N versus MJLab's 209 N. The MJCF excludes pelvis/hip-roll
and elbow/wrist-pitch self-collision pairs; the URDF did not. Global Isaac
self-collision disablement raised mean first-episode length from 12.58 to 50.79
steps, close to MJLab's 48.46, but it is broader than an acceptable fix.

The four narrow MJCF-equivalent exclusions now lower portably and passed cloned
relationship checks plus a 4,096-environment PhysX capacity probe. Training
behavior still needs a fresh A/B. Diagnostic reports and the reusable probe are:

```text
logs/diagnostics/perceptive_reset_bin12_model6000/
scripts/probe_perceptive_reset.py
logs/diagnostics/perceptive_collision_exclusions_4096_20260830.json
```

## Open work

1. **Complete platform lifecycle work before a 1.0 claim:** named clock domains,
   component timing/reset semantics, episode trace and replay, same-engine state
   snapshots, canonical additional-articulation schemas, stateful controller
   contracts, and construction/throughput/memory/reset benchmarks. Define a
   multi-agent API only when an actual multi-agent task establishes its policy,
   reward, termination, shared-state, and partial-reset needs.

Product-dependent work such as ROS 2/HIL, deformables, fluids, dynamic terrain,
visual-domain libraries, and hot service orchestration is not a current release
gate; add it only for a concrete task or deployment requirement.

## Bring-up and verification

Before starting any run, inspect existing training processes and GPU state, and
choose a new log directory:

```bash
pgrep -af scripts/train.py
nvidia-smi
```

Standard checks:

```bash
python scripts/install.py
PYTHONPATH=source/instinctlab \
  /root/miniconda3/envs/env_isaaclab/bin/python -m pytest -q tests
/root/miniconda3/envs/env_isaaclab/bin/python scripts/check_mjlab.py
PYTHONPATH=source/instinctlab \
  /root/miniconda3/envs/env_isaaclab/bin/python -m pytest -q \
  tests/test_parkour_g1_declaration.py -k contract
```

For term, timing, or physics changes, add fixed-state and temporal probes.
Compare episode length and termination behavior as well as reward. Terrain,
collision, constraint, solver, sensor, or actuator changes require selected-SDK
live and production-scale checks appropriate to the change. Cross-engine parity
is required at declared ordering, frame, time, effort, reset, and observable
interfaces, not for native solver internals.
