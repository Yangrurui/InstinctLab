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
- Latest verified implementation commits: `d343fdc` (locked-arm G1 robot and
  task onboarding), `c5fdfc5` (release/data/operator hardening), and `36f229d`
  (terrain/actuator extensions)
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
- Each G1 variant owns an explicit canonical DFS order in `RobotSpec`. The
  stock variants retain 29 joints; the locked-arm onboarding variant has 15
  movable joints. Isaac's articulation remains native BFS; every boundary
  resolves names and gathers/scatters explicitly. MJLab is naturally DFS.
- G1 native assets and every final actuator parameter remain explicit in
  `assets/unitree_g1/isaacsim.py` and `assets/unitree_g1/mjlab.py`. The neutral
  asset interface routes only `package/variant` ids.
- `unitree_g1/popsicle_torsobase_locked_arms_v1` is the concrete changed
  joint-count onboarding example. Its dedicated URDF and MJCF lock the 14 arm
  joints, both native configs expose the remaining 12 leg and 3 waist joints, and
  `Instinct-Velocity-Flat-G1-15DoF` owns a complete Locomotion config without
  an arm-joint reward.
- The application wheel publishes its `perlin_*` generated-terrain tiles via
  `instinctlab.terrains`; both SDK-free registrations point at lazy native
  builders that remain in the selected backend package. Parkour similarly
  selects the application semantic actuator id `instinctlab.delayed_pd.v1`
  through engine-scoped `instinctlab.actuators` entry points while retaining
  every final group parameter in the two G1 native asset modules.
- `perlin_wave` is the terrain-extension conformance example. It is registered
  by the application with one dedicated builder per backend, builds native
  height-field collision geometry on both engines without adding a scene-builder
  kind branch, and has a reproducible review render at
  `source/instinctlab/docs/perlin_wave_review.png`.
- Backends, assets, actuator models, native terms, native sensors, whole
  terrains, and generated-terrain tiles use transactional lazy registries.
  Discovery is SDK-free until the selected backend bootstraps and provenance is
  recorded for providers actually used by a compilation.
- Production train/play compilation is strict and clean by default. Omissions
  or emulation require an explicit override and remain visible in the manifest.
- Portable dataset resolution rejects decoded separators, traversal, absolute
  paths, and symlink escapes. Manifest resources and conversion targets must
  remain below their declared dataset root after canonical resolution.
- Release builds refuse dirty checkouts, stage only Git-tracked files, pin the
  complete build backend, and record their clean source commit. A `vX.Y.Z` tag
  must pass the fast, wheel, GPU-live, and protected operator-candidate workflows
  on that same commit before the PyPI job can run.
- Checkpoint manifests contain a readable versioned task, robot, policy-I/O,
  effective-agent, and training-semantic contract. Strict `--resume` rejects
  drift before environment construction and restores runner training state;
  explicit `--transfer` permits declaration drift and restarts learning at
  iteration zero. Both modes construct and reset a fresh environment and do not
  restore lifecycle snapshots or common RNG state.
- `scripts/train.py` and `scripts/play.py` select and bootstrap a backend before
  importing its SDK. Playback handlers are application-owned, not adapter APIs.
- Shared semantic meanings live in `spec/`; motion state lives in
  `motion_reference/`; narrow native-value readers live in `bridge/` and
  `compat/`. Native sensor, terrain, solver, and SDK lifecycle code stays under
  the selected backend package.

## Platform maturity

Overall verdict: stock Locomotion, Parkour, and Shadowing paths have strong
contract and construction coverage, and the external extension wheel proves
real native integration on both engines. The declared single-agent rigid-body
training architecture, including its lifecycle boundary and release-engineering
gates, is ready for a 1.0 claim. The coordinated packages intentionally remain
version `0.1.0`; no public tag, registry image, or package publication was
performed in this handoff. A maintainer may make that release decision using
the accepted process in `RELEASE.md`. This does not imply multi-agent, ROS/HIL,
deformable, fluid, or service-runtime APIs that no current task requires.

| Capability | Current state | Remaining boundary |
|---|---|---|
| Backend | Independent lazy plugins; the current four-way isolated wheel matrix and live external extension pass | Registry publication is an operator action |
| Task | Immutable application-owned registrations | External task catalogs are optional, not a current goal |
| Robot asset | Versioned native API and conformance command; stock 29-DoF and locked-arm 15-DoF G1 construct on both engines | Broader object/articulation catalogs |
| Actuator/controller | Lazy native model registry, explicit groups, live stiffness bindings, stateful `compute(command)`/`control_dt`/snapshot/reset contract | New controller families need native temporal evidence |
| Terrain | Whole-terrain and tile plugins | Region semantics; dynamic/deformable terrain only on demand |
| Sensor | Built-ins plus lazy native builders with timing/reset capabilities | New sensor families need fixed-state and temporal evidence |
| Rigid object | Mesh-backed construction on both engines | Full HOI resources and behavioral evidence |
| Collision relation | Portable entity-pair exclusions lower to both engines | Broader materials and constraints |
| Multiple articulations | Primary robot plus canonical `ArticulationRef` entities, each with its own `RobotSpec`; entity-targeted selectors lower on both engines | Broader task evidence on demand |
| Lifecycle | Named exact-rational clocks, phase/latency, component state ownership, and full/partial reset semantics lower on both engines | No open 1.0 boundary |
| Record/replay | Versioned safe same-engine snapshots plus normalized asynchronous episode traces and strict/tolerant replay | Cross-engine/native-solver bit equality is intentionally not promised |
| Operator packaging | Coordinated clean-source wheels/sdists, contained dataset manifests, complete direct runtime lock/import smoke, and an immutable externally supplied dual-SDK container contract | The protected tag candidate still needs a Docker/GPU release runner; no registry image was built or published on this server |

### Lifecycle 1.0 acceptance

The lifecycle boundary is accepted through `1a62fe7` (implementation sequence
`95041c1..1a62fe7`). The committed surface provides:

- named physics, policy, and episode clocks resolved with rational integer
  periods and phases, with explicit global or per-episode reset behavior;
- a resolved contract for every component's clock, step phase, latency, reset
  ownership, and snapshot/stateless behavior, attached exactly once to native
  environments on both engines;
- schema-, task-, engine-, environment-count-, provider-, and provider-version
  checked same-engine snapshots. Archives are atomic NPZ documents loaded with
  `allow_pickle=False`; providers restore native integration/scene state,
  managers, sensors, actuators/controllers, motion-reference state, environment
  buffers, lifecycle clocks, and common RNG state;
- selected-environment asynchronous episode traces at the normalized RL
  boundary, including full-vector actions, observations, rewards, done and
  timeout causes, active masks, and a recoverable initial snapshot. Replay is
  strict by default at absolute/relative tolerance `1e-5`, offers explicit
  per-field tolerances, and reports the first differing index and values without
  weakening done/timeout equality;
- canonical additional articulations through `ArticulationRef`, with one
  complete `RobotSpec` per entity and no engine import or concrete asset policy
  in `TaskSpec`;
- fail-closed stateful controller validation requiring `compute(command)`, a
  numeric `control_dt` equal to the selected clock, and declared reset/snapshot
  hooks;
- `scripts/benchmark_lifecycle.py`, which emits the stable
  `lifecycle_benchmark_v1` report for compilation, environment/wrapper
  construction, policy/environment/physics throughput, PyTorch allocator and
  whole-device resident memory, full/partial reset, snapshot capture/restore/
  persistence, and trace persistence/replay. Optional threshold documents match
  metadata and bound named metrics; unknown fields fail closed, and a failed
  Isaac threshold was verified to exit with code 2 rather than being hidden by
  SDK shutdown.

The retained two-environment smoke reports are:

```text
logs/diagnostics/lifecycle_benchmark_mjlab_smoke_20260831.json
logs/diagnostics/lifecycle_benchmark_isaacsim_smoke_20260831.json
```

Both reports have status `ok`, clean preflight/resolution, snapshot archive
round trips, and matched trace replay. MJLab matched the default observation and
reward tolerance of `1e-5`. Isaac explicitly used observation absolute
tolerance `1.5` and reward absolute tolerance `0.02`; its GPU policy
observation includes random corruption and its public scene state does not
expose PhysX contact-solver caches. These tolerances are recorded in the report
and are not library defaults. Same-engine snapshot means restoration of the
declared state contract, not bit-exact native solver history. The smoke reports
exercise the full path but are deliberately not production-scale performance
baselines; release hardware must supply reviewed threshold documents and an
appropriate environment count.

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

Application actuator aliases use identity-scoped runtime adapters rather than
registering the backend adapter object a second time. An unlabelled native PD
group therefore continues to match exactly one backend model, while a group
built through `instinctlab.delayed_pd.v1` resolves the application alias and
delegates its observable stiffness and effort-limit behavior to the selected
backend. The wrapper carries both `model_id` and `delegate_path`, so a second
application actuator identity can reuse the same boundary without matching the
delayed-PD identity.

## Accepted verification snapshot

The latest committed platform evidence is:

```text
1522 passed, 13 skipped, 33 deselected, 1 warning (full tests/ suite)
Parkour contract subset: 1 passed, 24 deselected
Pyright: 0 errors, 140 warnings under the Python 3.11 configuration
Ruff ratchet: 677 findings, below the reviewed maximum of 694
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
lifecycle benchmark smoke: MJLab and Isaac constructed native environments,
  measured construction/throughput/memory/reset, round-tripped snapshots and
  traces, and matched replay under each report's declared tolerance; the Isaac
  fail-threshold probe returned exit code 2
core-only, Isaac-only, MJLab-only, and dual-backend isolated wheels passed at
  the current report schema using a pinned isolated build toolchain
external fixture installed, passed SDK-free probes, constructed two real
  environments per engine, combined startup gain randomization with a live
  stiffness-normalized reward, passed native action/delay/reset checks, then
  uninstalled without breaking built-in backends
four coordinated wheels and four sdists passed `twine check`; their checksum
  manifest and the installed dual-runtime lock passed the container verifier
five required datasets and 78 resources passed the versioned SHA-256 manifest
release/data re-acceptance: 14 dataset tests and 19 release/operator tests passed;
  encoded-authority and symlink escapes fail closed, arbitrary untracked build
  files are excluded, dirty release checkouts are rejected, and workflow YAML
  parses successfully
clean clone at `c5fdfc5` built four wheels and four sdists with the fully pinned
  backend, all eight passed `twine check`, the manifest recorded the exact clean
  source commit, and the runtime verifier passed every locked distribution and
  isolated module import
application extension wiring: 122 focused tests passed; both Parkour preflights
  reported `ok`; MJLab generated a 25,600-vertex/50,562-face `perlin_wave`
  collision surface; an Isaac Kit test generated the corresponding native mesh;
  the dual-backend wheel matrix plus external live actuator extension passed;
  and the worktree-accessible non-live suite passed 1,400 tests with 3 skipped
  and 33 deliberately deselected
G1 asset conformance passed on both engines
locked-arm G1 onboarding: both SDK-free conformance reports found exactly 15
  canonical joints and complete actuator coverage; MJLab and Isaac Sim each
  constructed four environments with a 15-dimensional action axis, reset, and
  stepped five times
Isaac collision exclusions passed cloned-target validation and a
  4,096-environment, five-step capacity probe
```

These results establish declaration, compilation, construction, targeted
runtime behavior, and a reproducible release path. They do not establish
long-run convergence or native-physics equality. Repository-wide Ruff is a
ratcheted gate rather than a zero-finding gate; changes may not exceed the
reviewed baseline. The simulator environment deliberately retains
`packaging==23.0`; release and wheel builders use isolated tooling instead of
mutating that SDK environment.

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

This server has no Docker CLI, so no final image build or registry push is
claimed. The operator boundary was accepted through clean tracked-source release
builds, four isolated wheel matrices, real installed-SDK/source-receipt
verification, the complete direct application dependency lock and isolated
import smoke, and fail-closed tests for mutable image references, dirty or wrong
backend sources, source-commit drift, SDK-version drift, and artifact checksum
drift. The application Dockerfile consumes prebuilt coordinated artifacts and a
site-managed Isaac Sim/MJLab base by immutable digest; the protected
`release-candidate` workflow is the required Docker-capable acceptance path. The
rationale and required receipt are documented in `README.md`.

## External checkouts and runtime stack

These sibling checkouts are not submodules:

| Checkout | Revision | Purpose |
|---|---|---|
| `/root/InstinctLab-main` | `ba28d3d2655b15a19b729476a630937a19610a3b` | Isaac/main reference |
| `/root/InstinctMJ` | `8d05c122c78714ef7a00d5dc3cfe61787a767a5a` | MJLab reference plus play-device fix |
| `/root/IsaacLab` | `f73c33173801f5f8afea4142482e47b7710c2b75` | Isaac Lab dependency |
| `/root/mjlab` | `08090e8a77228e733373f3b5c54f8b5a68d19d9d` | MJLab dependency |
| `/root/instinct_rl` | `64d7e01` (detached HEAD) | RL runner and batched rollout logging |

The former uncommitted `/root/InstinctMJ` changes are preserved in `8d05c12`:
terrain debug visualization is capability-checked and play maps the selected
CUDA device to EGL/Warp before environment/viewer construction. That checkout
is clean and one commit ahead of its `origin/main`; it was not pushed.

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

Task declarations use `dataset://` URIs resolved below
`INSTINCTLAB_DATA_ROOT` (default `~/Datasets`). The versioned contract is
`datasets/manifest.json`; `scripts/verify_datasets.py` checks required paths,
SHA-256 values, and conversion indexes and can write a run/release receipt. The
current `/root/Datasets` verification accepted 5 datasets and 78 resources.

Parkour resolves directly to the release directory:

```text
dataset://parkour_release/parkour_motion_reference
  -> /root/Datasets/parkour_release/parkour_motion_reference
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
dataset://deep_whole_body_parkour_g1_release/
  20251116_50cm_kneeClimbStep1/20251106_diveroll4_roadRamp_noWall
  -> /root/Datasets/deep_whole_body_parkour_g1_release/
     20251116_50cm_kneeClimbStep1/20251106_diveroll4_roadRamp_noWall
diveroll4-ziwen-0-retargeted.npz
  751 frames, 29 joints
  SHA-256 8274d93046811824640ad373bba13ecd46ed347af8cc6d3d7c116df35a1bec59
```

Older compatibility symlinks may remain on this server, but active task
declarations no longer depend on them.

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

The active logical root is
`dataset://UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz`.
Manifest verification reads `conversion_manifest.json` and checks every
declared converted output, rather than accepting only the index file hash.

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
| Locomotion Flat, locked-arm G1 15-DoF | Declaration, preflight, native construction, reset, and five-step smoke accepted on both engines; no convergence claim |
| Locomotion Rough | Construction/stepping accepted; post-unification convergence pending |
| Parkour | Declarations and pre-terrain reproduction accepted on both engines; current terrain/sensor live construction passes; post-unification convergence pending |
| Whole Body plane Shadowing | Short-horizon parity accepted; retained Isaac 4,096-environment, 50,000-iteration run audited and accepted |
| Perceptive Shadowing | Four narrow collision exclusions accepted for contact/termination semantics; fixed-bin and 256-environment seed-42 A/B complete, but no post-fix long or multi-seed convergence claim |
| BeyondMimic | Official-data L7, 256 environments, seed 42, 700 iterations accepted on both engines; multi-seed and 4,096-environment baselines not promoted |
| Perceptive VAE | Canonical data and two-iteration training-chain smoke passed on both engines; no accepted production reproduction |
| Perceptive HOI | Declarations/fixed-state coverage only; motions and meshes missing |
| Multi-node training | Not validated |

Play variants consume their paired Train checkpoints and do not need separate
training reproductions. Construction smoke is never convergence evidence.

## Retained runs and diagnostics

Process check at 2026-08-31 07:37 UTC found no active `scripts/train.py`
process. The runs listed below are retained outputs, not active jobs. Do not
restart them without explicit operator approval.

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

The retained Whole Body Shadowing run completed 50,000 iterations and is now
audited. Across its last 1,000 TensorBoard samples, mean episode length was
261.54, mean reward per step was 0.07394, mean episode reward was 19.33, and
mean FPS was 32,479. All 40 scalar tags were finite. Mean termination tags were
0.96575 dataset exhausted, 0.02971 link position, 0.00667 base position,
0.00113 projected gravity, and zero each for timeout and out of border. This is
accepted retained Isaac convergence evidence for that run's declaration; it
does not by itself establish multi-seed or cross-engine long-run equality.

```text
logs/isaacsim/g1_shadowing/
  20260827_153315_jointref_fixed_final_long_4096_gpu5_20260827
```

The retained Perceptive run below completed 50,000 iterations and has now been
audited. Across its last 1,000 TensorBoard samples, mean episode length was
206.14, mean reward per step was 0.07267, and mean FPS was 15,181. Selected
loss, gradient, reward, and termination metrics contained no non-finite values.
Mean termination tags were 0.6664 dataset-exhausted, 0.2240 illegal reset
contact, 0.0745 link position, 0.0270 timeout, and 0.0074 each for base position
and projected gravity. This pre-exclusion run proves that the old configuration
eventually learned, not that its early contact semantics were correct, and is
not a post-fix convergence baseline.

```text
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
relationship checks plus a 4,096-environment PhysX capacity probe. A native
PhysX A/B on CUDA 2 is complete:

- The fixed-bin probe used the same `model_6000.pt`, 2,048 environments, seed
  42, bin 12, 400 steps, and identical initial-start hash. Filtered versus
  unfiltered step-1 non-support contact median was 84.75 versus 5,969.41 N, and
  the fraction above 500 N was 27.88% versus 90.77%. Mean first-episode length
  was 48.03 versus 12.58 steps, and first illegal-reset-contact terminations
  were 226 versus 1,737. Per-step reward was slightly lower at 0.05310 versus
  0.05453, which is why reward alone was not used as the verdict.
- The training A/B used 256 environments, seed 42, 700 iterations, the same
  CUDA device, and byte-identical effective agent configs. All 42 scalar tags
  had 70 samples through iteration 690 and no non-finite values. In the final
  100-iteration window, filtered versus strict unfiltered mean episode length
  was 43.312 versus 46.296, per-step reward was 0.023091 versus 0.023898, and
  illegal-reset-contact was 0.026058 versus 0.046517. FPS was 2,373.7 versus
  2,383.6. A second unfiltered control produced the same overall learning and
  illegal-contact pattern.

The exclusions are therefore accepted as a physical/contact and termination
semantics fix, not as a short-training performance optimization. This one-seed
700-iteration run does not show a reward or episode-length improvement, and no
post-fix long or multi-seed convergence claim is accepted. During the first
treatment startup, a concurrent lifecycle commit changed only the order of the
empty `lifecycle` and `engine_extras` declaration fields. The strict control
has the same empty values; after normalizing that order, repository paths, and
the intended collision field, the task declarations and datasets match.

Diagnostic reports, runs, and the reusable probe are:

```text
logs/diagnostics/perceptive_reset_bin12_model6000/
scripts/probe_perceptive_reset.py
logs/diagnostics/perceptive_collision_exclusions_4096_20260830.json
logs/diagnostics/perceptive_collision_ab_20260831/
  treatment_fixed_bin12_retry1.json
  control_fixed_bin12.json
  training_treatment_retry1/20260831_120042_narrow4_256_seed42_700_cuda2
  training_control_95041_retry1/20260831_130618_no_exclusions_95041_256_seed42_700_cuda2
  training_control/20260831_123247_no_exclusions_256_seed42_700_cuda2
```

## Open work

No platform implementation item remains open for the declared scope. The
former release-hardening list is closed by these independently verified
changes:

1. `4008f07` aligned the external fixture with `preflight_v1`; `d34ecdd`
   isolated and pinned its build tools. All four wheel matrices and both live
   native extension paths pass.
2. `d1bacb0` separated fail-closed resume from explicit permissive transfer and
   recorded the fresh-environment/RNG semantics.
3. `8d459e2` made dependency checkout selection fail closed and expanded
   editable-source runtime provenance.
4. `3a58365` established coordinated package/API metadata, fast/wheel/GPU/
   release workflows, the Python 3.11 type gate, Ruff ratchet, isolated release
   builder, and `RELEASE.md`. `3399713`, `4edeefe`, `4e5e23f`, and `c5fdfc5`
   then made the builder tracked-source and fully pinned, normalized the broken
   LFS asset, added exact-tag/same-commit publication gates, and added the
   protected full operator candidate.
5. `3e05864` flattened concrete Locomotion configs and made selectors local and
   explicit; `f6f2860` records the reviewed duplicate native-friction overlay
   required by those independent declarations.
6. `81a4d42` added portable dataset resolution and checksum manifests;
   `42cc328` added coordinated operator artifacts and the immutable external
   simulator-runtime contract. `51a8618` closed decoded and symlink traversal in
   both URI and manifest paths; `4e5e23f` locked every direct application/runtime
   dependency, isolated its import smoke, and bound artifacts to the requested
   clean source commit.

Remaining actions are operational choices, not unfinished implementation: push
the local InstinctLab branch and the one local InstinctMJ commit; configure the
protected `release-candidate` environment with the data root, immutable runtime
digest, intended environment count, and reviewed lifecycle thresholds; choose
and push a public version tag; let all four exact-commit workflows build and
verify the application image on a Docker/GPU runner; and only then dispatch
publication from that tag. None was performed implicitly.

A multi-agent API is intentionally not part of that scope. Define one only when
an actual multi-agent task establishes policy, reward, termination,
shared-state, observation, and partial-reset requirements; do not pre-design it
from a single-agent abstraction.

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

Release/operator checks:

```bash
/root/miniconda3/envs/env_isaaclab/bin/python scripts/check_release.py \
  --expected-version 0.1.0
/root/miniconda3/envs/env_isaaclab/bin/python scripts/check_release_handoff.py
/root/miniconda3/envs/env_isaaclab/bin/python scripts/check_ruff_ratchet.py
INSTINCTLAB_DATA_ROOT=/root/Datasets PYTHONPATH=source/instinctlab_engine/src \
  /root/miniconda3/envs/env_isaaclab/bin/python scripts/verify_datasets.py
/root/miniconda3/envs/env_isaaclab/bin/python scripts/verify_wheel_matrix.py \
  --live-extension --live-device cuda:0
```

CI installs `pyright[nodejs]` and runs `pyright`. This server's system Node 12
is too old for the current Pyright bundle; when rechecking locally, set
`PYRIGHT_PYTHON_GLOBAL_NODE=0` so the installed Python wrapper uses its managed
Node environment.

For term, timing, or physics changes, add fixed-state and temporal probes.
Compare episode length and termination behavior as well as reward. Terrain,
collision, constraint, solver, sensor, or actuator changes require selected-SDK
live and production-scale checks appropriate to the change. Cross-engine parity
is required at declared ordering, frame, time, effort, reset, and observable
interfaces, not for native solver internals.
