# InstinctLab current handoff

Updated: 2026-08-31 UTC

This is the authoritative record for the current repository, server, datasets,
live experiments, accepted baselines, and unresolved work. Historical audit
narratives are in Git history rather than duplicated here.

Every architecture or code review must be persisted here in the same turn.
Record its verdict, evidence, severity, unresolved risks, and re-acceptance
conditions; add durable enforcement to tests or code comments when appropriate.
Do not leave review findings only in chat and do not create one document per
audit phase.

## Repository

- Repository: `/root/InstinctLab`
- Branch: `feat/unified-engine`
- Current verified production code: `f1ff261`
- Local `origin`: `git@github.com:Yangrurui/InstinctLab.git`
- Export repository: `git@github.com:Yangrurui/XLab.git`; its `main` was synced
  through `348a73d`. Later local audit commits still need an explicit push.

Every active task is registered once in
`source/instinctlab/instinctlab/tasks/registry.py` and compiled by the selected
engine:

```text
asset plugin -> selected backend -> engine-core RobotSpec
task application -----------------> engine-core TaskSpec
                                     -> selected backend -> native environment
                                     -> common runner/checkpoint interface
```

Current code organization:

- Locomotion, Parkour, and Shadowing use named `EnvCfg`, reward, and observation
  config classes. Each family owns its shared `*_env_cfg.py`; concrete G1
  datasets, robots, and public factories remain in the corresponding
  `config/g1/*_cfg.py`. There is no central family dispatcher or `make_task`
  function.
- Each task family owns a complete local `mdp/` package. Locomotion, Parkour,
  and Shadowing repeat even generic observations, rewards, curricula, and
  terminations instead of importing a global term catalog. The former
  `instinctlab/mdp/` package has been removed, and task-local `mdp/__init__.py`
  files do not maintain `__all__` export catalogs.
- Locomotion's robot-independent observations, rewards, command, events, action,
  and environment skeleton live in `config/locomotion_env_cfg.py`. That file
  contains no G1 name, concrete joint/link selector, native asset, or G1 runner.
  `config/g1/flat_env_cfg.py` supplies the G1 contact and joint selections and
  explicitly assembles the final ordered reward table; only `flat_g1()` and
  `rough_g1()` convert complete configs to `TaskSpec` at the registry boundary.
  Repeated configuration values are written at their use sites rather than
  hidden behind constants such as `COMMAND`, `ROBOT`, or `FEET_CONTACT`.
- Parkour follows the same boundary. `config/parkour_env_cfg.py` owns only the
  robot-independent Parkour task classes and contains no configuration helper
  functions, G1 selector, dataset, asset, or G1 runner. The complete motion
  reference, camera, foot scanners, volume points, contact/joint selections and
  runner are written explicitly in `config/g1/g1_parkour_target_amp_cfg.py`.
  Parkour configuration values are repeated at their use sites instead of
  hidden behind task constants.
- Shadowing follows the Parkour declaration style across Whole Body,
  BeyondMimic, Perceptive, Perceptive VAE, and Perceptive HOI. Named command,
  action, observation, event, reward, termination, curriculum, and environment
  config classes contain the concrete values; G1 train/play classes inherit
  once, and only registry factories convert them to `TaskSpec`. Selectors and
  contact references are written on the term that consumes them rather than
  hidden behind configuration aliases.
- `source/instinctlab_engine` is the independently packaged shared boundary.
  Its `spec/` defines schemas and validation; `bridge/` owns narrow native-value
  compatibility readers; `motion_reference/` owns shared reference state; and
  its root owns adapter/compiler/registry infrastructure. It contains no task
  declarations, application import, or simulator SDK import.
- The retired `SimulatorBackend` stack and the empty `sim/` package have been
  removed. `instinctlab_engine/spec/capability.py` owns the task/engine
  capability protocol, `spec/robot.py` owns the shared `RobotSpec`, and
  `instinctlab_engine/assets.py` owns plugin-based native asset routing. Shared
  motion-reference state and reusable sensor math/timing live in the engine-core
  distribution. The retired Isaac-only motion-reference, environment, manager,
  and monitor stacks have been removed; no active code imports them.
- Task modules do not import engine implementations. Engine packages do not
  import task modules or one another; tests enforce both boundaries.
- `source/instinctlab_engine_isaacsim` and
  `source/instinctlab_engine_mjlab` are independently packaged backends. Each
  wheel contains only its own native adapter, terms, sensors, terrain, and SDK
  dependency pins. `source/instinctlab/instinctlab/engines` contains no tracked
  source. The backends register through the `instinctlab.engines` entry-point
  group, so adding a backend does not edit engine core or the task application.
- Adding a portable callable does not require a backend edit. A genuinely new
  native operation may live in a separate package registered under
  `instinctlab.engine_terms` as `<engine>.<extension>`; only the selected
  engine's registrar is imported.
- Backend RL wrappers and Isaac-only camera noise/buffer implementations live
  in their backend distributions. Shared Warp geometry and the dual-engine
  contact overflow guard live in `instinctlab_engine/geometry/` and
  `instinctlab_engine/diagnostics/`. The only SDK imports outside backend
  distributions are the explicitly native asset modules and application-level
  playback handlers.
- Terrain dispatch now has a public lazy extension registry. A new complete
  terrain uses `instinctlab_engine.register_terrain(engine, kind,
  "module:builder")`; a new generator tile uses
  `instinctlab_engine.register_sub_terrain(...)`. Installed
  extension packages may publish a registrar in the `instinctlab.terrains`
  entry-point group. Builders are resolved only for the selected engine, so a
  new terrain can support Isaac Sim, MJLab, or both without editing either
  existing scene builder.
- The four in-tree terrain kinds and all generated tile kinds now use that same
  registry. `scene.py` performs only a lookup and contains no terrain kind
  branch or native terrain class map; Isaac and MJLab native construction is
  isolated in their `terrain_builders.py`. The former duplicate
  `rough_terrain.py` bridges were removed.
- Shadowing-specific command, event, reward, observation, and termination
  builders have been removed from both engines. Portable task callables carry
  task behavior. The engines expose generic manager wrappers and only the
  native randomization, reset, sensor, and terrain translations that differ
  between SDKs.
- G1 has no `catalog.py` and there is no `sim/` package.
  `assets/unitree_g1/isaacsim.py` and `assets/unitree_g1/mjlab.py` each contain
  a complete, explicit main-style or InstinctMJ-style native configuration,
  including canonical metadata, model paths, variants, and only that engine's
  actuator values. Neither native module constructs a shared `RobotSpec`.
  `assets/unitree_g1/interface.py` only routes an explicit engine and variant;
  it contains no robot or actuator values. The package registers that router
  with engine core, and the application distribution publishes the same
  resolver under the `instinctlab.assets` entry-point group. The selected
  backend's `assets.py` converts its native configuration to the shared runtime
  `RobotSpec`; engine core has no `instinctlab.assets.*` import convention and
  generic adapters do not name G1.
- Playback is not part of `EngineAdapter`. `play/dispatch.py` lazily selects
  native or Viser handlers registered under the application-level `play/`
  package. Viser's MJLab-backed environment selection belongs to
  `play/viser.py`; neither engine adapter imports the playback layer.
- `scripts/train.py` and `scripts/play.py` are the only production
  train/play entry points. The obsolete `scripts/instinct_rl/` copies and
  their `sys.path` workarounds were removed.
- `TaskSpec` remains an engine-neutral interface. Concrete rewards, MDP
  values, solver profiles, and runner selection belong to task configuration,
  not to the schema module.
- The unused pre-unification `envs/`, `managers/`, and `monitors/` packages were
  removed together with `engines/isaacsim/legacy_motion_reference/`. Their only
  callers were inside that retired stack, and the old examples in `DOCS.md`
  were replaced with the current `TaskSpec`, shared motion runtime, and terrain
  extension interfaces.

### Independent engine packages (2026-08-30)

The engine boundary is now a physical packaging boundary, completed in three
independently verified commits:

- `e19241f` created `instinctlab-engine-core==0.1.0` and moved the shared
  schemas, compiler, registries, bridges, motion-reference runtime, geometry,
  diagnostics, and neutral asset interface into the `instinctlab_engine`
  namespace.
- `2e2e624` created `instinctlab-engine-isaacsim==0.1.0` and
  `instinctlab-engine-mjlab==0.1.0`, moved every native implementation into its
  own namespace, and changed backend discovery to the `instinctlab.engines`
  entry-point group. Application extras select backend distributions rather
  than listing simulator SDK packages directly.
- `5ff4367` replaced the engine core's application-specific asset import
  convention with the `instinctlab.assets` entry-point group and added
  `instinctlab.engine_terms` for independently shipped native term/randomizer
  lowering. Architecture tests reject any import of the `instinctlab`
  application from all three engine distributions.

The application still owns task declarations and the concrete Unitree G1 asset
catalog. That ownership is intentional; the engine packages consume only the
asset resolver and `TaskSpec`/`RobotSpec` interfaces. Installing a new asset,
terrain, native term extension, or backend does not require editing an existing
engine package.

Verification after the package and plugin extraction:

```text
1248 passed, 3 skipped, 26 deselected
all three wheels built successfully with independent top-level namespaces
core + Isaac-only isolated environment discovered only ('isaacsim',)
core + MJLab-only isolated environment discovered only ('mjlab',)
neither isolated discovery imported a simulator SDK module
```

## Architecture assessment (2026-08-30)

This assessment reviews the repository as a software platform rather than as a
single-task implementation. The inspected code includes the independently
packaged engine core, both backend distributions, the application task catalog,
the current plugin boundaries, checkpoint contracts, shared train/play entry
points, and the architecture test suite.

### Overall assessment

The architecture is a sound modular monolith using a ports-and-adapters design
and a compiler-style intermediate representation:

```text
task config -> TaskSpec -> engine-core validation/compiler -> selected adapter
asset plugin -> native asset -> selected adapter -> RobotSpec -> task config
```

The central design should be retained. `TaskSpec` is a useful stable contract,
the backends are genuine adapters rather than task dispatchers, and the physical
package split now reinforces the intended dependency direction. This is a
stronger foundation than a shared entry point containing Isaac/MJLab branches
or a direct engine-to-engine translation layer.

The design is mature enough for continued investment, but the platform should
not yet be treated as a stable 1.0 extension ecosystem. The remaining work is
mostly contract governance, failure behavior, and release hardening rather than
a need to reorganize the main layers.

### Architectural strengths

- `TaskSpec`, `RobotSpec`, and the term schemas form an explicit intermediate
  representation. Cross-object validation catches engine-key typos, invalid
  selectors, missing scene bindings, and canonical-axis violations before a
  native environment is constructed.
- `CompileCtx` is the single lowering point for canonical selectors. Expanding
  ordered joint selectors there prevents Isaac BFS and canonical DFS order from
  becoming a distributed convention that each task or backend must remember.
- Compilation produces a `Resolution` ledger for every resolved, skipped,
  emulated, and omitted term. The startup table and checkpoint manifest make
  capability loss visible instead of allowing a silently changed objective to
  look like a healthy training run.
- Tasks, engine core, the two backend packages, concrete assets, and playback
  have clear dependency directions. Static AST guards and blocked-import tests
  enforce those directions; the boundaries are not documentation-only.
- Lazy task, backend, terrain, asset, and native-term discovery avoids importing
  an unselected simulator SDK. The shared train/play entry points select and
  bootstrap the backend before importing runtime-dependent application code.
- Explicit task configurations preserve declaration order and make tuned
  values auditable against `/root/InstinctLab-main` and `/root/InstinctMJ`.
  Their verbosity is an intentional reproducibility tradeoff and should not be
  "fixed" with generated declarations or hidden override tables.

### Priority risks

#### 1. Separate policy compatibility from experiment provenance

`instinctlab/checkpoint.py` computes one hash over the complete `TaskSpec`, but
`validate_checkpoint_contract()` intentionally does not reject hash drift. It
only rejects task identity, contract-version, canonical-joint-order, and robot
schema changes. This keeps checkpoints usable across implementation-only
refactors, but it also permits shape-compatible changes to observation order,
observation scaling, history, action layout, or other policy-facing semantics.

Replace the single all-purpose hash with at least three contracts:

- a policy I/O compatibility hash covering ordered observations and actions,
  history, scaling, normalization, symmetry augmentation, and runner/network
  input structure; a mismatch must reject load;
- an experiment-semantics hash covering reward, termination, command,
  randomization, terrain, simulation, and agent-training values; resume should
  require an explicit override when this differs;
- a full provenance hash that includes callable identities and the complete
  declaration for audit and comparison, without being a load gate.

Train, resume, play, and export may apply different policies to the semantics
hash, but all four must enforce the policy I/O contract.

#### 2. Make production compilation fail closed by default

The train and play launchers currently default `--strict` to false, while
reward terms are optional unless a task promotes them. A missing optional
builder is recorded, but construction and a long production run may continue.
The current registered tasks compile cleanly in strict verification, so this is
primarily a safety-default problem for future terms and external backends.

Project-owned production tasks should require a clean resolution by default.
Permissive compilation should require an explicit flag such as
`--allow-optional-omissions`, and the manifest should record both the override
and the exact omissions. Diagnostic and compatibility probes may opt out.

#### 3. Make plugin discovery atomic and attributable

The engine, asset, terrain, and native-term registries set their one-time
`loaded` state before every entry point has loaded successfully. If a later
plugin raises, a process can retain a partially populated registry and refuse
to retry discovery. This is especially risky in notebooks, test workers, and
long-lived orchestration processes.

Load extensions transactionally or cache and consistently re-raise the first
discovery failure. Registration conflicts should identify both distributions,
and compilation manifests should include the distribution name, version,
entry-point name, and registered keys for every plugin that affected a run.
Wheel-install tests should also cover a broken plugin, a duplicate registration,
an unsupported core API version, and two extensions for the same backend.

#### 4. Control backend-name overlays

The task tree currently contains 40 uses of `engine_params`, `profiles`,
`engine_overrides`, or `engine_extras` across nine files. These overlays are a
pragmatic representation of real native differences, but each one couples a
task declaration to known backend names and weakens the claim that adding a
third backend is always additive.

Continue translating genuinely shared meanings into typed semantic fields and
backend builders. Retain an overlay only when the values intentionally express
different native semantics or capacity. Report portability in separate
dimensions instead of one boolean: contract portability, semantic overlays,
native extras, and clean resolution. Do not move task policy into a backend
merely to reduce the overlay count.

#### 5. Freeze the final declaration deeply

The spec dataclasses use `frozen=True`, but nested mappings are copied into
ordinary mutable dictionaries. A caller can therefore mutate a term, group, or
profile after validation or after a contract hash has been calculated. That is
inconsistent with treating `TaskSpec` as a stable intermediate representation.

Keep task configuration classes mutable while they are assembled, then create
a deeply immutable snapshot at the registry boundary. Validation, hashing, and
compilation should all consume that same snapshot. Avoid making native config
objects immutable; the boundary applies only to the engine-neutral declaration.

#### 6. Finish release and metadata hardening

The independent wheel layout is correct and isolated wheel discovery has been
verified. Publication metadata is still inconsistent: the application setup
declares `MIT` while the repository license is CC BY-NC 4.0, and
`config/extension.toml` still contains Isaac Lab template title, author,
repository, and dependency metadata. The README task example must also continue
to reflect that registry factories accept an engine-normalized `RobotSpec`.

Before publishing the split packages, make package metadata authoritative in
one place and verify these install matrices from built wheels rather than source
paths: core only, application plus Isaac only, application plus MJLab only, and
application plus both backends. Each environment should test discovery, SDK
import isolation, task materialization, and an appropriate construction smoke
test.

### Recommended sequence

1. Split checkpoint policy compatibility, experiment semantics, and provenance
   contracts, then add negative load tests for observation/action drift.
2. Make clean strict compilation the production default and require an explicit
   opt-out for omissions or emulation.
3. Make plugin discovery atomic and record plugin provenance in manifests.
4. Deep-freeze the registry-produced `TaskSpec` snapshot.
5. Align license, extension, wheel, and README metadata and run the four wheel
   installation matrices.
6. Track semantic overlays as an architecture metric and reduce only those that
   represent a shared meaning already understood by both builders.

Focused read-only verification performed during this assessment reported:

```text
206 passed (spec isolation, task/engine isolation, plugin registries,
            task registry, shared compiler)
48 passed, 1 skipped (checkpoint, train entry, package pins,
                      version guards, agent configs)
```

These checks support the dependency and contract assessment. They do not add
new live-physics, rollout-parity, or convergence evidence beyond the results
recorded elsewhere in this handoff.

### Architecture assessment remediation (2026-08-30)

The six release-hardening priorities above are now implemented:

- `725ac33` split checkpoint metadata into mandatory policy-I/O compatibility,
  explicitly governed experiment semantics, and audit-only full provenance.
  Observation/action drift has negative load tests; legacy v1 manifests remain
  loadable with a warning.
- `6e7e3fc` made production train/play compilation strict and clean by default.
  Skips, emulations, and profile omissions require the explicit
  `--allow-nonclean-resolution` override and remain recorded in the manifest.
- `c77fab9` made engine, asset, terrain, and native-term discovery transactional,
  cached the first failure, added core-API constraints, attributed conflicts to
  both distributions, and recorded distribution/version/entry-point/key
  provenance. `96e6ea9` scoped that provenance to one compilation so a long-lived
  process cannot leak plugins used by an earlier task into a later manifest.
- `a3ea9d2` added the deep immutable `TaskSpec` snapshot at the application
  registry boundary. Configuration classes remain mutable while assembled;
  validation, checkpoint hashing, and compilation consume the same frozen copy.
- `40e4e03` aligned application/extension/backend metadata with Project Instinct
  and CC BY-NC 4.0, corrected the README factory signature, and included the G1
  native resources in the application wheel. `scripts/verify_wheel_matrix.py`
  builds from temporary source copies and verifies core-only, Isaac-only,
  MJLab-only, and dual-backend installations without consulting editable source
  entry points.
- `60b4403` reports contract portability, semantic overlays, native extras, and
  clean compilation separately in checkpoint manifests. The reviewed source
  overlay budget is 40 use sites: 25 `engine_params`, 11 `profiles`, and 4
  `engine_overrides`; `engine_extras` remains zero. The CI budget may decrease
  without review but cannot increase silently. Current registered tasks report
  2--6 semantic overlay sites each and no native extras.

Final architecture verification:

```text
1278 passed, 3 skipped, 26 deselected (full tests/ suite)
1 passed, 24 deselected (Parkour contract subset)
four built-wheel matrices passed:
  core only             -> no backend discovered
  application + Isaac   -> only isaacsim; Flat G1 materialized from wheel assets
  application + MJLab   -> only mjlab; Flat G1 materialized from wheel assets
  application + both    -> both backends; both Flat G1 variants materialized
all matrix discovery phases imported no simulator SDK module
```

No simulator physics, task term, or native actuator value changed in this
remediation, so no new rollout-parity or convergence claim is made. The
GPU-dependent construction checks were not rerun while the live experiments
below remain active; the wheel matrices stop at native asset/task materialization
and engine contract reports.

### Architecture assessment follow-up: startup and runtime lifecycle (2026-08-30)

This second pass reviews failure paths after the six remediations above. The
main `TaskSpec -> compiler -> adapter` structure remains the right design; the
new findings are boundary-contract gaps, not a reason to reorganize the task or
engine layers. They should nevertheless be resolved before treating the split
packages and checkpoint format as a stable 1.0 platform.

#### 1. P0: restore the pre-bootstrap import barrier

The shared launchers and `EngineAdapter.bootstrap()` explicitly require Isaac
Sim's `AppLauncher` to run before `torch` is imported. The current implementation
violates that invariant during the first argument-parse pass: a fresh process
that only imports `instinctlab_engine` already has `torch` in `sys.modules`, and
calling `names()` retains that state even though neither simulator SDK has been
imported.

There are two pre-bootstrap import paths to remove:

- the engine-core package imports `bridge.entity`; importing its
  `instinctlab_engine.spec.entity` dependency first initializes
  `spec/__init__.py`, which imports `spec.robot`, whose module imports `torch` at
  top level;
- installed backend entry points target each backend package root. Those roots
  import the full adapter and term registry, and the adapters import scene
  modules whose shared sensor bridges also import `torch`.

The fact that no `isaaclab` or `mjlab` module is imported is not sufficient:
the documented ordering contract specifically includes `torch`. Existing wheel
isolation tests therefore guard too weak a boundary.

Make backend discovery and the CLI/bootstrap facade a genuinely lightweight
layer. Backend entry points should target a registrar that stores lazy dotted
paths without importing the package root, and adapter imports needed only for
compilation should move behind post-bootstrap methods or local imports. Remove
the eager `torch` dependency from declaration-only robot imports. Add fresh
subprocess tests asserting that all three stages -- `import instinctlab_engine`,
`names()`, and selecting an adapter plus adding its CLI arguments -- import
neither `torch` nor a simulator SDK. For Isaac, only `bootstrap()` may cross that
barrier.

#### 2. P1: checkpoint loading should not be hash-driven

The project decision is that a checkpoint does not require a file-content hash
or a `TaskSpec`-derived compatibility hash. The current `policy_io`,
`experiment_semantics`, and `provenance` fingerprints therefore should not be
expanded into a complete model-state hashing system. Doing so would duplicate
the runner's native state contract, couple checkpoint reuse to declaration
serialization, and still require special rules for inference-only, AMP, VAE,
teacher/student, and future auxiliary state.

Use an explicit checkpoint format version and narrowly scoped schema/version
fields only where a migration needs them. Let the runner's normal strict tensor
loading report module, key, and shape incompatibility. Task identity, experiment
configuration, and source provenance may remain readable manifest metadata for
operators, but they should not be hash gates on checkpoint loading. If play only
needs an inference policy, a policy-only load/export path is still preferable to
constructing and restoring unrelated training state.

This decision supersedes the earlier follow-up recommendation to enlarge
`policy_io_v1`; the observed AMP discriminator omission is evidence that the
hash is the wrong ownership boundary, not a request to add more fields to it.

#### 3. P1: record the effective run configuration as data

`task_contract(spec)` reconstructs the runner config from
`spec.agent.overrides`. The adapters, however, construct the actual runner with
`spec.agent.resolved_overrides(engine)`, and `train.py` then applies CLI seed,
maximum-iteration, device, resume, run, and checkpoint values directly to
`compiled.agent_cfg`. The manifest still calls `add_task_contract(..., spec)`,
so its embedded declaration fingerprints describe defaults rather than
necessarily describing the runner that produced the checkpoint.

No registered task currently uses `AgentSpec.engine_overrides`, but the public
contract explicitly supports it and validation does not restrict the override
keys to rollout length. The current CLI path already creates this difference
for `--seed` and `--max_iterations`. `agent.json` records the effective config
separately, but the primary manifest does not identify it as the authoritative
run configuration.

After engine, CLI, and distributed-rank overrides, serialize the selected
runner configuration directly into the run manifest as readable data. Do not
turn it into a checkpoint load gate. The manifest should also record normalized
`argv`, environment count, Python and installed package versions,
torch/CUDA/driver/device information, source commit and dirty state, and
declared dataset identifiers and versions where available. The runner's later
Git diff capture is useful evidence but is not a structured, complete run
record.

#### 4. P2: keep process lifecycle and fail-fast behavior behind the adapter

Both shared launchers unconditionally call `os._exit(0)` after their
`ExitStack` closes. The train comment attributes this to an Isaac Sim shutdown
hang, but the hard exit also applies to MJLab. It bypasses normal interpreter
and `atexit` teardown, flushes stdout but not every other buffered producer, and
places an Isaac-specific lifecycle workaround in an otherwise engine-neutral
entry point.

Move exceptional process-finalization policy into an adapter-owned lifecycle
hook or bootstrap resource without adding an engine branch to the launcher.
Normal backends should return normally; a backend that demonstrably requires a
hard exit should own and test that behavior. In play, also resolve and validate
a trained-agent checkpoint path, readability, and explicit format version
before `compiled.make_env()`: the current order constructs the expensive native
environment before reporting even a missing checkpoint. Tensor compatibility
remains the runner loader's responsibility.

#### 5. P2: consolidate catalog records and state the registry concurrency model

The task catalog repeats each task id across `TASKS`, `TASK_ASSETS`, and, for
play variants, `PLAY_CHECKPOINT_TASKS`. Tests catch several consistency errors,
but adding or renaming a task still requires coordinated edits to parallel
maps. Replace them with one immutable `TaskRegistration` record containing the
factory path, asset id, and optional checkpoint task id; compatibility views can
preserve the current public helpers.

Engine and plugin discovery also mutate process-global dictionaries, one-time
flags, active-plugin markers, and provenance cursors without a synchronization
contract. Current CLI launch is single-threaded, so this is not a production
fault today. Before advertising the plugin layer for concurrent service or
notebook orchestration, either document single-threaded discovery/compilation as
part of the API or protect discovery and provenance with locks/context-local
state and add a concurrent discovery test.

Recommended remediation order:

1. Re-establish and test the pre-bootstrap `torch` barrier.
2. Remove checkpoint hash gates; retain only explicit format/schema versions
   and the runner's strict tensor loading.
3. Record one effective, post-override agent configuration and structured
   runtime provenance as readable manifest data.
4. Move hard-exit policy behind the backend lifecycle and validate play
   checkpoints before environment construction.
5. Consolidate task registration records and define the concurrency contract
   before a 1.0 plugin API commitment.

Read-only evidence collected for this follow-up:

```text
fresh process: import instinctlab_engine -> torch imported
fresh process: names() -> ('isaacsim', 'mjlab'); torch imported;
               neither isaaclab nor mjlab imported
code inspection: task_contract rebuilds declaration defaults while train mutates
                 and writes the effective runner configuration separately
```

No training process was started, stopped, or signaled during this assessment,
and no simulator physics, task declaration, or production behavior was changed.

### Startup/runtime lifecycle remediation completed (2026-08-30)

The follow-up findings above are resolved through `8195d3e`:

- Engine core, `names()`, and adapter CLI setup now stay on the pre-bootstrap
  side of the `torch` and simulator-SDK boundary. Backend entry points load an
  SDK-free registrar and facade; full compile modules remain lazy. Fresh-process
  tests cover import, discovery, and CLI setup.
- Checkpoint manifests use readable `task_manifest_v3` metadata and the explicit
  `instinct_rl_on_policy_runner_v1` format version. They contain the complete
  effective post-engine/post-CLI agent configuration and task declaration, but
  no `TaskSpec`-, policy-, experiment-, or provenance-hash load gate. Pre-load
  validation checks file readability and manifest/format versions; the runner's
  strict state-dict load owns tensor key and shape compatibility.
- Run manifests record normalized argv, environment count, Python/package
  versions, torch/CUDA/driver/device data, Git commit/dirty state, and resolved
  motion-dataset paths with SHA-256 for concrete files. The same detached agent
  snapshot is written to the manifest and `agent.json` and passed to the runner.
- Shared launchers no longer call `os._exit`. MJLab returns normally; only the
  Isaac adapter owns the documented hard-exit workaround. Trained playback
  resolves and validates its checkpoint before native environment construction.
- One immutable `TaskRegistration` now owns factory path, asset id, and optional
  checkpoint-task id. The former public maps are read-only compatibility views.
- Plugin discovery/rollback uses one re-entrant transaction lock. Provenance
  usage cursors are thread-local, so concurrent compilations cannot attribute
  one another's providers. Concurrent discovery and ledger tests exercise both
  guarantees.
- Direct backend imports remain self-contained: SDK-free package registration
  installs selectors without relying on earlier launcher/test order. Compiler
  lowering also thaws frozen declaration mappings into detached native-config
  copies, allowing MJLab managers to deepcopy configs without weakening the
  immutable registry snapshot.

Verification from `8195d3e`:

```text
1288 passed, 3 skipped, 26 deselected
Parkour contract: 1 passed, 24 deselected
MJLab Flat G1: native construction, reset, canonical DFS action order, and one CPU step passed
wheel matrices: core-only, Isaac-only, MJLab-only, and both passed
fresh-process import/discovery/CLI probes imported no torch or simulator SDK
all isolated wheel discovery phases imported no simulator SDK
```

No training process was started, stopped, restarted, or signaled. No task term,
solver profile, native actuator value, or production physics behavior changed;
the MJLab smoke run was construction/reset/one-step verification only.

### Native actuator plugin protocol design (2026-08-30)

The actuator extension seam should be formalized without introducing a portable
actuator-parameter schema. An actuator implementation is native physics/runtime
code; a robot asset owns the concrete selection, joint groups, and every final
parameter. `TaskSpec` and `RobotSpec` must not become a second actuator catalog.

The intended dependency flow is:

```text
task registration -> package/variant asset id
asset plugin       -> engine-native asset module
actuator plugin    -> lazy engine-native config/model implementation
native asset module + explicit group values -> native articulation/entity
engine-term plugin -> optional new action or randomization lowering
```

#### Scope and non-goals

The protocol must support a reusable custom actuator model shipped outside this
repository, an asset using several actuator models at once, and an actuator
implemented on only one engine. Discovery must remain free of `torch` and SDK
imports until the selected backend has bootstrapped.

The protocol does not attempt to:

- run one actuator implementation unchanged on Isaac Sim and MJLab;
- normalize stiffness, damping, motor, tendon, network, thermal, or transmission
  parameters into a shared schema;
- infer native groups or parameters from `JointProperties`;
- select an actuator independently of the asset at runtime;
- move task action policy or task-specific reward behavior into the plugin.

Production selection remains an explicit asset variant. A new actuator on an
otherwise identical robot receives a new `package/variant` asset id and an
explicit task registration. There is deliberately no `--actuator` override that
can silently change the plant underneath an existing task id.

#### Discovery and registration API

Add one engine-core registry in `instinctlab_engine/actuators.py` and one entry
point group:

```toml
[project.entry-points."instinctlab.actuators"]
"isaacsim.acme_series_elastic" = "acme_actuators.registration:register_isaacsim"
"mjlab.acme_series_elastic" = "acme_actuators.registration:register_mjlab"
```

An entry-point name is `<engine>.<extension>`. Only entries for the selected
engine are loaded. The registration module must be SDK-free and register lazy
dotted paths rather than importing its implementation:

```python
def register_isaacsim(registry):
    registry.register(
        model_id="acme.series_elastic.v1",
        config_factory="acme_actuators.isaacsim:SeriesElasticActuatorCfg",
        runtime_adapter="acme_actuators.isaacsim:SERIES_ELASTIC_RUNTIME",
        capabilities={
            "joint_position_command",
            "applied_effort",
            "effort_limits",
            "stateful_reset",
        },
    )

register_isaacsim.instinctlab_engine_api = ">=0.1,<0.2"
```

The registry key is `(engine, model_id)`. A registration contains only identity,
lazy implementation paths, and declared capabilities. It contains no joint
selector and no actuator parameter value. `config_factory` resolves after
bootstrap to a callable producing that engine's native actuator config. A
plugin may use an SDK config class directly as the callable.

Discovery follows the existing engine/asset/term guarantees: load
transactionally, cache and consistently re-raise the first failure, reject
duplicate keys with both distributions named, enforce the core API range, and
record distribution/version/entry-point/model provenance for each provider
actually used by a compilation.

#### Asset-owned construction

The native asset module chooses the model and passes every final value directly
at each group declaration:

```python
series_elastic_cfg = native_actuator_factory(
    "isaacsim", "acme.series_elastic.v1"
)

hip_actuators = series_elastic_cfg(
    joint_names_expr=[".*_hip_pitch_joint", ".*_hip_yaw_joint"],
    effort_limit=88.0,
    stiffness=40.17923847137318,
    damping=2.5578897650279457,
    armature=0.01017752,
    spring_stiffness=1200.0,
)
```

The example name binding is only the resolved native config type. It must not
become a helper that supplies, transforms, or infers values. In adapted G1
modules, every group continues to repeat its complete selector and final native
parameters; loops, shared parameter dictionaries, name-based inference, and
construction from `JointProperties` remain prohibited.

An external robot/actuator distribution normally publishes both entry-point
groups: `instinctlab.assets` routes its `package/variant`, while
`instinctlab.actuators` publishes reusable native model implementations. A
model used by only one asset may live in that same distribution.

#### Runtime capability adapter

The current shared actuator readers rely on field duck typing and one class-name
test for MJLab's `BuiltinPdActuator`. Replace that implicit contract with an
optional lazy `runtime_adapter` registered beside the model. It is selected by
the native actuator config/instance type after bootstrap and may expose only
the capabilities it declares:

```python
class ActuatorRuntimeAdapter(Protocol):
    def matches(self, actuator: object) -> bool: ...
    def stiffness_groups(self, actuator: object): ...
    def effort_limits(self, env: object, asset: object, actuator: object): ...
```

The initial capability vocabulary is intentionally small and derived from
current consumers:

```text
joint_position_command  native model accepts the existing ordered position target
applied_effort          native asset data exposes the task's declared effort quantity
effort_limits           limit/ratio rewards can obtain limits in canonical joint order
stiffness               power-style rewards can obtain position-control stiffness
gain_randomization      the generic gain-randomization event is valid for this model
stateful_reset          model owns buffers/state and resets the selected environments
```

Unsupported capabilities fail closed with the engine, model id, native group,
and requesting term in the error. They must not return zero, skip a reward, or
pretend that a non-PD model has PD gains. Built-in Isaac and MJLab actuator
types should be registered through the same runtime adapter mechanism before
removing the existing compatibility reader, so the migration is behavior-neutral.

The runtime adapter normalizes observable interfaces only. It does not compute
the actuator's control law and does not translate actuator parameters between
engines. `compute()` and buffered-state `reset()` remain native SDK lifecycle
methods on Isaac `ActuatorBase` and MJLab `Actuator` implementations.

#### Action and randomization boundary

An actuator that still consumes ordered joint-position targets uses the existing
`joint_position` action lowering. If it consumes current, torque, muscle
activation, tendon length, or another command, the extension declares a new
`ActionTermSpec.kind` and supplies one native lowering per supported engine via
`instinctlab.engine_terms`. The actuator registry must not acquire task action
builders.

Likewise, the existing `randomize_actuator_gains` term may be used only when the
model declares `gain_randomization`. A different parameterization supplies a new
semantic event kind and native term lowering. Task configs continue to state the
randomization ranges; plugins only translate the declared operation to the
native implementation.

#### Failure behavior and lifecycle

- Unknown model ids fail while the native asset is materialized, before an
  environment is constructed.
- A provider for an unselected engine is never imported.
- A stateful actuator must reset per-environment state through the native SDK
  reset hook; process-global model state is forbidden.
- Mixed actuator groups dispatch runtime readers per native instance. A missing
  reader fails only when a task requests that capability, but never degrades
  silently.
- Breaking control/state semantics require a new major `model_id` and a new
  asset variant. Package version and readable model/resource version are
  recorded in the manifest; no checkpoint hash is introduced.
- Engine adapters remain unaware of concrete asset packages and actuator model
  names. They perform registry lookup and capability validation only.

#### Acceptance tests

The protocol is ready only when the following tests exist:

1. Core-only discovery of actuator metadata imports neither `torch` nor an SDK.
2. Selected-engine lazy loading imports only that engine's provider.
3. Broken, duplicate, wrong-API, and partially registering plugins roll back
   atomically and identify their distributions.
4. An external fixture package supplies one custom stateful actuator for Isaac
   and MJLab without editing either backend or engine core.
5. Native construction preserves the canonical DFS action order and exact
   joint-group coverage.
6. Fixed-state probes verify command-to-effort output, clipping/limits, and any
   learned or nonlinear model formula against an independent reference.
7. Temporal probes verify delay/history and partial-environment reset behavior.
8. Reward and event probes cover applied effort, effort limits, stiffness, and
   gain randomization; unsupported combinations fail loudly.
9. The compilation manifest records only the actuator providers and model ids
   used by that asset variant.
10. A two-engine model claims parity only at its declared command, effort, reset,
    ordering, and timing interfaces; native solver internals need not match.

Recommended implementation sequence:

1. Add the SDK-free registry, lazy resolution, provenance, and failure tests.
2. Register adapters for the existing built-in actuator types and migrate the
   shared reader away from class-name checks without changing native values.
3. Add a fixture custom stateful actuator package and construction/temporal
   tests on each backend.
4. Add new action/event kinds only for command or randomization semantics that
   the fixture actually requires.
5. Introduce the first production asset variant only after fixed-state,
   temporal, and production-scale plant probes pass.

### Simulation platform capability roadmap (2026-08-30)

InstinctLab should be evaluated as an engine-neutral robotics simulation and
training platform above Isaac Sim and MJLab, not as a replacement physics
solver. Its responsibility is to make task meaning, assets, native extensions,
runtime lifecycle, and evidence portable where portability is real; native
solver, renderer, deformable-body, and fluid behavior remain backend
capabilities and must fail closed when unavailable.

The current extension foundation is strong for independently installed
backends, terrain builders, native term lowering, and engine-neutral asset
routing. It is not yet a stable general simulation ecosystem. A mature
platform must make four kinds of contract explicit:

```text
entities     robot, sensor, rigid object, articulation, terrain
relations    attachment, frame, collision, material, constraint, control
lifecycle    construct, tick, reset, spawn/despawn, record, destroy
evidence     capability report, conformance, deterministic trace, replay
```

Adding another class in an existing scene builder does not count as an
extension seam. A first-class extension is independently installable, imports
only the selected backend after bootstrap, has explicit capability and failure
behavior, carries provider/version provenance, and passes a repository-external
fixture without an edit to engine core or an existing backend.

#### Current capability baseline

| Capability | Current state | Boundary still missing |
|---|---|---|
| Backend | First-class plugin; all isolated wheel matrices pass | Publication remains a release-process gate |
| Terrain | First-class whole-terrain and tile plugin | Region semantics and dynamic/deformable terrain are optional later work |
| Native task term | First-class engine-term plugin | Stateful controller timing must use an explicit lifecycle contract |
| Robot asset | Versioned native API, conformance command, and external wheel fixture | Object/articulation catalogs beyond the primary robot remain later work |
| Actuator | Lazy native model registry, runtime capability adapters, and explicit asset-owned groups | Additional controller clock semantics belong to P1 |
| Sensor | Built-in families plus lazy native builders with timing/reset capabilities | More native sensor families require their own fixed-state evidence |
| Scene object | Production-valid mesh-backed rigid objects on both backends | Object asset routing, articulated objects, and full HOI behavioral evidence |
| Collision/material | Portable entity-pair exclusions with native lowering | Body material bindings and broader contact/constraint relations |
| Multiple articulations | One canonical `TaskSpec.robot` | Canonical schemas and action/observation targeting for additional articulations |
| Time/lifecycle | Physics step, decimation, and selected sensor periods | Named clock domains, timestamps, latency, and per-environment component reset |
| Record/replay | Diagnostics and run manifests exist | Episode trace and engine-native state snapshot/restore interfaces |
| Task distribution | Immutable application-owned task registrations | An external task catalog is optional and must not weaken task ownership |

#### P0: close the advertised extension ecosystem

These goals are required before describing the plugin surface as a stable 1.0
third-party ecosystem.

1. **Implement the native actuator protocol.** Follow the preceding design:
   add SDK-free discovery, lazy native factories, runtime capability adapters,
   provenance, built-in adapters, an external stateful fixture, and fixed-state
   and temporal probes. Asset variants retain every final native group and
   parameter; no portable actuator-parameter schema or checkpoint hash is
   introduced.
2. **Formalize native robot asset onboarding.** Replace the current implicit
   native-module duck typing with a small documented contract at each backend's
   asset boundary. Provide a linter/conformance command for canonical DFS joint
   order, body/frame names, collision geometry, joint coverage, limits,
   actuator groups, units, and selected-engine resources. The acceptance
   fixture is an external robot wheel installed through `instinctlab.assets`
   without an engine-core or backend edit.
3. **Make sensors an additive native extension.** Define a stable semantic
   sensor reference only for meanings shared across engines and a lazy native
   builder registry for SDK implementations. The lifecycle must state attached
   entity/frame, pose convention, sample period, timestamp, latency/history,
   device, and partial-environment reset. IMU or force/torque is the initial
   external fixture; unsupported engines fail before environment construction.
4. **Establish scene relation contracts.** Add narrow engine-neutral
   declarations for collision layers or entity-pair exclusions, body material
   binding, and constraints. A shared semantic contract does not require both
   backends to implement it; an unsupported selected backend must fail in
   preflight. Do not encode a task name or concrete asset package in a scene
   builder. The Perceptive MJCF pair exclusions are the first collision-filter
   acceptance case, followed by a 4,096-environment contact/constraint probe.
5. **Make existing rigid-object/HOI construction production-valid.** Replace
   the removed Isaac `sim.MeshFileCfg` path, validate native object resource
   loading on both backends, and define explicit object spawn/reset and
   collision/material behavior. This goal does not claim HOI behavioral parity
   until the missing OMOMO motions and object meshes are installed.
6. **Provide one extension conformance kit and preflight report.** Before native
   construction, report the selected backend, asset, actuator and sensor
   providers, requested capabilities, native-only features, omissions, and
   incompatibilities. Test broken, duplicate, wrong-API, partial-registration,
   uninstall, and unselected-backend import cases consistently across every
   plugin group.

The following changes are the implementation evidence for P0. The 2026-08-31
audit below supersedes the earlier completion claim; P0 is not accepted while
its two production-entry regressions remain open.

- `c7e18ea` and `e7573b0` added the transactional native actuator registry,
  capability adapters, built-in registrations, exact native group validation,
  and asset-owned model routing without moving parameters into engine core.
- `d4462f2` defined native asset API `0.1` and added
  `scripts/asset_conformance.py`; both G1 backends report canonical DFS order,
  complete names/resources/units, and exact actuator-group coverage.
- `44f1903` added selected-engine native sensor discovery and explicit
  attachment, period, timestamp, latency/history, device, and partial-reset
  capabilities.
- `810af58` lowers portable `CollisionExclusionRef` pairs to Isaac USD
  `FilteredPairs` relationships and MJLab excludes without naming Perceptive or
  G1 in either scene builder. `2294868` verifies that Isaac cloning rewrites
  every relationship target into the owning environment.
- `e7cffb4` replaced the removed Isaac mesh-config path and made both rigid
  object builders validate resources before SDK import, then apply explicit
  spawn/reset, collision, mass, scale, friction, and kinematic semantics.
- `88e286b` unified preflight across assets, actuators, sensors, terrain, rigid
  objects, capabilities, omissions, incompatibilities, and selected-provider
  provenance.
- `ba6bfb3` added a separately built external fixture wheel. It installs one
  robot, stateful actuator, delayed/history sensor, and terrain for both
  backends without editing core or either backend; canonical group coverage,
  actuator clipping/delay/partial reset, sensor timing/partial reset,
  selected-provider provenance, unselected implementation imports, and clean
  uninstall all pass.

The production-scale Isaac relation probe in `6bc3b89` constructed and stepped
4,096 Perceptive environments for five zero-action steps on CUDA 2. It checked
16,384 cloned filtered-pair targets, found no cross-environment target, no
contact-drop warning, and no overflow at reset or any step. Peak reported
occupancy was 19,871,488 / 134,217,728 collision-stack bytes, 81,701 / 327,680
rigid patches, and 216,777 rigid contacts. PhysX API constraint counters were
also added in `d79d295`; this installed runtime returned zero for those optional
counters, while the populated native contact buffers supplied the capacity
evidence. The report is
`logs/diagnostics/perceptive_collision_exclusions_4096_20260830.json` (ignored
diagnostic output). No training process was stopped, restarted, or signaled.

The production probe also exposed an import leak outside the relation code:
the shared URDF motion-reference path entered `pytorch_kinematics`, whose eager
MuJoCo import is invalid inside the selected Isaac runtime on this server.
`3b7bd61` now uses a direct batched URDF fixed/revolute/prismatic FK path and
retains the existing MJCF implementation. It imports neither MuJoCo nor
`pytorch_kinematics` for URDF and agrees with the former G1 calculation within
`1.79e-7` maximum matrix error.

These changes cover the intended implementation scope, but the audit below
shows that P0 acceptance is still open and does not label the APIs stable 1.0.
The external wheel probe materializes native configs and exercises
deterministic lifecycle stand-ins rather than constructing a full third-party
simulator scene. The user-facing minimal extension example and a live native
external-asset construction gate are still release-hardening work.

Current review verification:

```text
1349 passed, 3 skipped, 28 deselected (full tests/ suite)
Parkour contract: 1 passed, 24 deselected
MJLab Flat G1: 16 environments constructed/reset and stepped five times;
  canonical DFS action order true, zero terminations
core-only, Isaac-only, MJLab-only, and dual-backend built-wheel matrices passed
external fixture wheel passed both engines and clean uninstall
G1 native asset conformance reported status ok on both engines
Isaac collision-relation live clone test and 4,096-environment probe passed
```

#### Extension platform implementation audit (2026-08-31)

Outcome: the implementation is not yet correct enough to accept P0 as
complete. The full default suite remains green, but direct production-path and
contract probes reproduced two P0 regressions and four protocol gaps. This
audit changed documentation only; it did not modify runtime code or start,
stop, restart, or signal a training process.

P0 regressions, in repair order:

1. **Generated-terrain preflight reads dictionary keys instead of tile kinds.**
   `instinctlab_engine/preflight.py` derives required sub-terrain kinds from
   `generator.sub_terrains`, which yields configuration labels. Native terrain
   discovery and lowering use each `SubTerrainSpec.kind`. Consequently the
   mandatory preflight in both `scripts/train.py` and `scripts/play.py` rejects
   the registered Rough and Parkour tasks on both Isaac Sim and MJLab before
   construction. Collect the value objects' `kind` fields, then enumerate every
   registered task against each declared engine in a regression test. HOI's
   failure without its external object resources is expected and should use
   fixtures when testing the registry matrix.
2. **The Isaac `VolumePoints` builder is truncated.**
   `_build_volume_points()` imports the grid generator and then returns
   implicitly; its former construction body now sits after the return in
   `_build_native_sensor()` and is unreachable. Isaac Parkour therefore
   receives `None` for `leg_volume_points`. Restore the builder and run a live
   Isaac Parkour construction/reset/step probe; the default suite's Parkour
   declaration test and the Perceptive collision probe do not exercise this
   path.

P1 contract gaps exposed by the same review:

1. **The external actuator fixture advertises capabilities it cannot serve.**
   Its `stiffness_groups()` returns scalar values rather than
   `(joint_ids, stiffness)` pairs, and it defines `effort_limits()` while the
   shared bridge calls `effort_limit_for_joint()`. The current wheel probe only
   calls `matches`, `compute`, and `reset`, so it does not prove the claimed
   runtime-capability integration. Exercise every advertised capability through
   the real shared bridge/reward consumer and either define capability-specific
   protocols or enforce each method and return shape at conformance time.
2. **Mixed actuator-model validation has no group-to-model identity.** Native
   assets report only unique model ids and a group count. Preflight therefore
   requires every model to satisfy the union of task capabilities, while group
   validation can neither prove which model built a group nor relate a selector
   to the requested capability. This can reject valid mixed assets and accept a
   group built by an unrelated native type. Add engine-native metadata mapping
   each group to its model id and selectors; keep all actuator parameter values
   in the native asset declarations.
3. **Dynamic scene names do not share one symbol table.** Rigid objects can use
   `robot`, `terrain`, or a sensor name and still pass `TaskSpec.validate()`.
   Both builders later materialize these components in overlapping namespaces,
   allowing silent overwrite. Validate robot, terrain, rigid-object, portable
   sensor, native-sensor, and backend-reserved names together before lowering.
4. **URDF conformance stops at fixed joints.** `_urdf_topology()` skips the
   fixed joint and its child subtree, so a movable joint below a fixed mount is
   omitted from canonical topology. Traverse fixed edges without adding the
   fixed joint to the canonical joint axis, and cover a
   `base -> fixed mount -> revolute child` fixture.

Evidence and limits:

```text
50 passed (focused extension/protocol tests)
1349 passed, 3 skipped, 28 deselected, 1 warning (full tests/ suite)
MJLab Flat G1: 16 environments constructed/reset and stepped five times;
  canonical DFS action order true, reward finite, zero terminations
compileall and git diff --check passed
all registered task ids were probed against their declared engines:
  Rough and Parkour reproduced the unexpected generated-terrain failure;
  HOI failed only for its known absent external resources
direct contract probes reproduced the actuator fixture shape/name mismatch
  and accepted scene-name collisions
```

Re-accept P0 only after both production regressions are fixed, each has a
targeted regression test, the complete registry-by-engine preflight matrix has
no unexpected failure, and live Isaac Parkour constructs, resets, and steps.
The four P1 gaps remain mandatory before declaring the extension APIs stable
1.0 even if those two blockers are repaired.

#### P1: make scenes and failures reproducible

These goals are needed for a mature robotics platform and become mandatory when
manipulation, multiple robots, or asynchronous sensing enter production scope.

1. **Define clock and component lifecycle semantics.** Name the physics,
   actuator/controller, policy, sensor, renderer, and recorder clock domains.
   Specify when a command becomes active, what time a sample represents, where
   delay/history is applied, and when reset occurs relative to termination and
   final observation. Stateful actuators, controllers, and sensors must reset
   only the selected environments and must not own process-global mutable state.
2. **Add trace recording and replay.** Record readable episode seeds, commands,
   actions, observations, rewards, terminations, references, and timestamps.
   Provide an engine-native state snapshot/restore interface for exact
   same-engine diagnosis and a portable command/reference trace for
   cross-engine comparison. Binary state equality across engines is not a
   goal, and checkpoint file hashes remain out of scope.
3. **Support additional articulations without destabilizing the primary robot
   contract.** Keep `TaskSpec.robot` as the primary policy robot initially;
   introduce explicit canonical topology for additional articulated entities
   and resolve their selectors by entity id. Generalize actions and
   observations only after this schema exists. Do not replace the typed scene
   with an unvalidated `tuple[Any, ...]` or a generic variant dispatcher.
4. **Define the multi-agent layer only on demonstrated need.** Agent identity,
   policy ownership, action/observation spaces, reward/termination ownership,
   shared state, and partial reset are separate from merely spawning two
   robots. Do not infer a multi-agent API from a multi-articulation scene.
5. **Formalize stateful controller extensions.** Existing
   `instinctlab.engine_terms` remains the lowering seam for a new command
   semantic. Torque, current, impedance, OSC, IK, MPC, tendon, or muscle
   controllers additionally need declared update rate, state, reset, limits,
   and observable-effort semantics; the actuator registry must not become a
   controller or task-policy registry.
6. **Benchmark platform behavior.** Maintain construction latency, steady-state
   environment steps/s, GPU memory/capacity, deterministic reset, trace replay,
   and partial-reset isolation checks. Construction smoke tests remain contract
   evidence only; contact, constraint, actuator, and timing changes require
   production-scale and temporal probes.

#### P2: product-dependent capabilities

These are not 1.0 release gates for the current locomotion, parkour, and
shadowing platform. Adopt them only when a concrete task or deployment path
requires them:

- external task-package discovery, while keeping each task family's complete
  configuration and MDP implementation application-owned;
- lighting, visual and contact material libraries, camera exposure/rolling
  shutter, calibration, synchronization, and reproducible visual domain
  randomization for sim-to-real perception;
- ROS 2, hardware-in-the-loop, and real-robot message/clock adapters;
- deformable bodies, cloth, cable, fluid, or dynamic terrain capability
  adapters where the selected native backend provides production support;
- distributed service orchestration and hot spawn/despawn beyond the current
  vectorized training-process model.

#### Architectural guardrails

- Do not create an entry-point group for every noun. Third-party native code is
  a plugin; portable frame, relation, time, and capability meanings are typed
  shared data; concrete physics and training values stay in task or native
  asset configuration.
- Preserve the existing dependency direction. Tasks and shared specs never
  import an engine; engines never import a task or a concrete asset package;
  different engines never depend on one another.
- Preserve explicit native asset declarations. No registry, shared helper, or
  `RobotSpec` field may infer, normalize, or translate actuator parameters.
- Fail closed at preflight for a missing capability, provider, resource,
  canonical schema, or lifecycle hook. Do not substitute zeros, skip policy
  inputs, suppress rewards, or silently change the simulated plant.
- Require parity only at a declared interface: canonical ordering, frame and
  timestamp meaning, command/effort/reset semantics, collision relation, and
  observable results. Native solver internals are intentionally different.

#### Maturity acceptance gate

The platform may be called a stable third-party simulation extension ecosystem
only after repository-external distributions can independently provide and an
external application can compose a robot with its native actuator, a native
sensor, and terrain without editing InstinctLab or either backend. Robot and
actuator support may live in one asset distribution where they are naturally
coupled; sensor and terrain fixtures must remain independently installable so
one omnibus wheel cannot hide plugin-to-plugin coupling. The following must
also hold:

1. unselected SDKs are not imported before backend bootstrap;
2. capability incompatibilities fail in preflight before native construction;
3. canonical ordering and entity/frame binding pass construction probes;
4. timing, delay/history, and partial reset pass temporal probes;
5. collision, constraint, and actuator behavior pass fixed-state and
   production-scale probes appropriate to the change;
6. the run manifest records readable provider, version, asset/model id,
   effective configuration, and resolution data without checkpoint hashes;
7. uninstalling the wheel leaves the core and existing backends functional;
8. documentation contains one minimal external-package example and a clear
   unsupported-capability failure example.

### Compat boundary cleanup (2026-08-30)

The isolated `codex/compat-boundary-cleanup` worktree branch was fast-forwarded
into `feat/unified-engine` at `f940fe6`. Four commits tightened the shared
runtime boundary without moving task policy into an engine:

- `79f5d64` moved `clip_frame` and `exhausted_envs` from `compat/` to their
  owning `motion_reference/buffers.py` module.
- `6f29ed0` removed root-field guessing from task code. Root angular velocity
  uses a narrow compat reader; root linear velocity requires an explicit
  COM/link anchor, and Locomotion declares the preserved Isaac/MJLab metric
  choice in `engine_params`. Shadowing now reads explicit `root_link_*` fields.
- `435a841` removed the obsolete test-only `compat/vocab.py` and
  `compat/denylist.py` catalogs. Runtime compatibility failures use the small
  `compat/errors.py` interface; task-owned formula guards remain in tests.
- `f940fe6` separated debug-image display into `utils/debug_image.py` from the
  delayed-observation reset bridge in `compat/observation_history.py`. The
  bridge requires an explicit opt-in protocol and fails loudly for incomplete
  implementations.

The relevant tests were synchronized while retaining the newer registry,
terrain, and sensor-builder test migrations already in the main checkout.
`tests/`, `pytest.ini`, and this handoff are now tracked. Post-merge
verification reports:

```text
1159 passed, 3 skipped, 26 deselected
15/15 strict MJLab task configs compiled clean in the isolated worktree
MJLab Flat constructed 16 CPU environments and stepped five finite steps
source compileall, fatal Ruff and import-direction searches passed
```

The 2026-08-30 engine-boundary cleanup was completed in `6dadf4a`, `1b59d7a`,
`e03823e`, and `10edfb2`. Built-in terrain dispatch now uses the public lazy
registry, playback no longer belongs to `EngineAdapter`, and backend-only
runtime helpers no longer leak through `utils/`. Verification after the final
commit:

```text
15/15 tasks compiled to clean strict MJLab configs
Flat, Rough, and Parkour compiled to clean strict Isaac configs after headless Kit bootstrap
scripts/check_mjlab.py constructed 16 Flat environments and stepped five times
137 passed, 2 skipped, 2 deselected (architecture, compiler, configclass, overflow, geometry, playback focus)
fatal Ruff, source compileall, SDK import-boundary audit, and diff checks passed
```

The engine-boundary extraction on 2026-08-30 made the package dependency
direction explicit without changing task values or native lowering behavior:

- shared motion clip loading, sampling, buffers, symmetry, and runtime state
  moved from `engines/motion_reference/` to the top-level `motion_reference/`
  package;
- ray alignment, volume-point geometry/kinematics, and contact-clock recurrence
  moved into `compat/sensors/`; the Parkour penetration formula remains in the
  owning task MDP;
- `RobotSpec` and native asset routing moved from `engines/assets.py` to
  `spec/robot.py` and `assets/registry.py`; the capability protocol moved from
  `engines/capabilities.py` to `spec/capability.py`;
- Isaac-native terrain and sensor implementations moved from top-level
  `terrains/` and `sensors/` into `engines/isaacsim/`; MJLab-native camera,
  terrain, raycast, event, and sensor lifecycles remain in `engines/mjlab/`;
- `assets/`, `compat/`, `motion_reference/`, `spec/`, and `tasks/` no longer import
  `engines`. The engine root now contains only `base.py`, `compile.py`,
  `registry.py`, and the lazy plugin registry in `__init__.py`.

Verification for that extraction:

```text
30/30 dual-engine task contracts complete with no missing terms
15/15 tasks compiled to strict MJLab configs with no skip/emulation/omission
Flat/Rough/Parkour compiled to strict Isaac configs after headless Kit bootstrap
scripts/check_mjlab.py constructed 16 Flat environments and stepped five times
107 passed (architecture, compiler, ray, contact, volume-point focus), 2 deselected
77 passed (motion-reference sampling, timing, symmetry, AMP and task contracts)
source compileall, focused fatal Ruff checks, import-boundary searches, and diff checks passed
```

The `tests/` suite was migrated on 2026-08-30 to the current
registry boundary. Tests now materialize the selected engine's `RobotSpec`, use
task-owned Parkour and Shadowing MDP callables, and import portable terrain and
motion helpers from their shared packages rather than deleted engine modules.
The latest default suite reports `1159 passed, 3 skipped, 26 deselected`; the remaining
deselections are explicitly marked live-engine probes, whose source has also
been migrated to the current APIs. `scripts/check_mjlab.py` additionally
resolved all 39 Flat terms, constructed 16 environments, and stepped five
times.

Current verification at `d8e8b24` plus the Perceptive diagnostic probe:

```text
1297 passed, 2 skipped, 32 deselected
python scripts/check_mjlab.py:
  Instinct-Velocity-Flat-G1 resolved all 39 terms
  constructed 16 MJLab environments and stepped 5 times
```

Locomotion base/G1 boundary verification at `7442f68`:

```text
Flat and Rough TaskSpec contract hashes matched 7c2179e for both engines
shared locomotion config imported without loading either engine SDK
MJLab Flat resolved all 39 terms and stepped one CPU environment
MJLab Rough resolved all 40 terms with none skipped, emulated, or omitted
```

Parkour base/G1 boundary verification at `c7b7fcf`:

```text
Isaac Sim and MJLab TaskSpec hashes matched 7442f68 exactly
all 26 rewards and four observation groups retained their declared order
shared Parkour config imported without loading either engine SDK
MJLab resolved all 64 Parkour terms with none skipped, emulated, or omitted
explicit-selector audit found no joint/body selector constructor aliases
```

Task-local MDP and explicit Shadowing config refactor at `2c54c64`:

```text
15/15 tasks materialized for both engines with no missing contract terms
15/15 tasks compiled to MJLab CPU configs in strict mode
every portable callable resolves to its owning task family's mdp/ package
97 passed (Shadowing declaration, fixed-state MDP, joint order, engine isolation)
96 passed (task-local MDP portability and numerical term checks)
1 passed (Parkour dual-engine contract), 24 deselected
source compileall, fatal Ruff checks, architecture searches, and diff checks passed
```

The engine/task isolation follow-up made contact thresholds, friction
randomization, reset/push entity selectors, motion-command entities, and MJLab
pinhole native settings explicit on the owning task configs. Engine builders
now reject missing task choices instead of inheriting values tuned for a
different task. Motion-reference buffer reads used by task terms moved to the
engine-neutral compatibility layer. Verification after that follow-up:

```text
15/15 tasks compiled to clean MJLab configs in strict mode
13/15 tasks compiled to clean Isaac configs in strict mode
the two HOI configs fail earlier in scene construction because the installed
  Isaac Lab has no sim.MeshFileCfg; this also fails at da6b214
20 engine/task-isolation and native-term fixed-state tests passed
MJLab Perceptive: 4 environments reset and stepped three times with no
  termination or truncation; all 10 rewards and seven terminations were active
```

The reward/termination ownership follow-up at `b6c420d` completed the same boundary on both
engines. All reward aggregation, force thresholds, clipping, normalization, and
termination policy now live in the owning Locomotion, Parkour, or Shadowing
`mdp/` package. `compat/robot.py` exposes only narrow native quantity reads:
Isaac keeps `applied_torque`, finite-difference `joint_acc`, and body COM
velocity, while MJLab keeps `qfrc_actuator`, analytic MuJoCo `qacc`, and body-link
velocity. Both engine reward and termination registries are empty apart from
their generic portable-family builders; `engines/mjlab/rewards.py` and the dead
Isaac-specific `envs/mdp/` catalog were removed. The two Isaac-only generic
randomization events still in use moved unchanged to `engines/isaacsim/events.py`.

No reward weight, term parameter, declaration order, Requirement level, or
native quantity was changed. Only `kind=` dispatch became a direct task-owned
`func=` reference, so informational task-contract hashes change because callable
identity is serialized even though the evaluated training formula is preserved.
Verification after the ownership move:

```text
28 passed (engine/task isolation, native fixed-state terms, Parkour ownership)
15/15 tasks compiled to MJLab configs in strict mode; none skipped or emulated
13/15 tasks compiled to Isaac configs in strict mode
the same two HOI configs hit the pre-existing missing sim.MeshFileCfg API
MJLab Perceptive: 4 environments reset and stepped three times with zero
  termination/truncation; rewards remained finite
source compileall, Ruff, architecture searches, and diff checks passed
```

The retained MJLab and Isaac production trainings on GPU 7/6 were not stopped,
restarted, or signalled. The short MJLab probe is runtime wiring evidence, not
new convergence evidence. The depth-camera hit-body list and its order were not
changed by the isolation follow-up.

The command/event registry boundary was tightened again on 2026-08-30. Native
event and domain-randomization lowering moved out of each general `terms.py`
into its engine's `event_terms.py`; the semantic registrations remain native
because PhysX and MuJoCo mutate different state. Portable events already use
`EventTermSpec(func=...)` and need no engine edit. Isaac's two project-specific
native event functions were folded into `event_terms.py`, so its former
`events.py` duplicate was removed. MJLab keeps the implementations in the
explicitly named `native_event_functions.py` because its
`requires_model_fields` decorators import the SDK at definition time;
`event_terms.py` remains SDK-free and loads those functions only during native
compilation.

Parkour pose velocity, Locomotion uniform velocity, and all Shadowing
motion-reference commands now live in their owning task `mdp/` packages and
are declared with `CommandTermSpec(func=...)`. Each engine retains only one
generic portable-command wrapper for its manager base and timer. Shadowing's
height scan likewise became a task-owned observation over the shared ray
sensor interface; its Isaac offset/miss semantics and MJLab multi-frame/miss
semantics are explicit task parameters. Neither engine registry contains a
concrete command or observation kind, and no task package is imported by an
engine.

The backend motion-reference and rough-terrain modules were retained because
they are genuinely native adapters, but renamed to make that responsibility
unambiguous: `motion_reference_sensor.py` implements each SDK's sensor lifecycle
around the shared runtime, and `rough_terrain.py` lowers the shared recipe to
native terrain/importer config classes. Shared clip behavior and terrain-column
queries live outside the backends.

Verification for this boundary change:

```text
30/30 dual-engine task contracts complete with no missing terms
15/15 tasks compiled to MJLab configs in strict mode
Flat Locomotion, Parkour, and Perceptive Shadowing compiled to strict Isaac configs
52 passed (engine isolation, compile, and capability focus)
fixed-state height-scan and velocity-command probes passed
MJLab Flat resolved all 39 terms, reset two CPU environments, and stepped twice
source compileall, focused Ruff, architecture searches, and diff checks passed
```

Moving defaults and callable ownership changes the informational full
`task_contract.hash` recorded in new manifests. Resume/play compatibility does
not compare that hash: it checks task identity, contract version, canonical
joint order, and robot schema. The current validator accepted the active MJLab
Perceptive `model_44000.pt`, so the structural refactor does not block that
checkpoint from loading; hash-based offline reports must not treat the new hash
as proof of changed training semantics.

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

Official BeyondMimic/LAFAN1 data is installed from Hugging Face dataset
`lvhaidong/LAFAN1_Retargeting_Dataset`, pinned at revision
`ce1572906efe6157840e8474d5a0d7aa87481e74`:

```text
/root/Datasets/LAFAN1_Retargeting_Dataset
  40 G1 CSV clips, 264,705 frames
/root/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz
  40 converted NPZ clips plus conversion_manifest.json
/root/Xyk/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz
  -> /root/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz
```

The selected production clip has these provenance hashes:

```text
sprint1_subject2.csv
  7babbd9d0a3cebf040709cb75fbf4268e925e337a2d44600dcce3d3b2d24a818
sprint1_subject2_retargetted.npz
  f1b1236d13f3f4d695ffb1b6ea8e7faf64363c419f7660336a4bd41da2bb7b55
```

Regenerate the NPZ set with `scripts/lafan1_csv_to_instinct.py`; its exact
invocation and format contract are in the BeyondMimic task README. The
converter changes the pelvis-root source to the production `torso_link` root,
corrects the three reversed waist axes, emits canonical DFS joints and `wxyz`
quaternions, and records source/target hashes atomically.

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
- **BeyondMimic: official-data L7 reproduction accepted on both engines.** The
  same selected LAFAN1 clip, 256 environments, seed 42, and 700 iterations ran
  to `model_700.pt` under both strict builders with all 49 terms resolved.
  This is a long regression, not a 4096-environment capacity or multi-seed
  performance claim.
- **Perceptive VAE: canonical data and a diagnostic MJLab teacher now complete
  the training chain; no accepted production reproduction on both engines.**
  HOI still has declarations only because its motions and object meshes are
  absent.
- Play variants use the corresponding train checkpoint; they do not need an
  independent training reproduction.
- Real multi-node distributed training remains an infrastructure validation
  item.

Known silent faults already fixed include canonical DFS/native BFS action
offset mapping, name-ordered joint-reference defaults, current-time AMP
references, link-origin velocity semantics, critic history/width, Perceptive
depth preprocessing, motion-terrain matching, reset sampling order, contact
sensor hot-path caching, and engine-native capacity profiles.

## G1 native asset ownership (2026-08-28)

The obsolete `assets/unitree_g1/catalog.py` layer was removed. The package
front is now a thin public-name export and contains no values or construction
logic. Isaac and MJLab each carry the complete G1 declaration in their own
native asset module, including an explicit ordered 29-joint property table,
three explicit `RobotSpec` variants, model paths, and native actuator groups.
Neither generic engine adapter derives gains or grouping from
`JointProperties`. The shared `actuator_group` field was removed, so native
lag correlation is also owned by those two asset modules. MJLab's seven gain
groups preserve five independent per-episode lag draws by sharing reset-only
periods across the two leg and two arm gain splits.

Verification:

```text
12 passed (G1 static/native actuator parity, excluding unrelated retired verify scene)
133 passed (task declarations, registry, and engine-isolation focus)
python scripts/check_mjlab.py resolved all 39 Locomotion terms,
constructed 16 MJLab environments and stepped 5 times
```

The retired sim2sim verification scene was then reconnected directly to
`assets.unitree_g1.make_g1_29dof_robot_spec`; the removed global asset registry
was not restored. Base, Shadowing, and Parkour are now explicit native asset
variants, and the shoe URDF/MJCF live with the other G1 resources rather than
under the Parkour task. The full default suite after this follow-up was:

```text
1297 passed, 2 skipped, 32 deselected
```

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
1 passed (fresh Isaac Whole Body reset/four-step rollout plus native BFS to
  policy DFS observation/action semantics)
1 passed (fresh full Isaac Parkour construction, eight-frame DFS policy
  history, AMP, sensors, terrain, volume points, and eight reward steps)
```

Fresh Isaac live probes now close the runtime gap. The lightweight Whole Body
probe verified reference reset writes and four steps while directly asserting
native BFS names and name-resolved DFS policy/action axes. The full Parkour
probe additionally filled and checked all eight frames of the DFS joint
position, joint velocity, and action histories before completing its sensor,
terrain, AMP, volume-point, and reward checks. Both probes emitted the optional
Iray/Neuray missing `libGLU.so.1` extension-load message and continued normally,
confirming that it is not a headless startup blocker. The three production
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

## Explicit G1 native asset configuration (2026-08-28)

The G1 native asset boundary was made explicit at `4ba3af0` and subsequently
flattened further. `catalog.py` and `sim/unitree_g1.py` do not exist.
`assets/unitree_g1/interface.py` is the neutral routing surface: it resolves an
explicit engine and variant to the native module; it owns no robot, model,
joint, or actuator values. `assets/unitree_g1/__init__.py` exports only that
router. Both native files are independently readable: each spells out the
canonical tensor names, ordered native joint configuration, three variants,
native model paths, and actuator configuration without constructing or
importing the shared `RobotSpec`.

- `assets/unitree_g1/isaacsim.py` states main's five native actuator groups and
  every effort, velocity, stiffness, damping, and armature value directly.
- `assets/unitree_g1/mjlab.py` states InstinctMJ's seven native
  `BuiltinPdActuatorCfg` groups directly, including the five explicit motor-bus
  delay periods that make the two leg configs and two arm configs share their
  respective per-episode lag draws.
- `asset_id` uses `package/variant` and is resolved through the small
  `assets/registry.py` interface. Engine adapters contain no G1 import, name,
  registration table, or file path; all three G1 variants are registered inside
  the corresponding native asset module.
- After the launcher selects an engine, that engine's `assets.py` reads its
  native configuration and converts it to the shared `RobotSpec`. The registry
  only checks the task's `asset_id` and injects the already-normalized robot
  into the concrete task config. Task declarations contain no G1 asset,
  actuator, model-path, or conversion values.
- Standard Locomotion, Shadowing, and Parkour use distinct variants, so model
  path, fixed-joint import, spawn height, and actuator delay are selected as one
  native configuration instead of being overlaid from `RobotSpec`.
- Isaac and MJLab each state their own motor constants, default joint pose,
  native model paths, actuator groups, and task variants. Neither imports these
  values from the portable G1 contract or from the other engine.
- The Parkour shoe URDF/MJCF moved out of `tasks/parkour` into the G1 asset
  resources; mesh references were rebased and the moved MJCF loads successfully.
- The shared 29-joint interface is also an explicit ordered table rather than an
  `if/elif` classification. Tests compare both native declarations against that
  interface and both reference repositories.

Native asset payloads are omitted from the engine-neutral checkpoint contract.
For all 15 registered tasks, Isaac and MJLab materializations have the same
contract hash while each contains only its selected backend asset.
Verification:

```text
15/15 tasks materialized for both engines with matching checkpoint contracts
fresh-process imports loaded only interface.py plus the selected native module
python scripts/check_mjlab.py resolved all 39 Locomotion terms,
  constructed 16 environments in canonical joint order, and stepped 5 times
Isaac native conversion and URDF generation completed; the live two-env probe
  was stopped after environment construction stalled with the server's
  existing Kit/libGLU warnings, so it is not step evidence
```

The test helpers deliberately select the native asset before
calling `registry.spec(task_id, robot)`. They do not restore the removed
package-level G1 exports or allow a task factory to infer an engine. The local
episode-length diagnostic follows the same boundary by asking the Isaac asset
adapter for the selected `RobotSpec` before reading the task simulation config.

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

No fresh Isaac rollout was run or claimed in this phase. The live GPU 5--7
training processes were not signaled or restarted. The configuration fixes
affect currently inactive VAE/HOI/BeyondMimic or Play variants; the active
Perceptive timeout behavior was already equivalent in the native builders.

## Event, randomization, and curriculum audit (2026-08-27)

The startup/reset/interval event sets and curriculum terms were compared
separately against main for Isaac and InstinctMJ for MJLab. The native manager
loops agree on the relevant order: curriculum records completed episodes
before reset events and manager reset, while interval events run after reset
in the same control step. A temporal probe now protects that failure counters
are recorded before the per-step EMA consumes and clears them.

Six silent differences were fixed:

- Perceptive, one-motion Perceptive, VAE, and HOI Play now remove training-only
  friction, default-joint, and COM randomization. Their reset pose, velocity,
  and joint ranges are explicit six-axis zeros, matching both references while
  preserving sensor, gain, and inertia randomization. Whole Body and
  BeyondMimic intentionally retain their Play randomization.
- OneMotion training no longer schedules adaptive sampling or its 0.02-second
  smoothing event when `motion_bin_length_s` is disabled.
- HOI Play keeps main's Isaac-only `(0, 1, 2)` visualization offset while
  MJLab retains InstinctMJ's zero offset. Fixed-state reset tests verify exact
  pose, velocity, and canonical joint writes on both lowerings.
- MJLab Shadow inertia randomization now samples mass scale uniformly in
  `[0.8, 1.2]` before converting to pseudo-inertia alpha. The old lowering
  sampled alpha uniformly and therefore produced a log-uniform mass ratio.
- Independent multi-motion adaptive sampling now reports the first four
  motions under the reference `motion_i_sampling_*` metric schema. Concatenated
  sampling keeps the unprefixed schema and all metrics still cross the device
  boundary in one transfer.
- MJLab camera calibration noise now perturbs camera-local ray starts and
  directions after the fixed mount pose, exactly like InstinctMJ. The reported
  camera pose remains the unperturbed mount pose; the old implementation
  composed rotation in the opposite order and reported the noisy pose.

Evidence:

```text
5 expected failures before the event/curriculum fixes
9 passed (new declaration, reset-effect, distribution, metric, and timing regressions)
287 passed, 1 deselected (Shadow/reference/compile/ray-caster focus)
1269 passed, 2 skipped, 32 deselected (full default suite)
1 passed, 6 deselected (MJLab CUDA fixed-state calibration-ray probe)
scripts/check_mjlab.py resolved all 39 Locomotion terms,
  constructed 16 environments in DFS action order, and stepped 5 times
```

No fresh Isaac environment was run or claimed in this phase. The three live
training processes were inspected and not signaled. In particular, the GPU 7
MJLab Perceptive process loaded the old inertia distribution and camera
calibration code before these commits; treat it as convergence diagnostics,
not post-fix randomization evidence.

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

## Reward, observation, command, and failure-term numerical audit (2026-08-27)

The active shared MDP terms were checked value-for-value against
`/root/InstinctLab-main` for Isaac and `/root/InstinctMJ` for MJLab. The
external RL runner was outside this audit. The reward, observation, and failure
pass found no production-code discrepancy.

Shadow imitation now has fixed-state guards for the reference XY/height anchor,
yaw-only relative-world correction, quaternion left-multiplication order,
base rotation error, and nonuniform multi-link `mean_prod` reduction. Failure
guards cover height-only versus full-distance checks, the selected link subset,
projected-gravity distance, and the first-reset-step accumulated contact gate.
These probes are nontrivial: the reference and robot have different world
origins and headings, and the selected links carry different errors.

The audit also reconfirmed that rewards read the separate current-time
`reference_frame`, while commands and failure checks read
`motion_reference.data` at its active aiming slot; both native builders resolve
the declared 14-link tuple in preserved order. Optional reference masking and
keyframe gating are inactive in the registered tasks (retargetted link/base
masks are all enabled and failure terms declare
`check_at_keyframe_threshold=-1`). Reward weights remain step-time scaled on
both engines.

The command follow-up found three interface differences whose active G1 values
had hidden them:

- The reference-anchored position command used `motion_reference.data[:, 0]`
  instead of the separate current-time `reference_frame`. Active Shadow data
  currently starts at `t`, so both happened to agree. The command now follows
  both references and uses the explicit current frame as its anchor.
- The joint-velocity command omitted the frozen default-velocity subtraction.
  G1 defaults are zero, so the omission was numerically silent. It now snapshots
  native defaults once and gathers them by joint name into canonical DFS order,
  matching the already-correct joint-position command.
- Non-realtime commands used `< step_dt` and reached through the sensor's
  private runtime. Both references use `< step_dt - 1e-6`. The shared command
  now reads the public `time_passed_from_update` and public `num_frames`
  properties with the reference boundary.

Evidence:

```text
5 passed (new fixed-state and temporal Shadow probes)
4 passed (new position-anchor, rotation-schema, DFS velocity-default, and refresh-window command probes)
128 passed (shared MDP, AMP, Parkour, and Shadow numerical focus)
136 passed, 1 deselected (reference declarations and native lowering focus)
115 passed (Shadow command, motion runtime, task declaration, and MDP focus)
1277 passed, 2 skipped, 32 deselected (full default suite)
scripts/check_mjlab.py resolved all 39 Locomotion terms,
  constructed 16 environments in canonical joint order, and stepped 5 times
```

## VAE and BeyondMimic runtime verification (2026-08-27)

The previously declaration-only VAE path was exercised through native MJLab
and Isaac environments. The diagnostic motion override now accepts a terrain
dataset directory and binds both the motion reference and motion-matched
terrain to that directory (`86cc601`). Using the installed
`deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1` data exposed
three consecutive VAE depth defects before the rollout could start:

- the shared depth term inherited Parkour's random one-frame delay even though
  both VAE references use non-delayed `visualizable_image`; ten history slots
  cannot hold frames `0, 3, 6, 9` plus that extra delay;
- the callable read `resize_shape` and `normalization_range` at construction but
  did not accept the observation manager's matching call arguments;
- its history ring was allocated at cropped source resolution even when the
  processed frame was resized to `18 x 32`.

Commit `082c2ae` fixes those contracts. VAE now has an explicit `(0, 0)` frame
delay and allocates its ring at the processed resolution. Commit `0a0f674`
adds a temporal probe proving that ten successive frames produce the ordered
policy sample `0, 3, 6, 9`, not merely the expected shape.

Fresh fixed-seed rollouts then completed on both engines:

```text
Perceptive VAE: 16 environments, reset plus 4 steps per engine
  policy depth (4, 18, 32), critic depth (1, 18, 32)
  identical initial DFS joint state/reference/action and identical done arrays
BeyondMimic: 16 environments, reset plus 4 steps per engine
  identical initial DFS joint state/reference/action and identical done arrays
1281 passed, 2 skipped, 32 deselected (full default suite)
scripts/check_mjlab.py resolved all 39 Locomotion terms,
  constructed 16 environments in canonical joint order, and stepped 5 times
```

Global root positions in the rollout files differ by engine terrain/environment
origin placement (up to 57 m for the VAE terrain grid). After subtracting the
motion-reference origin, the initial root error is exactly zero on both
engines. Four-step plant divergence is engine-native evidence, not an equality
requirement: VAE root-to-reference delta differs by at most `0.0321 m` and
BeyondMimic by `0.00372 m`; all initial joint and reference deltas are zero.

These were real environment construction and temporal rollouts. The released
VAE data is now the canonical engine-neutral dataset at
`/root/Datasets/deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1`
(`f0a9147`). Its metadata declares ten motions and six terrain meshes; every
declared file exists, all 8,816 declared frames load with finite tensors and
the exact G1 joint-name set, and URDF/MJCF kinematic reconstruction agrees to
within `5.513e-7 m` for link position, `7.451e-7` for link quaternion, and
`5.150e-5 rad/s` for link angular velocity. Both engine resolutions now point
to this same path.

A frozen diagnostic teacher bundle is installed at
`/root/Datasets/g1_perceptive_vae_teacher/mjlab_gpu7_iter22000`. It contains an
independent copy of the MJLab Perceptive `model_22000.pt`, the TPPO normalizer
metadata, source revisions, and an explicit `accepted_for_production: false`
manifest. The checkpoint SHA-256 is
`0c8c6a7a09cbc037e45f6bb2a36867e5400dfb325a30817932df6639a2e197da`;
strict teacher construction/loading succeeds and all checkpoint and
normalizer tensors are finite. The teacher predates the final Perceptive camera
and event-randomization fixes, so it is suitable for training-chain diagnosis,
not a promoted production teacher (`cdd5226`).

The first full-dataset 16-environment MJLab construction exposed sparse terrain
coverage: a small generated scene need not contain an origin for every motion
terrain ID. Main and InstinctMJ avoid those motions; the shared runtime instead
raised. Commit `979c030` now filters both independent motion weights and concat
motion-bin weights to compatible origins, preserving a matching start time,
while still rejecting a scene with no compatible motion at all.

The next run exposed that the VAE critic group was 1,966 values while TPPO
passes that group to the 1,990-value teacher policy. The declaration had both
omitted the teacher's 24-value projected-gravity history and used the world
position command instead of the teacher's body-relative position command.
Commit `d65b55b` restores the exact ordered teacher policy schema on the VAE
critic group; an effect-guard test fails if any ordered term is removed.

The corrected MJLab run
`20260828_111846_canonicaldata_mjteacher22000_smoke_retry2_16_seed123_gpu4_20260828`
completed two iterations and saved finite `model_2.pt` model, optimizer, and
normalizer state. Iteration 0 processed 384 transitions at 269 steps/s with
distillation/KL/total losses `5.6940/0.0283/5.7224`, mean reward `0.1906`, and
mean episode length `6.33`. This proves full-data construction, teacher action
generation, rollout, backpropagation, and checkpointing; it is not convergence
or teacher-quality evidence.

The same shared data and diagnostic teacher also completed the corresponding
16-environment, seed-123, two-iteration Isaac run
`20260828_113341_canonicaldata_mjteacher22000_smoke_isaac_16_seed123_gpu4_20260828`.
Its finite `model_2.pt` contains model, optimizer, and normalizer state.
Iteration 0 processed 384 transitions at 129 steps/s with
distillation/KL/total losses `3.2535/0.0168/3.2703`, mean reward `0.36`, and
mean episode length `7.20`. VAE therefore has a successful short training-chain
test on both engines; neither two-iteration run is a performance comparison.

HOI cannot be substituted safely: both OMOMO motion directories and all six
configured object meshes are absent, so it remains limited to the existing
fixed-state object-origin tests.

BeyondMimic's formal data gap was subsequently closed at `c68a1e6`. The
official LAFAN1 G1 CSV release was converted with a reproducible, tested
pelvis-to-torso-root transform. All 40 clips load under the production runtime,
contain finite values in canonical DFS order, have continuous base
quaternions, and remain within URDF joint position limits. The selected clip
contains 8,194 source frames; production resampling at 50 Hz produces 13,656
frames (273.1 seconds). A physical reconstruction test checks that the
conversion preserves pelvis, leg, and contact-link world transforms rather
than only checking array shape.

The same selected clip then completed the standard L7 long regression on both
engines:

```text
Task: Instinct-BeyondMimic-Plane-G1-v0
Configuration: 256 environments, seed 42, 700 iterations, strict mode
MJLab:    logs/mjlab/g1_beyondmimic/
          20260827_214821_official_lafan1_256_seed42_700_gpu1_retry1_20260827
Isaac Sim: logs/isaacsim/g1_beyondmimic/
          20260827_214841_official_lafan1_256_seed42_700_gpu0_retry1_20260827
Both: 49 terms resolved with none skipped, emulated, or omitted;
      finite model_700.pt written; task contract hash
      dc9a32ea68c2e87adf4bed552a2cc906aee00ab64ae44b984623685010c74420
```

At the last logged point (iteration 690), Isaac/MJLab reward was `1.33/1.34`,
episode length was `56.62/49.87`, and policy action standard deviation was
`0.550/0.551`. Over logged iterations 600--690, mean reward was `1.109/1.254`
and mean episode length was `47.26/47.43`. Both curves began near `-1.26`,
their non-timeout failure terms fired throughout the run, and no NaN, overflow,
or fatal simulator error occurred. The close return and episode curves are
cross-engine behavioral evidence; raw plant trajectories are intentionally not
required to match.

A formal BeyondMimic performance-baseline campaign started on 2026-08-28 from
the clean `9531ad4` worktree and runner `64d7e01`. It deliberately separates
two questions:

```text
production performance/capacity:
  4096 environments, seed 42, 30,000 iterations (the task's native maximum)
  GPU 0 Isaac Sim, GPU 1 MJLab, strict mode
seed noise at the comparable L7 scale:
  256 environments, seeds 43 and 44, 700 iterations
  GPU 2 Isaac Sim, GPU 3 MJLab, strict mode
  reuse the accepted seed-42 runs above as the third sample
```

Production run directories:

```text
logs/isaacsim/g1_beyondmimic/
  20260828_094446_official_lafan1_production_4096_seed42_30000_gpu0_20260828
logs/mjlab/g1_beyondmimic/
  20260828_094428_official_lafan1_production_4096_seed42_30000_gpu1_20260828
```

Both production builders resolved all 49 terms with none skipped, emulated, or
omitted and entered iteration zero. Initial Isaac/MJLab reward was
`-1.32/-1.31` and episode length was `16.99/17.26`. The initial throughput
was about 25,416/51,310 environment steps/s; the first ETA was approximately
32/16 hours. These runs are active, not yet accepted baselines. Promotion
requires the final checkpoint, finite optimization and observations, live
failure terms, no capacity warning, and comparison of normalized reward terms,
episode length, termination mix, action noise, and throughput.

An empty residual `/root/InstinctLab/scripts/instinct_rl/` directory initially
shadowed the installed runner as a Python namespace and prevented iteration
zero. It contained no files and was removed with `rmdir`; no runner source was
changed. Fresh clones do not contain an untracked empty directory, so the two
failed launch directories are retained only as diagnostics.

### Portable termination metric units

Commit `c90b24b` gives every `Episode_Termination/<term>` tag one shared
runner-level meaning: the fraction of all environments whose most recently
completed episode ended with that term. Values are therefore in `[0, 1]` and
independent of environment count. Terms can overlap, so their sum may exceed
one. Environments that have not yet completed an episode contribute zero.

This bridge is necessary because the native MJLab manager logs the raw number
of firings in the current reset batch, while the native Isaac manager logs a
last-completed-episode fraction. Both native managers remain unchanged; the two
`instinct_rl` environment wrappers replace those tags at their common boundary.
At matched iteration 390 of the 4,096-environment production runs, the old
Isaac/MJLab termination sums were `1.020467/42.958333`; MJLab's expected reset
batch size from episode length was `4096 / 95.67 = 42.81`, identifying the
factor as a unit mismatch rather than a termination-rate mismatch.

Fresh 16-environment, seed-123 BeyondMimic probes completed 30 iterations on
both engines and wrote finite `model_30.pt` checkpoints:

```text
MJLab: logs/mjlab/g1_beyondmimic/
  20260828_101553_termination_units_probe_16_seed123_gpu4_20260828
Isaac: logs/isaacsim/g1_beyondmimic/
  20260828_101716_termination_units_probe_16_seed123_gpu4_20260828
```

Every termination tag at iterations 0, 10, and 20 was within `[0, 1]` on both
engines. The focused wrapper/entry suite passed 45 tests, the full suite passed
1,288 tests with 2 skipped and 32 deselected, and a deliberate mean-to-sum
mutation failed the environment-count invariance tests.

The active 4,096-environment production processes were launched from
`9531ad4`, before this bridge existed, and were not restarted. Their existing
TensorBoard termination tags on port 6008 retain the old mixed units for the
life of those processes. Do not use those raw tags for cross-engine magnitude
comparison; new processes launched from `c90b24b` use the portable units.

## Live experiments

Snapshot at 2026-08-28 01:46 UTC. The old GPU 5 and GPU 6 runs were stopped and
replaced at explicit operator request. GPU 7 was not signaled.

| GPU | Run | Iteration | Reward | Episode length | Status |
|---:|---|---:|---:|---:|---|
| 0 | Isaac BeyondMimic `official_lafan1_production_4096_seed42_30000_gpu0_20260828` | 0 | -1.32 | 16.99 | live; formal production baseline, InstinctLab `9531ad4`, runner `64d7e01` |
| 1 | MJLab BeyondMimic `official_lafan1_production_4096_seed42_30000_gpu1_20260828` | 50 | 0.15 | 12.48 | live; formal production baseline, InstinctLab `9531ad4`, runner `64d7e01` |
| 2 | Isaac BeyondMimic L7 seed 43, then seed 44 | 30 | -0.47 | 7.90 | live; two sequential 256-environment noise-floor runs |
| 3 | MJLab BeyondMimic L7 seed 43, then seed 44 | 90 | -0.15 | 7.21 | live; two sequential 256-environment noise-floor runs |
| 5 | unified Isaac Whole Body `jointref_fixed_final_long_4096_gpu5_20260827` | 21360 | 19.96 | 279.51 | live; recovered from the temporary 3k--3.8k curriculum branch; fresh seed 42, InstinctLab `1ee8654`, runner `64d7e01` |
| 6 | unified Isaac Perceptive `jointref_fixed_final_long_4096_gpu6_20260827` | 9910 | 3.92 | 74.66 | live; fresh seed 42, InstinctLab `1ee8654`, runner `64d7e01` |
| 7 | unified MJLab Perceptive `stablecaps_final_long_4096_gpu7_20260826` | 21470 | 16.48 | 240.28 | live; natural DFS order avoids the Isaac BFS/DFS fault, but it predates camera and event-DR fixes |

Do not stop or restart these runs without an explicit operator request. Review
reward, episode length, termination mix, and action noise together before
promoting any run.

Current TensorBoard comparison links remain under:

```text
logs/tb_compare/g1_shadowing_diveroll/
logs/tb_compare/g1_perceptive_shadowing/
logs/tb_compare/g1_beyondmimic_production_4096_seed42/  (live on port 6008)
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
  `unspecified launch failure`, after 170,164,224 samples and about 10.6 hours.
  It is the only retained log with CUDA 719. Metrics immediately before the
  failure were finite and normal, and the error surfaced at a Torch
  `nonzero()` synchronization followed by repeated Warp free failures. No Xid
  or kernel log was retained, so neither that line nor the motion-reference
  code can be identified as the failing kernel. Classify it as an unreproduced
  asynchronous Warp/MuJoCo or device failure; keep the checkpoints and logs and
  do not describe it as a completed 50k run.
- The older MJLab Perceptive `perceptive_repro_4096_gpu2` reached iteration
  11750 and ended with `KeyboardInterrupt`. It predates final parity fixes and
  is not a baseline.
- The fixed-capacity MJLab Perceptive run failed after iteration 180 when the
  policy mean became NaN. Commit `6c472bf` restored the stable native capacity
  profile; the GPU 7 replacement above is the relevant run.
- Earlier Whole Body and Perceptive runs predating the final DFS reset,
  current-reference, history, terrain-match, or capacity fixes are diagnostic
  only. Do not resume or promote them.

## Perceptive bin-12 reset-contact diagnosis

The Isaac Perceptive stall is not a DFS/BFS or joint-initialization failure. A
fixed-policy, fixed-bin probe now evaluates the same `model_6000.pt` checkpoint
for 2,048 environments over 400 steps while holding adaptive sampling at global
bin 12 (`roadRamp_noWall`, diveroll 4--5 s). Same-engine variants have identical
start-time SHA-256 hashes.

| Engine/variant | Mean first episode steps | First `illegal_reset_contact` | Step 1/2 force >500 N |
|---|---:|---:|---:|
| Isaac default | 12.58 | 1,737 / 2,048 | 90.8% / 90.8% |
| Isaac MJ-style ground correction +0.1 m | 11.78 | 1,797 / 2,048 | not captured in this run |
| Isaac fixed +0.1 m only | 12.76 | 1,649 / 2,048 | 89.6% / 73.0% |
| Isaac fixed +0.2 m only | 6.60 | 1,336 / 2,048 | 88.6% / 76.2% |
| Isaac without the early-contact termination | 52.28 | disabled | 90.8% / 90.8% |
| Isaac with articulation self-collision off | 50.79 | 106 / 2,048 | 15.8% / 22.0% |
| MJLab default | 48.46 | 347 / 2,048 | 36.7% / 19.8% |
| MJLab without its reset lift | 44.13 | 419 / 2,048 | not captured in this run |

Isaac's step-one non-support contact median was 5,969 N versus 209 N on
MJLab. In Isaac, `pelvis`, `left_hip_roll_link`, and `right_hip_roll_link`
exceeded 500 N in 88.7%, 82.2%, and 71.9% of environments. MJLab's corresponding
rates were 7.1%, 0.5%, and below its top-six list. The MJCF explicitly excludes
`pelvis` against both hip-roll bodies (and elbow against wrist-pitch on both
arms); the Isaac URDF contains no pair filters while the articulation enables
self-collision. Disabling Isaac self-collision causally removes the episode
length gap, but is broader than an acceptable parity fix.

This difference already exists between `/root/InstinctLab-main` and
`/root/InstinctMJ`; the unified-task audit did not introduce it. Main and the
current Isaac asset both enable articulation self-collision. The current task's
Isaac reset values also match main (`ensure_link_below_zero_ground=False`,
height offset 0). The MJ reset lift explains only about 10% within MJLab and
does not fix Isaac.

The live Isaac log independently shows that while curriculum top-bin 12 held
from roughly iteration 4,900 through 10,000, its native early-contact
termination fraction stayed around 0.32--0.33 and mean episode length around
58--63. The raw-failure EMA then amplifies those resets into a curriculum lock;
random sampling determines how long recovery takes. Do not compare the old
MJLab termination tag numerically because that active run predates the portable
termination-unit bridge.

Probe reports and logs are under
`logs/diagnostics/perceptive_reset_bin12_model6000/`. The reusable probe is
`scripts/probe_perceptive_reset.py`; it hard-checks the fixed bin and reports
start hashes, early force quantiles, top bodies, survival, reward, and causes.
Commit `d8e8b24` fixes the Isaac profile interface so an explicit self-collision
override reaches both URDF conversion and articulation properties; it does not
change any task default. The four MJCF pair exclusions are now represented by
portable entity-pair relations and passed the 4,096-environment PhysX probe
recorded above. The next production experiment is therefore a fresh training
A/B using these narrow exclusions; global self-collision disablement is not the
production configuration.

## Open risks and next work

The two extension-platform P0 regressions recorded above take precedence before
using the production `train.py` or `play.py` entry points for Rough or Parkour;
do not treat the green default suite as acceptance evidence for those paths.

1. Let the retained GPU 6 Perceptive run continue; do not restart it or promote
   its checkpoint yet. The four MJCF-equivalent PhysX filtered pairs have passed
   cloning and 4,096-environment capacity probes. A fresh A/B is still required
   for training-behavior evidence; global self-collision disablement remains
   diagnostic evidence only.
2. The retained MJLab Whole Body CUDA 719 has no attributable source from the
   available log. A causal diagnosis requires an exact checkpoint/policy replay
   with synchronous CUDA diagnostics; do not blame the reported `nonzero()`
   synchronization point or resume the run as if it completed.
3. Train or recover a current post-fix Perceptive teacher, then run long VAE
   reproductions on both engines using the installed canonical dataset. The
   present MJLab teacher and two-iteration VAE run are diagnostic only. HOI
   still lacks both motions and all object meshes.
4. BeyondMimic has accepted official-data L7 evidence for seed 42. The
   multi-seed and 4,096-environment production campaign is running; wait for
   completion and evaluate the acceptance criteria above before promoting it as
   a production-throughput or paper-performance baseline. Its active processes
   predate the portable termination metric bridge, so normalize or regenerate
   those tags before comparing the termination mix.
5. Validate real multi-node distributed training.
6. Recover authoritative Parkour motion segment boundaries. The released NPZ
   concatenates clips without boundary metadata: 55 of 18,981 transitions
   exceed conservative discontinuity thresholds, and up to 2.81% of 10-frame
   AMP windows may cross a jump. Cross-engine parity is verified; source-data
   semantics are not.
7. Continue reducing engine-specific parameter overlays by translating shared
   semantic values in builders. Do not move task policy into an engine package
   to achieve this.
8. HOI rigid-object loading is now production-valid on Isaac Lab 5.1 and MJLab;
   missing resources fail in unified preflight before either SDK constructs a
   scene. Full HOI parity remains blocked by the absent OMOMO motion directories
   and six object meshes. After installing them, repeat strict preflight,
   construction, reset, object-reference, contact, and rollout probes on both
   engines.
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
