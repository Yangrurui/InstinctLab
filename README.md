# Project Instinct

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.2-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

InstinctLab is the environment layer of
[Project Instinct](https://project-instinct.github.io/). It declares humanoid
reinforcement-learning tasks once and compiles the same task contract for Isaac
Sim or MJLab. Training uses
[Instinct-RL](https://github.com/project-instinct/instinct_rl), and exported
policies can be deployed with
[instinct_onboard](https://github.com/project-instinct/instinct_onboard).

> [!WARNING]
> This project uses the [CC BY-NC 4.0 license](LICENSE), together with inherited
> Isaac Lab licensing. Commercial use is not permitted; this includes product
> advertising demos and commercial wrappers.

## Quick start

### 1. Install

Use pip (not uv) in a Python 3.11 conda environment. Clone InstinctLab outside
the Isaac Lab source tree:

```bash
conda create -n instinctlab python=3.11
conda activate instinctlab

git clone https://github.com/project-instinct/instinct_rl.git
python -m pip install -e instinct_rl

git clone https://github.com/project-instinct/instinctlab.git
cd instinctlab
python scripts/install.py
```

The installer prepares the engine core, both backends, and the task application.
It clones the pinned Isaac Lab (`f73c331738`) and MJLab (`v1.5.0`) revisions next
to this repository when they are absent. Existing dependency checkouts must be
clean and at those revisions. A deliberate local experiment can bypass that
check with `--allow-unverified-checkouts`; the installer records the exact
revision, dirty state, and override in
`<python-prefix>/share/instinctlab/install_provenance.json`.

Install [Isaac Sim 5.1.0](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
when using the Isaac Sim backend (`isaacsim[all,extscache]==5.1.0`). The MJLab
backend uses the InstinctMJ-compatible physics stack:
`mujoco==3.10.0`, `mujoco-warp==3.10.0.1`, and `warp-lang==1.14.0`. These exact
versions are part of the training plant because newer patch releases can change
contact and constraint kernels.

For a pip-only editable install without sibling checkouts, install the packages
in dependency order:

```bash
python -m pip install -e source/instinctlab_engine
python -m pip install -e source/instinctlab_engine_isaacsim
python -m pip install -e source/instinctlab_engine_mjlab
python -m pip install -e source/instinctlab
```

For a single backend, omit the other backend package. The application can also
install published backend extras with `instinctlab[isaaclab]`,
`instinctlab[mjlab]`, or `instinctlab[all]` once coordinated packages have been
published.

### 2. Configure datasets

Tasks use portable `dataset://...` URIs instead of server-specific home paths.
Set the data root, verify the versioned manifest, and keep the receipt with the
run:

```bash
export INSTINCTLAB_DATA_ROOT=/absolute/path/to/Datasets
PYTHONPATH=source/instinctlab_engine/src python scripts/verify_datasets.py \
  --receipt dataset-verification.json
```

`INSTINCTLAB_DATA_ROOT` defaults to `~/Datasets`. Expected relative paths and
SHA-256 values live in [datasets/manifest.json](datasets/manifest.json). URI
resolution is fail-closed: encoded separators, traversal, and symlink escapes
cannot leave the configured root, and manifest resources and conversion targets
must remain below that root. Run manifests retain both the portable URI and its
resolved local path.

The optional OMOMO entry is verified only when selected explicitly or when
`--include-optional` is passed.

### 3. List and train tasks

List every engine-neutral task ID without importing a simulator SDK:

```bash
python scripts/list_envs.py
```

Train the same task on either backend:

```bash
python scripts/train.py \
  --engine isaacsim \
  --task Instinct-Velocity-Flat-G1 \
  --num_envs 4096 \
  --device cuda:0 \
  --headless

python scripts/train.py \
  --engine mjlab \
  --task Instinct-Velocity-Flat-G1 \
  --num_envs 4096 \
  --device cuda:0
```

The selected engine contributes its own launch flags, so options such as
`--headless` are accepted only when that backend defines them. By default,
training requires a clean task compilation: no term resolution may be skipped,
emulated, or profile-omitted. `--allow-nonclean-resolution` is an explicit
diagnostic override and is recorded in the run's `manifest.json`.

Runs are written below `logs/<engine>/<experiment>/`. Useful shared options
include `--seed`, `--max_iterations`, `--logroot`, and `--run_name`; for
example, `python scripts/train.py --engine mjlab --help` includes the MJLab
backend's options.

Each primary training run stores checkpoints together with `git/*.diff`,
`params/env.yaml`, `params/agent.yaml`, and `manifest.json`. The YAML layout
matches InstinctMJ and records the final post-override configuration used to
construct the runner.

### 4. Resume or transfer a checkpoint

Use strict resume only when the task and effective agent contract are unchanged:

```bash
python scripts/train.py \
  --engine mjlab \
  --task Instinct-Velocity-Flat-G1 \
  --resume \
  --checkpoint /absolute/path/to/run/model_1000.pt \
  --device cuda:0
```

Strict resume requires the adjacent `manifest.json`. It verifies task identity,
the task contract, and the effective agent configuration before restoring the
model, optimizer, normalizers, and learning iteration. Supplying
`--load_run` or `--checkpoint` without a mode also defaults to strict resume.

Use transfer when a declared task or agent change is intentional, or when
loading a legacy checkpoint without the current strict metadata:

```bash
python scripts/train.py \
  --engine mjlab \
  --task Instinct-Velocity-Flat-G1 \
  --transfer \
  --checkpoint /absolute/path/to/run/model_1000.pt \
  --device cuda:0
```

Transfer performs a permissive runner-state load and starts the learning
iteration at zero. Both modes create and reset a fresh environment; neither
restores simulator lifecycle snapshots or common RNG state.

### 5. Play and export a policy

Play a trained checkpoint. `auto` chooses the native viewer when a display is
available and Viser otherwise:

```bash
python scripts/play.py \
  --engine mjlab \
  --task Instinct-Velocity-Flat-G1 \
  --checkpoint /absolute/path/to/run/model_1000.pt \
  --viewer auto \
  --device cuda:0
```

To smoke-test an environment without a checkpoint, use `--agent zero` or
`--agent random`. Export requires a trained policy and exactly one environment:

```bash
python scripts/play.py \
  --engine mjlab \
  --task Instinct-Velocity-Flat-G1 \
  --checkpoint /absolute/path/to/run/model_1000.pt \
  --num_envs 1 \
  --export-onnx \
  --export-only \
  --deployment-runtime onnxruntime \
  --export-dir /absolute/path/exported \
  --device cuda:0
```

Install the verifier with
`python -m pip install -e "source/instinctlab[deployment]"`. Export requires a
new or empty directory and produces exactly one self-contained file:
`/absolute/path/exported/policy.onnx`. Observation normalization, policy
encoders, provenance, the fixed I/O contract, and a numerical self-test are
embedded in the model; no data or normalizer sidecar is required.

Verify the copied model on the target machine. Supply a trusted SHA-256 from
release or transfer metadata when promoting it:

```bash
python scripts/verify_deployment.py /absolute/path/exported/policy.onnx \
  --runtime onnxruntime \
  --sha256 <64-hex-policy-digest> \
  --max-p95-latency-ms <target-budget-ms> \
  --report /absolute/path/release-evidence/policy-verification.json
```

The verifier checks the external digest when supplied, the embedded content
checksum, ONNX structure and fixed I/O shape, PyTorch/ONNX numerical parity,
finite outputs, and optional target-hardware p95 latency. It writes no sidecar
unless `--report` is explicitly given. Copy only `policy.onnx` to the robot
computer for the `instinct_onboard` workflow.

## Operator container

The operator image installs four coordinated InstinctLab artifacts onto an
externally supplied immutable dual-backend runtime. Isaac Sim is not rebuilt or
redistributed here: the runtime/EULA, NVIDIA driver boundary, and SDK caches
belong in the site-managed base image.

The base must contain the exact packages in
[docker/runtime-lock.json](docker/runtime-lock.json) and a receipt at
`/opt/instinctlab-runtime/runtime_provenance.json` matching
[docker/runtime-provenance.example.json](docker/runtime-provenance.example.json).
Create that receipt only from clean pinned dependency checkouts:

```bash
python scripts/write_container_runtime_provenance.py \
  --lock docker/runtime-lock.json \
  --isaaclab-checkout /workspace/IsaacLab \
  --mjlab-checkout /workspace/mjlab \
  --output /opt/instinctlab-runtime/runtime_provenance.json
```

Build from a clean InstinctLab checkout and an absent or empty `dist/release`
directory. Use a runtime digest, never a mutable image tag:

```bash
export INSTINCTLAB_RUNTIME_IMAGE='registry/image@sha256:<64-hex-digest>'
export INSTINCTLAB_HOST_DATA_ROOT=/absolute/path/to/Datasets
export INSTINCTLAB_BUILD_PYTHON=/path/to/python3.11
docker/docker-compose.sh

# Inside the container, validate the read-only dataset mount.
PYTHONPATH= python /opt/instinctlab/bin/verify_datasets.py \
  --manifest /opt/instinctlab/dataset-manifest.json \
  --receipt /workspace/logs/dataset-verification.json
```

The helper builds wheels and source archives only from Git-tracked files at the
current clean commit, verifies their versions and checksums, and binds that
commit into the image. The build fails when the runtime is not digest-pinned,
its provenance or locked dependencies drift, imports fail, or coordinated
artifacts disagree. The repository and sibling checkouts are not bind-mounted
into the running container.

## Development

### Add a task

Each public task has one engine-neutral configuration and one immutable
registration in
[`source/instinctlab/instinctlab/tasks/registry.py`](source/instinctlab/instinctlab/tasks/registry.py).
Do not add Gym registrations or backend-specific task entry points.

For Locomotion, Parkour, and Shadowing task families:

1. Put the robot-neutral base `EnvCfg`, `RewardsCfg`, observation configuration
   classes, and the complete family-owned MDP implementation outside robot
   directories.
2. Add one concrete robot `EnvCfg` that inherits from the base once. Write final
   parameter values and complete `EntityRef` selectors directly where they are
   consumed.
3. Add a small registry-boundary factory that instantiates the complete config
   and converts it to `TaskSpec`. Do not dispatch variants or carry reward and
   observation override dictionaries through the factory.
4. Register the factory together with its engine-neutral asset ID:

```python
_REGISTRATIONS["Instinct-My-Task"] = TaskRegistration(
    factory_path="instinctlab.tasks.my_task.config.my_robot:my_robot_task",
    asset_id="my_robot/default",
)
```

The factory accepts the `RobotSpec` already normalized by the selected engine:

```python
def my_robot_task(robot: RobotSpec) -> TaskSpec:
    config = MyRobotEnvCfg(robot)
    return TaskSpec(
        task_id="Instinct-My-Task",
        robot=config.robot,
        scene=config.scene,
        sim=config.sim,
        mdp=MdpSpec(...),
        agent=config.agent,
        engines=("isaacsim", "mjlab"),
    )
```

Both engines can then compile the same declaration:

```bash
python scripts/train.py --engine isaacsim --task Instinct-My-Task
python scripts/train.py --engine mjlab --task Instinct-My-Task
```

The checked-in onboarding example is
`Instinct-Velocity-Flat-G1-15DoF`. It selects
`unitree_g1/popsicle_torsobase_locked_arms_v1`, whose URDF and MJCF lock all 14
arm joints while retaining the 12 leg and 3 waist joints. The task therefore
has a real 15-dimensional native action axis on both engines; it is not a
29-DoF model with a masked policy output. Run its fast contract checks and
optional live construction probes with:

```bash
PYTHONPATH=source/instinctlab python -m pytest -q \
  tests/test_g1_15dof_onboarding.py

INSTINCTLAB_LIVE_DEVICE=cuda:2 python -m pytest -q -o addopts= \
  -m mjlab tests/test_g1_15dof_mjlab_live.py
INSTINCTLAB_LIVE_DEVICE=cuda:2 python -m pytest -q -o addopts= \
  -m isaacsim tests/test_g1_15dof_isaacsim_live.py
```

See [AGENTS.md](AGENTS.md) for the detailed dependency and configuration rules
used in this repository.

### Extend without editing a backend

Portable observations, rewards, terminations, commands, and events carry their
task-owned callables and need no backend registration. Native extensions ship
as separate Python packages using entry points:

```toml
[project.entry-points."instinctlab.assets"]
my_robot = "my_robot_assets.interface:native_module"

[project.entry-points."instinctlab.actuators"]
"mjlab.my_controller" = "my_extension.mjlab:register_actuators"

[project.entry-points."instinctlab.engine_terms"]
"mjlab.my_randomizer" = "my_extension.mjlab:register_terms"

[project.entry-points."instinctlab.terrains"]
my_terrain = "my_extension.terrain:register_terrains"

[project.entry-points."instinctlab.engines"]
myengine = "my_engine_backend:register"
```

Registrars receive the relevant shared registry. Keep simulator imports inside
native builders so task discovery remains SDK-free.

### Formatting

Install and run the repository hooks:

```bash
python -m pip install pre-commit
pre-commit run --all-files
pre-commit install
```

### VS Code setup (optional)

Open the command palette with `Ctrl+Shift+P`, choose `Tasks: Run Task`, and run
`setup_python_env`. Enter the absolute Isaac Sim path when prompted. The task
creates `.vscode/.python.env` so Pylance can index the simulator extensions.

If an extension is still missing from the index, add its source path under
`python.analysis.extraPaths` in `.vscode/settings.json`:

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/instinctlab",
        "<path-to-ext-repo>/source/instinctlab_engine/src",
        "<path-to-ext-repo>/source/instinctlab_engine_isaacsim/src",
        "<path-to-ext-repo>/source/instinctlab_engine_mjlab/src"
    ]
}
```

## Release operators

InstinctLab publishes four distributions with one coordinated semantic version:
engine core, Isaac Sim backend, MJLab backend, and the task application. Build
from a clean checkout into an absent or empty output directory:

```bash
python scripts/check_release.py --expected-version VERSION
python scripts/check_release_handoff.py
python scripts/build_release.py --expected-version VERSION
```

A `vX.Y.Z` tag runs four gates on the same commit: the fast suite, isolated
wheel matrix, live GPU checks, and the release-candidate operator workflow. The
protected release-candidate environment must configure `INSTINCTLAB_DATA_ROOT`,
`INSTINCTLAB_RUNTIME_IMAGE`, `INSTINCTLAB_RELEASE_DEVICE`,
`INSTINCTLAB_RELEASE_NUM_ENVS`, `INSTINCTLAB_ISAACSIM_LIFECYCLE_THRESHOLDS`, and
`INSTINCTLAB_MJLAB_LIFECYCLE_THRESHOLDS`. Publication is a separate dispatch
and fails closed unless all four tagged workflows succeeded.

See [RELEASE.md](RELEASE.md) for the complete gate and publication checklist.
Server locations, accepted evidence, unfinished operational work, and known
risks are maintained in [HANDOFF.md](HANDOFF.md).

## Documentation and contributing

- [InstinctLab critical components](DOCS.md)
- [Instinct-RL documentation](https://github.com/project-instinct/instinct_rl/blob/main/README.md)
- [Contributor Agreement](CONTRIBUTOR_AGREEMENT.md)
- [Acknowledged contributors](CONTRIBUTORS.md)

By contributing or submitting a pull request, you agree to the contributor
agreement and transfer copyright ownership of the contribution to the project
maintainers.
