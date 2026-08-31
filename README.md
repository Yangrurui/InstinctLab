# Project Instinct

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.2-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

## Overview

This repository is the environment side of [Project-Instinct](https://project-instinct.github.io/).

We aim at industralize Reinforcement Learning for Humanoid (legged robots) whole-body control.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.
- `Unified Ecosystem` This repository is a part of the Project-Instinct ecosystem, which includes the [instinct_rl](https://github.com/project-instinct/instinct_rl) and [instinct_onboard](https://github.com/project-instinct/instinct_onboard) repositories.
    - The core design of this ecosystem is to treat each experiment as a standalone structured folder, which start with a timestamp as a unique identifier.
    - Adding `--exportonnx` flag to the `play.py` script will export the policy as an ONNX model. After that, you should directly copy the logdir to the robot computer and use the `instinct_onboard` workflow to run the policy on the real robot.

**Keywords:** humanoid, reinforcement learning, Isaac Sim, MJLab

## Warning
This codebase is under [CC BY-NC 4.0 license](LICENSE), with inherited license in IsaacLab. You may not use the material for commercial purposes, e.g., to make demos to advertise your commercial products or wrap the code for your own commercial purposes.

## Contributing
See our [Contributor Agreement](CONTRIBUTOR_AGREEMENT.md) for contribution guidelines. By contributing or submitting a pull request, you agree to transfer copyright ownership of your contributions to the project maintainers.

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for a list of acknowledged contributors.

## Installation

Use **pip** (not uv) in a Python 3.11 conda environment. Installing this project can pull **Isaac Lab** and **MJLab** in the same command.

- Install [Isaac Sim 5.1.0](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) if you need the Isaac Sim backend (`isaacsim[all,extscache]==5.1.0`).

- Install Instinct-RL:

    ```bash
    git clone https://github.com/project-instinct/instinct_rl.git
    python -m pip install -e instinct_rl
    ```

- Clone this repository **outside** the Isaac Lab tree:

    ```bash
    git clone https://github.com/project-instinct/instinctlab.git
    cd instinctlab
    ```

- Install InstinctLab **with both simulator backends** (recommended). This clones Isaac Lab (`f73c331738`) and MJLab (`v1.5.0`) next to this repo if they are missing, then installs engine core, both backend packages, and the task application:

    ```bash
    python scripts/install.py
    ```

    Existing Isaac Lab and MJLab checkouts must be at the pinned revision and
    clean. The installer refuses a mismatch by default. For a deliberate local
    experiment, `--allow-unverified-checkouts` overrides the refusal and records
    the exact commit, dirty state, and override in
    `<python-prefix>/share/instinctlab/install_provenance.json`.

    The MJLab backend is installed with the InstinctMJ-compatible physics stack
    (`mujoco==3.10.0`, `mujoco-warp==3.10.0.1`, `warp-lang==1.14.0`). These are exact pins: newer
    patch releases change contact and constraint kernels and are not treated as
    the same training plant. The shared Isaac/MJLab environment keeps its existing
    compatible Torch build; Torch is not upgraded by these pins.

    Equivalent pip-only installation (no sibling checkouts; Isaac Lab comes
    from git):

    ```bash
    python -m pip install -e source/instinctlab_engine
    python -m pip install -e source/instinctlab_engine_isaacsim
    python -m pip install -e source/instinctlab_engine_mjlab
    python -m pip install -e source/instinctlab
    ```

    Install a single backend extra if needed:

    ```bash
    python -m pip install -e source/instinctlab_engine
    python -m pip install -e source/instinctlab_engine_isaacsim
    python -m pip install -e source/instinctlab  # Isaac only

    python -m pip install -e source/instinctlab_engine_mjlab
    python -m pip install -e source/instinctlab  # MJLab only
    ```

### Dataset mounts

Task declarations use portable `dataset://...` paths instead of server-specific
home-directory links. They resolve below `INSTINCTLAB_DATA_ROOT`, which defaults
to `~/Datasets`. Verify the mounted data and write a portable receipt before a
production run:

```bash
export INSTINCTLAB_DATA_ROOT=/absolute/path/to/Datasets
PYTHONPATH=source/instinctlab_engine python scripts/verify_datasets.py \
  --receipt dataset-verification.json
```

The versioned expected paths and SHA-256 values live in
`datasets/manifest.json`. Run manifests retain both the readable `dataset://`
declaration and its resolved local path. The optional OMOMO entry is recorded
as unavailable and is checked only when explicitly selected or when
`--include-optional` is used.

### Container image

The application image builds the four coordinated InstinctLab wheels in an
isolated Python 3.11 stage and installs them onto an externally supplied,
immutable dual-backend runtime. Isaac Sim is not rebuilt or redistributed here:
its runtime/EULA, NVIDIA driver boundary, and large SDK caches belong in the
site-managed base image. That base must contain the exact packages in
`docker/runtime-lock.json` and an
`/opt/instinctlab-runtime/runtime_provenance.json` receipt matching
`docker/runtime-provenance.example.json`.

The maintainer of that base writes the receipt only from clean pinned source
checkouts:

```bash
python scripts/write_container_runtime_provenance.py \
  --lock docker/runtime-lock.json \
  --isaaclab-checkout /workspace/IsaacLab \
  --mjlab-checkout /workspace/mjlab \
  --output /opt/instinctlab-runtime/runtime_provenance.json
```

Use an immutable digest, not a mutable tag:

```bash
export INSTINCTLAB_RUNTIME_IMAGE='registry/image@sha256:<64-hex-digest>'
export INSTINCTLAB_WHEEL_BUILDER_IMAGE='python:3.11-slim@sha256:<64-hex-digest>'
export INSTINCTLAB_HOST_DATA_ROOT=/absolute/path/to/Datasets
docker/docker-compose.sh

# Inside the container, validate the read-only dataset mount.
PYTHONPATH= python /opt/instinctlab/bin/verify_datasets.py \
  --manifest /opt/instinctlab/dataset-manifest.json \
  --receipt /workspace/logs/dataset-verification.json
```

The image build fails if the base reference is not digest-pinned, its receipt
does not name the pinned IsaacLab/MJLab commits, any locked SDK distribution is
missing or has drifted, or a coordinated wheel checksum/version is wrong. The
repository itself and sibling checkouts are not bind-mounted into the operator
container.

- Train any registered unified task on either engine. One declaration, compiled by the chosen engine's
  adapter; `--headless` and the other launch flags belong to the engine and are accepted only where
  it defines them.

    ```bash
    python scripts/train.py --engine isaacsim --task Instinct-Velocity-Flat-G1 --num_envs 4096 --device cuda:0 --headless
    python scripts/train.py --engine mjlab    --task Instinct-Velocity-Flat-G1 --num_envs 4096 --device cuda:1
    ```

## Documentation of Critical Components

- [Instinct-RL Documentation](https://github.com/project-instinct/instinct_rl/blob/main/README.md)
- [InstinctLab Documentation](DOCS.md)

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu. When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory. The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse. This helps in indexing all the python modules for intelligent suggestions while writing code.


## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

To make the `pre-commit` run automatically on every commit, you can use the following command in your repository:

```bash
pre-commit install
```

## Add a task

Define one engine-neutral task factory under `source/instinctlab/instinctlab/tasks/`. Keep the
concrete environment values together in that task's config file and return a `TaskSpec`. Register
the factory path once in `tasks/registry.py`; do not add a Gym registration or an engine-specific
entry point.

```python
# source/instinctlab/instinctlab/tasks/my_task/config.py
from instinctlab_engine.spec import RobotSpec, TaskSpec

def my_task(robot: RobotSpec) -> TaskSpec:
    return TaskSpec(robot=robot, ...)

# source/instinctlab/instinctlab/tasks/registry.py
TASKS["Instinct-My-Task"] = "instinctlab.tasks.my_task.config:my_task"
```

Both engines then use the same commands:

```bash
python scripts/train.py --engine isaacsim --task Instinct-My-Task
python scripts/train.py --engine mjlab --task Instinct-My-Task
```

## Extend without editing a backend

Portable observations, rewards, terminations, commands, and events carry their
task-owned callable and need no backend registration. Native extensions ship as
separate Python packages using entry points:

```toml
[project.entry-points."instinctlab.assets"]
my_robot = "my_robot_assets.interface:native_module"

[project.entry-points."instinctlab.engine_terms"]
"mjlab.my_randomizer" = "my_extension.mjlab:register_terms"

[project.entry-points."instinctlab.terrains"]
my_terrain = "my_extension.terrain:register_terrains"

[project.entry-points."instinctlab.engines"]
myengine = "my_engine_backend:register"
```

An engine-term registrar receives the selected backend's `TermRegistry`; a
terrain registrar receives the shared terrain registry. Keep simulator imports
inside native builders so discovery remains SDK-free.

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing. In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

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
