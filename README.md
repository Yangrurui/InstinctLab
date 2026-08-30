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

**Keywords:** extension, template, isaaclab

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

- Install InstinctLab **with both simulator backends** (recommended). This clones Isaac Lab (`f73c331738`) and MJLab (`v1.5.0`) next to this repo if they are missing, then `pip install -e` all three:

    ```bash
    python scripts/install.py
    ```

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
def my_task() -> TaskSpec:
    return TaskSpec(...)

# source/instinctlab/instinctlab/tasks/registry.py
TASKS["Instinct-My-Task"] = "instinctlab.tasks.my_task.config:my_task"
```

Both engines then use the same commands:

```bash
python scripts/train.py --engine isaacsim --task Instinct-My-Task
python scripts/train.py --engine mjlab --task Instinct-My-Task
```

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
