# InstinctLab core concepts

InstinctLab declares each task once as an engine-neutral `TaskSpec`, then lowers
that declaration to Isaac Sim or MJLab after the launcher selects a backend.
`README.md` contains installation and command-line setup. This document covers
the package boundaries used when extending the codebase.

## Dependency direction

```text
task config and task-local MDP
              |
              v
       spec / compat / assets
              |
              v
        selected engine adapter
              |
              v
       Isaac Lab or MJLab SDK
```

- Task declarations never import an engine implementation.
- Engine implementations never import a task package or one another.
- `spec/` contains declarations and validation, not native SDK objects.
- `compat/` contains small engine-neutral readers and tensor operations.
- Native assets live in `assets/<robot>/<engine>.py` and are selected only
  after the engine is known.
- Native scene, event, sensor, and simulator lowering lives under
  `engines/<engine>/`.

The shared launchers are `scripts/train.py` and `scripts/play.py`. They select
the engine before importing its SDK.

Playback is an application layer, not part of `EngineAdapter`. Native and
Viser handlers are registered lazily under `instinctlab.play`; therefore an
engine adapter can be packaged and used for compilation or training without
depending on the playback UI. An external viewer can register a handler with
`register_player(engine, viewer, "module:function")`.

## Task ownership

Every active task ID is registered in
`source/instinctlab/instinctlab/tasks/registry.py`. A task family owns its
complete MDP implementation under `tasks/<family>/mdp/` and keeps its concrete
configuration in named environment, reward, and observation classes.

Use `TaskSpec` for the stable contract and selected components. Reward weights,
physics values, datasets, native profiles, and runner settings belong to the
owning task configuration or backend builder, not to the shared schema.

## Motion reference

Tasks declare motion input with `MotionReferenceRef`. Shared clip loading,
sampling, buffers, symmetry, and timing live in the top-level
`motion_reference/` package. Each engine supplies only the native sensor
lifecycle that exposes that shared runtime to its scene.

The ordered reference contract is the same on both engines:

- `reference_frame` is the current-time sample at `t`, used by AMP;
- sensor `data` starts at `t + dt` and contains look-ahead frames;
- joint axes follow `RobotSpec.joint_names`, the canonical DFS order;
- engine-native articulation indices are resolved by name rather than written
  positionally.

Motion commands, rewards, observations, reset events, and terminations are
implemented by the owning task family's `mdp/` package.

## Reward groups

Reward groups are declared directly in `MdpSpec`. Both backends preserve group
and term declaration order, then lower the groups to the native reward-manager
shape. There is no separate InstinctLab reward manager or environment subclass.

## Terrain extensions

The built-in scene adapters understand the common plane and generated-terrain
contracts. A new independent terrain does not require editing an existing
engine scene builder. Register lazy native builders instead:

```python
from instinctlab.engines import register_sub_terrain, register_terrain

register_terrain(
    "isaacsim",
    "my_world",
    "my_package.isaacsim:build_world",
)
register_terrain(
    "mjlab",
    "my_world",
    "my_package.mjlab:build_world",
)

register_sub_terrain(
    "isaacsim",
    "my_tile",
    "my_package.isaacsim:build_tile",
)
register_sub_terrain(
    "mjlab",
    "my_tile",
    "my_package.mjlab:build_tile",
)
```

A whole-terrain builder receives `(TerrainSpec, engine_profile)`. A tile
builder receives `(SubTerrainSpec, TerrainGeneratorSpec)`. Dotted builder paths
are resolved only for the selected backend, so registering MJLab support does
not import MJLab while Isaac Sim is starting.

Installed extension packages may publish a registrar through the
`instinctlab.terrains` Python entry-point group. The registrar receives the
terrain extension registry and registers the backend implementations it
provides. If a kind is not registered for the selected backend, compilation
fails explicitly.

## Verification

Choose checks appropriate to the change:

```bash
PYTHONPATH=source/instinctlab /root/miniconda3/envs/env_isaaclab/bin/python -m pytest -q tests
PYTHONPATH=source/instinctlab /root/miniconda3/envs/env_isaaclab/bin/python scripts/check_mjlab.py
PYTHONPATH=source/instinctlab /root/miniconda3/envs/env_isaaclab/bin/python -m pytest -q tests/test_engine_task_layer_isolation.py
```

Construction alone is not physics parity evidence. Term, timer, contact,
terrain, and motion-reference changes need fixed-state or temporal probes in
addition to configuration compilation.
