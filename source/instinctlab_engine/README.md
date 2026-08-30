# InstinctLab Engine Core

Stable, engine-neutral contracts used by task packages and native simulator
backends. Importing `instinctlab_engine` does not import Isaac Sim, Isaac Lab,
MJLab, MuJoCo, or Warp.

Native backend implementations are installed separately. Task packages should
depend on this package rather than importing a simulator SDK.

External packages extend the boundary through Python entry points:

- `instinctlab.engines` registers an independently installed backend.
- `instinctlab.assets` registers an engine-neutral asset package resolver.
- `instinctlab.engine_terms` registers native term lowering under an
  `<engine>.<extension>` name.
- `instinctlab.terrains` registers whole-terrain and generated-tile builders.
