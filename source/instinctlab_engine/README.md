# InstinctLab Engine Core

Stable, engine-neutral contracts used by task packages and native simulator
backends. Importing `instinctlab_engine` does not import Isaac Sim, Isaac Lab,
MJLab, MuJoCo, or Warp.

Native backend implementations are installed separately. Task packages should
depend on this package rather than importing a simulator SDK.
