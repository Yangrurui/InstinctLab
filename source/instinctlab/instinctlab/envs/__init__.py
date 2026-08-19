"""Isaac Lab environment classes, exported lazily.

The unified environment that used to be re-exported here eagerly is gone; main's task registers
``InstinctRlEnv``, and the compiler stack builds Isaac Lab's own ``ManagerBasedRLEnv``. Lazy so
that importing this package does not pull in Isaac Sim.
"""

def __getattr__(name: str):
    if name == "InstinctRlEnv":
        from .manager_based_rl_env import InstinctRlEnv

        return InstinctRlEnv
    if name == "InstinctLabRLEnvCfg":
        from .manager_based_rl_env_cfg import InstinctLabRLEnvCfg

        return InstinctLabRLEnvCfg
    raise AttributeError(name)


__all__ = ["InstinctLabRLEnvCfg", "InstinctRlEnv"]
