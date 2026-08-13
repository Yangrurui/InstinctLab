"""Unified environment plus lazy legacy Isaac exports."""

from .unified_manager_based_rl_env import (
    InstinctManagerBasedRLEnv,
    InstinctManagerBasedRlEnv,
    UnifiedManagerBasedRLEnv,
    UnifiedManagerBasedRLEnvCfg,
)


def __getattr__(name: str):
    if name == "InstinctRlEnv":
        from .manager_based_rl_env import InstinctRlEnv

        return InstinctRlEnv
    if name == "InstinctLabRLEnvCfg":
        from .manager_based_rl_env_cfg import InstinctLabRLEnvCfg

        return InstinctLabRLEnvCfg
    raise AttributeError(name)


__all__ = [
    "InstinctManagerBasedRLEnv",
    "InstinctManagerBasedRlEnv",
    "InstinctRlEnv",
    "InstinctLabRLEnvCfg",
    "UnifiedManagerBasedRLEnv",
    "UnifiedManagerBasedRLEnvCfg",
]
