"""Isaac Lab manager extensions, exported lazily.

Only the multi-reward pieces live here now; the unified managers they used to sit beside are gone.
Lazy so that importing this package does not pull in Isaac Sim, which ``InstinctRlEnv`` relies on
when it asks for ``MultiRewardCfg`` before the app has started.
"""

def __getattr__(name: str):
    if name in {"DummyRewardCfg", "MultiRewardCfg"}:
        from . import manager_term_cfg

        return getattr(manager_term_cfg, name)
    if name == "MultiRewardManager":
        from .reward_manager import MultiRewardManager

        return MultiRewardManager
    raise AttributeError(name)


__all__ = ["DummyRewardCfg", "MultiRewardCfg", "MultiRewardManager"]
