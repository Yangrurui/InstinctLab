"""Engine-neutral managers with lazy Isaac compatibility exports."""

from .unified import *
from .unified import __all__ as _UNIFIED_EXPORTS

_LEGACY_EXPORTS = {"DummyRewardCfg", "MultiRewardCfg", "MultiRewardManager"}


def __getattr__(name: str):
    if name in {"DummyRewardCfg", "MultiRewardCfg"}:
        from . import manager_term_cfg

        return getattr(manager_term_cfg, name)
    if name == "MultiRewardManager":
        from .reward_manager import MultiRewardManager

        return MultiRewardManager
    raise AttributeError(name)


__all__ = [*_UNIFIED_EXPORTS, *_LEGACY_EXPORTS]
