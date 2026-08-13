"""Engine-neutral locomotion MDP terms.

Legacy Isaac-specific terms remain available through lazy attribute lookup.
"""

from . import unified
from .unified import *
from .unified import __all__ as _UNIFIED_EXPORTS


def __getattr__(name: str):
    from importlib import import_module

    for module_name in (
        "instinctlab.tasks.locomotion.mdp.rewards",
        "instinctlab.tasks.locomotion.mdp.curriculums",
        "isaaclab.envs.mdp",
    ):
        module = import_module(module_name)
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(name)


__all__ = [*_UNIFIED_EXPORTS, "unified"]
