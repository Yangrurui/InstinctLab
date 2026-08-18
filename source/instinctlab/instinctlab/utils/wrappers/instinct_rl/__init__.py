"""Instinct-RL configuration classes and environment wrappers.

The configuration modules are engine-free; the vecenv wrappers are not, since each one adapts a
particular engine's environment to the ``instinct_rl`` contract. Importing a wrapper eagerly here
would put an engine behind every hyperparameter read, so the wrappers resolve on attribute access
instead. ``from ... import InstinctRlVecEnvWrapper`` still works and still imports Isaac Lab; it
just no longer happens to callers who only wanted a learning rate.
"""

from typing import TYPE_CHECKING

from .module_cfg import *  # noqa: F403
from .rl_cfg import *  # noqa: F403

if TYPE_CHECKING:
    from .vecenv_wrapper import InstinctRlVecEnvWrapper

_LAZY = {
    "InstinctRlVecEnvWrapper": ".vecenv_wrapper",
    "MjlabVecEnvWrapper": ".mjlab_vecenv_wrapper",
}


def __getattr__(name: str):
    if name in _LAZY:
        from importlib import import_module

        return getattr(import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
