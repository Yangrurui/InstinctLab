"""Environment wrappers, resolved on access rather than on import.

Each wrapper binds to one engine's environment classes, so importing them all eagerly makes every
module downstream of this package depend on every engine being installed. Attribute access below
imports only the wrapper actually asked for; ``from instinctlab.utils.wrappers import
InstinctRlVecEnvWrapper`` behaves as it always did.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .instinct_rl import InstinctRlVecEnvWrapper
    from .rsl_rl_env_wrappers import RslRlVecEnvWrapper

_LAZY = {
    "InstinctRlVecEnvWrapper": ".instinct_rl",
    "MjlabVecEnvWrapper": ".instinct_rl",
    "RslRlVecEnvWrapper": ".rsl_rl_env_wrappers",
}


def __getattr__(name: str):
    if name in _LAZY:
        from importlib import import_module

        return getattr(import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
