"""Backends: the only place an engine SDK is imported.

``engines/base.py``, ``registry.py`` and ``compile.py`` are the shared machinery and stay
engine-free, so a task's compilation can be checked against an engine that is not installed.
The engine-specific code lives in ``engines/<name>/`` and is imported only once an engine has been
chosen -- importing this package must never pull in Isaac Sim or MuJoCo, since the launcher has to
be able to inspect the available adapters before deciding which one to bootstrap.
"""

from __future__ import annotations

from importlib import import_module

from .base import CompiledTask, EngineAdapter, Resolution, UnsupportedTerm
from .compile import CompileCtx, compile_family, compile_mdp, observation_group_settings, qualname_of
from .registry import FAMILIES, TermRegistry

ADAPTERS: dict[str, str] = {
    "isaacsim": "instinctlab.engines.isaacsim.adapter:IsaacSimAdapter",
    "mjlab": "instinctlab.engines.mjlab.adapter:MjlabAdapter",
}
"""Engine name -> dotted path of its adapter class.

Paths rather than classes: naming an engine must not import it. A launcher reads this table to
know what ``--engine`` accepts, then imports exactly the one it was given.
"""


def names() -> tuple[str, ...]:
    """Every engine with an adapter, whether or not it is installed here."""
    return tuple(sorted(ADAPTERS))


def adapter(engine: str) -> EngineAdapter:
    """Import ``engine``'s adapter and return an instance of it.

    Importing the adapter module is safe on a machine without that engine -- adapters import their
    SDK inside ``bootstrap`` and ``compile``, never at module level -- so this also serves the
    checks that report what an engine would do with a task it will never run.
    """
    try:
        path = ADAPTERS[engine]
    except KeyError:
        raise KeyError(f"unknown engine {engine!r}; known engines are {', '.join(names())}") from None
    module_path, _, attr = path.partition(":")
    return getattr(import_module(module_path), attr)()


__all__ = [
    "ADAPTERS",
    "FAMILIES",
    "CompileCtx",
    "CompiledTask",
    "EngineAdapter",
    "Resolution",
    "TermRegistry",
    "UnsupportedTerm",
    "adapter",
    "compile_family",
    "compile_mdp",
    "names",
    "observation_group_settings",
    "qualname_of",
]
