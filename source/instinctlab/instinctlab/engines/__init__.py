"""Backends: the only place an engine SDK is imported.

``engines/base.py``, ``registry.py`` and ``compile.py`` are the shared machinery and stay
engine-free, so a task's compilation can be checked against an engine that is not installed.
The engine-specific code lives in ``engines/<name>/`` and is imported only once an engine has been
chosen -- importing this package must never pull in Isaac Sim or MuJoCo, since the launcher has to
be able to inspect the available adapters before deciding which one to bootstrap.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from instinctlab.compat import entity as _entity

if TYPE_CHECKING:
    from .base import EngineAdapter

ADAPTERS: dict[str, str] = {}
"""Engine name -> dotted path of its adapter class.

Paths rather than classes: naming an engine must not import it. A launcher reads this table to
know what ``--engine`` accepts, then imports exactly the one it was given.
"""


def register_adapter(engine: str, path: str) -> None:
    """Register an engine plugin without editing the shared compiler."""
    module, separator, attribute = path.partition(":")
    if not engine.isidentifier() or not separator or not module or not attribute.isidentifier():
        raise ValueError(f"invalid engine adapter registration {engine!r} -> {path!r}")
    existing = ADAPTERS.get(engine)
    if existing is not None and existing != path:
        raise ValueError(f"engine {engine!r} is already registered as {existing!r}")
    ADAPTERS[engine] = path
    _entity.register_packages({engine: module.rpartition(".")[0]})


register_adapter("isaacsim", "instinctlab.engines.isaacsim.adapter:IsaacSimAdapter")
register_adapter("mjlab", "instinctlab.engines.mjlab.adapter:MjlabAdapter")


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


_SHARED_EXPORTS = {
    "CompileCtx": ("instinctlab.engines.compile", "CompileCtx"),
    "CompiledTask": ("instinctlab.engines.base", "CompiledTask"),
    "EngineAdapter": ("instinctlab.engines.base", "EngineAdapter"),
    "FAMILIES": ("instinctlab.engines.registry", "FAMILIES"),
    "Resolution": ("instinctlab.engines.base", "Resolution"),
    "TermRegistry": ("instinctlab.engines.registry", "TermRegistry"),
    "UnsupportedTerm": ("instinctlab.engines.base", "UnsupportedTerm"),
    "compile_family": ("instinctlab.engines.compile", "compile_family"),
    "compile_mdp": ("instinctlab.engines.compile", "compile_mdp"),
    "observation_group_settings": (
        "instinctlab.engines.compile",
        "observation_group_settings",
    ),
    "qualname_of": ("instinctlab.engines.compile", "qualname_of"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _SHARED_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


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
    "register_adapter",
]
