"""Stable task/engine contracts and lazy backend selection.

This distribution owns the public boundary used by both task packages and
native simulator backends. It deliberately contains no task declarations and
imports no simulator SDK. Backends are resolved lazily after an application has
selected one.
"""

from __future__ import annotations

from importlib import import_module, metadata
from typing import TYPE_CHECKING, Any

from instinctlab_engine.bridge import entity as _entity

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


_ENGINE_ENTRY_POINT_GROUP = "instinctlab.engines"
_engine_entry_points_loaded = False


def _load_installed_engines() -> None:
    """Load installed backend registrars without importing a simulator SDK."""
    global _engine_entry_points_loaded
    if _engine_entry_points_loaded:
        return
    _engine_entry_points_loaded = True
    entry_points = metadata.entry_points(group=_ENGINE_ENTRY_POINT_GROUP)
    for entry_point in sorted(entry_points, key=lambda item: item.name):
        registrar = entry_point.load()
        if not callable(registrar):
            raise TypeError(
                f"engine entry point {entry_point.name!r} must load a callable registrar"
            )
        registrar()


def names() -> tuple[str, ...]:
    """Every engine with an adapter, whether or not it is installed here."""
    _load_installed_engines()
    return tuple(sorted(ADAPTERS))


def adapter(engine: str) -> EngineAdapter:
    """Import ``engine``'s adapter and return an instance of it.

    Importing the adapter module is safe on a machine without that engine -- adapters import their
    SDK inside ``bootstrap`` and ``compile``, never at module level -- so this also serves the
    checks that report what an engine would do with a task it will never run.
    """
    _load_installed_engines()
    try:
        path = ADAPTERS[engine]
    except KeyError:
        raise KeyError(f"unknown engine {engine!r}; known engines are {', '.join(names())}") from None
    module_path, _, attr = path.partition(":")
    return getattr(import_module(module_path), attr)()


_SHARED_EXPORTS = {
    "CompileCtx": ("instinctlab_engine.compile", "CompileCtx"),
    "CompiledTask": ("instinctlab_engine.base", "CompiledTask"),
    "EngineAdapter": ("instinctlab_engine.base", "EngineAdapter"),
    "FAMILIES": ("instinctlab_engine.registry", "FAMILIES"),
    "Resolution": ("instinctlab_engine.base", "Resolution"),
    "TermRegistry": ("instinctlab_engine.registry", "TermRegistry"),
    "TerrainExtensionRegistry": (
        "instinctlab_engine.registry",
        "TerrainExtensionRegistry",
    ),
    "UnsupportedTerm": ("instinctlab_engine.base", "UnsupportedTerm"),
    "compile_family": ("instinctlab_engine.compile", "compile_family"),
    "compile_mdp": ("instinctlab_engine.compile", "compile_mdp"),
    "observation_group_settings": (
        "instinctlab_engine.compile",
        "observation_group_settings",
    ),
    "qualname_of": ("instinctlab_engine.compile", "qualname_of"),
}


def register_terrain(engine: str, kind: str, builder: Any) -> None:
    """Register a lazy whole-terrain builder from an application or plugin."""
    from .registry import TERRAIN_EXTENSIONS

    TERRAIN_EXTENSIONS.register_terrain(engine, kind, builder)


def register_sub_terrain(engine: str, kind: str, builder: Any) -> None:
    """Register a lazy generated-terrain tile builder from an application or plugin."""
    from .registry import TERRAIN_EXTENSIONS

    TERRAIN_EXTENSIONS.register_sub_terrain(engine, kind, builder)


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
    "TerrainExtensionRegistry",
    "UnsupportedTerm",
    "adapter",
    "compile_family",
    "compile_mdp",
    "names",
    "observation_group_settings",
    "qualname_of",
    "register_adapter",
    "register_sub_terrain",
    "register_terrain",
]
