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
from instinctlab_engine.plugins import (
    PluginDiscoveryError,
    _restore_provenance,
    _snapshot_provenance,
    _plugin_locked,
    entry_point_description,
    load_plugin_callable,
    mark_plugin_used,
    record_plugin,
)

if TYPE_CHECKING:
    from .base import EngineAdapter

ADAPTERS: dict[str, str] = {}
"""Engine name -> dotted path of its adapter class.

Paths rather than classes: naming an engine must not import it. A launcher reads this table to
know what ``--engine`` accepts, then imports exactly the one it was given.
"""

_ADAPTER_SOURCES: dict[str, str] = {}
_active_engine_plugin: str | None = None


@_plugin_locked
def register_adapter(engine: str, path: str) -> None:
    """Register an engine plugin without editing the shared compiler."""
    module, separator, attribute = path.partition(":")
    if (
        not engine.isidentifier()
        or not separator
        or not module
        or not attribute.isidentifier()
    ):
        raise ValueError(f"invalid engine adapter registration {engine!r} -> {path!r}")
    existing = ADAPTERS.get(engine)
    if existing is not None and existing != path:
        existing_source = _ADAPTER_SOURCES.get(engine, "a direct registration")
        incoming_source = _active_engine_plugin or "a direct registration"
        raise ValueError(
            f"engine {engine!r} is already registered as {existing!r} by "
            f"{existing_source}; conflicting registration is from {incoming_source}"
        )
    ADAPTERS[engine] = path
    if existing is None and _active_engine_plugin is not None:
        _ADAPTER_SOURCES[engine] = _active_engine_plugin
    _entity.register_packages({engine: module.rpartition(".")[0]})


_ENGINE_ENTRY_POINT_GROUP = "instinctlab.engines"
_engine_entry_points_loaded = False
_engine_entry_point_error: PluginDiscoveryError | None = None


@_plugin_locked
def _load_installed_engines() -> None:
    """Load installed backend registrars without importing a simulator SDK."""
    global _active_engine_plugin, _engine_entry_points_loaded, _engine_entry_point_error
    if _engine_entry_point_error is not None:
        raise _engine_entry_point_error
    if _engine_entry_points_loaded:
        return
    from .registry import TERRAIN_EXTENSIONS

    adapter_snapshot = dict(ADAPTERS)
    adapter_source_snapshot = dict(_ADAPTER_SOURCES)
    entity_snapshot = _entity._snapshot_registrations()
    terrain_snapshot = TERRAIN_EXTENSIONS._snapshot()
    provenance_snapshot = _snapshot_provenance()
    try:
        entry_points = metadata.entry_points(group=_ENGINE_ENTRY_POINT_GROUP)
        for entry_point in sorted(entry_points, key=lambda item: item.name):
            before = set(ADAPTERS)
            before_terrains = set(TERRAIN_EXTENSIONS._terrains)
            before_sub_terrains = set(TERRAIN_EXTENSIONS._sub_terrains)
            description = entry_point_description(
                _ENGINE_ENTRY_POINT_GROUP, entry_point
            )
            try:
                _active_engine_plugin = description
                TERRAIN_EXTENSIONS._active_plugin = description
                registrar = load_plugin_callable(_ENGINE_ENTRY_POINT_GROUP, entry_point)
                registrar()
            except Exception as exc:
                raise PluginDiscoveryError(
                    f"Engine plugin registrar failed ({description}): {exc}"
                ) from exc
            finally:
                _active_engine_plugin = None
                TERRAIN_EXTENSIONS._active_plugin = None
            registered = set(ADAPTERS) - before
            if entry_point.name not in ADAPTERS:
                raise PluginDiscoveryError(
                    "Engine plugin did not register its entry-point name "
                    f"({entry_point_description(_ENGINE_ENTRY_POINT_GROUP, entry_point)})"
                )
            record_plugin(
                _ENGINE_ENTRY_POINT_GROUP,
                entry_point,
                [
                    *(registered or (entry_point.name,)),
                    *(
                        f"terrain:whole:{engine}:{kind}"
                        for engine, kind in set(TERRAIN_EXTENSIONS._terrains)
                        - before_terrains
                    ),
                    *(
                        f"terrain:sub:{engine}:{kind}"
                        for engine, kind in set(TERRAIN_EXTENSIONS._sub_terrains)
                        - before_sub_terrains
                    ),
                ],
            )
    except Exception as exc:  # noqa: BLE001 - plugin transactions must roll back any failure
        ADAPTERS.clear()
        ADAPTERS.update(adapter_snapshot)
        _ADAPTER_SOURCES.clear()
        _ADAPTER_SOURCES.update(adapter_source_snapshot)
        _active_engine_plugin = None
        _entity._restore_registrations(entity_snapshot)
        TERRAIN_EXTENSIONS._restore(terrain_snapshot)
        _restore_provenance(provenance_snapshot)
        error = (
            exc
            if isinstance(exc, PluginDiscoveryError)
            else PluginDiscoveryError(f"Engine plugin discovery failed: {exc}")
        )
        _engine_entry_point_error = error
        raise error
    _engine_entry_points_loaded = True


@_plugin_locked
def names() -> tuple[str, ...]:
    """Every engine with an adapter, whether or not it is installed here."""
    _load_installed_engines()
    return tuple(sorted(ADAPTERS))


@_plugin_locked
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
        raise KeyError(
            f"unknown engine {engine!r}; known engines are {', '.join(names())}"
        ) from None
    module_path, _, attr = path.partition(":")
    mark_plugin_used(_ENGINE_ENTRY_POINT_GROUP, engine)
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


def register_actuator(
    engine: str,
    model_id: str,
    config_factory: Any,
    *,
    runtime_adapter: Any | None = None,
    capabilities: Any = (),
) -> None:
    """Register one lazy engine-native actuator model."""
    from .actuators import register_actuator as _register_actuator

    _register_actuator(
        engine,
        model_id,
        config_factory,
        runtime_adapter=runtime_adapter,
        capabilities=capabilities,
    )


def register_sensor(
    engine: str,
    kind: str,
    builder: Any,
    *,
    capabilities: Any,
) -> None:
    """Register one lazy engine-native sensor builder."""
    from .sensors import register_sensor as _register_sensor

    _register_sensor(
        engine,
        kind,
        builder,
        capabilities=capabilities,
    )


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
    "register_actuator",
    "register_sub_terrain",
    "register_sensor",
    "register_terrain",
]
