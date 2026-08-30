"""Engine-owned registries for term lowering and externally supplied terrain builders.

There is no separate list of what an engine supports. The registry *is* the list: a lookup that
finds a builder means the engine can do it, a lookup that misses means it cannot, and
:meth:`TermRegistry.capabilities` derives the advertised capability set from the ``provides=``
annotations on the builders themselves.

That identity is the point. A hand-written capability list is a second copy of the truth and drifts
from the first -- this project already shipped a backend that wrote restitution values while
advertising that it could not randomise restitution, which is exactly the failure a task's
``Requirement`` checks are supposed to catch and could not.

Two ways in, matching the two ways a term is declared:

**By kind**, for families where the engines genuinely differ::

    @TERMS.event("randomize_friction", provides=(DR_SLIDING_FRICTION,))
    def _friction(spec, ctx): ...

**Per family**, for portable terms that arrive carrying their own function. The builder wraps that
function in the engine's native term class and does nothing else::

    @TERMS.portable("observation")
    def _obs(spec, ctx): return ObsTerm(func=spec.func, params=spec.resolved_params(ctx.engine), ...)

An engine may also register a kind in a portable family, which is how a term that happens to need
a native implementation on one engine gets one without making every engine special-case it.

Terrain extensions are different from MDP terms: they build native scene objects and therefore
provide one lazy builder path per supported engine. Their registry lives here so an application
can add a terrain without changing either engine package or importing an SDK during task discovery.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from importlib import import_module, metadata
from typing import Any

from instinctlab_engine.plugins import (
    PluginDiscoveryError,
    _restore_provenance,
    _snapshot_provenance,
    entry_point_description,
    load_plugin_callable,
    mark_plugin_used,
    record_plugin,
)
from instinctlab_engine.spec.capability import CapabilitySet

__all__ = [
    "FAMILIES",
    "TERRAIN_EXTENSIONS",
    "TermBuilder",
    "TermRegistry",
    "TerrainExtensionRegistry",
]

FAMILIES: tuple[str, ...] = (
    "observation",
    "action",
    "reward",
    "termination",
    "event",
    "command",
    "curriculum",
)
"""Term families, matching the fields of :class:`~instinctlab_engine.spec.mdp.MdpSpec`."""

TermBuilder = Callable[..., Any]
"""``(spec, ctx) -> native term config``."""

TerrainBuilder = Callable[..., Any]
"""A whole-terrain or sub-terrain builder supplied by an extension package."""


class TerrainExtensionRegistry:
    """Lazy native terrain builders supplied outside the engine library.

    A whole-terrain builder receives ``(TerrainSpec, engine_profile)``. A
    sub-terrain builder receives ``(SubTerrainSpec, TerrainGeneratorSpec)``.
    Builders may be callables or ``"module:attribute"`` paths; paths keep the
    selected engine SDK unloaded until compilation actually needs that builder.

    Installed packages may expose a registrar through the
    ``instinctlab.terrains`` entry-point group. The registrar is called with
    this registry and should register one implementation per supported engine.
    This lets a new terrain ship beside an application or plugin without
    editing InstinctLab's engine packages.
    """

    ENTRY_POINT_GROUP = "instinctlab.terrains"

    def __init__(self, *, load_entry_points: bool = True):
        self._terrains: dict[tuple[str, str], TerrainBuilder | str] = {}
        self._sub_terrains: dict[tuple[str, str], TerrainBuilder | str] = {}
        self._load_entry_points = load_entry_points
        self._entry_points_loaded = False
        self._entry_point_error: PluginDiscoveryError | None = None
        self._origins: dict[tuple[str, str], str] = {}
        self._active_plugin: str | None = None

    def _snapshot(
        self,
    ) -> tuple[dict, dict, dict, str | None, bool, PluginDiscoveryError | None]:
        return (
            dict(self._terrains),
            dict(self._sub_terrains),
            dict(self._origins),
            self._active_plugin,
            self._entry_points_loaded,
            self._entry_point_error,
        )

    def _restore(
        self,
        snapshot: tuple[
            dict,
            dict,
            dict,
            str | None,
            bool,
            PluginDiscoveryError | None,
        ],
    ) -> None:
        terrains, sub_terrains, origins, active, loaded, error = snapshot
        self._terrains = terrains
        self._sub_terrains = sub_terrains
        self._origins = origins
        self._active_plugin = active
        self._entry_points_loaded = loaded
        self._entry_point_error = error

    @staticmethod
    def _key(engine: str, kind: str) -> tuple[str, str]:
        if not engine or not engine.isidentifier():
            raise ValueError(f"invalid terrain engine name {engine!r}")
        if not kind or not kind.strip():
            raise ValueError("terrain kind must be a non-empty string")
        return engine, kind

    @staticmethod
    def _validate_builder(builder: TerrainBuilder | str) -> None:
        if callable(builder):
            return
        module, separator, attribute = builder.partition(":")
        if not separator or not module or not attribute.isidentifier():
            raise ValueError(
                f"invalid terrain builder {builder!r}; expected a callable or 'module:attribute'"
            )

    def _register(
        self,
        table: dict[tuple[str, str], TerrainBuilder | str],
        engine: str,
        kind: str,
        builder: TerrainBuilder | str,
        *,
        scope: str,
    ) -> None:
        key = self._key(engine, kind)
        self._validate_builder(builder)
        existing = table.get(key)
        if existing is not None and existing != builder:
            origin_key = (scope, f"{engine}:{kind}")
            existing_source = self._origins.get(origin_key, "a built-in registration")
            incoming_source = self._active_plugin or "a direct registration"
            raise ValueError(
                f"{engine}: {scope} terrain {kind!r} is already registered by "
                f"{existing_source}; conflicting registration is from {incoming_source}"
            )
        table[key] = builder
        if existing is None and self._active_plugin is not None:
            self._origins[(scope, f"{engine}:{kind}")] = self._active_plugin

    def register_terrain(
        self, engine: str, kind: str, builder: TerrainBuilder | str
    ) -> None:
        """Register a complete native terrain importer builder."""
        self._register(self._terrains, engine, kind, builder, scope="whole")

    def register_sub_terrain(
        self, engine: str, kind: str, builder: TerrainBuilder | str
    ) -> None:
        """Register one native tile builder for generated terrain."""
        self._register(self._sub_terrains, engine, kind, builder, scope="sub")

    def _load_installed_extensions(self) -> None:
        if self._entry_point_error is not None:
            raise self._entry_point_error
        if self._entry_points_loaded:
            return
        if not self._load_entry_points:
            self._entry_points_loaded = True
            return
        snapshot = self._snapshot()
        provenance_snapshot = _snapshot_provenance()
        try:
            entry_points = metadata.entry_points(group=self.ENTRY_POINT_GROUP)
            for entry_point in sorted(entry_points, key=lambda item: item.name):
                before_terrains = set(self._terrains)
                before_sub_terrains = set(self._sub_terrains)
                registrar = load_plugin_callable(self.ENTRY_POINT_GROUP, entry_point)
                description = entry_point_description(
                    self.ENTRY_POINT_GROUP, entry_point
                )
                try:
                    self._active_plugin = description
                    registrar(self)
                except Exception as exc:
                    raise PluginDiscoveryError(
                        "Terrain plugin registrar failed "
                        f"({entry_point_description(self.ENTRY_POINT_GROUP, entry_point)}): {exc}"
                    ) from exc
                finally:
                    self._active_plugin = None
                terrain_keys = set(self._terrains) - before_terrains
                sub_terrain_keys = set(self._sub_terrains) - before_sub_terrains
                registered = [
                    f"whole:{engine}:{kind}" for engine, kind in terrain_keys
                ] + [f"sub:{engine}:{kind}" for engine, kind in sub_terrain_keys]
                record_plugin(self.ENTRY_POINT_GROUP, entry_point, registered)
                for engine, kind in terrain_keys:
                    self._origins[("whole", f"{engine}:{kind}")] = description
                for engine, kind in sub_terrain_keys:
                    self._origins[("sub", f"{engine}:{kind}")] = description
        except Exception as exc:  # noqa: BLE001 - plugin transactions must roll back any failure
            self._restore(snapshot)
            _restore_provenance(provenance_snapshot)
            error = (
                exc
                if isinstance(exc, PluginDiscoveryError)
                else PluginDiscoveryError(f"Terrain plugin registrar failed: {exc}")
            )
            self._entry_point_error = error
            raise error
        self._entry_points_loaded = True

    @staticmethod
    def _resolve(builder: TerrainBuilder | str) -> TerrainBuilder:
        if callable(builder):
            return builder
        module_name, _, attribute = builder.partition(":")
        resolved = getattr(import_module(module_name), attribute)
        if not callable(resolved):
            raise TypeError(
                f"terrain builder {builder!r} resolved to a non-callable object"
            )
        return resolved

    def terrain(self, engine: str, kind: str) -> TerrainBuilder | None:
        """Return a registered whole-terrain builder, loading plugins once."""
        self._load_installed_extensions()
        builder = self._terrains.get((engine, kind))
        if builder is not None:
            mark_plugin_used(self.ENTRY_POINT_GROUP, f"whole:{engine}:{kind}")
        return None if builder is None else self._resolve(builder)

    def sub_terrain(self, engine: str, kind: str) -> TerrainBuilder | None:
        """Return a registered tile builder, loading plugins once."""
        self._load_installed_extensions()
        builder = self._sub_terrains.get((engine, kind))
        if builder is not None:
            mark_plugin_used(self.ENTRY_POINT_GROUP, f"sub:{engine}:{kind}")
        return None if builder is None else self._resolve(builder)

    def terrain_kinds(self, engine: str) -> frozenset[str]:
        self._load_installed_extensions()
        return frozenset(
            kind
            for registered_engine, kind in self._terrains
            if registered_engine == engine
        )

    def sub_terrain_kinds(self, engine: str) -> frozenset[str]:
        self._load_installed_extensions()
        return frozenset(
            kind
            for registered_engine, kind in self._sub_terrains
            if registered_engine == engine
        )


TERRAIN_EXTENSIONS = TerrainExtensionRegistry()


class TermRegistry:
    """One engine's term builders, keyed by family and kind.

    Installed packages can add native lowering without editing an engine package
    by exposing a registrar in ``instinctlab.engine_terms``. Entry-point names
    use ``<engine>.<extension>`` and values are callables accepting this registry.
    Only extensions for this registry's engine are imported.
    """

    ENTRY_POINT_GROUP = "instinctlab.engine_terms"

    def __init__(self, engine: str, *, load_entry_points: bool = True):
        self.engine = engine
        self._builders: dict[tuple[str, str], TermBuilder] = {}
        self._portable: dict[str, TermBuilder] = {}
        self._emulations: dict[tuple[str, str], TermBuilder] = {}
        self._provides: dict[tuple[str, str], tuple[str, ...]] = {}
        self._load_entry_points = load_entry_points
        self._entry_points_loaded = False
        self._entry_point_error: PluginDiscoveryError | None = None
        self._origins: dict[tuple[str, str, str], str] = {}
        self._active_plugin: str | None = None

    def __repr__(self) -> str:
        return f"TermRegistry({self.engine!r}, {len(self._builders)} kinds, {len(self._portable)} portable families)"

    def _check_family(self, family: str) -> None:
        if family not in FAMILIES:
            raise KeyError(
                f"Unknown term family {family!r}; known families are {list(FAMILIES)}."
            )

    def _load_installed_extensions(self) -> None:
        if self._entry_point_error is not None:
            raise self._entry_point_error
        if self._entry_points_loaded:
            return
        if not self._load_entry_points:
            self._entry_points_loaded = True
            return
        snapshot = (
            dict(self._builders),
            dict(self._portable),
            dict(self._emulations),
            dict(self._provides),
            dict(self._origins),
            self._active_plugin,
        )
        provenance_snapshot = _snapshot_provenance()
        try:
            entry_points = metadata.entry_points(group=self.ENTRY_POINT_GROUP)
            prefix = f"{self.engine}."
            for entry_point in sorted(entry_points, key=lambda item: item.name):
                if not entry_point.name.startswith(prefix):
                    continue
                before_builders = set(self._builders)
                before_portable = set(self._portable)
                before_emulations = set(self._emulations)
                registrar = load_plugin_callable(self.ENTRY_POINT_GROUP, entry_point)
                description = entry_point_description(
                    self.ENTRY_POINT_GROUP, entry_point
                )
                try:
                    self._active_plugin = description
                    registrar(self)
                except Exception as exc:
                    raise PluginDiscoveryError(
                        f"Term plugin registrar failed ({description}): {exc}"
                    ) from exc
                finally:
                    self._active_plugin = None
                builder_keys = set(self._builders) - before_builders
                portable_keys = set(self._portable) - before_portable
                emulation_keys = set(self._emulations) - before_emulations
                registered = (
                    [
                        f"{self.engine}:kind:{family}:{kind}"
                        for family, kind in builder_keys
                    ]
                    + [f"{self.engine}:portable:{family}" for family in portable_keys]
                    + [
                        f"{self.engine}:emulation:{family}:{kind}"
                        for family, kind in emulation_keys
                    ]
                )
                record_plugin(self.ENTRY_POINT_GROUP, entry_point, registered)
                for family, kind in builder_keys:
                    self._origins[("kind", family, kind)] = description
                for family in portable_keys:
                    self._origins[("portable", family, "")] = description
                for family, kind in emulation_keys:
                    self._origins[("emulation", family, kind)] = description
        except Exception as exc:  # noqa: BLE001 - plugin transactions must roll back any failure
            builders, portable, emulations, provides, origins, active = snapshot
            self._builders = builders
            self._portable = portable
            self._emulations = emulations
            self._provides = provides
            self._origins = origins
            self._active_plugin = active
            _restore_provenance(provenance_snapshot)
            error = (
                exc
                if isinstance(exc, PluginDiscoveryError)
                else PluginDiscoveryError(
                    f"Term plugin registrar failed for engine {self.engine!r}: {exc}"
                )
            )
            self._entry_point_error = error
            raise error
        self._entry_points_loaded = True

    def register(
        self,
        family: str,
        kind: str,
        builder: TermBuilder,
        *,
        provides: Iterable[str] = (),
        emulates: bool = False,
    ) -> TermBuilder:
        """Register one builder. Used by the decorators below; called directly by tests."""
        self._check_family(family)
        table = self._emulations if emulates else self._builders
        key = (family, kind)
        if key in table:
            scope = "emulation" if emulates else "kind"
            existing_source = self._origins.get(
                (scope, family, kind), "a built-in registration"
            )
            incoming_source = self._active_plugin or "a direct registration"
            raise ValueError(
                f"{self.engine}: {family}/{kind} is already registered by "
                f"{existing_source}; conflicting registration is from {incoming_source}"
            )
        table[key] = builder
        if self._active_plugin is not None:
            scope = "emulation" if emulates else "kind"
            self._origins[(scope, family, kind)] = self._active_plugin
        if provides:
            self._provides[key] = tuple(provides)
        return builder

    def portable(self, family: str) -> Callable[[TermBuilder], TermBuilder]:
        """Register the wrapper this engine uses for portable terms of ``family``."""

        def decorate(builder: TermBuilder) -> TermBuilder:
            self._check_family(family)
            if family in self._portable:
                existing_source = self._origins.get(
                    ("portable", family, ""), "a built-in registration"
                )
                incoming_source = self._active_plugin or "a direct registration"
                raise ValueError(
                    f"{self.engine}: a portable builder for {family!r} is already "
                    f"registered by {existing_source}; conflicting registration is "
                    f"from {incoming_source}"
                )
            self._portable[family] = builder
            if self._active_plugin is not None:
                self._origins[("portable", family, "")] = self._active_plugin
            return builder

        return decorate

    def _kind_decorator(
        self, family: str
    ) -> Callable[..., Callable[[TermBuilder], TermBuilder]]:
        def by_kind(
            kind: str, *, provides: Iterable[str] = ()
        ) -> Callable[[TermBuilder], TermBuilder]:
            def decorate(builder: TermBuilder) -> TermBuilder:
                return self.register(family, kind, builder, provides=provides)

            return decorate

        return by_kind

    def __getattr__(self, name: str) -> Any:
        """``registry.event(...)`` and friends, one decorator per family.

        Generated rather than written out so that adding a family to :data:`FAMILIES` cannot leave
        a decorator behind. ``__getattr__`` runs only for names not found normally, so it never
        shadows a real method.
        """
        if name in FAMILIES:
            return self._kind_decorator(name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def emulation(self, family: str, kind: str) -> Callable[[TermBuilder], TermBuilder]:
        """Register a stand-in used when a term asks for ``kind`` with ``Requirement.EMULATE``.

        Kept in a separate table from the real builders so that a compilation can report an
        emulated term as emulated. A run whose push events were approximated by writing root
        velocity is not the same experiment as one that applied real external wrenches, and the
        comparison between engines has to know which it is looking at.
        """

        def decorate(builder: TermBuilder) -> TermBuilder:
            return self.register(family, kind, builder, emulates=True)

        return decorate

    def lookup(self, family: str, kind: str) -> TermBuilder | None:
        """The builder for a named kind, or ``None`` when this engine has none."""
        self._check_family(family)
        self._load_installed_extensions()
        builder = self._builders.get((family, kind))
        if builder is not None:
            mark_plugin_used(
                self.ENTRY_POINT_GROUP,
                f"{self.engine}:kind:{family}:{kind}",
            )
        return builder

    def lookup_portable(self, family: str) -> TermBuilder | None:
        """The wrapper for portable terms of ``family``, or ``None``."""
        self._check_family(family)
        self._load_installed_extensions()
        builder = self._portable.get(family)
        if builder is not None:
            mark_plugin_used(
                self.ENTRY_POINT_GROUP,
                f"{self.engine}:portable:{family}",
            )
        return builder

    def lookup_emulation(self, family: str, kind: str) -> TermBuilder | None:
        """The stand-in for a named kind, or ``None``."""
        self._check_family(family)
        self._load_installed_extensions()
        builder = self._emulations.get((family, kind))
        if builder is not None:
            mark_plugin_used(
                self.ENTRY_POINT_GROUP,
                f"{self.engine}:emulation:{family}:{kind}",
            )
        return builder

    def kinds(self, family: str) -> frozenset[str]:
        """Kinds this engine implements in ``family``."""
        self._check_family(family)
        self._load_installed_extensions()
        return frozenset(
            kind for registered, kind in self._builders if registered == family
        )

    def capabilities(self) -> CapabilitySet:
        """Every capability the registered builders claim, and nothing else.

        Derived, never declared. An engine that implements something without saying ``provides=``
        does not advertise it, which is the safe direction: a task requiring the capability fails
        at startup with a fixable message instead of a backend quietly doing something a task
        believed it could not.
        """
        self._load_installed_extensions()
        return CapabilitySet.of(cap for caps in self._provides.values() for cap in caps)

    def provides(self) -> Mapping[str, tuple[str, ...]]:
        """``family/kind`` -> the capabilities it claims, for the contract report."""
        self._load_installed_extensions()
        return {
            f"{family}/{kind}": caps
            for (family, kind), caps in sorted(self._provides.items())
        }
