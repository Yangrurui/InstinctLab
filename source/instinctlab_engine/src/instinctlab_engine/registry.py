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
            raise ValueError(
                f"{engine}: {scope} terrain {kind!r} is already registered as {existing!r}"
            )
        table[key] = builder

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
        if self._entry_points_loaded:
            return
        self._entry_points_loaded = True
        if not self._load_entry_points:
            return
        entry_points = metadata.entry_points(group=self.ENTRY_POINT_GROUP)
        for entry_point in sorted(entry_points, key=lambda item: item.name):
            registrar = entry_point.load()
            if not callable(registrar):
                raise TypeError(
                    f"terrain entry point {entry_point.name!r} must load a callable registrar"
                )
            registrar(self)

    @staticmethod
    def _resolve(builder: TerrainBuilder | str) -> TerrainBuilder:
        if callable(builder):
            return builder
        module_name, _, attribute = builder.partition(":")
        resolved = getattr(import_module(module_name), attribute)
        if not callable(resolved):
            raise TypeError(f"terrain builder {builder!r} resolved to a non-callable object")
        return resolved

    def terrain(self, engine: str, kind: str) -> TerrainBuilder | None:
        """Return a registered whole-terrain builder, loading plugins once."""
        self._load_installed_extensions()
        builder = self._terrains.get((engine, kind))
        return None if builder is None else self._resolve(builder)

    def sub_terrain(self, engine: str, kind: str) -> TerrainBuilder | None:
        """Return a registered tile builder, loading plugins once."""
        self._load_installed_extensions()
        builder = self._sub_terrains.get((engine, kind))
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
            kind for registered_engine, kind in self._sub_terrains if registered_engine == engine
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

    def __repr__(self) -> str:
        return f"TermRegistry({self.engine!r}, {len(self._builders)} kinds, {len(self._portable)} portable families)"

    def _check_family(self, family: str) -> None:
        if family not in FAMILIES:
            raise KeyError(f"Unknown term family {family!r}; known families are {list(FAMILIES)}.")

    def _load_installed_extensions(self) -> None:
        if self._entry_points_loaded:
            return
        self._entry_points_loaded = True
        if not self._load_entry_points:
            return
        entry_points = metadata.entry_points(group=self.ENTRY_POINT_GROUP)
        prefix = f"{self.engine}."
        for entry_point in sorted(entry_points, key=lambda item: item.name):
            if not entry_point.name.startswith(prefix):
                continue
            registrar = entry_point.load()
            if not callable(registrar):
                raise TypeError(
                    f"term entry point {entry_point.name!r} must load a callable registrar"
                )
            registrar(self)

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
            raise ValueError(f"{self.engine}: {family}/{kind} is already registered.")
        table[key] = builder
        if provides:
            self._provides[key] = tuple(provides)
        return builder

    def portable(self, family: str) -> Callable[[TermBuilder], TermBuilder]:
        """Register the wrapper this engine uses for portable terms of ``family``."""

        def decorate(builder: TermBuilder) -> TermBuilder:
            self._check_family(family)
            if family in self._portable:
                raise ValueError(f"{self.engine}: a portable builder for {family!r} is already registered.")
            self._portable[family] = builder
            return builder

        return decorate

    def _kind_decorator(self, family: str) -> Callable[..., Callable[[TermBuilder], TermBuilder]]:
        def by_kind(kind: str, *, provides: Iterable[str] = ()) -> Callable[[TermBuilder], TermBuilder]:
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
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

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
        return self._builders.get((family, kind))

    def lookup_portable(self, family: str) -> TermBuilder | None:
        """The wrapper for portable terms of ``family``, or ``None``."""
        self._check_family(family)
        self._load_installed_extensions()
        return self._portable.get(family)

    def lookup_emulation(self, family: str, kind: str) -> TermBuilder | None:
        """The stand-in for a named kind, or ``None``."""
        self._check_family(family)
        self._load_installed_extensions()
        return self._emulations.get((family, kind))

    def kinds(self, family: str) -> frozenset[str]:
        """Kinds this engine implements in ``family``."""
        self._check_family(family)
        self._load_installed_extensions()
        return frozenset(kind for registered, kind in self._builders if registered == family)

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
        return {f"{family}/{kind}": caps for (family, kind), caps in sorted(self._provides.items())}
