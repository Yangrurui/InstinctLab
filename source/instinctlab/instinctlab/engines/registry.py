"""Each engine's term registry, which is also its capability matrix.

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
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from instinctlab.spec.capability import CapabilitySet

__all__ = ["FAMILIES", "TermBuilder", "TermRegistry"]

FAMILIES: tuple[str, ...] = (
    "observation",
    "action",
    "reward",
    "termination",
    "event",
    "command",
    "curriculum",
)
"""Term families, matching the fields of :class:`~instinctlab.spec.mdp.MdpSpec`."""

TermBuilder = Callable[..., Any]
"""``(spec, ctx) -> native term config``."""


class TermRegistry:
    """One engine's term builders, keyed by family and kind."""

    def __init__(self, engine: str):
        self.engine = engine
        self._builders: dict[tuple[str, str], TermBuilder] = {}
        self._portable: dict[str, TermBuilder] = {}
        self._emulations: dict[tuple[str, str], TermBuilder] = {}
        self._provides: dict[tuple[str, str], tuple[str, ...]] = {}

    def __repr__(self) -> str:
        return f"TermRegistry({self.engine!r}, {len(self._builders)} kinds, {len(self._portable)} portable families)"

    def _check_family(self, family: str) -> None:
        if family not in FAMILIES:
            raise KeyError(f"Unknown term family {family!r}; known families are {list(FAMILIES)}.")

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
        return self._builders.get((family, kind))

    def lookup_portable(self, family: str) -> TermBuilder | None:
        """The wrapper for portable terms of ``family``, or ``None``."""
        self._check_family(family)
        return self._portable.get(family)

    def lookup_emulation(self, family: str, kind: str) -> TermBuilder | None:
        """The stand-in for a named kind, or ``None``."""
        self._check_family(family)
        return self._emulations.get((family, kind))

    def kinds(self, family: str) -> frozenset[str]:
        """Kinds this engine implements in ``family``."""
        self._check_family(family)
        return frozenset(kind for registered, kind in self._builders if registered == family)

    def capabilities(self) -> CapabilitySet:
        """Every capability the registered builders claim, and nothing else.

        Derived, never declared. An engine that implements something without saying ``provides=``
        does not advertise it, which is the safe direction: a task requiring the capability fails
        at startup with a fixable message instead of a backend quietly doing something a task
        believed it could not.
        """
        return CapabilitySet.of(cap for caps in self._provides.values() for cap in caps)

    def provides(self) -> Mapping[str, tuple[str, ...]]:
        """``family/kind`` -> the capabilities it claims, for the contract report."""
        return {f"{family}/{kind}": caps for (family, kind), caps in sorted(self._provides.items())}
