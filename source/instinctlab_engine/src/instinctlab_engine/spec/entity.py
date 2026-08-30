"""References to a subset of a scene entity, stated without naming an engine.

An :class:`EntityRef` says *which parts of which entity* a term acts on -- these joints, those
bodies -- and leaves it to the backend to turn that into the engine's own selector configuration.
The translation happens once at compile time, so the runtime cost is nil and terms keep reading
tensors directly.

Selector kinds are open on purpose. Isaac Lab and mjlab agree on only two of them:

===============  ==========================================================================
engine           kinds its ``SceneEntityCfg`` accepts
===============  ==========================================================================
both             ``joint``, ``body``
Isaac Lab only   ``fixed_tendon``, ``object_collection``
mjlab only       ``actuator``, ``camera``, ``geom``, ``light``, ``material``, ``pair``,
                 ``site``, ``tendon``
===============  ==========================================================================

Two kinds out of twelve. A fixed pair of ``joints`` / ``bodies`` fields would therefore be able to
express Isaac Lab tasks and quietly lose everything an mjlab task says about geoms and sites, which
is the direction this project has to support as well. So the common two get named fields for
legibility and everything else goes in :attr:`other`, where the backend can either translate it or
refuse loudly.

Note that Isaac Lab's ``fixed_tendon`` and mjlab's ``tendon`` are related but not the same kind,
and are deliberately not unified here; naming them apart keeps a backend from silently accepting a
selector it cannot honour.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

__all__ = ["UNIVERSAL_KINDS", "EntityRef", "resolve_entity_names"]

UNIVERSAL_KINDS: tuple[str, ...] = ("joint", "body")
"""The only selector kinds every supported engine can express. Checked in the tests."""


def _normalise(patterns: str | Sequence[str] | None) -> tuple[str, ...] | None:
    """Accept a bare string the way both engines' ``SceneEntityCfg`` does."""
    if patterns is None:
        return None
    if isinstance(patterns, str):
        return (patterns,)
    return tuple(patterns)


def resolve_entity_names(
    patterns: str | Sequence[str],
    available_names: Sequence[str],
    *,
    preserve_order: bool,
) -> tuple[str, ...]:
    """Resolve selector patterns with the semantics shared by both engines.

    ``preserve_order=False`` follows ``available_names``.  ``True`` groups matches by
    pattern order while retaining ``available_names`` order inside each pattern.  A name may
    match only one pattern and every pattern must match, matching both native selector helpers.
    Keeping this tiny resolver in the declaration layer lets validation and compilation reason
    about the selected tensor axis before either engine SDK is imported.
    """
    expressions = (patterns,) if isinstance(patterns, str) else tuple(patterns)
    if not expressions:
        raise ValueError("A selector was given no patterns.")

    matches_by_pattern: list[list[str]] = [[] for _ in expressions]
    source_order: list[str] = []
    for name in available_names:
        matching = [index for index, expression in enumerate(expressions) if re.fullmatch(expression, name)]
        if len(matching) > 1:
            duplicate_patterns = tuple(expressions[index] for index in matching)
            raise ValueError(f"Entity name {name!r} matches multiple selector patterns: {duplicate_patterns!r}.")
        if matching:
            matches_by_pattern[matching[0]].append(name)
            source_order.append(name)

    unmatched = [expression for expression, matches in zip(expressions, matches_by_pattern) if not matches]
    if unmatched:
        raise ValueError(
            f"Selector patterns match no entity names: {unmatched!r}. Available names: {tuple(available_names)!r}."
        )
    if not preserve_order:
        return tuple(source_order)
    return tuple(name for matches in matches_by_pattern for name in matches)


@dataclass(frozen=True)
class EntityRef:
    """A subset of one scene entity, named by pattern rather than by index.

    Patterns are regular expressions matched against the entity's own names, which is what both
    engines do; the matching helper is byte-identical between them, so a pattern selects the same
    thing either way.

    Args:
        entity: Key of the entity in the scene.
        joints: Joint name patterns, or a single pattern.
        bodies: Body name patterns, or a single pattern.
        other: Patterns for selector kinds outside :data:`UNIVERSAL_KINDS`, keyed by kind. A
            backend that cannot express a kind must reject the reference rather than drop it.
        preserve_order: When true the selection follows the order of the patterns; when false it
            follows the entity's own order. Note what the entity's own order is not: it is whatever
            the engine built, which is a breadth-first walk under PhysX and model-file order under
            MuJoCo, and those two disagree. D1's canonical depth-first order is therefore reached
            only by passing the catalog's joint names explicitly with this flag set -- a bare ``.*``
            preserves the order of a one-element pattern list and so changes nothing.
    """

    entity: str = "robot"
    joints: str | Sequence[str] | None = None
    bodies: str | Sequence[str] | None = None
    other: Mapping[str, str | Sequence[str]] = field(default_factory=dict)
    preserve_order: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "joints", _normalise(self.joints))
        object.__setattr__(self, "bodies", _normalise(self.bodies))
        normalised = {kind: _normalise(patterns) for kind, patterns in dict(self.other).items()}
        for kind, patterns in normalised.items():
            if kind in UNIVERSAL_KINDS:
                raise ValueError(f"'{kind}' has its own field on EntityRef; do not pass it through 'other'.")
            if not patterns:
                raise ValueError(f"Selector '{kind}' was given no patterns.")
        # Sorted so that two references built from equivalent mappings compare equal.
        object.__setattr__(self, "other", dict(sorted(normalised.items())))

    def selectors(self) -> dict[str, tuple[str, ...]]:
        """Every selector on this reference, keyed by kind, in a single mapping."""
        out: dict[str, tuple[str, ...]] = {}
        if self.joints is not None:
            out["joint"] = self.joints
        if self.bodies is not None:
            out["body"] = self.bodies
        out.update(self.other)  # type: ignore[arg-type]
        return out

    def kinds(self) -> frozenset[str]:
        """The selector kinds this reference uses. A backend checks it against what it supports."""
        return frozenset(self.selectors())
