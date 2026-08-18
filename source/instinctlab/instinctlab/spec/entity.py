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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

__all__ = ["UNIVERSAL_KINDS", "EntityRef"]

UNIVERSAL_KINDS: tuple[str, ...] = ("joint", "body")
"""The only selector kinds every supported engine can express. Checked in the tests."""


def _normalise(patterns: str | Sequence[str] | None) -> tuple[str, ...] | None:
    """Accept a bare string the way both engines' ``SceneEntityCfg`` does."""
    if patterns is None:
        return None
    if isinstance(patterns, str):
        return (patterns,)
    return tuple(patterns)


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
            follows the entity's own order. This is the switch decision D1 hangs on, and it is the
            caller's to make -- an entity's natural order is a depth-first walk of the kinematic
            tree, which is not the order a policy's action vector is necessarily in.
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
