"""Lowering an :class:`~instinctlab.spec.entity.EntityRef` onto each engine's selector config.

Both engines converged on a class called ``SceneEntityCfg`` with the same shape -- ``name``,
``preserve_order``, and a ``<kind>_names`` / ``<kind>_ids`` pair per selector kind -- so lowering is
a rename-free field mapping. The name resolution underneath is the same code on both sides:
``resolve_matching_names`` is byte-identical apart from its docstring, and behaves identically for
every pattern order and both settings of ``preserve_order``. None of that needs reimplementing here.

What does need handling is where the two genuinely differ.

**Selector kinds.** Only ``joint`` and ``body`` are common. Isaac Lab adds ``fixed_tendon`` and
``object_collection``; mjlab adds eight more. A reference naming a kind the target engine cannot
express is rejected here rather than dropped, because dropping it produces a task that runs and
means something else.

**What ``<kind>_names`` holds after ``resolve()``.** This is a trap of exactly the kind
:mod:`~instinctlab.compat.denylist` exists for, one level up from the data attributes it covers.
Isaac Lab leaves the *user's patterns* in the field (it discards the matched names) while mjlab
overwrites it with the *matched names*. So a term reading ``asset_cfg.body_names`` gets
``[".*_ankle_roll_link"]`` under one engine and ``["left_ankle_roll_link", ...]`` under the other.
Real code reads it -- Isaac Lab's own ``events.py`` joins the field back into a regex to match USD
prim paths, and this repository stores it in an observation term. :func:`resolved_names` is the
portable way to ask the question, and it happens to need no engine-specific branch at all.

This module imports no engine at module scope; each lowering imports its own engine when called.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import MappingProxyType
from typing import Any, Mapping

from instinctlab.spec.entity import UNIVERSAL_KINDS, EntityRef

from .denylist import PortabilityError

__all__ = [
    "SELECTOR_KINDS",
    "UnsupportedSelector",
    "lower",
    "resolved_names",
    "selector_field",
    "universal",
]

SELECTOR_KINDS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "isaacsim": frozenset({"joint", "body", "fixed_tendon", "object_collection"}),
        "mjlab": frozenset(
            {
                "joint",
                "body",
                "geom",
                "site",
                "actuator",
                "tendon",
                "camera",
                "light",
                "material",
                "pair",
            }
        ),
    }
)
"""Selector kinds each engine's ``SceneEntityCfg`` accepts, checked against the installed engines.

Isaac Lab's ``fixed_tendon`` and mjlab's ``tendon`` are listed apart on purpose. They are not known
to be the same selector, and treating them as one would let a reference through that the target
engine resolves against a different set of elements.
"""

# Isaac Lab annotates its name fields as ``list[str]`` and mjlab as ``tuple[str, ...]``. Both accept
# either at runtime, but matching the declared type keeps the produced config indistinguishable from
# a hand-written one, which is what the golden diff in P4 compares against.
_CONTAINER: Mapping[str, Any] = MappingProxyType({"isaacsim": list, "mjlab": tuple})


class UnsupportedSelector(PortabilityError):
    """Raised when an engine has no selector for a kind the reference names."""


def selector_field(kind: str) -> str:
    """Name of the config field carrying patterns for ``kind``.

    Both engines follow the same convention for all twelve kinds between them, so this is one
    function rather than a per-engine table.
    """
    return f"{kind}_names"


def _cfg_type(engine: str) -> type:
    """Import the engine's selector config lazily, so this module stays engine-free."""
    if engine == "isaacsim":
        from isaaclab.managers import SceneEntityCfg
    elif engine == "mjlab":
        from mjlab.managers.scene_entity_config import SceneEntityCfg
    else:
        raise KeyError(f"unknown engine {engine!r}; known engines are {sorted(SELECTOR_KINDS)}")
    return SceneEntityCfg


def lower(ref: EntityRef, engine: str) -> Any:
    """Compile ``ref`` into ``engine``'s native ``SceneEntityCfg``.

    Args:
        ref: The engine-agnostic reference.
        engine: Target engine key, one of :data:`SELECTOR_KINDS`.

    Returns:
        The engine's own ``SceneEntityCfg``, ready to be handed to its manager. Resolution to
        indices happens later, inside the engine, against the real scene.

    Raises:
        UnsupportedSelector: ``ref`` names a selector kind this engine cannot express.
        KeyError: ``engine`` is not a known engine.
    """
    try:
        supported = SELECTOR_KINDS[engine]
    except KeyError:
        raise KeyError(f"unknown engine {engine!r}; known engines are {sorted(SELECTOR_KINDS)}") from None

    missing = sorted(ref.kinds() - supported)
    if missing:
        raise UnsupportedSelector(
            f"{engine} has no selector for {missing} (entity {ref.entity!r}). "
            f"It supports {sorted(supported)}. Express this per-engine, or drop the selector in the "
            "task spec so the omission is recorded rather than inferred."
        )

    container = _CONTAINER[engine]
    kwargs: dict[str, Any] = {"name": ref.entity, "preserve_order": ref.preserve_order}
    for kind, patterns in ref.selectors().items():
        kwargs[selector_field(kind)] = container(patterns)
    return _cfg_type(engine)(**kwargs)


def resolved_names(entity: Any, cfg: Any, kind: str = "body") -> list[str]:
    """The names a resolved selector actually selected, in the order it selected them.

    Read this instead of ``cfg.<kind>_names``, which means different things on the two engines: Isaac
    Lab leaves the caller's patterns in place, mjlab replaces them with what matched. Going through
    the indices sidesteps the difference entirely, because the indices are the thing both engines
    agree on.

    Args:
        entity: The scene entity the config was resolved against.
        cfg: An engine ``SceneEntityCfg``, already resolved.
        kind: Selector kind to read.

    Returns:
        Matched names, ordered as the selection is ordered -- which follows the patterns when
        ``preserve_order`` was set and the entity's own order otherwise.
    """
    all_names: Sequence[str] = getattr(entity, f"{kind}_names")
    ids = getattr(cfg, f"{kind}_ids")
    if isinstance(ids, slice):
        return list(all_names[ids])
    if isinstance(ids, int):
        return [all_names[ids]]
    return [all_names[index] for index in ids]


def universal(ref: EntityRef) -> bool:
    """Whether every kind on ``ref`` is one all engines can express."""
    return ref.kinds() <= frozenset(UNIVERSAL_KINDS)
