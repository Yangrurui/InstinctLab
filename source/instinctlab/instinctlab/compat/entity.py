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

import importlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from instinctlab.spec.entity import UNIVERSAL_KINDS, EntityRef

from .denylist import PortabilityError

__all__ = [
    "UnsupportedSelector",
    "lower",
    "register",
    "resolved_names",
    "selector_field",
    "selector_kinds",
    "universal",
]


@dataclass(frozen=True)
class _Selectors:
    """What one engine can select, and how its config wants to be built."""

    kinds: frozenset[str]
    cfg: tuple[str, str]
    container: type


_ENGINES: dict[str, _Selectors] = {}


def register(engine: str, *, kinds: Iterable[str], cfg: tuple[str, str], container: type) -> None:
    """Declare what ``engine`` can select. Called by that engine's package when it is imported.

    Args:
        engine: The engine key, matching its entry in :data:`instinctlab.engines.ADAPTERS`.
        kinds: Selector kinds its ``SceneEntityCfg`` accepts. Kinds two engines spell the same are
            not assumed to mean the same thing -- Isaac Lab's ``fixed_tendon`` and mjlab's
            ``tendon`` are registered apart, because treating them as one would let a reference
            through that the target resolves against a different set of elements.
        cfg: Module path and attribute name of the engine's selector config, imported on use so
            that this module stays importable without any engine present.
        container: Sequence type the engine annotates its name fields with. Isaac Lab says
            ``list[str]`` and mjlab ``tuple[str, ...]``; both accept either at runtime, but matching
            the declaration keeps the produced config indistinguishable from a hand-written one,
            which is what the golden diff compares against.

    This is a registration rather than a table in this file for the reason decision S2 gives: an
    engine whose selectors nobody here anticipated should cost a call in its own package, not an
    edit to the shared layer. The shared layer still decides what happens to a kind it has never
    heard of, which is what :class:`UnsupportedSelector` is.
    """
    _ENGINES[engine] = _Selectors(frozenset(kinds), cfg, container)


def _ensure_registered() -> None:
    """Import the adapter packages, since registration is a side effect of importing them.

    Adapters do not import their SDK at module scope, so this is safe on a machine with neither
    engine installed -- which is the case this whole layer is built to keep working.
    """
    from instinctlab.engines import ADAPTERS

    for engine, path in ADAPTERS.items():
        if engine not in _ENGINES:
            importlib.import_module(path.partition(":")[0].rpartition(".")[0])


def selector_kinds() -> Mapping[str, frozenset[str]]:
    """Selector kinds every known engine accepts, keyed by engine."""
    _ensure_registered()
    return MappingProxyType({engine: entry.kinds for engine, entry in _ENGINES.items()})


class UnsupportedSelector(PortabilityError):
    """Raised when an engine has no selector for a kind the reference names."""


def selector_field(kind: str) -> str:
    """Name of the config field carrying patterns for ``kind``.

    Both engines follow the same convention for all twelve kinds between them, so this is one
    function rather than a per-engine table.
    """
    return f"{kind}_names"


def _registered(engine: str) -> _Selectors:
    _ensure_registered()
    try:
        return _ENGINES[engine]
    except KeyError:
        raise KeyError(f"unknown engine {engine!r}; known engines are {sorted(_ENGINES)}") from None


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
    entry = _registered(engine)

    missing = sorted(ref.kinds() - entry.kinds)
    if missing:
        raise UnsupportedSelector(
            f"{engine} has no selector for {missing} (entity {ref.entity!r}). "
            f"It supports {sorted(entry.kinds)}. Express this per-engine, or drop the selector in the "
            "task spec so the omission is recorded rather than inferred."
        )

    kwargs: dict[str, Any] = {"name": ref.entity, "preserve_order": ref.preserve_order}
    for kind, patterns in ref.selectors().items():
        kwargs[selector_field(kind)] = entry.container(patterns)
    module, attribute = entry.cfg
    return getattr(importlib.import_module(module), attribute)(**kwargs)


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
