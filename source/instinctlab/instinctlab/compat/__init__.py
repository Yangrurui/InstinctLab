"""Thin compatibility layer shared by the portable MDP terms and every engine adapter.

:mod:`~instinctlab.compat.vocab` is the single named source of truth for what a physical quantity
means. :mod:`~instinctlab.compat.denylist` records the attributes whose names agree across engines
while their semantics do not. :mod:`~instinctlab.compat.math` carries the tensor math both engines
already share, so that a term can do frame arithmetic without importing either one; its names are
left on the submodule rather than re-exported here, since callers read better as
``math_utils.quat_apply_inverse``. :mod:`~instinctlab.compat.entity` lowers an ``EntityRef`` onto
each engine's selector config, which is where the engines diverge far more than their data
attributes do, and :mod:`~instinctlab.compat.sensors` reads contact sensors, the one place that
does need a runtime shim because the two engines disagree on tensor layout as well as on names.
"""

from __future__ import annotations

from .denylist import DENYLIST, LEGACY_COM_ALIASES, DenylistEntry, PortabilityError, assert_portable
from .entity import SELECTOR_KINDS, UnsupportedSelector, lower, resolved_names
from .vocab import (
    CANONICAL_QUATERNION,
    ENGINES,
    HUB,
    Anchor,
    Frame,
    HubEntry,
    RotationConvention,
    Spoke,
    hub_entry,
    spoke_attr,
)

__all__ = [
    "CANONICAL_QUATERNION",
    "DENYLIST",
    "ENGINES",
    "HUB",
    "LEGACY_COM_ALIASES",
    "SELECTOR_KINDS",
    "Anchor",
    "DenylistEntry",
    "Frame",
    "HubEntry",
    "PortabilityError",
    "RotationConvention",
    "Spoke",
    "UnsupportedSelector",
    "assert_portable",
    "hub_entry",
    "lower",
    "resolved_names",
    "spoke_attr",
]
