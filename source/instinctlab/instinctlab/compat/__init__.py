"""Thin compatibility layer shared by the portable MDP terms and every engine adapter.

:mod:`~instinctlab.compat.vocab` is the single named source of truth for what a physical quantity
means. :mod:`~instinctlab.compat.denylist` records the attributes whose names agree across engines
while their semantics do not. :mod:`~instinctlab.compat.math` carries the tensor math both engines
already share, so that a term can do frame arithmetic without importing either one; its names are
left on the submodule rather than re-exported here, since callers read better as
``math_utils.quat_apply_inverse``.
"""

from __future__ import annotations

from .denylist import DENYLIST, LEGACY_COM_ALIASES, DenylistEntry, PortabilityError, assert_portable
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
    "Anchor",
    "DenylistEntry",
    "Frame",
    "HubEntry",
    "PortabilityError",
    "RotationConvention",
    "Spoke",
    "assert_portable",
    "hub_entry",
    "spoke_attr",
]
