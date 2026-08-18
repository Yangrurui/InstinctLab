"""Thin compatibility layer shared by the portable MDP terms and every engine adapter.

:mod:`~instinctlab.compat.vocab` is the single named source of truth for what a physical quantity
means. :mod:`~instinctlab.compat.denylist` records the attributes whose names agree across engines
while their semantics do not.
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
