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
:mod:`~instinctlab.compat.env` is the smallest of them, because the two environment classes turned
out to agree on nearly everything a term reads; it covers the command lookup, which fails
differently on each engine, and names the environment type portably.
"""

from __future__ import annotations

from .denylist import DENYLIST, LEGACY_COM_ALIASES, DenylistEntry, PortabilityError, assert_portable
from .entity import UnsupportedSelector, lower, resolved_names, selector_kinds
from .env import ENV_TYPE_NAMES, RlEnv, command_names, env_engine, get_command, has_command
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
    "ENV_TYPE_NAMES",
    "HUB",
    "LEGACY_COM_ALIASES",
    "selector_kinds",
    "Anchor",
    "DenylistEntry",
    "Frame",
    "HubEntry",
    "PortabilityError",
    "RlEnv",
    "RotationConvention",
    "Spoke",
    "UnsupportedSelector",
    "assert_portable",
    "command_names",
    "env_engine",
    "get_command",
    "has_command",
    "hub_entry",
    "lower",
    "resolved_names",
    "spoke_attr",
]
