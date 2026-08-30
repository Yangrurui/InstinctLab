"""Small engine-neutral interfaces for native values that actually differ.

Task formulas stay in their owning task family. This package contains only
runtime boundaries that normalize native names, shapes, failure behavior, or
quantity selection without importing an engine SDK.
"""

from __future__ import annotations

from .entity import UnsupportedSelector, lower, resolved_names, selector_kinds
from .env import (
    ENV_TYPE_NAMES,
    RlEnv,
    command_names,
    env_engine,
    get_command,
    has_command,
)
from .errors import PortabilityError

__all__ = [
    "ENV_TYPE_NAMES",
    "PortabilityError",
    "RlEnv",
    "UnsupportedSelector",
    "command_names",
    "env_engine",
    "get_command",
    "has_command",
    "lower",
    "resolved_names",
    "selector_kinds",
]
