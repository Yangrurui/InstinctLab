"""Isaac Sim backend public package.

The package root intentionally exposes only lazy attributes. Backend discovery and
argument parsing happen before Isaac Sim starts, so importing this module must not
import torch, Isaac Lab, or compile-time scene implementations.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


def register() -> None:
    """Register the lightweight backend facade and lazy native builders."""
    from .plugin import register as register_plugin

    register_plugin()


register.instinctlab_engine_api = ">=0.1,<0.2"

_EXPORTS = {
    "IsaacSimAdapter": ("adapter", "IsaacSimAdapter"),
    "IsaacSimCompileCtx": ("adapter", "IsaacSimCompileCtx"),
    "TERMS": ("terms", "TERMS"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value


__all__ = ["TERMS", "IsaacSimAdapter", "IsaacSimCompileCtx", "register"]
