"""Resolve neutral asset identifiers to one engine's native asset module."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

__all__ = ["native_asset_module"]


def native_asset_module(asset_id: str, engine: str) -> tuple[ModuleType, str]:
    """Resolve ``package/variant`` through the asset package's neutral interface."""
    package, separator, variant = asset_id.partition("/")
    if not separator or not package.isidentifier() or not variant:
        raise ValueError(
            f"asset_id must be 'package/variant' with a Python package name, got {asset_id!r}"
        )
    if not engine.isidentifier():
        raise ValueError(f"engine must be a Python module name, got {engine!r}")
    interface = import_module(f"instinctlab.assets.{package}.interface")
    return interface.native_module(engine, variant)
