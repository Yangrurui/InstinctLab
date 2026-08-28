"""Resolve an engine-neutral asset id to its engine-owned asset module."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

__all__ = ["native_asset_module"]


def native_asset_module(asset_id: str, engine: str) -> tuple[ModuleType, str]:
    """Resolve ``package/variant`` through the package's neutral interface.

    The resolver knows the package convention, not any concrete robot. Each
    package interface only forwards to its selected native module, which keeps
    engine adapters generic while native registrations and paths stay local.
    """
    package, separator, variant = asset_id.partition("/")
    if not separator or not package.isidentifier() or not variant:
        raise ValueError(
            f"asset_id must be 'package/variant' with a Python package name, got {asset_id!r}"
        )
    if not engine.isidentifier():
        raise ValueError(f"engine must be a Python module name, got {engine!r}")
    interface = import_module(f"instinctlab.assets.{package}.interface")
    return interface.native_module(engine, variant)
