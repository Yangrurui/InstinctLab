"""Resolve an engine-neutral asset id to its engine-owned asset module."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

__all__ = ["native_asset_module"]


def native_asset_module(asset_id: str, engine: str) -> tuple[ModuleType, str]:
    """Return ``(assets.<package>.<engine>, variant)`` for ``package/variant``.

    The resolver knows the package convention, not any concrete robot.  This
    keeps engine adapters generic while each asset package owns its native
    registrations and model paths.
    """
    package, separator, variant = asset_id.partition("/")
    if not separator or not package.isidentifier() or not variant:
        raise ValueError(
            f"asset_id must be 'package/variant' with a Python package name, got {asset_id!r}"
        )
    if not engine.isidentifier():
        raise ValueError(f"engine must be a Python module name, got {engine!r}")
    return import_module(f"instinctlab.assets.{package}.{engine}"), variant
