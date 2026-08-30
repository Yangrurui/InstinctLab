"""Resolve neutral asset identifiers through installed asset plugins."""

from __future__ import annotations

from collections.abc import Callable
from importlib import metadata
from types import ModuleType

from instinctlab_engine.plugins import (
    PluginDiscoveryError,
    _PLUGIN_LOCK,
    _restore_provenance,
    _snapshot_provenance,
    _plugin_locked,
    entry_point_description,
    load_plugin_callable,
    mark_plugin_used,
    record_plugin,
)

AssetResolver = Callable[[str, str], tuple[ModuleType, str]]
"""``(engine, variant) -> (native module, native variant)``."""


class AssetRegistry:
    """Engine-neutral asset package registry.

    An asset distribution exposes one resolver per package through the
    ``instinctlab.assets`` entry-point group. The entry-point name is the first
    component of an asset ID and its value is a callable resolver. For example::

        [project.entry-points."instinctlab.assets"]
        my_robot = "my_robot_assets.interface:native_module"

    This keeps the engine core independent of the application that happens to
    supply a robot catalog.
    """

    ENTRY_POINT_GROUP = "instinctlab.assets"

    def __init__(self, *, load_entry_points: bool = True):
        self._resolvers: dict[str, AssetResolver] = {}
        self._resolver_sources: dict[str, str] = {}
        self._load_entry_points = load_entry_points
        self._entry_points_loaded = False
        self._entry_point_error: PluginDiscoveryError | None = None
        self._active_plugin: str | None = None

    @staticmethod
    def _validate_package(package: str) -> None:
        if not package or not package.isidentifier():
            raise ValueError(f"invalid asset package name {package!r}")

    @_plugin_locked
    def register(self, package: str, resolver: AssetResolver) -> None:
        """Register one asset package resolver without importing an engine SDK."""
        self._validate_package(package)
        if not callable(resolver):
            raise TypeError(f"asset resolver for {package!r} must be callable")
        existing = self._resolvers.get(package)
        if existing is not None and existing is not resolver:
            source = self._resolver_sources.get(package, "a direct registration")
            incoming = self._active_plugin or "a direct registration"
            raise ValueError(
                f"asset package {package!r} is already registered by {source}; "
                f"conflicting registration is from {incoming}"
            )
        self._resolvers[package] = resolver
        if existing is None and self._active_plugin is not None:
            self._resolver_sources[package] = self._active_plugin

    @_plugin_locked
    def _load_installed_assets(self) -> None:
        if self._entry_point_error is not None:
            raise self._entry_point_error
        if self._entry_points_loaded:
            return
        if not self._load_entry_points:
            self._entry_points_loaded = True
            return
        snapshot = (
            dict(self._resolvers),
            dict(self._resolver_sources),
            self._active_plugin,
        )
        provenance_snapshot = _snapshot_provenance()
        try:
            entry_points = metadata.entry_points(group=self.ENTRY_POINT_GROUP)
            for entry_point in sorted(entry_points, key=lambda item: item.name):
                resolver = load_plugin_callable(self.ENTRY_POINT_GROUP, entry_point)
                description = entry_point_description(
                    self.ENTRY_POINT_GROUP, entry_point
                )
                try:
                    self._active_plugin = description
                    self.register(entry_point.name, resolver)
                except Exception as exc:
                    raise PluginDiscoveryError(
                        "Asset plugin registration failed "
                        f"({entry_point_description(self.ENTRY_POINT_GROUP, entry_point)}): {exc}"
                    ) from exc
                finally:
                    self._active_plugin = None
                self._resolver_sources[entry_point.name] = description
                record_plugin(
                    self.ENTRY_POINT_GROUP,
                    entry_point,
                    (entry_point.name,),
                )
        except Exception as exc:  # noqa: BLE001 - plugin transactions must roll back any failure
            self._resolvers, self._resolver_sources, self._active_plugin = snapshot
            _restore_provenance(provenance_snapshot)
            error = (
                exc
                if isinstance(exc, PluginDiscoveryError)
                else PluginDiscoveryError(f"Asset plugin registrar failed: {exc}")
            )
            self._entry_point_error = error
            raise error
        self._entry_points_loaded = True

    def packages(self) -> tuple[str, ...]:
        """Return every registered or installed asset package name."""
        with _PLUGIN_LOCK:
            self._load_installed_assets()
            return tuple(sorted(self._resolvers))

    def native_module(self, asset_id: str, engine: str) -> tuple[ModuleType, str]:
        """Resolve ``package/variant`` through its neutral asset resolver."""
        package, separator, variant = asset_id.partition("/")
        if not separator or not package.isidentifier() or not variant:
            raise ValueError(
                f"asset_id must be 'package/variant' with a Python package name, got {asset_id!r}"
            )
        if not engine.isidentifier():
            raise ValueError(f"engine must be a Python module name, got {engine!r}")
        with _PLUGIN_LOCK:
            self._load_installed_assets()
            try:
                resolver = self._resolvers[package]
            except KeyError:
                known = ", ".join(sorted(self._resolvers)) or "none"
                raise KeyError(
                    f"unknown asset package {package!r}; installed packages are {known}"
                ) from None
        mark_plugin_used(self.ENTRY_POINT_GROUP, package)
        return resolver(engine, variant)


ASSETS = AssetRegistry()


def register_asset(package: str, resolver: AssetResolver) -> None:
    """Register an asset resolver from an application or plugin."""
    ASSETS.register(package, resolver)


def asset_packages() -> tuple[str, ...]:
    """Return every asset package known to the shared registry."""
    return ASSETS.packages()


def native_asset_module(asset_id: str, engine: str) -> tuple[ModuleType, str]:
    """Resolve an asset ID without assuming which application supplies it."""
    return ASSETS.native_module(asset_id, engine)


__all__ = [
    "ASSETS",
    "AssetRegistry",
    "AssetResolver",
    "asset_packages",
    "native_asset_module",
    "register_asset",
]
