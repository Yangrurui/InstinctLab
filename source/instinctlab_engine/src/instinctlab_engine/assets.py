"""Resolve neutral asset identifiers through installed asset plugins."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
import re
from types import ModuleType
from typing import Any
from xml.etree import ElementTree

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

NATIVE_ASSET_API_VERSION = "0.1"


class NativeAssetContractError(RuntimeError):
    """An asset plugin does not satisfy the selected backend's native boundary."""


@dataclass(frozen=True, slots=True)
class NativeAssetDefinition:
    """Validated native module surface, still free of simulator SDK objects."""

    asset_id: str
    engine: str
    variant: str
    module: ModuleType
    config: Any
    builder: Callable[..., Any]
    resource_path: Path
    resource_kind: str


_COMMON_CONFIG_FIELDS = (
    "name",
    "schema_version",
    "asset_id",
    "root_body",
    "joint_names",
    "body_names",
    "frame_names",
    "collision_body_names",
    "joint_properties",
    "contact_body_aliases",
    "default_root_pos",
    "default_root_quat_wxyz",
    "soft_joint_pos_limit_factor",
    "actuator_delay",
    "actuator_model_ids",
    "actuator_group_count",
    "length_unit",
    "angle_unit",
    "effort_unit",
)


def _require_attributes(value: Any, names: tuple[str, ...], *, context: str) -> None:
    missing = [name for name in names if not hasattr(value, name)]
    if missing:
        raise NativeAssetContractError(
            f"{context} is missing required fields: {', '.join(missing)}"
        )


def native_asset_definition(
    asset_id: str,
    engine: str,
    *,
    builder_name: str,
    resource_field: str,
    resource_kind: str,
    registry: "AssetRegistry | None" = None,
) -> NativeAssetDefinition:
    """Validate one installed native asset module before native construction.

    The module and its ``native_config`` function must remain SDK-free. The
    returned builder is invoked later by the selected backend, after bootstrap.
    """
    selected_registry = registry or ASSETS
    module, variant = selected_registry.native_module(asset_id, engine)
    api_version = getattr(module, "INSTINCTLAB_NATIVE_ASSET_API", None)
    if api_version != NATIVE_ASSET_API_VERSION:
        raise NativeAssetContractError(
            f"Native asset module {module.__name__!r} for {engine!r} declares API "
            f"{api_version!r}; required {NATIVE_ASSET_API_VERSION!r}."
        )
    native_config = getattr(module, "native_config", None)
    if not callable(native_config):
        raise NativeAssetContractError(
            f"Native asset module {module.__name__!r} does not define "
            "native_config(variant)."
        )
    builder = getattr(module, builder_name, None)
    if not callable(builder):
        raise NativeAssetContractError(
            f"Native asset module {module.__name__!r} does not define "
            f"{builder_name}(variant, robot)."
        )
    config = native_config(variant)
    _require_attributes(
        config,
        (*_COMMON_CONFIG_FIELDS, resource_field),
        context=f"Native config {module.__name__}:{variant}",
    )
    if config.asset_id != asset_id:
        raise NativeAssetContractError(
            f"Native config {module.__name__}:{variant} declares "
            f"{config.asset_id!r}, expected {asset_id!r}."
        )
    resource_path = Path(getattr(config, resource_field))
    if not resource_path.is_file():
        raise NativeAssetContractError(
            f"Native asset resource for {asset_id!r} on {engine!r} is missing: "
            f"{resource_path}"
        )
    definition = NativeAssetDefinition(
        asset_id=asset_id,
        engine=engine,
        variant=variant,
        module=module,
        config=config,
        builder=builder,
        resource_path=resource_path,
        resource_kind=resource_kind,
    )
    validate_native_asset_definition(definition)
    return definition


def _duplicates(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return tuple(sorted(duplicates))


def _urdf_topology(path: Path, root_body: str) -> tuple[tuple[str, ...], set[str], set[str]]:
    root = ElementTree.parse(path).getroot()
    bodies = {link.get("name") for link in root.findall("link") if link.get("name")}
    collision_bodies = {
        link.get("name")
        for link in root.findall("link")
        if link.get("name") and link.find("collision") is not None
    }
    children: dict[str, list[tuple[str, str]]] = {}
    for joint in root.findall("joint"):
        if joint.get("type") == "fixed":
            continue
        parent = joint.find("parent")
        child = joint.find("child")
        name = joint.get("name")
        if parent is None or child is None or not name:
            continue
        children.setdefault(parent.get("link", ""), []).append(
            (name, child.get("link", ""))
        )
    joints: list[str] = []

    def visit(body: str) -> None:
        for joint_name, child_body in children.get(body, ()):
            joints.append(joint_name)
            visit(child_body)

    visit(root_body)
    return tuple(joints), bodies, collision_bodies


def _mjcf_topology(path: Path) -> tuple[tuple[str, ...], set[str], set[str]]:
    root = ElementTree.parse(path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise NativeAssetContractError(f"MJCF resource has no <worldbody>: {path}")
    joints: list[str] = []
    bodies: set[str] = set()
    collision_bodies: set[str] = set()

    def visit(body: Any) -> None:
        name = body.get("name")
        if name:
            bodies.add(name)
            if body.find("geom") is not None:
                collision_bodies.add(name)
        for joint in body.findall("joint"):
            if joint.get("type", "hinge") != "free" and joint.get("name"):
                joints.append(joint.get("name"))
        for child in body.findall("body"):
            visit(child)

    for body in worldbody.findall("body"):
        visit(body)
    return tuple(joints), bodies, collision_bodies


def validate_native_asset_definition(definition: NativeAssetDefinition) -> None:
    """Fail closed on the SDK-free portion of native robot onboarding."""
    config = definition.config
    names = {
        "joint_names": tuple(config.joint_names),
        "body_names": tuple(config.body_names),
        "frame_names": tuple(config.frame_names),
        "collision_body_names": tuple(config.collision_body_names),
    }
    for field, values in names.items():
        duplicates = _duplicates(values)
        if duplicates:
            raise NativeAssetContractError(
                f"{definition.asset_id} has duplicate {field}: {', '.join(duplicates)}"
            )
    if config.root_body not in names["body_names"]:
        raise NativeAssetContractError(
            f"{definition.asset_id} root body {config.root_body!r} is absent from body_names"
        )
    for field in ("frame_names", "collision_body_names"):
        unknown = set(names[field]) - set(names["body_names"])
        if unknown:
            raise NativeAssetContractError(
                f"{definition.asset_id} {field} contains unknown bodies: "
                f"{', '.join(sorted(unknown))}"
            )

    properties = tuple(config.joint_properties)
    property_names = tuple(item.name for item in properties)
    if property_names != names["joint_names"]:
        raise NativeAssetContractError(
            f"{definition.asset_id} joint_properties must exactly follow canonical DFS "
            "joint_names"
        )
    for joint in properties:
        if joint.effort_limit <= 0.0 or joint.velocity_limit <= 0.0:
            raise NativeAssetContractError(
                f"{definition.asset_id} joint {joint.name!r} has non-positive limits"
            )
        if joint.armature < 0.0 or joint.action_scale <= 0.0:
            raise NativeAssetContractError(
                f"{definition.asset_id} joint {joint.name!r} has invalid native values"
            )
    if not 0.0 < config.soft_joint_pos_limit_factor <= 1.0:
        raise NativeAssetContractError(
            f"{definition.asset_id} soft_joint_pos_limit_factor must be in (0, 1]"
        )
    delay_min, delay_max = config.actuator_delay
    if delay_min < 0 or delay_max < delay_min:
        raise NativeAssetContractError(
            f"{definition.asset_id} has invalid actuator_delay {config.actuator_delay!r}"
        )
    model_ids = tuple(config.actuator_model_ids)
    if not model_ids or _duplicates(model_ids):
        raise NativeAssetContractError(
            f"{definition.asset_id} must declare unique actuator_model_ids"
        )
    if not isinstance(config.actuator_group_count, int) or config.actuator_group_count <= 0:
        raise NativeAssetContractError(
            f"{definition.asset_id} actuator_group_count must be a positive integer"
        )
    from instinctlab_engine.actuators import ACTUATORS

    known_models = set(ACTUATORS.registrations(definition.engine))
    unknown_models = set(model_ids) - known_models
    if unknown_models:
        raise NativeAssetContractError(
            f"{definition.asset_id} requests unavailable actuator models for "
            f"{definition.engine!r}: {', '.join(sorted(unknown_models))}"
        )
    expected_units = {"length_unit": "m", "angle_unit": "rad", "effort_unit": "N*m"}
    wrong_units = {
        field: getattr(config, field)
        for field, expected in expected_units.items()
        if getattr(config, field) != expected
    }
    if wrong_units:
        raise NativeAssetContractError(
            f"{definition.asset_id} uses unsupported native units: {wrong_units}"
        )

    if definition.resource_kind == "urdf":
        resource_joints, resource_bodies, collision_bodies = _urdf_topology(
            definition.resource_path, config.root_body
        )
    elif definition.resource_kind == "mjcf":
        resource_joints, resource_bodies, collision_bodies = _mjcf_topology(
            definition.resource_path
        )
    else:
        raise NativeAssetContractError(
            f"unsupported native resource kind {definition.resource_kind!r}"
        )
    if resource_joints != names["joint_names"]:
        raise NativeAssetContractError(
            f"{definition.asset_id} canonical joint_names do not match resource DFS order"
        )
    missing_bodies = set(names["body_names"]) - resource_bodies
    if missing_bodies:
        raise NativeAssetContractError(
            f"{definition.asset_id} body_names are missing from its resource: "
            f"{', '.join(sorted(missing_bodies))}"
        )
    missing_collision = set(names["collision_body_names"]) - collision_bodies
    if missing_collision:
        raise NativeAssetContractError(
            f"{definition.asset_id} declares bodies without collision geometry: "
            f"{', '.join(sorted(missing_collision))}"
        )


def native_asset_conformance_report(
    definition: NativeAssetDefinition,
) -> dict[str, Any]:
    """Readable evidence from the validated SDK-free asset boundary."""
    config = definition.config
    return {
        "asset_id": definition.asset_id,
        "engine": definition.engine,
        "module": definition.module.__name__,
        "variant": definition.variant,
        "native_asset_api": NATIVE_ASSET_API_VERSION,
        "resource": str(definition.resource_path),
        "resource_kind": definition.resource_kind,
        "schema_version": config.schema_version,
        "canonical_order": "dfs",
        "joint_count": len(config.joint_names),
        "body_count": len(config.body_names),
        "frame_count": len(config.frame_names),
        "collision_body_count": len(config.collision_body_names),
        "actuator_model_ids": list(config.actuator_model_ids),
        "actuator_group_count": config.actuator_group_count,
        "units": {
            "length": config.length_unit,
            "angle": config.angle_unit,
            "effort": config.effort_unit,
        },
        "status": "ok",
    }


def validate_native_actuator_groups(
    asset_id: str,
    groups: Any,
    joint_names: tuple[str, ...],
    *,
    selector_field: str,
    expected_group_count: int,
) -> None:
    """Validate native group selectors after SDK config construction.

    This checks coverage only. Every concrete parameter remains written on the
    native group object owned by the asset module.
    """
    if isinstance(groups, dict):
        named_groups = tuple((str(name), value) for name, value in groups.items())
    else:
        named_groups = tuple((str(index), value) for index, value in enumerate(groups))
    if len(named_groups) != expected_group_count:
        raise NativeAssetContractError(
            f"{asset_id} constructed {len(named_groups)} actuator groups; native config "
            f"declares {expected_group_count}"
        )
    coverage = {name: [] for name in joint_names}
    for group_name, group in named_groups:
        selectors = getattr(group, selector_field, None)
        if not selectors:
            raise NativeAssetContractError(
                f"{asset_id} actuator group {group_name!r} has no {selector_field}"
            )
        matched = [
            name
            for name in joint_names
            if any(re.fullmatch(pattern, name) for pattern in selectors)
        ]
        if not matched:
            raise NativeAssetContractError(
                f"{asset_id} actuator group {group_name!r} matches no canonical joint"
            )
        for name in matched:
            coverage[name].append(group_name)
    uncovered = [name for name, owners in coverage.items() if not owners]
    duplicated = {
        name: owners for name, owners in coverage.items() if len(owners) > 1
    }
    if uncovered or duplicated:
        raise NativeAssetContractError(
            f"{asset_id} actuator joint coverage is not exact; uncovered={uncovered}, "
            f"multiple_groups={duplicated}"
        )


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
    "NATIVE_ASSET_API_VERSION",
    "NativeAssetContractError",
    "NativeAssetDefinition",
    "asset_packages",
    "native_asset_conformance_report",
    "native_asset_definition",
    "native_asset_module",
    "register_asset",
    "validate_native_asset_definition",
    "validate_native_actuator_groups",
]
