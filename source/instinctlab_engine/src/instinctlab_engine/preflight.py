"""SDK-free task extension compatibility report before native construction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from instinctlab_engine.actuators import ACTUATORS
from instinctlab_engine.plugins import (
    plugin_provenance_since,
    plugin_usage_snapshot,
)
from instinctlab_engine.registry import TERRAIN_EXTENSIONS
from instinctlab_engine.sensors import SENSORS, required_sensor_capabilities
from instinctlab_engine.spec.capability import Requirement
from instinctlab_engine.spec.entity import EntityRef, resolve_entity_names

if TYPE_CHECKING:
    from instinctlab_engine.base import EngineAdapter
    from instinctlab_engine.spec import TaskSpec


class PreflightError(RuntimeError):
    """The selected task cannot be constructed under its declared contracts."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        problems = [*report["incompatibilities"]]
        if report["omissions"] and not report["allow_nonclean"]:
            problems.extend(report["omissions"])
        details = "\n".join(f"  - {problem}" for problem in problems)
        super().__init__(
            f"Preflight failed for {report['task_id']!r} on {report['engine']!r}:\n"
            f"{details}"
        )


def _entity_refs(value: Any) -> Iterable[EntityRef]:
    if isinstance(value, EntityRef):
        yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from _entity_refs(nested)
    elif isinstance(value, (tuple, list)):
        for nested in value:
            yield from _entity_refs(nested)


def _term_joint_names(
    spec: TaskSpec,
    engine: str,
    term_key: str,
) -> dict[str, tuple[str, ...]]:
    """Resolve canonical joints by articulation for one actuator-aware term."""
    term = spec.mdp.terms().get(term_key)
    if term is None:
        raise ValueError(f"actuator requirements reference unknown term {term_key!r}")
    values = (term.target, term.resolved_params(engine))
    joint_refs = [
        ref
        for value in values
        for ref in _entity_refs(value)
        if ref.joints is not None
    ]
    if not joint_refs:
        return {"robot": tuple(spec.robot.joint_names)}
    selected: dict[str, set[str]] = {}
    for ref in joint_refs:
        schema = spec.articulation_schema(ref.entity)
        selected.setdefault(ref.entity, set()).update(
            resolve_entity_names(
                ref.joints,
                schema.joint_names,
                preserve_order=ref.preserve_order,
            )
        )
    return {
        entity: tuple(
            name
            for name in spec.articulation_schema(entity).joint_names
            if name in names
        )
        for entity, names in selected.items()
    }


def _provider_keys(
    *,
    engine: str,
    asset_ids: Iterable[str],
    actuator_model_ids: Iterable[str],
    sensor_kinds: Iterable[str],
    terrain_kind: str,
    sub_terrain_kinds: Iterable[str],
) -> tuple[tuple[str, str], ...]:
    keys: list[tuple[str, str]] = [
        ("instinctlab.engines", engine),
        ("instinctlab.terrains", f"whole:{engine}:{terrain_kind}"),
        ("instinctlab.engines", f"terrain:whole:{engine}:{terrain_kind}"),
    ]
    keys.extend(
        ("instinctlab.assets", package)
        for package in dict.fromkeys(
            asset_id.partition("/")[0] for asset_id in asset_ids
        )
    )
    keys.extend(
        ("instinctlab.actuators", f"{engine}:{model_id}")
        for model_id in actuator_model_ids
    )
    keys.extend(
        ("instinctlab.sensors", f"{engine}:{kind}") for kind in sensor_kinds
    )
    for kind in sub_terrain_kinds:
        keys.extend(
            (
                ("instinctlab.terrains", f"sub:{engine}:{kind}"),
                ("instinctlab.engines", f"terrain:sub:{engine}:{kind}"),
            )
        )
    return tuple(keys)


def _native_only_features(spec: TaskSpec, engine: str) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    profile = spec.sim.profiles.get(engine, {})
    if profile:
        features.append({"kind": "simulation_profile", "keys": sorted(profile)})
    extras = spec.engine_extras.get(engine, {})
    if extras:
        features.append({"kind": "engine_extras", "keys": sorted(extras)})
    term_overrides = sorted(
        key
        for key, term in spec.mdp.terms().items()
        if engine in term.engine_params
    )
    if term_overrides:
        features.append({"kind": "term_engine_params", "terms": term_overrides})
    if spec.scene.native_sensors:
        features.append(
            {
                "kind": "native_sensors",
                "sensors": [sensor.name for sensor in spec.scene.native_sensors],
            }
        )
    object_overrides = [
        obj.name for obj in spec.scene.rigid_objects if engine in obj.engine_meshes
    ]
    if object_overrides:
        features.append(
            {"kind": "rigid_object_resources", "objects": object_overrides}
        )
    return features


def preflight_report(
    spec: TaskSpec,
    engine: str,
    *,
    selected_adapter: EngineAdapter | None = None,
    allow_nonclean: bool = False,
) -> dict[str, Any]:
    """Collect selected providers and fail-closed compatibility evidence.

    This function resolves registration metadata and local resources only. It
    never resolves actuator factories, sensor builders, terrain builders, or a
    simulator SDK object.
    """
    usage_start = plugin_usage_snapshot()
    if selected_adapter is None:
        from instinctlab_engine import adapter

        selected_adapter = adapter(engine)
    if selected_adapter.name != engine:
        raise ValueError(
            f"preflight engine {engine!r} does not match adapter "
            f"{selected_adapter.name!r}"
        )

    incompatibilities: list[str] = []
    omissions: list[str] = []
    try:
        spec.validate_for_engine(engine)
    except Exception as exc:  # noqa: BLE001 - report all boundary failures uniformly
        incompatibilities.append(f"task contract: {exc}")

    try:
        contract = selected_adapter.contract_report(spec)
    except Exception as exc:  # noqa: BLE001
        contract = {
            "engine": engine,
            "task_id": spec.task_id,
            "capabilities": [],
            "missing": {},
            "omitted": [],
            "engine_extras_used": [],
            "requested_capabilities": {},
        }
        incompatibilities.append(f"term contract: {exc}")
    for term_key, reason in contract.get("missing", {}).items():
        term = spec.mdp.terms().get(term_key)
        if term is not None and term.level is Requirement.REQUIRED:
            incompatibilities.append(f"required term {term_key}: {reason}")
        else:
            omissions.append(f"term {term_key}: {reason}")
    omissions.extend(
        f"profile omission {term_key}" for term_key in contract.get("omitted", ())
    )

    asset_reports: dict[str, dict[str, Any] | None] = {}
    for entity_name, schema in spec.articulation_schemas.items():
        try:
            asset_reports[entity_name] = selected_adapter.asset_conformance(
                schema.asset_id
            )
        except Exception as exc:  # noqa: BLE001
            asset_reports[entity_name] = None
            incompatibilities.append(
                f"native asset {entity_name!r}/{schema.asset_id}: {exc}"
            )
    asset_report = asset_reports["robot"]

    try:
        actuator_by_term = selected_adapter.actuator_requirements(spec)
    except Exception as exc:  # noqa: BLE001
        actuator_by_term = {}
        incompatibilities.append(f"actuator term requirements: {exc}")
    actuator_model_ids = tuple(
        dict.fromkeys(
            model_id
            for report in asset_reports.values()
            if report is not None
            for model_id in report.get("actuator_model_ids", ())
        )
    )
    actuator_groups = tuple(
        (entity_name, group)
        for entity_name, report in asset_reports.items()
        if report is not None
        for group in report.get("actuator_groups", ())
    )
    actuator_reports: list[dict[str, Any]] = []
    joint_names_by_term: dict[str, dict[str, tuple[str, ...]]] = {}
    try:
        actuator_registrations = ACTUATORS.registrations(engine)
        if actuator_model_ids and not actuator_groups:
            incompatibilities.append(
                "native asset report has actuator models but no group-to-model metadata"
            )
        for term_key in actuator_by_term:
            try:
                joint_names_by_term[term_key] = _term_joint_names(
                    spec, engine, term_key
                )
            except Exception as exc:  # noqa: BLE001
                incompatibilities.append(
                    f"actuator joint selection for {term_key}: {exc}"
                )
        for entity_name, group in actuator_groups:
            group_name = str(group["name"])
            model_id = str(group["model_id"])
            group_joints = tuple(group["joint_names"])
            requirements_by_term = {
                term_key: capabilities
                for term_key, capabilities in actuator_by_term.items()
                if set(group_joints).intersection(
                    joint_names_by_term.get(term_key, {}).get(entity_name, ())
                )
            }
            requested = sorted(
                {
                    capability
                    for capabilities in requirements_by_term.values()
                    for capability in capabilities
                }
            )
            registration = actuator_registrations.get(model_id)
            if registration is None:
                incompatibilities.append(
                    f"actuator group {group_name!r} model {model_id!r} has no provider "
                    f"for {engine!r}"
                )
                continue
            missing = set(requested) - registration.capabilities
            if missing:
                incompatibilities.append(
                    f"actuator group {group_name!r} model {model_id!r} lacks requested "
                    f"capabilities: {', '.join(sorted(missing))}"
                )
            actuator_reports.append(
                {
                    "entity": entity_name,
                    "group": group_name,
                    "model_id": model_id,
                    "selectors": list(group.get("selectors", ())),
                    "joint_names": list(group_joints),
                    "capabilities": sorted(registration.capabilities),
                    "requirements_by_term": requirements_by_term,
                    "requested_capabilities": requested,
                    "missing_capabilities": sorted(missing),
                }
            )
    except Exception as exc:  # noqa: BLE001
        incompatibilities.append(f"actuator providers: {exc}")

    sensor_reports: list[dict[str, Any]] = []
    sensor_kinds = tuple(sensor.kind for sensor in spec.scene.native_sensors)
    try:
        sensor_registrations = SENSORS.registrations(engine)
        for sensor in spec.scene.native_sensors:
            required = required_sensor_capabilities(sensor)
            registration = sensor_registrations.get(sensor.kind)
            if registration is None:
                incompatibilities.append(
                    f"native sensor {sensor.name!r} kind {sensor.kind!r} has no "
                    f"provider for {engine!r}"
                )
                sensor_reports.append(
                    {
                        "name": sensor.name,
                        "kind": sensor.kind,
                        "requested_capabilities": sorted(required),
                        "missing_capabilities": sorted(required),
                    }
                )
                continue
            missing = required - registration.capabilities
            if missing:
                incompatibilities.append(
                    f"native sensor {sensor.name!r} lacks lifecycle capabilities: "
                    f"{', '.join(sorted(missing))}"
                )
            sensor_reports.append(
                {
                    "name": sensor.name,
                    "kind": sensor.kind,
                    "capabilities": sorted(registration.capabilities),
                    "requested_capabilities": sorted(required),
                    "missing_capabilities": sorted(missing),
                }
            )
    except Exception as exc:  # noqa: BLE001
        incompatibilities.append(f"sensor providers: {exc}")

    terrain_kind = spec.scene.terrain.kind
    generator = spec.scene.terrain.generator
    sub_terrain_kinds = (
        tuple(tile.kind for tile in generator.sub_terrains.values())
        if generator is not None
        else ()
    )
    try:
        available_terrains = TERRAIN_EXTENSIONS.terrain_kinds(engine)
        if terrain_kind not in available_terrains:
            incompatibilities.append(
                f"terrain kind {terrain_kind!r} has no provider for {engine!r}"
            )
        available_sub_terrains = TERRAIN_EXTENSIONS.sub_terrain_kinds(engine)
        missing_tiles = set(sub_terrain_kinds) - available_sub_terrains
        if missing_tiles:
            incompatibilities.append(
                "generated terrain tile kinds have no provider for "
                f"{engine!r}: {', '.join(sorted(missing_tiles))}"
            )
    except Exception as exc:  # noqa: BLE001
        incompatibilities.append(f"terrain providers: {exc}")

    rigid_objects: list[dict[str, Any]] = []
    for ref in spec.scene.rigid_objects:
        try:
            rigid_objects.append(selected_adapter.rigid_object_conformance(ref))
        except Exception as exc:  # noqa: BLE001
            rigid_objects.append(ref.resource_report(engine))
            incompatibilities.append(f"rigid object {ref.name!r}: {exc}")

    provider_keys = _provider_keys(
        engine=engine,
        asset_ids=(
            schema.asset_id for schema in spec.articulation_schemas.values()
        ),
        actuator_model_ids=actuator_model_ids,
        sensor_kinds=sensor_kinds,
        terrain_kind=terrain_kind,
        sub_terrain_kinds=sub_terrain_kinds,
    )
    providers = plugin_provenance_since(
        usage_start,
        engine=engine,
        include_keys=provider_keys,
    )
    failed = bool(incompatibilities or (omissions and not allow_nonclean))
    return {
        "schema_version": "preflight_v1",
        "status": "failed" if failed else "ok",
        "engine": engine,
        "task_id": spec.task_id,
        "asset": asset_report,
        "additional_articulations": [
            {
                "name": ref.name,
                "asset_id": ref.schema.asset_id,
                "asset": asset_reports[ref.name],
            }
            for ref in spec.scene.articulations
        ],
        "actuators": actuator_reports,
        "native_sensors": sensor_reports,
        "terrain": {
            "kind": terrain_kind,
            "sub_terrain_kinds": list(sub_terrain_kinds),
        },
        "rigid_objects": rigid_objects,
        "requested_capabilities": {
            "engine_terms": contract.get("requested_capabilities", {}),
            "actuator_by_term": actuator_by_term,
            "actuator_joint_names_by_term": {
                term_key: {
                    entity: list(joint_names)
                    for entity, joint_names in selections.items()
                }
                for term_key, selections in joint_names_by_term.items()
            },
            "native_sensors": {
                report["name"]: report["requested_capabilities"]
                for report in sensor_reports
            },
        },
        "engine_capabilities": contract.get("capabilities", []),
        "native_only_features": _native_only_features(spec, engine),
        "omissions": omissions,
        "incompatibilities": incompatibilities,
        "allow_nonclean": allow_nonclean,
        "providers": providers,
        "selected_components": {
            "asset_id": spec.robot.asset_id,
            "articulation_asset_ids": {
                entity: schema.asset_id
                for entity, schema in spec.articulation_schemas.items()
            },
            "actuator_model_ids": list(actuator_model_ids),
            "actuator_groups": [
                {
                    "entity": entity,
                    "name": group["name"],
                    "model_id": group["model_id"],
                }
                for entity, group in actuator_groups
            ],
            "sensor_kinds": list(sensor_kinds),
            "terrain_kind": terrain_kind,
            "sub_terrain_kinds": list(sub_terrain_kinds),
        },
    }


def require_preflight(
    spec: TaskSpec,
    engine: str,
    *,
    selected_adapter: EngineAdapter | None = None,
    allow_nonclean: bool = False,
) -> dict[str, Any]:
    """Return a passing report or raise before a native config is constructed."""
    report = preflight_report(
        spec,
        engine,
        selected_adapter=selected_adapter,
        allow_nonclean=allow_nonclean,
    )
    if report["status"] != "ok":
        raise PreflightError(report)
    return report


__all__ = ["PreflightError", "preflight_report", "require_preflight"]
