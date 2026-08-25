"""Cross-object validation for engine-neutral task declarations.

Individual spec dataclasses validate their own fields in ``__post_init__``.  This module owns the
checks that need a complete :class:`TaskSpec`: engine-key consistency, sensor-to-robot bindings,
and references from MDP terms back into the declared scene.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from .mdp import walk_parameter_values
from .sensor import ContactSensorRef, MotionReferenceRef, RayCasterRef, VolumePointsRef

if TYPE_CHECKING:
    from .task import TaskSpec

_RESERVED_SCENE_NAMES = frozenset({"robot", "terrain"})


def _validate_engine_keys(spec: TaskSpec) -> None:
    declared = set(spec.engines)
    asset_backends = {asset.backend for asset in spec.robot.assets}
    missing_assets = declared - asset_backends
    if missing_assets:
        raise ValueError(
            f"Task {spec.task_id!r} declares engines {sorted(missing_assets)} without a robot asset for them."
        )

    keyed_settings = (
        ("sim.profiles", spec.sim.profiles),
        ("engine_extras", spec.engine_extras),
        ("agent.engine_overrides", spec.agent.engine_overrides),
    )
    for source, settings in keyed_settings:
        unknown = set(settings) - declared
        if unknown:
            raise ValueError(
                f"Task {spec.task_id!r} keys {source} by {sorted(unknown)}, which is not in "
                f"engines={sorted(declared)}. A misspelled engine key is silently ignored "
                "otherwise, and the override never applies."
            )

    for key, term in spec.mdp.terms().items():
        unknown = term.engines_named() - declared
        if unknown:
            raise ValueError(
                f"Term {key!r} has engine_params for {sorted(unknown)}, which is not in engines={sorted(declared)}."
            )


def _matching_names(patterns: Iterable[str], names: Iterable[str]) -> set[str]:
    patterns = tuple(patterns)
    return {name for name in names if any(re.fullmatch(pattern, name) for pattern in patterns)}


def _validate_scene_bindings(spec: TaskSpec) -> None:
    scene = spec.scene
    all_sensors = (
        *scene.contact_sensors,
        *scene.ray_casters,
        *scene.motion_references,
        *scene.volume_points,
    )
    collisions = sorted({sensor.name for sensor in all_sensors} & _RESERVED_SCENE_NAMES)
    if collisions:
        raise ValueError(f"Scene sensor names collide with scene entities: {collisions}.")

    body_names = tuple(spec.robot.body_names)
    body_name_set = set(body_names)
    joint_names = set(spec.robot.joint_names)

    for sensor in scene.contact_sensors:
        if sensor.entity != "robot":
            raise ValueError(f"Contact sensor {sensor.name!r} refers to unknown entity {sensor.entity!r}.")
        unmatched = [pattern for pattern in sensor.elements if not _matching_names((pattern,), body_names)]
        if unmatched:
            raise ValueError(f"Contact sensor {sensor.name!r} patterns match no robot body: {unmatched}.")

    for sensor in scene.ray_casters:
        if sensor.entity != "robot" or sensor.attach not in body_name_set:
            raise ValueError(
                f"Ray caster {sensor.name!r} attaches to {sensor.entity!r}/{sensor.attach!r}, "
                "which is not a declared robot body."
            )
        scene_object_names = {obj.name for obj in scene.rigid_objects}
        unknown_hits = sorted(set(sensor.hit_bodies()) - body_name_set - scene_object_names)
        if unknown_hits:
            raise ValueError(f"Ray caster {sensor.name!r} names unknown hit bodies: {unknown_hits}.")

    for sensor in scene.motion_references:
        unknown_joints = sorted(set(sensor.joints) - joint_names)
        unknown_links = sorted(set(sensor.links) - body_name_set)
        if sensor.entity != "robot" or unknown_joints or unknown_links:
            raise ValueError(
                f"Motion reference {sensor.name!r} does not match the robot: "
                f"entity={sensor.entity!r}, unknown joints={unknown_joints}, unknown links={unknown_links}."
            )

    for sensor in scene.volume_points:
        unknown_bodies = sorted(set(sensor.bodies) - body_name_set)
        if sensor.entity != "robot" or unknown_bodies:
            raise ValueError(
                f"Volume points {sensor.name!r} does not match the robot: "
                f"entity={sensor.entity!r}, unknown bodies={unknown_bodies}."
            )


def _sensor_maps(spec: TaskSpec) -> dict[type, Mapping[str, Any]]:
    scene = spec.scene
    return {
        ContactSensorRef: {sensor.name: sensor for sensor in scene.contact_sensors},
        RayCasterRef: {sensor.name: sensor for sensor in scene.ray_casters},
        MotionReferenceRef: {sensor.name: sensor for sensor in scene.motion_references},
        VolumePointsRef: {sensor.name: sensor for sensor in scene.volume_points},
    }


def _validate_contact_request(
    *,
    term_key: str,
    requested: ContactSensorRef,
    declared: ContactSensorRef,
    body_names: tuple[str, ...],
) -> None:
    tracked_bodies = _matching_names(declared.elements, body_names)
    requested_bodies = _matching_names(requested.elements, body_names)
    if not requested_bodies or not requested_bodies <= tracked_bodies:
        raise ValueError(
            f"Term {term_key!r} requests contact bodies {sorted(requested_bodies)} outside sensor "
            f"{requested.name!r}'s tracked bodies {sorted(tracked_bodies)}."
        )


def _validate_sensor_reference(
    *,
    term_key: str,
    value: Any,
    sensors: Mapping[type, Mapping[str, Any]],
    body_names: tuple[str, ...],
) -> None:
    labels = {
        ContactSensorRef: "contact sensor",
        RayCasterRef: "ray caster",
        MotionReferenceRef: "motion reference",
        VolumePointsRef: "volume-points sensor",
    }
    value_type = type(value)
    declared_by_name = sensors.get(value_type)
    if declared_by_name is None:
        return
    if value.name not in declared_by_name:
        label = labels[value_type]
        raise ValueError(
            f"Term {term_key!r} reads {label} {value.name!r}, which the scene does not declare. "
            f"Declared: {sorted(declared_by_name) or 'none'}."
        )

    declared = declared_by_name[value.name]
    if isinstance(value, ContactSensorRef):
        _validate_contact_request(
            term_key=term_key,
            requested=value,
            declared=declared,
            body_names=body_names,
        )
    elif value != declared:
        label = labels[value_type]
        raise ValueError(f"Term {term_key!r} uses a {label} declaration different from scene sensor {value.name!r}.")


def _validate_mdp_references(spec: TaskSpec) -> None:
    sensors = _sensor_maps(spec)
    body_names = tuple(spec.robot.body_names)
    for key, term in spec.mdp.terms().items():
        parameter_sets = (term.params, *term.engine_params.values())
        for params in parameter_sets:
            command_name = params.get("command_name")
            if command_name is not None and command_name not in spec.mdp.commands:
                raise ValueError(
                    f"Term {key!r} reads command {command_name!r}, which the MDP does not declare. "
                    f"Declared: {sorted(spec.mdp.commands) or 'none'}."
                )

        values = (value for params in parameter_sets for value in params.values())
        for value in walk_parameter_values(values):
            _validate_sensor_reference(term_key=key, value=value, sensors=sensors, body_names=body_names)


def validate_task(spec: TaskSpec) -> None:
    """Validate relationships that span a complete task declaration."""
    spec.robot.validate()
    _validate_engine_keys(spec)
    _validate_scene_bindings(spec)
    _validate_mdp_references(spec)


__all__ = ["validate_task"]
