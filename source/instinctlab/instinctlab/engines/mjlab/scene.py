"""Assembling an mjlab scene from a declared one.

Two places where this reads differently from the Isaac Sim side, and both are the design working
rather than an inconvenience.

The solver profile has no counterpart on the other engine at all -- Newton with an iteration budget
and a CCD budget is not a translation of PhysX's TGS with position and velocity iterations. Neither
task states them; each engine's profile carries its own reference implementation's values.

Contact sensors go the other way. Isaac Lab has one sensor per prim pattern and terms slice it;
mjlab has explicitly matched sensors. A single sensor over every body works here too, so the
declaration's one sensor lowers to one sensor, and terms slice it by element index exactly as they
do on Isaac Lab. InstinctMJ instead declares two narrow sensors, one per contact group -- the same
elements measured, arranged differently.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from instinctlab.compat.sensors.ray import refuse_unhonored_ray_alignment
from instinctlab.engines.registry import TERRAIN_EXTENSIONS
from instinctlab.spec.sensor import ContactSensorRef, RayCasterRef, VolumePointsRef
from instinctlab.spec.task import SceneSpec, TerrainSpec

from .assets import entity as build_entity

__all__ = ["PROFILE_DEFAULTS", "build_scene"]

PROFILE_DEFAULTS: Mapping[str, Any] = {
    "solver": "newton",
    "iterations": 10,
    "ls_iterations": 20,
    "ccd_iterations": 500,
    "njmax": 300,
}
"""Native simulator defaults shared by every task unless its profile overrides them."""


def _terrain(spec: TerrainSpec, profile: Mapping[str, Any]) -> Any:
    builder = TERRAIN_EXTENSIONS.terrain("mjlab", spec.kind)
    if builder is None:
        available = sorted(TERRAIN_EXTENSIONS.terrain_kinds("mjlab"))
        raise NotImplementedError(
            f"The MJLab backend has no terrain {spec.kind!r}; "
            f"registered terrains are {available}."
        )
    return builder(spec, profile)


def _build_volume_points(sensor: VolumePointsRef) -> Any:
    from .volume_points import build_sensor

    return build_sensor(sensor)


def _build_contact_sensor(sensor: ContactSensorRef) -> Any:
    """One mjlab contact sensor covering the declared elements.

    ``reduce="netforce"`` gives one force per element, which is the layout the portable accessors
    expect and the only one comparable to Isaac Lab's per-body net force.

    The air-time clock runs off net force rather than mjlab's ``found``, because ``found`` counts
    any contact the solver reports at any force while Isaac Lab requires
    ``ContactSensorRef.air_time_force_threshold`` newtons. Left alone the two engines hand the same
    portable ``feet_air_time`` two different gaits -- see :mod:`.contact_sensor`.

    ``"found"`` is still requested rather than inherited from mjlab's default, because the stock
    clock returns early without it and leaves both timers at zero for the whole run: no
    ``illegal_contact``, no ``feet_air_time`` payout, training proceeding on an episode that can
    only time out. Keeping the field means a sensor that falls back to the stock class still ticks.
    """
    sensor = sensor.for_engine("mjlab")
    from mjlab.sensor import ContactMatch

    from .contact_sensor import thresholded_contact_sensor_cfg

    elements = (
        (sensor.elements,)
        if isinstance(sensor.elements, str)
        else tuple(sensor.elements)
    )
    return thresholded_contact_sensor_cfg(
        name=sensor.name,
        primary=ContactMatch(mode="body", pattern=elements, entity=sensor.entity),
        secondary=None
        if sensor.against is None
        else ContactMatch(mode="body", pattern=(sensor.against,)),
        fields=("found", "force"),
        reduce="netforce",
        track_air_time=sensor.track_air_time,
        history_length=sensor.history_length,
        force_threshold=sensor.air_time_force_threshold,
    )


def _build_ray_caster(
    sensor: RayCasterRef, profile: Mapping[str, Any] | None = None
) -> Any:
    """mjlab does not ship Isaac's sky-origin scanner or world-convention camera."""
    sensor = sensor.for_engine("mjlab")
    refuse_unhonored_ray_alignment(sensor)
    if sensor.pattern.kind == "pinhole":
        from .camera import pinhole_ray_caster

        return pinhole_ray_caster(sensor, profile)
    if sensor.mode == "terrain_height":
        from .raycast import terrain_height_scanner

        return terrain_height_scanner(sensor)
    from .raycast import terrain_ray_caster

    return terrain_ray_caster(sensor)


def build_scene(
    spec: SceneSpec, robot: Any, profile: Mapping[str, Any], *, num_envs: int
) -> Any:
    """A ``SceneCfg`` holding the robot, terrain and sensors."""
    from mjlab.scene import SceneCfg

    sensors = (
        tuple(_build_contact_sensor(sensor) for sensor in spec.contact_sensors)
        + tuple(_build_ray_caster(sensor, profile) for sensor in spec.ray_casters)
        + tuple(
            _build_motion_reference(sensor, robot) for sensor in spec.motion_references
        )
        + tuple(_build_volume_points(sensor) for sensor in spec.volume_points)
    )
    entities = {"robot": build_entity(robot)}
    for obj in spec.rigid_objects:
        import os

        import mujoco
        from mjlab.entity import EntityCfg

        resolved = obj.for_engine("mjlab")

        def object_spec(resolved=resolved):
            native = mujoco.MjSpec()
            mesh = native.add_mesh(
                name="object_mesh",
                file=os.path.expanduser(resolved.mesh),
                scale=resolved.scale,
            )
            body = native.worldbody.add_body(name="object", mocap=resolved.kinematic)
            body.add_geom(
                name="object_geom",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh.name,
                mass=resolved.mass,
                group=2,
                friction=(1.0, 0.005, 0.0001),
            )
            return native

        entities[resolved.name] = EntityCfg(spec_fn=object_spec)
    return SceneCfg(
        num_envs=num_envs,
        env_spacing=spec.env_spacing,
        terrain=_terrain(spec.terrain, profile),
        entities=entities,
        sensors=sensors,
    )


def _build_motion_reference(sensor: Any, robot: Any) -> Any:
    """Build the clip-backed reference sensor for MJLab's lifecycle."""
    from .motion_reference_sensor import build_motion_reference_sensor

    return build_motion_reference_sensor(sensor, robot)
