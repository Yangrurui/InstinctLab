"""Assembling an Isaac Lab scene from a declared one.

The interesting part is that the declaration says almost nothing about physics. Self-collision,
solver iteration counts and depenetration velocity live in this engine's profile rather than in the
task, because they are PhysX settings with no counterpart to state portably -- a task that named
them would be an Isaac Lab task wearing a portable coat.

Contact sensors are the exception in the other direction. Isaac Lab needs
``activate_contact_sensors`` set on the spawn or the sensor silently reports zeros, and that flag
is a consequence of the scene declaring a sensor at all. Deriving it here rather than asking the
task for it removes a way to declare a sensor that cannot work.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from instinctlab_engine.bridge.sensors.ray import refuse_unhonored_ray_alignment
from instinctlab_engine.registry import TERRAIN_EXTENSIONS
from instinctlab_engine.sensors import NativeSensorBuildContext, native_sensor_builder
from instinctlab_engine.spec.sensor import ContactSensorRef, RayCasterRef, VolumePointsRef
from instinctlab_engine.spec.task import SceneSpec, TerrainSpec

from .assets import articulation

__all__ = ["PROFILE_DEFAULTS", "build_scene"]

PROFILE_DEFAULTS: Mapping[str, Any] = {
    # ``None`` means "leave whatever the robot asset declares". These four used to hold literal
    # copies of the G1's own numbers, described as matching main -- and one of them, self_collision,
    # did not: main spawned the G1 with it off. The disagreement was settled the wrong way round, by
    # editing the env config that the parity check measured everything against so it would agree
    # with this table. Restating asset values here buys nothing and lets the two drift apart
    # silently, so the asset is now the only place they are written.
    "self_collision": None,
    "solver_position_iterations": None,
    "solver_velocity_iterations": None,
    "max_depenetration_velocity": None,
    "gpu_max_rigid_patch_count": None,
    "gpu_max_rigid_contact_count": None,
    "gpu_collision_stack_size": None,
    "use_terrain_physics_material": False,
}
"""Engine settings a task may state through ``spec.sim.profiles['isaacsim']``.

Values default to the asset or simulator settings; a task overrides one by naming it.
"""

_ROBOT_PRIM = "{ENV_REGEX_NS}/Robot"


def _terrain(spec: TerrainSpec, profile: Mapping[str, Any]) -> Any:
    builder = TERRAIN_EXTENSIONS.terrain("isaacsim", spec.kind)
    if builder is None:
        available = sorted(TERRAIN_EXTENSIONS.terrain_kinds("isaacsim"))
        raise NotImplementedError(
            f"The Isaac Sim backend has no terrain {spec.kind!r}; "
            f"registered terrains are {available}."
        )
    return builder(spec, profile)


def _build_volume_points(sensor: VolumePointsRef, *, sensor_period: float) -> Any:
    from instinctlab_engine_isaacsim.sensors.volume_points.points_generator_cfg import (
        Grid3dPointsGeneratorCfg,
    )


def _build_native_sensor(
    sensor: Any,
    robot: Any,
    profile: Mapping[str, Any],
    *,
    sensor_period: float,
    num_envs: int,
) -> Any:
    builder = native_sensor_builder("isaacsim", sensor)
    native = builder(
        sensor,
        NativeSensorBuildContext(
            engine="isaacsim",
            robot=robot,
            sensor_period=sensor_period,
            profile=profile,
            num_envs=num_envs,
        ),
    )
    if native is None:
        raise RuntimeError(
            f"Native sensor builder for {sensor.kind!r} returned no config for {sensor.name!r}."
        )
    return native
    from instinctlab_engine_isaacsim.sensors.volume_points.volume_points_cfg import (
        VolumePointsCfg,
    )

    period = sensor.update_period if sensor.update_period is not None else sensor_period
    bodies = f"({'|'.join(sensor.bodies)})"
    return VolumePointsCfg(
        prim_path=f"{_ROBOT_PRIM}/{bodies}",
        points_generator=Grid3dPointsGeneratorCfg(
            x_min=sensor.grid.x_min,
            x_max=sensor.grid.x_max,
            x_num=sensor.grid.x_num,
            y_min=sensor.grid.y_min,
            y_max=sensor.grid.y_max,
            y_num=sensor.grid.y_num,
            z_min=sensor.grid.z_min,
            z_max=sensor.grid.z_max,
            z_num=sensor.grid.z_num,
        ),
        debug_vis=False,
        update_period=period,
        body_order=list(sensor.bodies),
        velocity=sensor.velocity,
    )


def _build_contact_sensor(sensor: ContactSensorRef) -> Any:
    """One Isaac Lab contact sensor.

    Isaac Lab has no counterpart to mjlab's per-sensor element list: one sensor covers a prim
    pattern and terms slice it by body index, which is why the declared elements become a single
    pattern here and a list of sensors there. ``history_length`` maps across directly, but the axis
    order of what comes back does not, and ``compat.sensors`` is what hides that.

    ``against`` is refused, not ignored. Isaac Lab *has* ``filter_prim_paths_expr``, but it
    fills ``force_matrix_w`` only: ``net_forces_w`` and the air-time timers stay unfiltered,
    and those are what portable terms read. The filter is also documented not to work when
    ``prim_path`` matches several bodies, which is how this builder always emits the sensor.
    Honouring the field with that API would compile a filtered-looking sensor that still
    reports every contact -- the same shape of silent failure as a missing ``found`` field.
    mjlab's ``secondary`` actually filters; a task that sets ``against`` therefore cannot
    compile for Isaac until there is an implementation that changes the tensors terms read.
    """
    sensor = sensor.for_engine("isaacsim")
    if sensor.against is not None:
        raise ValueError(
            f"Isaac Lab cannot honor ContactSensorRef.against={sensor.against!r} on "
            f"{sensor.name!r}. filter_prim_paths_expr does not change net_forces_w or "
            "air-time (portable terms read those), and it does not work on a multi-body "
            "prim_path, which is how this backend builds the sensor. Leave against unset "
            "or compile for an engine that can filter the contact signal itself."
        )
    from isaaclab.sensors import ContactSensorCfg

    elements = (
        sensor.elements
        if isinstance(sensor.elements, str)
        else "|".join(sensor.elements)
    )
    return ContactSensorCfg(
        prim_path=f"{_ROBOT_PRIM}/{elements}",
        history_length=sensor.history_length,
        track_air_time=sensor.track_air_time,
        # Passed explicitly after per-engine resolution: main uses the 1 N default for
        # Locomotion/Parkour and writes 10 N for Shadowing. Inheriting the SDK default is how
        # reference-specific clocks otherwise disappear from an apparently valid config.
        force_threshold=sensor.air_time_force_threshold,
    )


def _build_ray_caster(sensor: RayCasterRef, *, sensor_period: float) -> Any:
    """Isaac Lab's native ray caster, or the grouped camera that can see the robot.

    A grid is terrain-only (``mesh_prim_paths=['/World/ground']``). A pinhole uses
    ``GroupedRayCasterCamera`` so listed link visuals move with the robot -- stock
    ``RayCasterCamera`` only hits static meshes, and the robot would not see its
    own legs. The camera keeps unclipped depth so a miss remains non-finite, as
    required by the portable sensor contract. The observation term maps misses
    to the normalisation ceiling before blur.
    """
    refuse_unhonored_ray_alignment(sensor)
    if sensor.miss != "infinity":
        raise ValueError(
            f"Isaac ray caster {sensor.name!r} has miss={sensor.miss!r}; the portable contract is +inf."
        )
    if sensor.pattern.kind == "pinhole":
        return _build_pinhole_camera(sensor, sensor_period=sensor_period)
    if sensor.pattern.kind != "grid":
        raise ValueError(
            f"Isaac ray caster {sensor.name!r} has pattern.kind={sensor.pattern.kind!r}."
        )
    if sensor.hit != "terrain":
        raise ValueError(
            f"Isaac ray caster {sensor.name!r} has hit={sensor.hit!r}; only 'terrain' is implemented for a grid."
        )
    from isaaclab.sensors import RayCasterCfg, patterns

    period = sensor.update_period if sensor.update_period is not None else sensor_period
    return RayCasterCfg(
        prim_path=f"{_ROBOT_PRIM}/{sensor.attach}",
        offset=RayCasterCfg.OffsetCfg(pos=sensor.offset),
        ray_alignment=sensor.ray_alignment,
        pattern_cfg=patterns.GridPatternCfg(
            resolution=sensor.pattern.resolution,
            size=list(sensor.pattern.size),
            direction=sensor.direction,
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=sensor.max_distance,
        update_period=period,
    )


def _build_pinhole_camera(sensor: RayCasterRef, *, sensor_period: float) -> Any:
    """The parkour depth camera: world-convention pinhole, terrain plus listed links.

    ``GroupedRayCasterCamera`` always applies the attach body's full rotation
    (``attach_yaw_only=False``). ``ray_alignment`` is not read. A pinhole
    declared as anything other than ``base`` is refused by
    ``refuse_unhonored_ray_alignment`` before this runs.
    """
    from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg

    from instinctlab_engine_isaacsim.sensors.grouped_ray_caster.grouped_ray_caster_camera_cfg import (
        GroupedRayCasterCameraCfg,
    )
    from instinctlab_engine_isaacsim.sensors.grouped_ray_caster.grouped_ray_caster_cfg import (
        get_link_prim_targets,
    )

    meshes: list[Any] = []
    if sensor.hits_terrain():
        meshes.append("/World/ground")
    bodies = sensor.hit_bodies()
    if bodies:
        meshes.extend(get_link_prim_targets(list(bodies)))
    if not meshes:
        raise ValueError(f"Isaac camera {sensor.name!r} names nothing to hit.")
    period = sensor.update_period if sensor.update_period is not None else sensor_period
    return GroupedRayCasterCameraCfg(
        prim_path=f"{_ROBOT_PRIM}/{sensor.attach}",
        mesh_prim_paths=meshes,
        pattern_cfg=PinholeCameraPatternCfg(
            focal_length=sensor.pattern.focal_length,
            horizontal_aperture=sensor.pattern.horizontal_aperture,
            vertical_aperture=sensor.pattern.vertical_aperture,
            width=sensor.pattern.width,
            height=sensor.pattern.height,
        ),
        debug_vis=False,
        data_types=["distance_to_image_plane"],
        update_period=period,
        depth_clipping_behavior="none",
        offset=GroupedRayCasterCameraCfg.OffsetCfg(
            pos=sensor.offset,
            rot=sensor.offset_rot,
            convention=sensor.offset_convention,
        ),
        min_distance=sensor.min_distance,
        max_distance=sensor.max_distance,
    )


def _sky_light() -> Any:
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sim import DomeLightCfg
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    return AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


def _spawn_overrides(
    spawn: Any, spec: SceneSpec, profile: Mapping[str, Any]
) -> dict[str, Any]:
    """What this task changes about the robot's spawn, and nothing else.

    Anything the profile leaves at ``None`` is not mentioned, so the asset's value survives instead
    of being overwritten with a restatement of itself that can fall out of step with it.
    """
    overrides: dict[str, Any] = {"activate_contact_sensors": bool(spec.contact_sensors)}
    articulation = {}
    if profile["self_collision"] is not None:
        overrides["self_collision"] = profile["self_collision"]
        # URDF conversion and the spawned articulation each carry a self-collision switch.  The
        # popsicle asset declares the latter explicitly, so changing only the converter option
        # leaves the runtime articulation unchanged.
        articulation["enabled_self_collisions"] = profile["self_collision"]

    if profile["max_depenetration_velocity"] is not None:
        overrides["rigid_props"] = spawn.rigid_props.replace(
            max_depenetration_velocity=profile["max_depenetration_velocity"]
        )

    for field, key in (
        ("solver_position_iteration_count", "solver_position_iterations"),
        ("solver_velocity_iteration_count", "solver_velocity_iterations"),
    ):
        if profile[key] is not None:
            articulation[field] = profile[key]
    if articulation:
        overrides["articulation_props"] = spawn.articulation_props.replace(
            **articulation
        )
    return overrides


def build_scene(
    spec: SceneSpec,
    robot: Any,
    profile: Mapping[str, Any],
    *,
    num_envs: int,
    sensor_period: float,
) -> Any:
    """An ``InteractiveSceneCfg`` holding the robot, terrain, sensors and light.

    Built by assignment rather than by declaring a subclass, because the set of sensors is only
    known once the task is read. Isaac Lab's scene walks ``cfg.__dict__``, so assigned attributes
    are found exactly as declared ones are.
    """
    from isaaclab.scene import InteractiveSceneCfg

    articulation_cfg = articulation(robot)
    spawn = articulation_cfg.spawn.replace(
        **_spawn_overrides(articulation_cfg.spawn, spec, profile)
    )
    from .relations import with_collision_exclusions

    spawn = with_collision_exclusions(spawn, spec.collision_exclusions)

    scene = InteractiveSceneCfg(num_envs=num_envs, env_spacing=spec.env_spacing)
    scene.lazy_sensor_update = True
    scene.replicate_physics = True
    scene.filter_collisions = True
    scene.terrain = _terrain(spec.terrain, profile)
    scene.robot = articulation_cfg.replace(prim_path=_ROBOT_PRIM, spawn=spawn)
    from .rigid_objects import rigid_object_cfg

    for obj in spec.rigid_objects:
        setattr(
            scene,
            obj.name,
            rigid_object_cfg(obj, prim_path=f"{{ENV_REGEX_NS}}/{obj.name}"),
        )
    for sensor in spec.contact_sensors:
        cfg = _build_contact_sensor(sensor)
        # Every physics step, matching the contact durations the timing terms read. The default of
        # zero means "once per rendering step", which quietly undersamples air time.
        cfg.update_period = sensor_period
        setattr(scene, sensor.name, cfg)
    for sensor in spec.ray_casters:
        sensor = sensor.for_engine("isaacsim")
        setattr(
            scene, sensor.name, _build_ray_caster(sensor, sensor_period=sensor_period)
        )
    for sensor in spec.motion_references:
        setattr(scene, sensor.name, _build_motion_reference(sensor, robot))
    for sensor in spec.volume_points:
        setattr(
            scene,
            sensor.name,
            _build_volume_points(sensor, sensor_period=sensor_period),
        )
    for sensor in spec.native_sensors:
        setattr(
            scene,
            sensor.name,
            _build_native_sensor(
                sensor,
                robot,
                profile,
                sensor_period=sensor_period,
                num_envs=num_envs,
            ),
        )
    scene.sky_light = _sky_light()
    return scene


def _build_motion_reference(sensor: Any, robot: Any) -> Any:
    """Build the clip-backed reference sensor for Isaac's lifecycle."""
    from .motion_reference_sensor import build_motion_reference_sensor

    return build_motion_reference_sensor(sensor, robot)
