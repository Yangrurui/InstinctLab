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

from instinctlab.engines.ray_alignment import refuse_unhonored_ray_alignment
from instinctlab.spec.sensor import ContactSensorRef, RayCasterRef, VolumePointsRef
from instinctlab.spec.task import SceneSpec, TerrainGeneratorSpec, TerrainSpec

from .assets import entity as build_entity

__all__ = ["GENERATOR_KINDS", "PROFILE_DEFAULTS", "build_scene"]

GENERATOR_KINDS: frozenset[str] = frozenset(
    {
        "pyramid_stairs",
        "pyramid_stairs_inv",
        "boxes",
        "random_rough",
        "hf_pyramid_slope",
        "hf_pyramid_slope_inv",
    }
)
"""Semantic tile kinds this adapter can lower onto mjlab terrain configs."""

PROFILE_DEFAULTS: Mapping[str, Any] = {
    "solver": "newton",
    "iterations": 10,
    "ls_iterations": 20,
    "ccd_iterations": 500,
    "njmax": 300,
    "friction_dr": {
        # One interval, because MuJoCo has one sliding friction coefficient where PhysX has a
        # static and a dynamic one. This is the union of main's two intervals -- min of the lows,
        # max of the highs -- which is what InstinctMJ's port does, and the closest thing to "the
        # same randomisation" that a single coefficient can be. Restitution has no per-geom
        # counterpart at all and is dropped rather than approximated.
        "ranges": (0.2, 0.8),
        "operation": "abs",
        "shared_random": True,
    },
}
"""Solver settings a task does not state, matching InstinctMJ's values for flat locomotion."""


def _sub_terrain(kind: str, proportion: float, params: Mapping[str, Any], generator: TerrainGeneratorSpec) -> Any:
    """One mjlab tile config. Imports stay in the function so the module stays engine-free."""
    import mjlab.terrains as terrain_gen

    fields = dict(params)
    if kind in {"random_rough", "hf_pyramid_slope", "hf_pyramid_slope_inv"}:
        fields.setdefault("horizontal_scale", generator.horizontal_scale)
        fields.setdefault("vertical_scale", generator.vertical_scale)
    if kind == "hf_pyramid_slope_inv":
        fields["inverted"] = True
        return terrain_gen.HfPyramidSlopedTerrainCfg(proportion=proportion, **fields)
    classes = {
        "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg,
        "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg,
        "boxes": terrain_gen.BoxRandomGridTerrainCfg,
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg,
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg,
    }
    try:
        cls = classes[kind]
    except KeyError:
        raise NotImplementedError(
            f"The mjlab adapter has no generator tile {kind!r}. It builds {sorted(GENERATOR_KINDS)}."
        ) from None
    return cls(proportion=proportion, **fields)


def _generator(spec: TerrainGeneratorSpec) -> Any:
    from mjlab.terrains import TerrainGeneratorCfg

    return TerrainGeneratorCfg(
        seed=spec.seed,
        curriculum=spec.curriculum,
        size=spec.size,
        border_width=spec.border_width,
        num_rows=spec.num_rows,
        num_cols=spec.num_cols,
        add_lights=True,
        sub_terrains={
            name: _sub_terrain(tile.kind, tile.proportion, tile.params, spec)
            for name, tile in spec.sub_terrains.items()
        },
    )


def _terrain(spec: TerrainSpec, profile: Mapping[str, Any]) -> Any:
    from mjlab.terrains import TerrainEntityCfg

    if spec.kind == "plane":
        return TerrainEntityCfg(terrain_type="plane")
    if spec.kind == "generator":
        if spec.generator is None:
            raise ValueError("kind='generator' needs a TerrainGeneratorSpec.")
        return TerrainEntityCfg(
            terrain_type="generator",
            terrain_generator=_generator(spec.generator),
            max_init_terrain_level=spec.generator.max_init_level,
        )
    if spec.kind == "rough":
        from .rough import rough_importer_cfg

        num_cols = int(profile.get("num_cols", 20))
        return _attach_virtual_obstacles(rough_importer_cfg(spec, num_cols=num_cols), spec)
    raise NotImplementedError(
        f"The mjlab adapter builds 'plane', 'generator' and 'rough' terrain; the task asked for {spec.kind!r}."
    )


def _attach_virtual_obstacles(cfg: Any, spec: TerrainSpec) -> Any:
    """Parkour's edge cylinders. Terrain constants stay in rough.py.

    The generated set will not match Isaac's — that divergence is recorded
    next to the stairs 6-vs-5 in :mod:`instinctlab.engines.mjlab.rough`.
    """
    if not spec.virtual_obstacles:
        return cfg
    from .terrains.virtual_obstacle.edge_cylinder_cfg import GreedyconcatEdgeCylinderCfg

    obstacles: dict[str, Any] = {}
    for obstacle in spec.virtual_obstacles:
        if obstacle.kind != "greedy_edge_cylinder":
            raise NotImplementedError(f"mjlab has no virtual obstacle {obstacle.kind!r} for {obstacle.name!r}.")
        obstacles[obstacle.name] = GreedyconcatEdgeCylinderCfg(
            cylinder_radius=obstacle.cylinder_radius,
            min_points=obstacle.min_points,
            angle_threshold=obstacle.angle_threshold,
            # InstinctMJ parkour: mesh source, hfield repair, collinear merge.
            component_workers=0,
            merge_collinear_gap=0.09,
            merge_collinear_angle_threshold=30.0,
            merge_collinear_line_distance=0.04,
        )
    cfg.virtual_obstacles = obstacles
    cfg.virtual_obstacle_source = "mesh"
    cfg.virtual_obstacle_hfield_height_threshold = 0.04
    return cfg


def _volume_points(sensor: VolumePointsRef) -> Any:
    from .volume_points import build_sensor

    return build_sensor(sensor)


def _contact_sensor(sensor: ContactSensorRef) -> Any:
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
    from mjlab.sensor import ContactMatch

    from .contact_sensor import thresholded_contact_sensor_cfg

    elements = (sensor.elements,) if isinstance(sensor.elements, str) else tuple(sensor.elements)
    return thresholded_contact_sensor_cfg(
        name=sensor.name,
        primary=ContactMatch(mode="body", pattern=elements, entity=sensor.entity),
        secondary=None if sensor.against is None else ContactMatch(mode="body", pattern=(sensor.against,)),
        fields=("found", "force"),
        reduce="netforce",
        track_air_time=sensor.track_air_time,
        history_length=sensor.history_length,
        force_threshold=sensor.air_time_force_threshold,
    )


def _ray_caster(sensor: RayCasterRef) -> Any:
    """mjlab does not ship Isaac's sky-origin scanner or world-convention camera."""
    refuse_unhonored_ray_alignment(sensor)
    if sensor.pattern.kind == "pinhole":
        from .camera import pinhole_ray_caster

        return pinhole_ray_caster(sensor)
    from .raycast import terrain_sky_ray_caster

    return terrain_sky_ray_caster(sensor)


def build_scene(spec: SceneSpec, robot: Any, profile: Mapping[str, Any], *, num_envs: int) -> Any:
    """A ``SceneCfg`` holding the robot, terrain and sensors."""
    from mjlab.scene import SceneCfg

    sensors = (
        tuple(_contact_sensor(sensor) for sensor in spec.contact_sensors)
        + tuple(_ray_caster(sensor) for sensor in spec.ray_casters)
        + tuple(_motion_reference(sensor, robot) for sensor in spec.motion_references)
        + tuple(_volume_points(sensor) for sensor in spec.volume_points)
    )
    return SceneCfg(
        num_envs=num_envs,
        env_spacing=spec.env_spacing,
        terrain=_terrain(spec.terrain, profile),
        entities={"robot": build_entity(robot)},
        sensors=sensors,
    )


def _motion_reference(sensor: Any, robot: Any) -> Any:
    """Clip-backed reference. Separate from the ray builders another increment owns."""
    from .motion_reference import build_sensor

    return build_sensor(sensor, robot)
