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

from instinctlab.spec.sensor import ContactSensorRef
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


def _terrain(spec: TerrainSpec) -> Any:
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

        return rough_importer_cfg(spec)
    raise NotImplementedError(
        f"The mjlab adapter builds 'plane', 'generator' and 'rough' terrain; the task asked for {spec.kind!r}."
    )


def _contact_sensor(sensor: ContactSensorRef) -> Any:
    """One mjlab contact sensor covering the declared elements.

    ``reduce="netforce"`` gives one force per element, which is the layout the portable accessors
    expect and the only one comparable to Isaac Lab's per-body net force. A plain
    :class:`ContactSensorCfg` rather than a force-thresholded one: the portable terms decide contact
    from the sensor's own contact duration, so imposing a newton threshold here would reintroduce
    the very quantity the two engines disagree about.

    ``"found"`` is requested and not merely inherited from mjlab's default, because everything that
    decides contact here depends on it and its absence does not announce itself. mjlab accumulates
    air and contact time from ``found`` and returns early when the field was not requested, leaving
    both timers at zero for the whole run -- so ``illegal_contact`` never fires and
    ``feet_air_time`` pays nothing. Training still proceeds, on an episode that can only time out.
    InstinctMJ can ask for force alone because its sensor subclass rederives contact from a force
    threshold; the portable terms deliberately do not.
    """
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    elements = (sensor.elements,) if isinstance(sensor.elements, str) else tuple(sensor.elements)
    return ContactSensorCfg(
        name=sensor.name,
        primary=ContactMatch(mode="body", pattern=elements, entity=sensor.entity),
        secondary=None if sensor.against is None else ContactMatch(mode="body", pattern=(sensor.against,)),
        fields=("found", "force"),
        reduce="netforce",
        track_air_time=sensor.track_air_time,
        history_length=sensor.history_length,
    )


def build_scene(spec: SceneSpec, robot: Any, profile: Mapping[str, Any], *, num_envs: int) -> Any:
    """A ``SceneCfg`` holding the robot, terrain and sensors."""
    from mjlab.scene import SceneCfg

    return SceneCfg(
        num_envs=num_envs,
        env_spacing=spec.env_spacing,
        terrain=_terrain(spec.terrain),
        entities={"robot": build_entity(robot)},
        sensors=tuple(_contact_sensor(sensor) for sensor in spec.contact_sensors),
    )
