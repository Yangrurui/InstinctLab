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

from instinctlab.spec.sensor import ContactSensorRef
from instinctlab.spec.task import SceneSpec, TerrainGeneratorSpec, TerrainSpec

from .assets import articulation

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
"""Semantic tile kinds this adapter can lower onto Isaac Lab terrain configs."""

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
    "friction_dr": {
        "static_friction_range": (0.25, 0.8),
        "dynamic_friction_range": (0.2, 0.6),
        "restitution_range": (0.0, 0.8),
        "num_buckets": 64,
    },
}
"""Engine settings a task may state through ``spec.sim.profiles['isaacsim']``.

``friction_dr`` is main's randomisation for the flat locomotion task and has no asset to come from.
The rest default to the asset's own values; a task overrides one by naming it.
"""

_ROBOT_PRIM = "{ENV_REGEX_NS}/Robot"


def _physics_material(spec: TerrainSpec) -> Any:
    from isaaclab.sim import RigidBodyMaterialCfg

    return RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=spec.static_friction,
        dynamic_friction=spec.dynamic_friction,
        restitution=spec.restitution,
    )


def _visual_material() -> Any:
    from isaaclab.sim import MdlFileCfg
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

    return MdlFileCfg(
        mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        project_uvw=True,
        texture_scale=(0.25, 0.25),
    )


def _sub_terrain(kind: str, proportion: float, params: Mapping[str, Any]) -> Any:
    """One Isaac Lab tile config. Imports stay in the function so the module stays engine-free."""
    import isaaclab.terrains as terrain_gen

    classes = {
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg,
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg,
        "boxes": terrain_gen.MeshRandomGridTerrainCfg,
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg,
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg,
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg,
    }
    try:
        cls = classes[kind]
    except KeyError:
        raise NotImplementedError(
            f"The Isaac Sim adapter has no generator tile {kind!r}. It builds {sorted(classes)}."
        ) from None
    return cls(proportion=proportion, **params)


def _generator(spec: TerrainGeneratorSpec) -> Any:
    from isaaclab.terrains import TerrainGeneratorCfg

    return TerrainGeneratorCfg(
        seed=spec.seed,
        curriculum=spec.curriculum,
        size=spec.size,
        border_width=spec.border_width,
        num_rows=spec.num_rows,
        num_cols=spec.num_cols,
        horizontal_scale=spec.horizontal_scale,
        vertical_scale=spec.vertical_scale,
        slope_threshold=spec.slope_threshold,
        use_cache=False,
        sub_terrains={
            name: _sub_terrain(tile.kind, tile.proportion, tile.params) for name, tile in spec.sub_terrains.items()
        },
    )


def _terrain(spec: TerrainSpec, profile: Mapping[str, Any]) -> Any:
    from isaaclab.terrains import TerrainImporterCfg

    del profile
    if spec.kind == "plane":
        return TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=_physics_material(spec),
            debug_vis=False,
        )
    if spec.kind == "generator":
        if spec.generator is None:
            raise ValueError("kind='generator' needs a TerrainGeneratorSpec.")
        return TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=_generator(spec.generator),
            max_init_terrain_level=spec.generator.max_init_level,
            collision_group=-1,
            physics_material=_physics_material(spec),
            visual_material=_visual_material(),
            debug_vis=False,
        )
    if spec.kind == "rough":
        from .rough import rough_importer_cfg

        return rough_importer_cfg(spec)
    raise NotImplementedError(
        f"The Isaac Sim adapter builds 'plane', 'generator' and 'rough' terrain; the task asked for {spec.kind!r}."
    )


def _contact_sensor(sensor: ContactSensorRef) -> Any:
    """One Isaac Lab contact sensor.

    Isaac Lab has no counterpart to mjlab's per-sensor element list: one sensor covers a prim
    pattern and terms slice it by body index, which is why the declared elements become a single
    pattern here and a list of sensors there. ``history_length`` maps across directly, but the axis
    order of what comes back does not, and ``compat.sensors`` is what hides that.
    """
    from isaaclab.sensors import ContactSensorCfg

    elements = sensor.elements if isinstance(sensor.elements, str) else "|".join(sensor.elements)
    return ContactSensorCfg(
        prim_path=f"{_ROBOT_PRIM}/{elements}",
        history_length=sensor.history_length,
        track_air_time=sensor.track_air_time,
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


def _spawn_overrides(spawn: Any, spec: SceneSpec, profile: Mapping[str, Any]) -> dict[str, Any]:
    """What this task changes about the robot's spawn, and nothing else.

    Anything the profile leaves at ``None`` is not mentioned, so the asset's value survives instead
    of being overwritten with a restatement of itself that can fall out of step with it.
    """
    overrides: dict[str, Any] = {"activate_contact_sensors": bool(spec.contact_sensors)}
    if profile["self_collision"] is not None:
        overrides["self_collision"] = profile["self_collision"]

    if profile["max_depenetration_velocity"] is not None:
        overrides["rigid_props"] = spawn.rigid_props.replace(
            max_depenetration_velocity=profile["max_depenetration_velocity"]
        )

    solver = {
        field: profile[key]
        for field, key in (
            ("solver_position_iteration_count", "solver_position_iterations"),
            ("solver_velocity_iteration_count", "solver_velocity_iterations"),
        )
        if profile[key] is not None
    }
    if solver:
        overrides["articulation_props"] = spawn.articulation_props.replace(**solver)
    return overrides


def build_scene(spec: SceneSpec, robot: Any, profile: Mapping[str, Any], *, num_envs: int, sensor_period: float) -> Any:
    """An ``InteractiveSceneCfg`` holding the robot, terrain, sensors and light.

    Built by assignment rather than by declaring a subclass, because the set of sensors is only
    known once the task is read. Isaac Lab's scene walks ``cfg.__dict__``, so assigned attributes
    are found exactly as declared ones are.
    """
    from isaaclab.scene import InteractiveSceneCfg

    articulation_cfg = articulation(robot)
    spawn = articulation_cfg.spawn.replace(**_spawn_overrides(articulation_cfg.spawn, spec, profile))

    scene = InteractiveSceneCfg(num_envs=num_envs, env_spacing=spec.env_spacing)
    scene.lazy_sensor_update = True
    scene.replicate_physics = True
    scene.filter_collisions = True
    scene.terrain = _terrain(spec.terrain, profile)
    scene.robot = articulation_cfg.replace(prim_path=_ROBOT_PRIM, spawn=spawn)
    for sensor in spec.contact_sensors:
        cfg = _contact_sensor(sensor)
        # Every physics step, matching the contact durations the timing terms read. The default of
        # zero means "once per rendering step", which quietly undersamples air time.
        cfg.update_period = sensor_period
        setattr(scene, sensor.name, cfg)
    scene.sky_light = _sky_light()
    return scene
