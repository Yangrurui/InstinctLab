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
from instinctlab.spec.task import SceneSpec, TerrainSpec

from .assets import articulation

__all__ = ["PROFILE_DEFAULTS", "build_scene"]

PROFILE_DEFAULTS: Mapping[str, Any] = {
    "self_collision": True,
    "solver_position_iterations": 8,
    "solver_velocity_iterations": 4,
    "max_depenetration_velocity": 1.0,
    "friction_dr": {
        "static_friction_range": (0.25, 0.8),
        "dynamic_friction_range": (0.2, 0.6),
        "restitution_range": (0.0, 0.8),
        "num_buckets": 64,
    },
}
"""Solver settings a task does not state, matching main's values for the flat locomotion task."""

_ROBOT_PRIM = "{ENV_REGEX_NS}/Robot"


def _terrain(spec: TerrainSpec, profile: Mapping[str, Any]) -> Any:
    from isaaclab.sim import RigidBodyMaterialCfg
    from isaaclab.terrains import TerrainImporterCfg

    if spec.kind != "plane":
        raise NotImplementedError(
            f"The Isaac Sim adapter builds 'plane' terrain; the task asked for {spec.kind!r}. "
            "Generated terrain is P6 work."
        )
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=spec.static_friction,
            dynamic_friction=spec.dynamic_friction,
        ),
        debug_vis=False,
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


def build_scene(spec: SceneSpec, robot: Any, profile: Mapping[str, Any], *, num_envs: int, sensor_period: float) -> Any:
    """An ``InteractiveSceneCfg`` holding the robot, terrain, sensors and light.

    Built by assignment rather than by declaring a subclass, because the set of sensors is only
    known once the task is read. Isaac Lab's scene walks ``cfg.__dict__``, so assigned attributes
    are found exactly as declared ones are.
    """
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import ArticulationRootPropertiesCfg, RigidBodyPropertiesCfg

    articulation_cfg = articulation(robot)
    spawn = articulation_cfg.spawn.replace(
        self_collision=profile["self_collision"],
        activate_contact_sensors=bool(spec.contact_sensors),
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=profile["max_depenetration_velocity"],
        ),
        articulation_props=ArticulationRootPropertiesCfg(
            enabled_self_collisions=profile["self_collision"],
            solver_position_iteration_count=profile["solver_position_iterations"],
            solver_velocity_iteration_count=profile["solver_velocity_iterations"],
        ),
    )

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
