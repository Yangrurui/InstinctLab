"""A whole task, declared once and compiled per engine.

:class:`TaskSpec` is the intermediate representation the whole design turns on. A frontend reads
some project's native task definition and produces one of these; a backend consumes it and emits
that engine's native environment config. With ``N`` frontends and ``M`` backends that is ``N + M``
pieces of work rather than the ``N × M`` a direct engine-to-engine converter would need, and the
saving is the reason this layer exists at all rather than being a translation helper.

Two rules keep it honest, both enforced by tests:

**No engine imports, anywhere below this package.** A declaration that reaches for an engine is no
longer a declaration.

**No branching on engine name.** Differences between engines live in data keyed by engine --
``SimSpec.profiles``, ``TermSpec.engine_params``, :attr:`TaskSpec.engine_extras` -- never in an
``if engine == ...``. The distinction matters for the third engine: data keys are additive, and a
conditional is an edit to every task file that ever used one.

"""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from instinctlab_engine.spec.robot import RobotSpec

from .mdp import MdpSpec
from .rigid_object import RigidObjectRef
from .relations import CollisionExclusionRef
from .sensor import (
    ContactSensorRef,
    MotionReferenceRef,
    NativeSensorRef,
    RayCasterRef,
    VirtualObstacleRef,
    VolumePointsRef,
)

__all__ = [
    "AgentSpec",
    "SceneSpec",
    "SimSpec",
    "SubTerrainSpec",
    "TaskSpec",
    "TerrainGeneratorSpec",
    "TerrainSpec",
]


@dataclass(frozen=True)
class SimSpec:
    """Timing, and the per-engine solver settings that timing cannot capture.

    Args:
        physics_dt: Physics step, in seconds.
        decimation: Physics steps per environment step.
        episode_length_s: Episode length, in seconds.
        is_finite_horizon: Whether the time limit is part of the MDP rather than a truncation.
        scale_rewards_by_dt: Whether reward weights are multiplied by the step time, so that a
            weight means the same thing at different control rates.
        profiles: Per-engine **overrides** of solver settings, keyed by engine name.

    The word "overrides" in ``profiles`` is load-bearing. Defaults come from the adapter, and each
    adapter's defaults are the values that engine's own reference implementation uses -- Isaac Lab's
    PhysX solver counts, mjlab's MuJoCo iterations. A task that sets nothing therefore gets, on each
    engine, what a native task on that engine would have got. Filling these in with a single set of
    "unified" numbers would make the two engines equally wrong instead of each natively right.
    """

    physics_dt: float
    decimation: int
    episode_length_s: float
    is_finite_horizon: bool = False
    scale_rewards_by_dt: bool = True
    profiles: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.physics_dt <= 0.0:
            raise ValueError(f"physics_dt must be positive, got {self.physics_dt}.")
        if self.decimation <= 0:
            raise ValueError(f"decimation must be positive, got {self.decimation}.")
        if self.episode_length_s <= 0.0:
            raise ValueError(f"episode_length_s must be positive, got {self.episode_length_s}.")
        object.__setattr__(self, "profiles", {k: dict(v) for k, v in self.profiles.items()})

    @property
    def step_dt(self) -> float:
        """Seconds per environment step. Both engines expose this as ``env.step_dt``."""
        return self.physics_dt * self.decimation

    def profile_for(self, engine: str) -> dict[str, Any]:
        """This engine's overrides, empty when the task states none."""
        return dict(self.profiles.get(engine, {}))


@dataclass(frozen=True)
class SubTerrainSpec:
    """One tile type in a generated grid, named by intent rather than by a native class.

    ``kind`` is a semantic name -- ``pyramid_stairs``, ``random_rough`` -- that each adapter maps
    onto its own primitive. Isaac Lab builds stairs as a triangle mesh; mjlab builds them from
    boxes. The numbers in ``params`` are the ones both mappings understand (step height, slope,
    noise amplitude). A field only one engine has does not belong here.
    """

    kind: str
    proportion: float = 1.0
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("SubTerrainSpec.kind must name a terrain type.")
        if self.proportion <= 0.0:
            raise ValueError(f"SubTerrainSpec.proportion must be positive, got {self.proportion}.")
        object.__setattr__(self, "params", dict(self.params))


@dataclass(frozen=True)
class TerrainGeneratorSpec:
    """A grid of sub-terrains. The layout is portable; how each tile is built is not.

    ``num_rows`` is difficulty under curriculum on both engines. ``num_cols`` is the grid
    width both adapters honor: columns are assigned by Isaac Lab's cumulative-proportion
    formula. Difficulty along a row is still not the same number
    — Isaac jitters ``(row + U) / num_rows`` and never hits 1.0; mjlab uses
    ``row / (num_rows - 1)`` and hits both endpoints. Once a type occupies several columns,
    every mjlab duplicate at a given row therefore has identical difficulty.
    """

    size: tuple[float, float] = (8.0, 8.0)
    border_width: float = 20.0
    num_rows: int = 10
    num_cols: int = 20
    curriculum: bool = False
    max_init_level: int | None = 5
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float = 0.75
    seed: int | None = None
    sub_terrains: Mapping[str, SubTerrainSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sub_terrains", dict(self.sub_terrains))
        if self.size[0] <= 0.0 or self.size[1] <= 0.0:
            raise ValueError(f"Terrain patch size must be positive, got {self.size}.")
        if self.num_rows < 1 or self.num_cols < 1:
            raise ValueError(f"Terrain grid must be at least 1x1, got {self.num_rows}x{self.num_cols}.")
        if self.border_width < 0.0:
            raise ValueError(f"Terrain border width must be non-negative, got {self.border_width}.")
        if self.horizontal_scale <= 0.0 or self.vertical_scale <= 0.0:
            raise ValueError(
                "Terrain horizontal_scale and vertical_scale must be positive, "
                f"got {self.horizontal_scale} and {self.vertical_scale}."
            )
        if self.slope_threshold <= 0.0:
            raise ValueError(f"Terrain slope_threshold must be positive, got {self.slope_threshold}.")
        if not self.sub_terrains:
            raise ValueError("A generator with no sub-terrains produces an empty world.")


@dataclass(frozen=True)
class TerrainSpec:
    """The ground. Flat unless a task says otherwise.

    ``kind`` is open. ``"generator"`` and ``"rough"`` both carry the engine-neutral
    :attr:`generator` recipe. ``"rough"`` additionally selects the rough importer and its
    production contact-capacity profile. An unknown value fails in the adapter rather than
    here, so a third engine can add a kind without editing this module.
    """

    kind: str = "plane"
    static_friction: float = 1.0
    dynamic_friction: float = 1.0
    restitution: float = 0.0
    generator: TerrainGeneratorSpec | None = None
    virtual_obstacles: tuple[VirtualObstacleRef, ...] = ()
    """Edge cylinders generated from the terrain at import. Empty for locomotion."""
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(self, "virtual_obstacles", tuple(self.virtual_obstacles))
        material = (self.static_friction, self.dynamic_friction, self.restitution)
        if not all(math.isfinite(value) for value in material):
            raise ValueError("Terrain material coefficients must be finite")
        if self.static_friction < 0.0 or self.dynamic_friction < 0.0:
            raise ValueError("Terrain friction coefficients must be non-negative")
        if not 0.0 <= self.restitution <= 1.0:
            raise ValueError("Terrain restitution must be in [0, 1]")
        if self.kind == "plane" and self.generator is not None:
            raise ValueError(f"kind={self.kind!r} cannot carry a generator.")
        if self.kind in {"generator", "rough"} and self.generator is None:
            raise ValueError(f"kind={self.kind!r} needs a TerrainGeneratorSpec.")
        names = [obstacle.name for obstacle in self.virtual_obstacles]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Virtual obstacle names must be unique; repeated: {duplicates}.")


@dataclass(frozen=True)
class SceneSpec:
    """What is in the world besides the robot.

    ``num_envs`` is absent on purpose: how many copies to run is an argument to the compilation,
    not a property of the task, and putting it here would mean a task file that says how big a GPU
    you have.
    """

    terrain: TerrainSpec = field(default_factory=TerrainSpec)
    contact_sensors: tuple[ContactSensorRef, ...] = ()
    ray_casters: tuple[RayCasterRef, ...] = ()
    """Grid scanners and pinhole cameras. Distinguished by ``pattern.kind``, not by a second list."""
    motion_references: tuple[MotionReferenceRef, ...] = ()
    """Clip-backed motion references. A new sensor family, not a ray or a contact."""
    volume_points: tuple[VolumePointsRef, ...] = ()
    native_sensors: tuple[NativeSensorRef, ...] = field(
        default=(), metadata={"contract_omit_if_default": True}
    )
    collision_exclusions: tuple[CollisionExclusionRef, ...] = field(
        default=(), metadata={"contract_omit_if_default": True}
    )
    rigid_objects: tuple[RigidObjectRef, ...] = field(default=(), metadata={"contract_omit_if_default": True})
    """Ankle (or other) volume-point clouds. Penetration is not a raycast."""
    env_spacing: float = 2.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "contact_sensors", tuple(self.contact_sensors))
        object.__setattr__(self, "ray_casters", tuple(self.ray_casters))
        object.__setattr__(self, "motion_references", tuple(self.motion_references))
        object.__setattr__(self, "volume_points", tuple(self.volume_points))
        object.__setattr__(self, "native_sensors", tuple(self.native_sensors))
        object.__setattr__(self, "collision_exclusions", tuple(self.collision_exclusions))
        object.__setattr__(self, "rigid_objects", tuple(self.rigid_objects))
        names = (
            [sensor.name for sensor in self.contact_sensors]
            + [sensor.name for sensor in self.ray_casters]
            + [sensor.name for sensor in self.motion_references]
            + [sensor.name for sensor in self.volume_points]
            + [sensor.name for sensor in self.native_sensors]
        )
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Sensor names must be unique; repeated: {duplicates}.")
        object_names = [obj.name for obj in self.rigid_objects]
        if len(object_names) != len(set(object_names)):
            raise ValueError("Rigid object names must be unique.")
        if self.env_spacing <= 0.0:
            raise ValueError(f"env_spacing must be positive, got {self.env_spacing}.")
        exclusion_keys = [
            (exclusion.entity, exclusion.pair)
            for exclusion in self.collision_exclusions
        ]
        if len(exclusion_keys) != len(set(exclusion_keys)):
            raise ValueError("Collision exclusions must be unique unordered body pairs.")

    def sensor(self, name: str) -> ContactSensorRef:
        """The declared contact sensor called ``name``."""
        for sensor in self.contact_sensors:
            if sensor.name == name:
                return sensor
        have = ", ".join(sorted(s.name for s in self.contact_sensors)) or "none"
        raise KeyError(f"Scene declares no contact sensor {name!r}. Declared: {have}.")

    def ray_caster(self, name: str) -> RayCasterRef:
        """The declared ray caster called ``name``."""
        for sensor in self.ray_casters:
            if sensor.name == name:
                return sensor
        have = ", ".join(sorted(s.name for s in self.ray_casters)) or "none"
        raise KeyError(f"Scene declares no ray caster {name!r}. Declared: {have}.")

    def motion_reference(self, name: str) -> MotionReferenceRef:
        """The declared motion reference called ``name``."""
        for sensor in self.motion_references:
            if sensor.name == name:
                return sensor
        have = ", ".join(sorted(s.name for s in self.motion_references)) or "none"
        raise KeyError(f"Scene declares no motion reference {name!r}. Declared: {have}.")

    def volume_point(self, name: str) -> VolumePointsRef:
        """The declared volume-points sensor called ``name``."""
        for sensor in self.volume_points:
            if sensor.name == name:
                return sensor
        have = ", ".join(sorted(s.name for s in self.volume_points)) or "none"
        raise KeyError(f"Scene declares no volume-points sensor {name!r}. Declared: {have}.")


@dataclass(frozen=True)
class AgentSpec:
    """The learning agent, named rather than imported.

    The runner config is referenced by dotted path and imported only when a compilation asks for
    it. That indirection is not stylistic: the project's ``InstinctRlOnPolicyRunnerCfg`` is built
    on ``isaaclab.utils.configclass``, so importing it from this module would drag Isaac Lab into
    every task declaration and break the isolation the whole layer depends on. Naming it defers the
    import to the backend, which has already loaded an engine by then.

    Per decision D4 the project standardises on ``instinct_rl``; configs written for other
    frameworks are translated at the frontend rather than carried through to here.

    Args:
        runner: Dotted path to the runner config class, ``"pkg.module:ClassName"`` or
            ``"pkg.module.ClassName"``.
        overrides: Attributes set on the constructed config, applied on every engine.
        engine_overrides: Further attributes applied per engine. Present for the cases where the
            same policy needs a different rollout length to keep wall-clock comparable, and no
            more; hyperparameters that differ per engine make the two runs incomparable, which is
            the opposite of what this project is for.
    """

    runner: str
    overrides: Mapping[str, Any] = field(default_factory=dict)
    engine_overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.runner:
            raise ValueError("AgentSpec.runner must name a runner config class.")
        object.__setattr__(self, "overrides", dict(self.overrides))
        object.__setattr__(self, "engine_overrides", {k: dict(v) for k, v in self.engine_overrides.items()})

    def resolve(self) -> type:
        """Import and return the runner config class. Called by the backend, never at import time."""
        path = self.runner.replace(":", ".")
        module_name, _, class_name = path.rpartition(".")
        if not module_name:
            raise ValueError(f"AgentSpec.runner must be a dotted path, got {self.runner!r}.")
        return getattr(importlib.import_module(module_name), class_name)

    def resolved_overrides(self, engine: str) -> dict[str, Any]:
        """``overrides`` with this engine's additions applied."""
        merged = dict(self.overrides)
        merged.update(self.engine_overrides.get(engine, {}))
        return merged


@dataclass(frozen=True)
class TaskSpec:
    """One task, in the form every frontend produces and every backend consumes.

    Args:
        task_id: Registration id, unique across the project.
        robot: The robot's engine-independent contract from ``spec/robot.py``.
        scene: Terrain and sensors.
        sim: Timing and per-engine solver profiles.
        mdp: Every term of the MDP.
        agent: The learning agent.
        engines: Engines this task claims to run on. Compiling for an engine outside this tuple
            fails rather than being attempted, because the claim is what a reviewer reads.
        engine_extras: The escape hatch, keyed by engine -- Isaac Lab tiled cameras, USD authoring,
            anything with no counterpart elsewhere. A task that uses it is not portable, which is a
            legitimate thing to be; what is not legitimate is being unaware of it, so every use is
            recorded in the compilation's resolution and lands in the checkpoint manifest.
    """

    task_id: str
    robot: RobotSpec
    scene: SceneSpec
    sim: SimSpec
    mdp: MdpSpec
    agent: AgentSpec
    engines: tuple[str, ...]
    engine_extras: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "engines", tuple(self.engines))
        object.__setattr__(self, "engine_extras", {k: dict(v) for k, v in self.engine_extras.items()})
        if not self.engines:
            raise ValueError(f"Task {self.task_id!r} names no engines.")
        duplicates = sorted({e for e in self.engines if self.engines.count(e) > 1})
        if duplicates:
            raise ValueError(f"Task {self.task_id!r} repeats engines: {duplicates}.")

    def validate(self) -> None:
        """Check what can be checked without an engine present.

        Called by every backend before it compiles, and standalone in CI so that a task with a
        misspelled engine key fails in a unit test rather than after a simulator has booted.
        """
        from .validation import validate_task

        validate_task(self)

    def extras_for(self, engine: str) -> dict[str, Any]:
        """This engine's escape-hatch settings, empty when the task uses none."""
        return dict(self.engine_extras.get(engine, {}))

    def validate_for_engine(self, engine: str) -> None:
        """Validate the declaration and require it to opt in to ``engine``."""
        self.validate()
        if engine not in self.engines:
            raise ValueError(
                f"Task {self.task_id!r} does not declare engine {engine!r}; "
                f"declared engines are {sorted(self.engines)}."
            )
        robot_engines = {asset.backend for asset in self.robot.assets}
        if robot_engines != {engine}:
            raise ValueError(
                f"Task {self.task_id!r} was materialized for {sorted(robot_engines)}, not {engine!r}."
            )

    @property
    def is_portable(self) -> bool:
        """Legacy shorthand for multi-engine declaration without native extras.

        Use :func:`instinctlab_engine.spec.portability_report` when overlays or
        compilation cleanliness matter; those are independent dimensions.
        """
        return len(self.engines) > 1 and not self.engine_extras
