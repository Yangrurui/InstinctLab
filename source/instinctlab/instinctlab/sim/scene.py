"""Canonical scene specifications and runtime views."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from .robot_spec import RobotSpec
from .state import ArticulationState, ContactState


def resolve_names(patterns: str | Sequence[str], names: Sequence[str]) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Resolve regex patterns while preserving canonical ``names`` order."""
    expressions = (patterns,) if isinstance(patterns, str) else tuple(patterns)
    selected = tuple(index for index, name in enumerate(names) if any(re.fullmatch(expr, name) for expr in expressions))
    if not selected:
        raise ValueError(f"patterns {expressions!r} matched no canonical names")
    return selected, tuple(names[index] for index in selected)


@dataclass(frozen=True)
class SimulationSpec:
    sim_dt: float = 0.005
    decimation: int = 4
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    engine_options: Mapping[str, Any] = field(default_factory=dict)

    @property
    def policy_dt(self) -> float:
        return self.sim_dt * self.decimation

    def validate(self) -> None:
        if self.sim_dt <= 0.0:
            raise ValueError("sim_dt must be positive")
        if self.decimation <= 0:
            raise ValueError("decimation must be positive")


@dataclass(frozen=True)
class TerrainSpec:
    terrain_type: str = "plane"
    height: float = 0.0
    sliding_friction: float = 1.0
    restitution: float = 0.0


@dataclass(frozen=True)
class ContactSensorSpec:
    name: str
    entity_name: str
    body_names: tuple[str, ...]
    history_length: int = 3
    force_threshold: float = 1.0
    track_air_time: bool = True

    def validate(self, robot_spec: RobotSpec) -> None:
        unknown = set(self.body_names).difference(robot_spec.body_names)
        if unknown:
            raise ValueError(f"contact sensor {self.name!r} contains unknown bodies: {sorted(unknown)}")
        if self.history_length < 0:
            raise ValueError("contact history_length cannot be negative")
        if self.force_threshold < 0.0:
            raise ValueError("contact force_threshold cannot be negative")


@dataclass(frozen=True)
class SceneSpec:
    num_envs: int
    env_spacing: float
    robot: RobotSpec
    terrain: TerrainSpec = field(default_factory=TerrainSpec)
    contact_sensors: tuple[ContactSensorSpec, ...] = ()

    def validate(self) -> None:
        if self.num_envs <= 0:
            raise ValueError("SceneSpec num_envs must be positive")
        if self.env_spacing <= 0.0:
            raise ValueError("SceneSpec env_spacing must be positive")
        self.robot.validate()
        names = tuple(sensor.name for sensor in self.contact_sensors)
        if len(set(names)) != len(names):
            raise ValueError("contact sensor names must be unique")
        for sensor in self.contact_sensors:
            sensor.validate(self.robot)


@dataclass(frozen=True)
class SceneEntitySelector:
    name: str
    joint_names: str | tuple[str, ...] | None = None
    body_names: str | tuple[str, ...] | None = None

    def resolve(self, scene: "SceneView") -> "ResolvedSceneEntity":
        entity = scene.articulations[self.name]
        joint_ids: tuple[int, ...] | slice = slice(None)
        body_ids: tuple[int, ...] | slice = slice(None)
        if self.joint_names is not None:
            joint_ids, _ = entity.find_joints(self.joint_names)
        if self.body_names is not None:
            body_ids, _ = entity.find_bodies(self.body_names)
        return ResolvedSceneEntity(self.name, joint_ids, body_ids)


@dataclass(frozen=True)
class ResolvedSceneEntity:
    name: str
    joint_ids: tuple[int, ...] | slice
    body_ids: tuple[int, ...] | slice


@dataclass
class ArticulationView:
    """Canonical articulation names and mutable state."""

    name: str
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    data: ArticulationState

    def find_joints(self, patterns: str | Sequence[str]) -> tuple[tuple[int, ...], tuple[str, ...]]:
        return resolve_names(patterns, self.joint_names)

    def find_bodies(self, patterns: str | Sequence[str]) -> tuple[tuple[int, ...], tuple[str, ...]]:
        return resolve_names(patterns, self.body_names)


class SceneView(Mapping[str, ArticulationView | ContactState]):
    """Backend-independent runtime scene mapping."""

    def __init__(
        self,
        *,
        env_origins: torch.Tensor,
        articulations: Mapping[str, ArticulationView],
        sensors: Mapping[str, ContactState],
    ) -> None:
        self.env_origins = env_origins
        self.articulations = dict(articulations)
        self.sensors = dict(sensors)

    def __getitem__(self, key: str) -> ArticulationView | ContactState:
        if key in self.articulations:
            return self.articulations[key]
        return self.sensors[key]

    def __iter__(self) -> Iterator[str]:
        yield from self.articulations
        yield from self.sensors

    def __len__(self) -> int:
        return len(self.articulations) + len(self.sensors)

    @property
    def num_envs(self) -> int:
        return int(self.env_origins.shape[0])

    def reset(self, env_ids: torch.Tensor) -> None:
        for sensor in self.sensors.values():
            sensor.reset(env_ids)


__all__ = [
    "ArticulationView",
    "ContactSensorSpec",
    "ResolvedSceneEntity",
    "SceneEntitySelector",
    "SceneSpec",
    "SceneView",
    "SimulationSpec",
    "TerrainSpec",
    "resolve_names",
]
