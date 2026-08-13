"""Simulator backend contract and lazy backend registry."""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import torch

from .capabilities import Capability, CapabilitySet
from .control import JointControlTarget
from .scene import SceneSpec, SceneView, SimulationSpec


class SensorReadPhase(str, Enum):
    POST_PHYSICS = "post_physics"
    POST_RESET = "post_reset"
    POST_EVENT = "post_event"


@dataclass(frozen=True)
class RuntimeRequirements:
    capabilities: frozenset[Capability]
    randomization_fields: frozenset[str] = frozenset()


@dataclass
class MaterialProperties:
    entity_name: str
    body_ids: torch.Tensor
    env_ids: torch.Tensor
    sliding_friction: torch.Tensor
    restitution: torch.Tensor | None = None


@dataclass
class MassProperties:
    entity_name: str
    body_ids: torch.Tensor
    env_ids: torch.Tensor
    mass: torch.Tensor
    inertia: torch.Tensor
    center_of_mass: torch.Tensor


@dataclass(frozen=True)
class BackendMetadata:
    name: str
    version: str
    engine_version: str
    control_semantics: str
    contact_force_semantics: str
    physics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalIndexMap:
    """Map a frozen canonical name list onto one backend's native list."""

    canonical_names: tuple[str, ...]
    native_names: tuple[str, ...]
    native_ids_for_canonical: torch.Tensor

    @classmethod
    def build(
        cls,
        canonical_names: Iterable[str],
        native_names: Iterable[str],
        *,
        device: torch.device | str,
    ) -> "CanonicalIndexMap":
        canonical = tuple(canonical_names)
        native = tuple(native_names)
        if len(set(canonical)) != len(canonical):
            raise ValueError("canonical names must be unique")
        if len(set(native)) != len(native):
            raise ValueError("native names must be unique")
        missing = tuple(name for name in canonical if name not in native)
        if missing:
            raise ValueError(f"native asset is missing canonical names: {missing}")
        ids = torch.tensor([native.index(name) for name in canonical], dtype=torch.int64, device=device)
        return cls(canonical, native, ids)

    @property
    def is_identity(self) -> bool:
        if len(self.canonical_names) != len(self.native_names):
            return False
        expected = torch.arange(len(self.canonical_names), device=self.native_ids_for_canonical.device)
        return bool(torch.equal(self.native_ids_for_canonical, expected))

    def to_canonical(self, native_value: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
        return torch.index_select(native_value, dim, self.native_ids_for_canonical)

    def native_ids(self, canonical_ids: torch.Tensor | None = None) -> torch.Tensor:
        if canonical_ids is None:
            return self.native_ids_for_canonical
        return self.native_ids_for_canonical[canonical_ids]

    def copy_to_native(
        self,
        native_target: torch.Tensor,
        canonical_value: torch.Tensor,
        *,
        dim: int = -1,
    ) -> None:
        native_target.index_copy_(dim, self.native_ids_for_canonical, canonical_value)


@runtime_checkable
class SimulatorBackend(Protocol):
    capabilities: CapabilitySet
    metadata: BackendMetadata
    scene: SceneView
    num_envs: int
    device: torch.device
    sim_dt: float

    def initialize(
        self,
        scene_spec: SceneSpec,
        simulation_spec: SimulationSpec,
        requirements: RuntimeRequirements,
    ) -> None: ...

    def reset(self, env_ids: torch.Tensor) -> None: ...

    def write_root_state(self, entity_name: str, state_wxyz: torch.Tensor, env_ids: torch.Tensor) -> None: ...

    def write_joint_state(
        self,
        entity_name: str,
        position: torch.Tensor,
        velocity: torch.Tensor,
        env_ids: torch.Tensor,
        joint_ids: torch.Tensor | None = None,
    ) -> None: ...

    def set_joint_control_target(
        self,
        entity_name: str,
        target: JointControlTarget,
        env_ids: torch.Tensor | None = None,
    ) -> None: ...

    def set_external_wrench(
        self,
        entity_name: str,
        body_ids: torch.Tensor,
        force_w: torch.Tensor,
        torque_w: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None: ...

    def set_body_material(self, values: MaterialProperties) -> None: ...

    def set_body_mass_properties(self, values: MassProperties) -> None: ...

    def step(self) -> None: ...

    def synchronize(self, phase: SensorReadPhase) -> None: ...

    def render(self, mode: str) -> object | None: ...

    def close(self) -> None: ...


class BackendProvider(Protocol):
    """Bootstrap and construct a backend without importing another engine."""

    @staticmethod
    def add_cli_args(parser: Any) -> None: ...

    @staticmethod
    def bootstrap(args: Any) -> object | None: ...

    @staticmethod
    def create(*, device: str, bootstrap_context: object | None = None) -> SimulatorBackend: ...


class BackendRegistry:
    """Lazy provider registry used before engine-specific imports are legal."""

    def __init__(self) -> None:
        self._providers: dict[str, str] = {}

    def register(self, name: str, import_path: str) -> None:
        if not name or ":" not in import_path:
            raise ValueError("backend registration requires a name and 'module:attribute' import path")
        if name in self._providers and self._providers[name] != import_path:
            raise ValueError(f"backend {name!r} is already registered")
        self._providers[name] = import_path

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))

    def load(self, name: str) -> BackendProvider:
        try:
            import_path = self._providers[name]
        except KeyError as error:
            raise KeyError(f"unknown backend {name!r}; available: {', '.join(self.names())}") from error
        module_name, attribute = import_path.split(":", maxsplit=1)
        provider = getattr(importlib.import_module(module_name), attribute)
        return provider


BACKENDS = BackendRegistry()
BACKENDS.register("mock", "instinctlab.backends.mock:MockBackendProvider")
BACKENDS.register("isaacsim", "instinctlab.backends.isaacsim:IsaacSimBackendProvider")
BACKENDS.register("mjlab", "instinctlab.backends.mjlab:MjlabBackendProvider")


__all__ = [
    "BACKENDS",
    "BackendMetadata",
    "BackendProvider",
    "BackendRegistry",
    "CanonicalIndexMap",
    "MassProperties",
    "MaterialProperties",
    "RuntimeRequirements",
    "SensorReadPhase",
    "SimulatorBackend",
]
