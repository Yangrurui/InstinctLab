"""Simulator backend contract and lazy backend registry."""

from __future__ import annotations

import importlib
import torch
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from .capabilities import CapabilitySet
from .control import JointControlTarget
from .scene import SceneSpec, SceneView, SimulationSpec


class SensorReadPhase(str, Enum):
    """When canonical views are refreshed.

    ``POST_PHYSICS`` copies the buffers left by the last ``step()``. On MJLab
    those derived fields (``xpos`` / ``xquat`` / ``cvel``) lag ``qpos`` /
    ``qvel`` by one physics substep; reward and termination read that view.

    ``PRE_OBSERVATION`` is the InstinctMJ-equivalent refresh: MJLab runs
    ``sim.forward()`` and recopies so heading commands and policy / critic IMU
    terms match current ``qpos``. Isaac already has current poses after
    ``step()``, so it only recopies. Skip this call when ``POST_RESET`` already
    forwarded the batch in the same env step.

    After an interval write, ``POST_INTERVAL`` recopies without ``forward()``.
    InstinctMJ and Isaac Lab also leave derived quantities as the engine left
    them: MJLab ``cvel`` can still be pre-push, Isaac buffers already hold the
    written velocity. ``POST_EVENT`` still forwards, for write-then-read helpers
    that need kinematics to catch up.
    """

    POST_PHYSICS = "post_physics"
    POST_RESET = "post_reset"
    POST_EVENT = "post_event"
    POST_INTERVAL = "post_interval"
    PRE_OBSERVATION = "pre_observation"


JOINT_ACC_SOURCES = frozenset({"qacc_v1", "isaaclab_lazy_fd_v1", "fd_v1"})


@dataclass(frozen=True)
class RuntimeRequirements:
    capabilities: frozenset[str]
    optional_capabilities: frozenset[str] = frozenset()
    randomization_fields: frozenset[str] = frozenset()
    accepted_joint_acc_sources: frozenset[str] = frozenset()

    def validate_backend_metadata(self, metadata: BackendMetadata) -> None:
        if not self.accepted_joint_acc_sources:
            return
        if metadata.joint_acc_source not in self.accepted_joint_acc_sources:
            allowed = ", ".join(sorted(self.accepted_joint_acc_sources))
            raise RuntimeError(
                f"backend {metadata.name!r} joint_acc_source {metadata.joint_acc_source!r} "
                f"is not in accepted_joint_acc_sources: {allowed}"
            )


MATERIAL_LAYOUTS = frozenset({"body", "shape"})


@dataclass
class MaterialProperties:
    entity_name: str
    body_ids: torch.Tensor
    env_ids: torch.Tensor
    sliding_friction: torch.Tensor
    dynamic_friction: torch.Tensor | None = None
    restitution: torch.Tensor | None = None
    # "body": (N, n_bodies), broadcast onto every shape of that body.
    # "shape": (N, n_shapes), columns follow body_ids then native shapes on each body.
    layout: str = "body"


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
    joint_acc_source: str
    physics: Mapping[str, Any] = field(default_factory=dict)


def contiguous_index_range(
    ids: torch.Tensor,
    *,
    expected_count: int | None = None,
    require_positive_start: bool = False,
) -> tuple[int, int] | None:
    """Return ``(start, count)`` when ``ids`` is a contiguous arithmetic range.

    Body ids must use ``require_positive_start=True`` so world body 0 cannot be
    sliced in. Joint q/v addresses may start at 0 on fixed-base robots.
    """
    if ids.ndim != 1 or ids.numel() == 0:
        return None
    count = int(ids.numel())
    if expected_count is not None and count != expected_count:
        return None
    start = int(ids[0].item())
    if require_positive_start and start <= 0:
        return None
    expected = torch.arange(start, start + count, device=ids.device, dtype=ids.dtype)
    if not torch.equal(ids, expected):
        return None
    return start, count


@dataclass(frozen=True)
class CanonicalIndexMap:
    """Map a frozen canonical name list onto one backend's native list."""

    canonical_names: tuple[str, ...]
    native_names: tuple[str, ...]
    native_ids_for_canonical: torch.Tensor
    _is_identity: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_is_identity", self.canonical_names == self.native_names)

    @classmethod
    def build(
        cls,
        canonical_names: Iterable[str],
        native_names: Iterable[str],
        *,
        device: torch.device | str,
    ) -> CanonicalIndexMap:
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
        return self._is_identity

    def to_canonical(self, native_value: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
        if self._is_identity:
            return native_value
        return torch.index_select(native_value, dim, self.native_ids_for_canonical)

    def copy_to_canonical(
        self,
        native_value: torch.Tensor,
        out: torch.Tensor,
        *,
        dim: int = -1,
    ) -> None:
        """Gather native columns into a preallocated canonical buffer.

        Read direction is ``out[i] = native[index[i]]``. Do not use
        ``index_copy_`` here; that primitive scatters into native order.
        """
        if self._is_identity:
            out.copy_(native_value)
            return
        resolved_dim = dim if dim >= 0 else native_value.ndim + dim
        torch.index_select(native_value, resolved_dim, self.native_ids_for_canonical, out=out)

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

    def material_shape_counts(self, entity_name: str, body_ids: torch.Tensor) -> torch.Tensor: ...

    def set_body_mass_properties(self, values: MassProperties) -> None: ...

    def get_body_mass_properties(
        self,
        entity_name: str,
        env_ids: torch.Tensor,
        body_ids: torch.Tensor,
    ) -> MassProperties: ...

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
    "contiguous_index_range",
    "JOINT_ACC_SOURCES",
    "MATERIAL_LAYOUTS",
    "MassProperties",
    "MaterialProperties",
    "RuntimeRequirements",
    "SensorReadPhase",
    "SimulatorBackend",
]
