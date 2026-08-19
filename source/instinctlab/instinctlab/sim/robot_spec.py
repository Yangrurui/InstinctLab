"""Canonical robot descriptions shared by all simulator backends."""

from __future__ import annotations

import torch
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

_ALLOWED_LOAD_MODES = frozenset({"default", "strip_visual_meshes"})


@dataclass(frozen=True)
class BackendAsset:
    """Per-backend asset binding for one robot.

    ``import_options`` holds topology-changing loader flags such as URDF
    ``merge_fixed_joints`` or ``prim_path``. Runtime solver and scene
    profiles belong on the task ``SceneSpec`` / ``SimulationSpec``.
    """

    backend: str
    path: str
    contact_body_aliases: Mapping[str, str] = field(default_factory=dict)
    load_mode: str = "default"
    import_options: Mapping[str, Any] = field(default_factory=dict)

    def resolve_contact_body_names(self, body_names: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(self.contact_body_aliases.get(name, name) for name in body_names)

    def validate_against(self, body_names: tuple[str, ...]) -> None:
        if self.load_mode not in _ALLOWED_LOAD_MODES:
            raise ValueError(f"BackendAsset {self.backend!r} has unsupported load_mode {self.load_mode!r}")
        unknown = set(self.contact_body_aliases).difference(body_names)
        if unknown:
            raise ValueError(f"BackendAsset {self.backend!r} aliases unknown canonical bodies: {sorted(unknown)}")
        empty = [name for name, native in self.contact_body_aliases.items() if not native]
        if empty:
            raise ValueError(f"BackendAsset {self.backend!r} has empty native aliases for: {empty}")
        native_names = tuple(self.contact_body_aliases.values())
        if len(set(native_names)) != len(native_names):
            raise ValueError(f"BackendAsset {self.backend!r} maps multiple canonical bodies to the same native name")


@dataclass(frozen=True)
class JointProperties:
    name: str
    default_pos: float
    stiffness: float
    damping: float
    armature: float
    effort_limit: float
    velocity_limit: float
    action_scale: float


@dataclass(frozen=True)
class RobotSpec:
    """Engine-independent physical and indexing contract for one robot."""

    name: str
    schema_version: str
    asset_id: str
    root_body: str
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    joint_properties: tuple[JointProperties, ...]
    assets: tuple[BackendAsset, ...]
    default_root_pos: tuple[float, float, float]
    default_root_quat_wxyz: tuple[float, float, float, float]
    soft_joint_pos_limit_factor: float
    frame_names: tuple[str, ...] = field(default_factory=tuple)
    collision_body_names: tuple[str, ...] = field(default_factory=tuple)

    @property
    def physical_body_names(self) -> tuple[str, ...]:
        """Canonical bodies that are not declared sensor/visual frames."""
        frames = frozenset(self.frame_names)
        return tuple(name for name in self.body_names if name not in frames)

    @property
    def material_body_names(self) -> tuple[str, ...]:
        """Bodies that own collision geometry and may receive material writes."""
        return self.collision_body_names or self.physical_body_names

    def validate(self) -> None:
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("RobotSpec joint_names must be non-empty and unique")
        if not self.body_names or len(set(self.body_names)) != len(self.body_names):
            raise ValueError("RobotSpec body_names must be non-empty and unique")
        if self.root_body != self.body_names[0]:
            raise ValueError("RobotSpec root_body must be the first canonical body")
        if len(set(self.frame_names)) != len(self.frame_names):
            raise ValueError("RobotSpec frame_names must be unique")
        unknown_frames = set(self.frame_names).difference(self.body_names)
        if unknown_frames:
            raise ValueError(f"RobotSpec frame_names are not in body_names: {sorted(unknown_frames)}")
        if self.root_body in self.frame_names:
            raise ValueError("RobotSpec root_body cannot be a frame")
        if len(set(self.collision_body_names)) != len(self.collision_body_names):
            raise ValueError("RobotSpec collision_body_names must be unique")
        unknown_collision = set(self.collision_body_names).difference(self.physical_body_names)
        if unknown_collision:
            raise ValueError(
                f"RobotSpec collision_body_names must be physical bodies, not frames: {sorted(unknown_collision)}"
            )
        property_names = tuple(item.name for item in self.joint_properties)
        if property_names != self.joint_names:
            raise ValueError("joint_properties must exactly follow canonical joint_names")
        asset_backends = tuple(asset.backend for asset in self.assets)
        if len(set(asset_backends)) != len(asset_backends):
            raise ValueError("RobotSpec may declare at most one asset per backend")
        for asset in self.assets:
            asset.validate_against(self.body_names)

    def asset_for(self, backend: str) -> BackendAsset:
        for asset in self.assets:
            if asset.backend == backend:
                return asset
        raise KeyError(f"RobotSpec {self.name!r} has no asset for backend {backend!r}")

    def joint_index(self, name: str) -> int:
        return self.joint_names.index(name)

    def materialize(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, torch.Tensor]:
        fields = (
            "default_pos",
            "stiffness",
            "damping",
            "armature",
            "effort_limit",
            "velocity_limit",
            "action_scale",
        )
        return {
            field: torch.tensor(
                [getattr(properties, field) for properties in self.joint_properties],
                device=device,
                dtype=dtype,
            )
            for field in fields
        }


__all__ = [
    "BackendAsset",
    "JointProperties",
    "RobotSpec",
]
