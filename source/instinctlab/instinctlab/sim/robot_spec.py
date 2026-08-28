"""Canonical robot descriptions shared by all simulator backends."""

from __future__ import annotations

import math
import torch
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
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
        if not self.backend or not self.path:
            raise ValueError("BackendAsset backend and path must be non-empty")
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
    actuator_delay: tuple[int, int] = (0, 0)
    """Command lag ``(min, max)`` in **physics steps**, inclusive, drawn once per episode.

    Hub semantics — see ``compat/denylist.py`` ``actuator_delay``. ``(0, 0)`` is the
    robot-interface default (no delay). A task that wants Isaac's DelayedPD / mjlab's
    BuiltinPD 0–2 step lag writes ``(0, 2)`` here; each native asset applies the
    same public meaning with its engine's actuator implementation.
    """

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
        if not self.name or not self.schema_version or not self.asset_id or not self.root_body:
            raise ValueError("RobotSpec name, schema_version, asset_id, and root_body must be non-empty")
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
        for item in self.joint_properties:
            values = {
                "default_pos": item.default_pos,
                "stiffness": item.stiffness,
                "damping": item.damping,
                "armature": item.armature,
                "effort_limit": item.effort_limit,
                "velocity_limit": item.velocity_limit,
                "action_scale": item.action_scale,
            }
            non_finite = [name for name, value in values.items() if not math.isfinite(value)]
            if non_finite:
                raise ValueError(f"Joint {item.name!r} has non-finite properties: {non_finite}")
            if min(item.stiffness, item.damping, item.armature, item.action_scale) < 0.0:
                raise ValueError(f"Joint {item.name!r} has a negative PD, armature, or action-scale value")
            if item.effort_limit <= 0.0 or item.velocity_limit <= 0.0:
                raise ValueError(f"Joint {item.name!r} effort and velocity limits must be positive")
        asset_backends = tuple(asset.backend for asset in self.assets)
        if len(set(asset_backends)) != len(asset_backends):
            raise ValueError("RobotSpec may declare at most one asset per backend")
        for asset in self.assets:
            asset.validate_against(self.body_names)
        root_values = (*self.default_root_pos, *self.default_root_quat_wxyz)
        if not all(math.isfinite(value) for value in root_values):
            raise ValueError("RobotSpec default root pose must be finite")
        quaternion_norm = math.sqrt(sum(value * value for value in self.default_root_quat_wxyz))
        if not math.isclose(quaternion_norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"RobotSpec default root quaternion must be unit length, got norm={quaternion_norm}")
        if not 0.0 < self.soft_joint_pos_limit_factor <= 1.0:
            raise ValueError(
                f"RobotSpec soft_joint_pos_limit_factor must be in (0, 1], got {self.soft_joint_pos_limit_factor}"
            )
        lo, hi = self.actuator_delay
        if lo < 0 or hi < lo:
            raise ValueError(
                "RobotSpec.actuator_delay must be inclusive physics-step bounds with "
                f"0 <= min <= max, got {self.actuator_delay!r}"
            )
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

    def overridden(
        self,
        *,
        default_root_pos: tuple[float, float, float] | None = None,
        actuator_delay: tuple[int, int] | None = None,
        asset_paths: Mapping[str, str] | None = None,
        import_options: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> RobotSpec:
        """A copy with task-level plant changes. Does not mutate the base interface.

        The task holds this object; adapters already read ``spec.robot``. A second
        override bag on ``TaskSpec`` would be another place to forget to look, and
        changing the asset factory would move every task that shares the robot.

        ``asset_paths`` and ``import_options`` are keyed by backend name — data, not
        ``if engine ==``. Unknown keys fail here so a misspelled engine cannot
        silently keep the base asset.
        """
        updates: dict[str, Any] = {}
        if default_root_pos is not None:
            updates["default_root_pos"] = default_root_pos
        if actuator_delay is not None:
            updates["actuator_delay"] = actuator_delay
        known = {asset.backend for asset in self.assets}
        for source, keys in (("asset_paths", asset_paths), ("import_options", import_options)):
            if keys:
                unknown = set(keys) - known
                if unknown:
                    raise ValueError(
                        f"RobotSpec.overridden {source} keys {sorted(unknown)}, which "
                        f"this robot does not declare. Declared: {sorted(known)}."
                    )
        if asset_paths or import_options:
            paths = dict(asset_paths or {})
            option_patches = {backend: dict(patch) for backend, patch in (import_options or {}).items()}
            updates["assets"] = tuple(
                replace(
                    asset,
                    path=paths.get(asset.backend, asset.path),
                    import_options={**dict(asset.import_options), **option_patches.get(asset.backend, {})},
                )
                for asset in self.assets
            )
        result = replace(self, **updates)
        result.validate()
        return result


__all__ = [
    "BackendAsset",
    "JointProperties",
    "RobotSpec",
]
