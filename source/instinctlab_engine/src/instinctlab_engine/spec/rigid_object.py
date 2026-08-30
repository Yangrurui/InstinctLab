"""Engine-neutral rigid mesh objects used by HOI shadowing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import math
from pathlib import Path


@dataclass(frozen=True)
class RigidObjectRef:
    """A mesh object with explicit spawn, reset, collision, and material meaning.

    ``friction`` is one isotropic Coulomb coefficient. Isaac applies the same
    value as static and dynamic friction; MuJoCo applies it as sliding friction.
    Backend-only contact models do not belong in this shared reference.
    """

    name: str
    mesh: str
    scale: tuple[float, float, float]
    engine_meshes: Mapping[str, str] = field(default_factory=dict)
    mass: float = 1.0
    kinematic: bool = True
    collision_enabled: bool = True
    friction: float = 1.0
    initial_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_quaternion_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    initial_linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        object.__setattr__(self, "engine_meshes", dict(self.engine_meshes))
        if not self.name or not self.mesh:
            raise ValueError("Rigid object name and mesh must be non-empty.")
        if len(self.scale) != 3 or not all(
            math.isfinite(value) and value > 0.0 for value in self.scale
        ):
            raise ValueError(f"Rigid object {self.name!r} scale must contain three positive values.")
        if not math.isfinite(self.mass) or self.mass <= 0.0:
            raise ValueError(f"Rigid object {self.name!r} mass must be positive.")
        if not math.isfinite(self.friction) or self.friction < 0.0:
            raise ValueError(f"Rigid object {self.name!r} friction must be non-negative.")
        vectors = (
            ("initial_position", self.initial_position, 3),
            ("initial_quaternion_wxyz", self.initial_quaternion_wxyz, 4),
            ("initial_linear_velocity", self.initial_linear_velocity, 3),
            ("initial_angular_velocity", self.initial_angular_velocity, 3),
        )
        for field_name, values, size in vectors:
            if len(values) != size or not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"Rigid object {self.name!r} {field_name} must contain {size} finite values."
                )
        quaternion_norm = math.sqrt(sum(value * value for value in self.initial_quaternion_wxyz))
        if not math.isclose(quaternion_norm, 1.0, rel_tol=1.0e-5, abs_tol=1.0e-6):
            raise ValueError(
                f"Rigid object {self.name!r} initial quaternion must be normalized."
            )

    def for_engine(self, engine: str) -> RigidObjectRef:
        return replace(self, mesh=self.engine_meshes.get(engine, self.mesh), engine_meshes={})

    def resource_path(self, engine: str, *, require_exists: bool = True) -> Path:
        """Resolve this engine's local mesh and optionally require it to exist."""
        path = Path(self.engine_meshes.get(engine, self.mesh)).expanduser()
        if require_exists and not path.is_file():
            raise FileNotFoundError(
                f"Rigid object {self.name!r} resource for {engine!r} does not exist: {path}."
            )
        return path

    def resource_report(self, engine: str) -> dict[str, object]:
        path = self.resource_path(engine, require_exists=False)
        return {
            "name": self.name,
            "resource": str(path),
            "exists": path.is_file(),
            "kinematic": self.kinematic,
            "collision_enabled": self.collision_enabled,
            "friction": self.friction,
            "reset": {
                "position": list(self.initial_position),
                "quaternion_wxyz": list(self.initial_quaternion_wxyz),
                "linear_velocity": list(self.initial_linear_velocity),
                "angular_velocity": list(self.initial_angular_velocity),
            },
        }


__all__ = ["RigidObjectRef"]
