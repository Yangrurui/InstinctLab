"""Engine-neutral kinematic mesh objects used by HOI shadowing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class RigidObjectRef:
    name: str
    mesh: str
    scale: tuple[float, float, float]
    engine_meshes: Mapping[str, str] = field(default_factory=dict)
    mass: float = 1.0
    kinematic: bool = True

    def __post_init__(self):
        object.__setattr__(self, "engine_meshes", dict(self.engine_meshes))
        if not self.name or not self.mesh:
            raise ValueError("Rigid object name and mesh must be non-empty.")

    def for_engine(self, engine: str) -> RigidObjectRef:
        return replace(self, mesh=self.engine_meshes.get(engine, self.mesh), engine_meshes={})


__all__ = ["RigidObjectRef"]
