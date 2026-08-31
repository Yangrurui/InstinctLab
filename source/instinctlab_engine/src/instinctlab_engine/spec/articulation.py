"""Additional articulated scene entities with canonical joint schemas."""

from __future__ import annotations

from dataclasses import dataclass

from .robot import RobotSpec


@dataclass(frozen=True)
class ArticulationRef:
    """A named non-primary articulation materialized from a :class:`RobotSpec`.

    ``RobotSpec.joint_names`` remains the canonical DFS axis. Reusing that
    schema makes policy and observation selectors independent of each native
    simulator's articulation order.
    """

    name: str
    schema: RobotSpec

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ValueError(
                f"Additional articulation names must be Python identifiers, got {self.name!r}."
            )
        if self.name in {"robot", "terrain"}:
            raise ValueError(
                f"Additional articulation name {self.name!r} is reserved by the scene."
            )
        self.schema.validate()


__all__ = ["ArticulationRef"]
