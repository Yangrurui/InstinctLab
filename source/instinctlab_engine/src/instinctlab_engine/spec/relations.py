"""Narrow engine-neutral relations whose physical meaning both backends preserve."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CollisionExclusionRef:
    """Disable collision generation for one unordered pair of entity bodies."""

    body_a: str
    body_b: str
    entity: str = "robot"

    def __post_init__(self) -> None:
        if not self.entity or not self.body_a or not self.body_b:
            raise ValueError("Collision exclusion entity and body names must be non-empty.")
        if self.body_a == self.body_b:
            raise ValueError("A collision exclusion must name two different bodies.")

    @property
    def pair(self) -> tuple[str, str]:
        return tuple(sorted((self.body_a, self.body_b)))


__all__ = ["CollisionExclusionRef"]
