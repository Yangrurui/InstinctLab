"""Backend capability declarations.

The common environment validates these capabilities before creating a simulator.
Unsupported task semantics must fail at startup instead of silently degrading.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class Capability(str, Enum):
    """Simulator features visible to backend-independent task code."""

    BATCHED_SIMULATION = "batched_simulation"
    GPU_SIMULATION = "gpu_simulation"
    PLANE_TERRAIN = "plane_terrain"
    ROOT_STATE = "root_state"
    JOINT_STATE = "joint_state"
    BODY_STATE = "body_state"
    IMPLICIT_POSITION_CONTROL = "implicit_position_control"
    EFFORT_CONTROL = "effort_control"
    CONTACT_ACTIVE = "contact_active"
    CONTACT_HISTORY = "contact_history"
    CONTACT_AIR_TIME = "contact_air_time"
    CONTACT_FORCE_VECTOR = "contact_force_vector"
    DR_SLIDING_FRICTION = "dr_sliding_friction"
    DR_RESTITUTION = "dr_restitution"
    BODY_MASS_PROPERTIES = "body_mass_properties"
    EXTERNAL_WRENCH = "external_wrench"
    ROOT_VELOCITY_WRITE = "root_velocity_write"
    HUMAN_VIEWER = "human_viewer"
    RGB_ARRAY = "rgb_array"


@dataclass(frozen=True)
class CapabilitySet:
    """Immutable capability set with descriptive validation errors."""

    values: frozenset[Capability]

    @classmethod
    def of(cls, values: Iterable[Capability]) -> "CapabilitySet":
        return cls(frozenset(values))

    def supports(self, capability: Capability) -> bool:
        return capability in self.values

    def require(self, required: Iterable[Capability], *, context: str) -> None:
        missing = frozenset(required).difference(self.values)
        if missing:
            names = ", ".join(sorted(capability.value for capability in missing))
            raise RuntimeError(f"{context} requires unsupported backend capabilities: {names}")


__all__ = ["Capability", "CapabilitySet"]
