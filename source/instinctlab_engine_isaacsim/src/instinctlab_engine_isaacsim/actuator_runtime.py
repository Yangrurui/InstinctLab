"""Observable runtime adapters for Isaac Lab's native PD actuators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class IsaacPdRuntimeAdapter:
    model: str

    def matches(self, actuator: object) -> bool:
        from isaaclab.actuators import DelayedPDActuator, ImplicitActuator

        if self.model == "implicit":
            return isinstance(actuator, ImplicitActuator)
        if self.model == "delayed":
            return isinstance(actuator, DelayedPDActuator)
        raise RuntimeError(f"unknown Isaac PD runtime model {self.model!r}")

    def stiffness_groups(self, env: Any, asset: Any, actuator: Any):
        del env, asset
        return ((actuator.joint_indices, actuator.stiffness),)


IMPLICIT_PD_RUNTIME = IsaacPdRuntimeAdapter("implicit")
DELAYED_PD_RUNTIME = IsaacPdRuntimeAdapter("delayed")

__all__ = ["DELAYED_PD_RUNTIME", "IMPLICIT_PD_RUNTIME", "IsaacPdRuntimeAdapter"]
