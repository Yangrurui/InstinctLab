"""Backend-independent joint control commands."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class ControlMode(str, Enum):
    POSITION = "position"
    VELOCITY = "velocity"
    EFFORT = "effort"


class ControlSemantics(str, Enum):
    NATIVE_IMPLICIT_V1 = "native_implicit_v1"
    EXPLICIT_EFFORT_V1 = "explicit_effort_v1"


@dataclass
class JointControlTarget:
    """Canonical joint command.

    Values are ordered by the controlled canonical joint list. Position control
    accepts an optional velocity target, which defaults to zero.
    """

    mode: ControlMode
    value: torch.Tensor
    joint_ids: torch.Tensor | None = None
    velocity: torch.Tensor | None = None

    def validate(self, *, num_envs: int, num_joints: int) -> None:
        expected_joints = num_joints if self.joint_ids is None else int(self.joint_ids.numel())
        expected = (num_envs, expected_joints)
        if tuple(self.value.shape) != expected:
            raise ValueError(f"control target has shape {tuple(self.value.shape)}, expected {expected}")
        if self.value.dtype != torch.float32:
            raise ValueError(f"control target must use float32, received {self.value.dtype}")
        if self.velocity is not None and tuple(self.velocity.shape) != expected:
            raise ValueError(f"velocity target has shape {tuple(self.velocity.shape)}, expected {expected}")
        if self.mode is not ControlMode.POSITION and self.velocity is not None:
            raise ValueError("velocity may only accompany a position control target")
        if self.joint_ids is not None:
            if self.joint_ids.dtype != torch.int64:
                raise ValueError("joint_ids must use int64")
            if torch.any(self.joint_ids < 0) or torch.any(self.joint_ids >= num_joints):
                raise ValueError("joint_ids contains an out-of-range canonical index")


__all__ = ["ControlMode", "ControlSemantics", "JointControlTarget"]
