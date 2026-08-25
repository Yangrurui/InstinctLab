"""MJLab actuator extensions used by the shared robot catalog."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mjlab.actuator import Actuator, ActuatorCfg, ActuatorCmd
from mjlab.utils.spec import create_motor_actuator

if TYPE_CHECKING:
    import mujoco
    from mjlab.entity import Entity


@dataclass(kw_only=True)
class JointVelocityLimiterCfg(ActuatorCfg):
    """Motor-side braking at a symmetric joint-speed limit.

    MuJoCo joints have no native equivalent of PhysX's ``velocity_limit_sim``.
    This actuator therefore runs beside the native PD actuator and adds a
    braking motor. It is inactive below the limit and does not replace the
    implicit-integration PD path.
    """

    velocity_limit: float
    effort_limit: float

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.velocity_limit <= 0.0:
            raise ValueError("velocity_limit must be positive")
        if self.effort_limit <= 0.0:
            raise ValueError("effort_limit must be positive")

    def build(
        self,
        entity: Entity,
        target_ids: list[int],
        target_names: list[str],
    ) -> JointVelocityLimiter:
        return JointVelocityLimiter(self, entity, target_ids, target_names)


class JointVelocityLimiter(Actuator[JointVelocityLimiterCfg]):
    """Apply full reverse motor effort once a joint reaches its speed cap."""

    def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
        # The native PD can simultaneously request +effort_limit. A 2x braking
        # channel makes the joint-level sum clamp resolve to -effort_limit at
        # the upper speed cap (and conversely at the lower cap).
        brake_effort = 2.0 * self.cfg.effort_limit
        for target_name in target_names:
            actuator = create_motor_actuator(spec, target_name, effort_limit=brake_effort)
            actuator.name = f"{target_name}_velocity_limiter"
            self._mjs_actuators.append(actuator)

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        brake_effort = 2.0 * self.cfg.effort_limit
        effort = torch.zeros_like(cmd.vel)
        effort = torch.where(cmd.vel >= self.cfg.velocity_limit, -brake_effort, effort)
        return torch.where(cmd.vel <= -self.cfg.velocity_limit, brake_effort, effort)


__all__ = ["JointVelocityLimiter", "JointVelocityLimiterCfg"]
