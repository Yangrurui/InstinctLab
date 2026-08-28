"""Generic bridge from ``RobotSpec.asset_id`` to an MJLab native asset."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from instinctlab.sim.native_asset import native_asset_module
from instinctlab.sim.robot_spec import RobotSpec

__all__ = ["entity"]


def entity(robot: RobotSpec, *, actuator_order: Sequence[str] | None = None) -> Any:
    module, variant = native_asset_module(robot.asset_id, "mjlab")
    try:
        builder = module.entity
    except AttributeError:
        raise AttributeError(
            f"Native asset module {module.__name__!r} does not define entity(variant, robot)"
        ) from None
    return builder(variant, robot, actuator_order=actuator_order)
