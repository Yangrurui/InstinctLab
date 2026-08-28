"""Generic bridge from ``RobotSpec.asset_id`` to an Isaac Lab native asset."""

from __future__ import annotations

from typing import Any

from instinctlab.sim.native_asset import native_asset_module
from instinctlab.sim.robot_spec import RobotSpec

__all__ = ["articulation"]


def articulation(robot: RobotSpec) -> Any:
    module, variant = native_asset_module(robot.asset_id, "isaacsim")
    try:
        builder = module.articulation
    except AttributeError:
        raise AttributeError(
            f"Native asset module {module.__name__!r} does not define articulation(variant, robot)"
        ) from None
    return builder(variant, robot)
