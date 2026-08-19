"""mjlab view of G1.

The MJCF already carries geometry, so this module does not restate an ``ArticulationCfg``.
It builds the entity from ``make_g1_29dof_robot_spec``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from instinctlab.assets.unitree_g1.isaacsim import make_g1_29dof_robot_spec
from instinctlab.sim.robot_spec import RobotSpec

__all__ = ["entity"]


def entity(robot: RobotSpec | None = None, *, actuator_order: Sequence[str] | None = None) -> Any:
    """The mjlab entity for G1, or for ``robot`` when the compile path passes one in."""
    from instinctlab.engines.mjlab.assets import entity as derive_entity

    return derive_entity(robot if robot is not None else make_g1_29dof_robot_spec(), actuator_order=actuator_order)
