"""Isaac Lab articulations for the robots in the catalog.

A lookup from a robot's ``asset_id`` to the hand-written ``ArticulationCfg`` that already describes
it, not a conversion. Decision D5 calls for generating this from :class:`RobotSpec` with numerical
validation, and until that pipeline exists the honest thing is to reuse what main already ships
rather than to hand-write a second description that can disagree with it.

What that costs is visible: adding a robot means adding an entry here *and* one in the mjlab
adapter, which is the ``N x M`` growth the rest of the design avoids. The asset pipeline is what
removes it, and this module is the place it will land.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from instinctlab.sim.robot_spec import RobotSpec

__all__ = ["ARTICULATIONS", "articulation"]

ARTICULATIONS: dict[str, str] = {
    "popsicle_torsobase_v1": "instinctlab.assets.unitree_g1.isaacsim:G1_29DOF_TORSOBASE_POPSICLE_CFG",
}
""":attr:`RobotSpec.asset_id` -> dotted path of the ``ArticulationCfg`` describing it."""


def articulation(robot: RobotSpec) -> Any:
    """The Isaac Lab articulation for ``robot``, copied so callers can retarget it safely."""
    try:
        path = ARTICULATIONS[robot.asset_id]
    except KeyError:
        have = ", ".join(sorted(ARTICULATIONS)) or "none"
        raise KeyError(
            f"No Isaac Lab articulation is registered for asset id {robot.asset_id!r} "
            f"(robot {robot.name!r}). Registered: {have}."
        ) from None
    module_path, _, attribute = path.partition(":")
    return getattr(import_module(module_path), attribute).copy()
