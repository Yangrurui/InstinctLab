"""Engine-neutral routing into one Unitree G1 native configuration.

This module owns no robot, model, or actuator values. It only resolves an
explicit engine and variant to the corresponding native module.
"""

from __future__ import annotations

from instinctlab.sim.native_asset import native_asset_module
from instinctlab.sim.robot_spec import RobotSpec

__all__ = ["robot_spec"]


def robot_spec(engine: str, variant: str) -> RobotSpec:
    """Return ``variant`` from the explicitly selected native G1 module."""
    module, resolved_variant = native_asset_module(f"unitree_g1/{variant}", engine)
    return module.robot_spec(resolved_variant)
