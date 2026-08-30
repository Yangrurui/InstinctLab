"""Test-only helpers for comparing the two native G1 declarations."""

from __future__ import annotations

from dataclasses import replace

from instinctlab.engines.isaacsim.assets import robot_spec as isaac_robot_spec
from instinctlab.engines.mjlab.assets import robot_spec as mjlab_robot_spec
from instinctlab_engine.spec.robot import RobotSpec


def paired_robot_spec(asset_id: str) -> RobotSpec:
    """Return one test view containing both independently materialized assets.

    Production selects an engine before materializing ``RobotSpec`` and therefore
    carries one native asset. Parity tests need both paths at once, so they join
    the two views locally without restoring a shared G1 catalog factory.
    """
    isaac = isaac_robot_spec(asset_id)
    mjlab = mjlab_robot_spec(asset_id)
    for field in (
        "name",
        "schema_version",
        "asset_id",
        "root_body",
        "joint_names",
        "body_names",
        "frame_names",
        "collision_body_names",
        "joint_properties",
        "default_root_pos",
        "default_root_quat_wxyz",
        "soft_joint_pos_limit_factor",
        "actuator_delay",
    ):
        assert getattr(isaac, field) == getattr(mjlab, field), field
    paired = replace(isaac, assets=(*isaac.assets, *mjlab.assets))
    paired.validate()
    return paired
