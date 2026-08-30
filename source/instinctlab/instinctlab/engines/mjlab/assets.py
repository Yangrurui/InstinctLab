"""Convert an MJLab-native asset configuration to the shared runtime spec."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from instinctlab.assets.registry import native_asset_module
from instinctlab.spec.robot import BackendAsset, JointProperties, RobotSpec

__all__ = ["entity", "robot_spec"]


def robot_spec(asset_id: str) -> RobotSpec:
    """Normalize one MJLab-native robot configuration after engine selection."""
    module, variant = native_asset_module(asset_id, "mjlab")
    native = module.native_config(variant)
    if native.asset_id != asset_id:
        raise ValueError(
            f"MJLab native variant {variant!r} declares {native.asset_id!r}, expected {asset_id!r}"
        )
    robot = RobotSpec(
        name=native.name,
        schema_version=native.schema_version,
        asset_id=native.asset_id,
        root_body=native.root_body,
        joint_names=native.joint_names,
        body_names=native.body_names,
        frame_names=native.frame_names,
        collision_body_names=native.collision_body_names,
        joint_properties=tuple(
            JointProperties(
                name=joint.name,
                default_pos=joint.default_pos,
                stiffness=joint.stiffness,
                damping=joint.damping,
                armature=joint.armature,
                effort_limit=joint.effort_limit,
                velocity_limit=joint.velocity_limit,
                action_scale=joint.action_scale,
            )
            for joint in native.joint_properties
        ),
        assets=(
            BackendAsset(
                backend="mjlab",
                path=native.mjcf_path,
                contact_body_aliases=native.contact_body_aliases,
                load_mode=native.load_mode,
            ),
        ),
        default_root_pos=native.default_root_pos,
        default_root_quat_wxyz=native.default_root_quat_wxyz,
        soft_joint_pos_limit_factor=native.soft_joint_pos_limit_factor,
        actuator_delay=native.actuator_delay,
    )
    robot.validate()
    return robot


def entity(robot: RobotSpec, *, actuator_order: Sequence[str] | None = None) -> Any:
    module, variant = native_asset_module(robot.asset_id, "mjlab")
    try:
        builder = module.entity
    except AttributeError:
        raise AttributeError(
            f"Native asset module {module.__name__!r} does not define entity(variant, robot)"
        ) from None
    return builder(variant, robot, actuator_order=actuator_order)
