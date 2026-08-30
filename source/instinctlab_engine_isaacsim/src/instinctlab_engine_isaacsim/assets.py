"""Convert an Isaac-native asset configuration to the shared runtime spec."""

from __future__ import annotations

from typing import Any

from instinctlab_engine.assets import (
    native_asset_conformance_report,
    native_asset_definition,
)
from instinctlab_engine.spec.robot import BackendAsset, JointProperties, RobotSpec

__all__ = ["articulation", "asset_conformance", "robot_spec"]


def _definition(asset_id: str):
    return native_asset_definition(
        asset_id,
        "isaacsim",
        builder_name="articulation",
        resource_field="urdf_path",
        resource_kind="urdf",
    )


def robot_spec(asset_id: str) -> RobotSpec:
    """Normalize one Isaac-native robot configuration after engine selection."""
    native = _definition(asset_id).config
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
                backend="isaacsim",
                path=native.urdf_path,
                contact_body_aliases=native.contact_body_aliases,
                import_options=native.import_options,
            ),
        ),
        default_root_pos=native.default_root_pos,
        default_root_quat_wxyz=native.default_root_quat_wxyz,
        soft_joint_pos_limit_factor=native.soft_joint_pos_limit_factor,
        actuator_delay=native.actuator_delay,
    )
    robot.validate()
    return robot


def articulation(robot: RobotSpec) -> Any:
    definition = _definition(robot.asset_id)
    return definition.builder(definition.variant, robot)


def asset_conformance(asset_id: str) -> dict[str, Any]:
    """Return SDK-free onboarding evidence for one Isaac-native robot."""
    return native_asset_conformance_report(_definition(asset_id))
