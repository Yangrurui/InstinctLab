"""End-to-end onboarding evidence for a robot with a different joint count."""

from __future__ import annotations

import instinctlab_engine
import pytest
from instinctlab.tasks import registry
from instinctlab_engine.preflight import require_preflight
from instinctlab_engine.spec import EntityRef

from tests.task_specs import task_spec


TASK_ID = "Instinct-Velocity-Flat-G1-15DoF"
ASSET_ID = "unitree_g1/popsicle_torsobase_locked_arms_v1"
EXPECTED_JOINTS = (
    "waist_pitch_joint",
    "waist_roll_joint",
    "waist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
LOCKED_ARM_JOINTS = {
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
}


def test_new_task_routes_to_the_new_robot_variant() -> None:
    registration = registry.REGISTRATIONS[TASK_ID]

    assert registration.asset_id == ASSET_ID
    assert registration.factory_path.endswith("g1_15dof:flat_g1_15dof")
    assert len(task_spec("Instinct-Velocity-Flat-G1").robot.joint_names) == 29


@pytest.mark.parametrize("engine", ("isaacsim", "mjlab"))
def test_locked_arm_asset_has_15_real_native_joints(engine: str) -> None:
    selected = instinctlab_engine.adapter(engine)
    report = selected.asset_conformance(ASSET_ID)
    robot = selected.robot_spec(ASSET_ID)

    assert report["status"] == "ok"
    assert report["joint_count"] == 15
    assert report["canonical_order"] == "dfs"
    assert robot.name == "unitree_g1_15dof_locked_arms"
    assert robot.joint_names == EXPECTED_JOINTS
    assert LOCKED_ARM_JOINTS.isdisjoint(robot.joint_names)
    assert tuple(joint.name for joint in robot.joint_properties) == EXPECTED_JOINTS
    assert {
        joint_name
        for group in report["actuator_groups"]
        for joint_name in group["joint_names"]
    } == set(EXPECTED_JOINTS)


@pytest.mark.parametrize("engine", ("isaacsim", "mjlab"))
def test_new_task_uses_the_15_joint_axis_and_passes_preflight(engine: str) -> None:
    task = task_spec(TASK_ID, engine)
    report = require_preflight(task, engine)
    action = task.mdp.actions["joint_pos"]

    assert report["status"] == "ok"
    assert task.robot.joint_names == EXPECTED_JOINTS
    assert action.target == EntityRef(
        "robot",
        joints=EXPECTED_JOINTS,
        preserve_order=True,
    )
    assert tuple(action.params["scale"]) == EXPECTED_JOINTS
    assert "joint_deviation_arms" not in task.mdp.rewards["rewards"]

    for group_name in ("policy", "critic"):
        for term_name in ("joint_pos", "joint_vel"):
            selector = task.mdp.observations[group_name].terms[term_name].params[
                "asset_cfg"
            ]
            assert selector == EntityRef(
                "robot",
                joints=EXPECTED_JOINTS,
                preserve_order=True,
            )


def test_15dof_robot_contract_is_equal_across_engines() -> None:
    isaac = task_spec(TASK_ID, "isaacsim").robot
    mjlab = task_spec(TASK_ID, "mjlab").robot

    assert isaac.joint_names == mjlab.joint_names == EXPECTED_JOINTS
    assert isaac.body_names == mjlab.body_names
    assert isaac.frame_names == mjlab.frame_names
    assert isaac.collision_body_names == mjlab.collision_body_names
    assert isaac.joint_properties == mjlab.joint_properties
    assert isaac.default_root_pos == mjlab.default_root_pos
    assert isaac.default_root_quat_wxyz == mjlab.default_root_quat_wxyz
