from __future__ import annotations

import torch

from instinctlab.backends.mock import MockSimulatorBackend
from instinctlab.sim.backend import CanonicalIndexMap, RuntimeRequirements, SensorReadPhase
from instinctlab.sim.capabilities import Capability
from instinctlab.sim.control import ControlMode, JointControlTarget
from instinctlab.sim.rng import RngManager
from instinctlab.sim.robot_spec import G1_29DOF_DFS_JOINT_NAMES, make_g1_29dof_robot_spec
from instinctlab.sim.scene import ContactSensorSpec, SceneSpec, SimulationSpec
from instinctlab.sim.schema import locomotion_flat_schema


ISAAC_BFS_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "waist_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "waist_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "waist_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
)


def test_canonical_index_map_round_trip() -> None:
    mapping = CanonicalIndexMap.build(G1_29DOF_DFS_JOINT_NAMES, ISAAC_BFS_JOINT_NAMES, device="cpu")
    assert not mapping.is_identity
    native = torch.arange(29, dtype=torch.float32).repeat(2, 1)
    canonical = mapping.to_canonical(native)
    rebuilt = torch.zeros_like(native)
    mapping.copy_to_native(rebuilt, canonical)
    torch.testing.assert_close(rebuilt, native)


def test_g1_robot_spec_and_schema() -> None:
    robot = make_g1_29dof_robot_spec()
    assert robot.root_body == "torso_link"
    assert robot.joint_names == G1_29DOF_DFS_JOINT_NAMES
    assert robot.joint_properties[robot.joint_index("right_shoulder_pitch_joint")].default_pos == 0.2
    schema = locomotion_flat_schema(len(robot.joint_names))
    assert schema.observation_group("policy").flat_dim == 96
    assert schema.observation_group("critic").flat_dim == 99
    assert len(schema.hash) == 64


def test_named_rng_streams_are_isolated() -> None:
    first = RngManager(42, "cpu")
    second = RngManager(42, "cpu")
    ignored = first.uniform("backend_should_not_use", 0.0, 1.0, (100,))
    del ignored
    torch.testing.assert_close(
        first.uniform("reset.root", -1.0, 1.0, (4, 3)),
        second.uniform("reset.root", -1.0, 1.0, (4, 3)),
    )


def test_mock_backend_contract() -> None:
    robot = make_g1_29dof_robot_spec()
    contact = ContactSensorSpec(
        name="contact_forces",
        entity_name="robot",
        body_names=robot.body_names,
        history_length=3,
        force_threshold=1.0,
    )
    scene_spec = SceneSpec(num_envs=4, env_spacing=2.5, robot=robot, contact_sensors=(contact,))
    simulation_spec = SimulationSpec()
    backend = MockSimulatorBackend(device="cpu")
    backend.initialize(
        scene_spec,
        simulation_spec,
        RuntimeRequirements(
            capabilities=frozenset(
                {
                    Capability.BATCHED_SIMULATION,
                    Capability.ROOT_STATE,
                    Capability.JOINT_STATE,
                    Capability.CONTACT_ACTIVE,
                }
            )
        ),
    )
    state = backend.scene.articulations["robot"].data
    target = state.default_joint_pos + 0.1
    backend.set_joint_control_target(
        "robot",
        JointControlTarget(mode=ControlMode.POSITION, value=target.clone()),
    )
    backend.step()
    backend.synchronize(SensorReadPhase.POST_PHYSICS)
    assert torch.any(state.joint_pos != state.default_joint_pos)
    assert tuple(state.joint_pos.shape) == (4, 29)
    assert tuple(backend.scene.sensors["contact_forces"].net_forces_w_history.shape) == (4, 3, 40, 3)
    state.validate()
    backend.scene.sensors["contact_forces"].validate()
