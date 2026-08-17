from __future__ import annotations

import torch
from dataclasses import replace

import pytest

from instinctlab.assets.unitree_g1 import G1_29DOF_DFS_JOINT_NAMES, make_g1_29dof_robot_spec
from instinctlab.backends.mock import MockSimulatorBackend
from instinctlab.sim.backend import CanonicalIndexMap, RuntimeRequirements, SensorReadPhase
from instinctlab.sim.capabilities import Capability
from instinctlab.sim.control import ControlMode, JointControlTarget
from instinctlab.sim.rng import RngManager
from instinctlab.sim.robot_spec import BackendAsset
from instinctlab.sim.scene import ContactSensorSpec, SceneSpec, SimulationSpec
from instinctlab.sim.schema import locomotion_flat_schema
from instinctlab.sim.state import ContactState
from instinctlab.tasks.locomotion.mdp.unified import _material_params_for_backend, _slide_friction_range

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


def test_canonical_index_map_identity_avoids_reordering() -> None:
    names = ("first", "second")
    mapping = CanonicalIndexMap.build(names, names, device="cpu")
    native = torch.arange(4, dtype=torch.float32).reshape(2, 2)

    assert mapping.is_identity
    assert mapping.to_canonical(native) is native


def test_canonical_index_map_copy_to_canonical_is_in_place() -> None:
    mapping = CanonicalIndexMap.build(G1_29DOF_DFS_JOINT_NAMES, ISAAC_BFS_JOINT_NAMES, device="cpu")
    native = torch.arange(29, dtype=torch.float32).repeat(2, 1)
    out = torch.zeros_like(native)
    mapping.copy_to_canonical(native, out)
    torch.testing.assert_close(out, mapping.to_canonical(native))
    rebuilt = torch.zeros_like(native)
    mapping.copy_to_native(rebuilt, out)
    torch.testing.assert_close(rebuilt, native)


def test_canonical_index_map_copy_to_canonical_rank3() -> None:
    mapping = CanonicalIndexMap.build(("a", "c"), ("a", "b", "c"), device="cpu")
    native = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    out = torch.zeros(2, 2, 2)
    mapping.copy_to_canonical(native, out, dim=1)
    torch.testing.assert_close(out, native[:, (0, 2)])


def test_canonical_index_map_copy_to_canonical_identity() -> None:
    names = ("first", "second")
    mapping = CanonicalIndexMap.build(names, names, device="cpu")
    native = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    out = torch.zeros_like(native)
    mapping.copy_to_canonical(native, out)
    torch.testing.assert_close(out, native)


def test_simulation_engine_options_are_backend_scoped() -> None:
    spec = SimulationSpec(
        engine_options={
            "isaacsim": {"physx": {"solver_type": 1}},
            "mjlab": {"iterations": 10},
        }
    )

    spec.validate()
    assert spec.engine_options_for("isaacsim") == {"physx": {"solver_type": 1}}
    assert spec.engine_options_for("mjlab") == {"iterations": 10}
    assert spec.engine_options_for("mock") == {}


def test_simulation_engine_options_reject_unscoped_values() -> None:
    spec = SimulationSpec(engine_options={"iterations": 10})  # type: ignore[dict-item]

    with pytest.raises(ValueError, match="scoped by backend name"):
        spec.validate()


def test_scene_backend_options_are_backend_scoped() -> None:
    robot = make_g1_29dof_robot_spec()
    spec = SceneSpec(
        num_envs=2,
        env_spacing=1.0,
        robot=robot,
        backend_options={
            "isaacsim": {"scene": {"lazy_sensor_update": True}},
            "mjlab": {"unused": True},
        },
    )

    spec.validate()
    assert spec.backend_options_for("isaacsim") == {"scene": {"lazy_sensor_update": True}}
    assert spec.backend_options_for("mock") == {}


def test_scene_backend_options_reject_unscoped_values() -> None:
    robot = make_g1_29dof_robot_spec()
    spec = SceneSpec(
        num_envs=2,
        env_spacing=1.0,
        robot=robot,
        backend_options={"lazy_sensor_update": True},  # type: ignore[dict-item]
    )

    with pytest.raises(ValueError, match="scoped by backend name"):
        spec.validate()


def test_scene_rejects_contact_sensor_on_non_primary_entity() -> None:
    robot = make_g1_29dof_robot_spec()
    spec = SceneSpec(
        num_envs=2,
        env_spacing=1.0,
        robot=robot,
        contact_sensors=(
            ContactSensorSpec(
                name="contact_forces",
                entity_name="other",
                body_names=("torso_link",),
            ),
        ),
    )

    with pytest.raises(ValueError, match="primary_entity"):
        spec.validate()


def test_backend_asset_rejects_unknown_contact_aliases() -> None:
    robot = make_g1_29dof_robot_spec()
    bad = BackendAsset(
        backend="mjlab",
        path=robot.asset_for("mjlab").path,
        contact_body_aliases={"not_a_body": "pelvis"},
    )
    with pytest.raises(ValueError, match="unknown canonical bodies"):
        bad.validate_against(robot.body_names)


def test_backend_asset_rejects_duplicate_native_aliases() -> None:
    robot = make_g1_29dof_robot_spec()
    bad = BackendAsset(
        backend="mjlab",
        path=robot.asset_for("mjlab").path,
        contact_body_aliases={"LL_FOOT": "left_ankle_roll_link", "LR_FOOT": "left_ankle_roll_link"},
    )
    with pytest.raises(ValueError, match="same native name"):
        bad.validate_against(robot.body_names)


def test_backend_asset_rejects_unknown_load_mode() -> None:
    robot = make_g1_29dof_robot_spec()
    bad = BackendAsset(backend="mjlab", path=robot.asset_for("mjlab").path, load_mode="explode")
    with pytest.raises(ValueError, match="unsupported load_mode"):
        bad.validate_against(robot.body_names)


def test_robot_spec_rejects_duplicate_backend_assets() -> None:
    robot = make_g1_29dof_robot_spec()
    duplicate = replace(robot, assets=(robot.assets[0], robot.assets[0]))
    with pytest.raises(ValueError, match="at most one asset per backend"):
        duplicate.validate()


def test_robot_spec_rejects_frame_as_collision_body() -> None:
    robot = make_g1_29dof_robot_spec()
    bad = replace(robot, collision_body_names=robot.collision_body_names + ("LL_FOOT",))
    with pytest.raises(ValueError, match="physical bodies"):
        bad.validate()


def test_g1_assets_declare_contact_aliases_and_load_mode() -> None:
    robot = make_g1_29dof_robot_spec()
    isaac = robot.asset_for("isaacsim")
    mjlab = robot.asset_for("mjlab")

    assert isaac.resolve_contact_body_names(("torso_link", "LL_FOOT", "LR_FOOT")) == (
        "torso_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )
    assert isaac.import_options["fix_base"] is False
    assert isaac.import_options["merge_fixed_joints"] is False
    assert isaac.import_options["replace_cylinders_with_capsules"] is True
    assert mjlab.load_mode == "strip_visual_meshes"
    assert mjlab.contact_body_aliases == isaac.contact_body_aliases


def test_g1_robot_spec_and_schema() -> None:
    robot = make_g1_29dof_robot_spec()
    assert robot.root_body == "torso_link"
    assert robot.joint_names == G1_29DOF_DFS_JOINT_NAMES
    assert robot.joint_properties[robot.joint_index("right_shoulder_pitch_joint")].default_pos == 0.2
    assert "LL_FOOT" in robot.frame_names
    assert "imu_in_pelvis" in robot.frame_names
    assert "torso_link" in robot.collision_body_names
    assert "LL_FOOT" not in robot.physical_body_names
    assert robot.material_body_names == robot.collision_body_names
    schema = locomotion_flat_schema(len(robot.joint_names))
    assert schema.observation_group("policy").flat_dim == 96
    assert schema.observation_group("critic").flat_dim == 99
    assert len(schema.hash) == 64


def test_g1_assets_are_pinned_and_verify() -> None:
    robot = make_g1_29dof_robot_spec()
    isaac = robot.asset_for("isaacsim")
    mjlab = robot.asset_for("mjlab")
    assert isaac.checksum is not None
    assert mjlab.checksum is not None
    robot.verify_assets()


def test_material_params_are_backend_scoped() -> None:
    backend_params = {
        "default": {"shared_random": False, "restitution_range": (0.0, 0.8)},
        "mjlab": {"shared_random": True, "restitution_range": None},
        "isaacsim": {"separate_dynamic_friction": True},
    }
    mjlab = _material_params_for_backend("mjlab", backend_params, shared_random=True)
    isaac = _material_params_for_backend("isaacsim", backend_params, shared_random=True)
    assert mjlab["shared_random"] is True
    assert mjlab["restitution_range"] is None
    assert isaac["shared_random"] is False
    assert isaac["restitution_range"] == (0.0, 0.8)
    assert isaac["separate_dynamic_friction"] is True


def test_instinctmj_material_ranges_merge_to_slide_friction() -> None:
    assert _slide_friction_range(None, (0.25, 0.8), (0.2, 0.6)) == (0.2, 0.8)
    assert _slide_friction_range((0.3, 0.7), None, None) == (0.3, 0.7)
    with pytest.raises(ValueError, match="either friction_range or static/dynamic"):
        _slide_friction_range((0.3, 0.7), (0.25, 0.8), None)


def test_contact_state_air_time_uses_force_threshold_edges() -> None:
    sensor = ContactState.allocate(num_envs=1, body_names=("left_foot", "right_foot"), history_length=0, device="cpu")
    contact = torch.tensor([[False, True]])
    sensor.update_air_time(contact, 0.005)
    torch.testing.assert_close(sensor.current_air_time, torch.tensor([[0.005, 0.0]]))
    torch.testing.assert_close(sensor.current_contact_time, torch.tensor([[0.0, 0.005]]))

    sensor.update_air_time(torch.tensor([[True, False]]), 0.005)
    torch.testing.assert_close(sensor.last_air_time, torch.tensor([[0.010, 0.0]]))
    torch.testing.assert_close(sensor.last_contact_time, torch.tensor([[0.0, 0.010]]))
    torch.testing.assert_close(sensor.current_air_time, torch.tensor([[0.0, 0.005]]))
    torch.testing.assert_close(sensor.current_contact_time, torch.tensor([[0.005, 0.0]]))


def test_named_rng_streams_are_isolated() -> None:
    first = RngManager(42, "cpu")
    second = RngManager(42, "cpu")
    ignored = first.uniform("backend_should_not_use", 0.0, 1.0, (100,))
    del ignored
    torch.testing.assert_close(
        first.uniform("reset.root", -1.0, 1.0, (4, 3)),
        second.uniform("reset.root", -1.0, 1.0, (4, 3)),
    )
    torch.testing.assert_close(
        first.integers("event.material.bucket_ids", 0, 64, (4, 8)),
        second.integers("event.material.bucket_ids", 0, 64, (4, 8)),
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
