from __future__ import annotations

import json
import re
import torch
from dataclasses import replace
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest

from instinctlab.assets.unitree_g1.isaacsim import (
    G1_29DOF_DFS_JOINT_NAMES,
    G1_29DOF_ISAAC_BFS_JOINT_NAMES,
)
from instinctlab.checkpoint import add_task_contract, validate_checkpoint_contract
from instinctlab.tasks.shadowing.mdp.commands import (
    JointPositionReference,
    JointVelocityReference,
    PositionReference,
    RotationReference,
)
from instinctlab_engine_mjlab.assets import robot_spec
from instinctlab.tasks import registry
from instinctlab.tasks.shadowing.mdp.events import reset_robot_from_reference
from instinctlab_engine.name_order import copy_named_columns_, resolve_name_indices


def _task(task_id: str):
    return registry.spec(task_id, robot_spec(registry.asset_id(task_id)))


SHADOWING_IDS = tuple(
    task_id
    for task_id in registry.ids()
    if any(
        term.kind == "motion_reference_joint_position"
        for term in _task(task_id).mdp.commands.values()
    )
)


class _ResetAsset:
    def __init__(self, native_joint_names: tuple[str, ...]) -> None:
        self.joint_names = native_joint_names
        self.joint_pos = torch.zeros(1, len(native_joint_names))
        self.joint_vel = torch.zeros_like(self.joint_pos)

    def find_joints(self, names, preserve_order=False):
        assert preserve_order is True
        resolved = tuple(name for name in names if name in self.joint_names)
        return [self.joint_names.index(name) for name in resolved], list(resolved)

    def write_root_link_pose_to_sim(self, pose, env_ids=None):
        self.root_pose = pose.clone()

    def write_root_link_velocity_to_sim(self, velocity, env_ids=None):
        self.root_velocity_writer = "link"
        self.root_velocity = velocity.clone()

    def write_root_com_velocity_to_sim(self, velocity, env_ids=None):
        self.root_velocity_writer = "com"
        self.root_velocity = velocity.clone()

    def write_joint_state_to_sim(
        self, position, velocity, joint_ids=None, env_ids=None
    ):
        self.joint_pos[env_ids[:, None], joint_ids[None, :]] = position
        self.joint_vel[env_ids[:, None], joint_ids[None, :]] = velocity


def _reset_env(
    canonical_joint_names: tuple[str, ...], native_joint_names: tuple[str, ...]
):
    asset = _ResetAsset(native_joint_names)
    joint_pos = torch.arange(len(canonical_joint_names), dtype=torch.float32) + 10.0
    joint_vel = torch.arange(len(canonical_joint_names), dtype=torch.float32) + 1.0
    state = SimpleNamespace(
        base_pos_w=torch.zeros(1, 1, 3),
        base_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        base_lin_vel_w=torch.zeros(1, 1, 3),
        base_ang_vel_w=torch.zeros(1, 1, 3),
        joint_pos=joint_pos.reshape(1, 1, -1),
        joint_vel=joint_vel.reshape(1, 1, -1),
    )
    sensor = SimpleNamespace(
        init_reference_state=state, joint_names=canonical_joint_names
    )
    env = SimpleNamespace(
        device="cpu", scene={"robot": asset, "motion_reference": sensor}
    )
    reset_robot_from_reference(
        env, torch.tensor([0]), randomize_joint_pos_range=(0.0, 0.0)
    )
    return asset


def test_shadowing_reset_writes_canonical_values_to_native_joint_indices() -> None:
    canonical = G1_29DOF_DFS_JOINT_NAMES
    native = G1_29DOF_ISAAC_BFS_JOINT_NAMES

    asset = _reset_env(canonical, native)

    expected_pos = torch.tensor([[canonical.index(name) + 10.0 for name in native]])
    expected_vel = torch.tensor([[canonical.index(name) + 1.0 for name in native]])
    torch.testing.assert_close(asset.joint_pos, expected_pos)
    torch.testing.assert_close(asset.joint_vel, expected_vel)


def test_shadowing_reset_uses_the_declared_root_velocity_point() -> None:
    canonical = G1_29DOF_DFS_JOINT_NAMES
    asset = _reset_env(canonical, canonical)
    assert asset.root_velocity_writer == "link"

    state = SimpleNamespace(
        base_pos_w=torch.zeros(1, 1, 3),
        base_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        base_lin_vel_w=torch.zeros(1, 1, 3),
        base_ang_vel_w=torch.zeros(1, 1, 3),
        joint_pos=torch.zeros(1, 1, len(canonical)),
        joint_vel=torch.zeros(1, 1, len(canonical)),
    )
    sensor = SimpleNamespace(init_reference_state=state, joint_names=canonical)
    env = SimpleNamespace(
        device="cpu", scene={"robot": asset, "motion_reference": sensor}
    )
    reset_robot_from_reference(
        env,
        torch.tensor([0]),
        randomize_joint_pos_range=(0.0, 0.0),
        root_velocity_frame="com",
    )
    assert asset.root_velocity_writer == "com"


@pytest.mark.parametrize(
    ("engine", "expected_position", "expected_velocity_writer"),
    (
        ("isaacsim", (1.0, 3.0, 5.0), "com"),
        ("mjlab", (1.0, 2.0, 3.0), "link"),
    ),
)
def test_hoi_play_reset_has_no_state_noise_and_applies_only_the_isaac_offset(
    engine: str,
    expected_position: tuple[float, float, float],
    expected_velocity_writer: str,
) -> None:
    canonical = G1_29DOF_DFS_JOINT_NAMES
    asset = _ResetAsset(canonical)
    joint_pos = torch.arange(len(canonical), dtype=torch.float32).reshape(1, 1, -1)
    joint_vel = joint_pos + 100.0
    state = SimpleNamespace(
        base_pos_w=torch.tensor([[[1.0, 2.0, 3.0]]]),
        base_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        base_lin_vel_w=torch.tensor([[[4.0, 5.0, 6.0]]]),
        base_ang_vel_w=torch.tensor([[[7.0, 8.0, 9.0]]]),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
    )
    sensor = SimpleNamespace(init_reference_state=state, joint_names=canonical)
    env = SimpleNamespace(
        device="cpu", scene={"robot": asset, "motion_reference": sensor}
    )
    reset = _task("Instinct-Perceptive-HOI-Shadowing-G1-Play-v0").mdp.events[
        "reset_robot"
    ]

    reset_robot_from_reference(env, torch.tensor([0]), **reset.resolved_params(engine))

    torch.testing.assert_close(asset.root_pose[0, :3], torch.tensor(expected_position))
    torch.testing.assert_close(asset.root_pose[0, 3:], state.base_quat_w[0, 0])
    torch.testing.assert_close(
        asset.root_velocity[0],
        torch.cat((state.base_lin_vel_w[0, 0], state.base_ang_vel_w[0, 0])),
    )
    torch.testing.assert_close(asset.joint_pos, joint_pos[:, 0])
    torch.testing.assert_close(asset.joint_vel, joint_vel[:, 0])
    assert asset.root_velocity_writer == expected_velocity_writer


def test_shadowing_reset_rejects_an_incomplete_joint_mapping() -> None:
    with pytest.raises(ValueError, match="do not resolve one-to-one"):
        _reset_env(
            ("waist_joint", "missing_joint", "right_joint"),
            ("right_joint", "waist_joint"),
        )


@pytest.mark.parametrize("task_id", SHADOWING_IDS)
def test_every_shadowing_tensor_uses_the_canonical_joint_order(task_id: str) -> None:
    task = _task(task_id)
    action = task.mdp.actions["joint_pos"]

    assert task.robot.joint_names == G1_29DOF_DFS_JOINT_NAMES
    assert task.robot.actuator_delay == (0, 0)
    assert action.target is not None
    assert action.target.joints == G1_29DOF_DFS_JOINT_NAMES
    assert action.target.preserve_order is True
    assert tuple(action.params["scale"]) == G1_29DOF_DFS_JOINT_NAMES
    assert tuple(action.params["scale"].values()) == tuple(
        joint.action_scale for joint in task.robot.joint_properties
    )
    assert task.scene.motion_references[0].joints == G1_29DOF_DFS_JOINT_NAMES

    for group in task.mdp.observations.values():
        for name in ("joint_pos", "joint_vel"):
            term = group.terms.get(name)
            if term is not None:
                selector = term.params["asset_cfg"]
                assert selector.joints == G1_29DOF_DFS_JOINT_NAMES
                assert selector.preserve_order is True


def test_isaac_bfs_to_policy_dfs_mapping_is_name_based_and_reversible() -> None:
    native_ids = resolve_name_indices(
        G1_29DOF_ISAAC_BFS_JOINT_NAMES, G1_29DOF_DFS_JOINT_NAMES
    )
    native = torch.arange(len(G1_29DOF_ISAAC_BFS_JOINT_NAMES), dtype=torch.float32)
    canonical = native[list(native_ids)]
    rebuilt = torch.empty_like(native)
    rebuilt[list(native_ids)] = canonical

    torch.testing.assert_close(rebuilt, native)
    for name in ("waist_pitch_joint", "left_hip_pitch_joint", "right_wrist_yaw_joint"):
        assert canonical[
            G1_29DOF_DFS_JOINT_NAMES.index(name)
        ] == G1_29DOF_ISAAC_BFS_JOINT_NAMES.index(name)


def test_default_randomization_scatter_maps_native_columns_into_action_order() -> None:
    """Isaac startup DR must not write BFS offsets positionally into DFS actions."""
    native_names = ("left_hip", "right_hip", "waist")
    action_names = ("waist", "left_hip", "right_hip")
    offsets = torch.zeros(2, 3)
    native_values = torch.tensor([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]])

    copy_named_columns_(
        offsets,
        native_values,
        torch.tensor([0, 1]),
        value_names=native_names,
        target_names=action_names,
    )

    torch.testing.assert_close(
        offsets, torch.tensor([[30.0, 10.0, 20.0], [31.0, 11.0, 21.0]])
    )


def test_position_reference_anchor_uses_the_separate_current_frame() -> None:
    """The anchor is sampled at t even when the command window starts later."""

    identity = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    yaw_90 = torch.tensor([[[2.0**-0.5, 0.0, 0.0, 2.0**-0.5]]])
    data = SimpleNamespace(
        base_pos_w=torch.tensor([[[10.0, 2.0, 0.0], [7.0, 0.0, 0.0]]]),
        base_quat_w=identity.expand(1, 2, 4),
        validity=torch.ones(1, 2),
    )
    current = SimpleNamespace(
        base_pos_w=torch.tensor([[[10.0, 0.0, 0.0]]]),
        base_quat_w=yaw_90,
    )
    motion = SimpleNamespace(
        data=data,
        reference_frame=current,
        num_frames=2,
    )
    env = SimpleNamespace(num_envs=1, device="cpu", scene={"motion_reference": motion})
    command = PositionReference(
        env,
        {
            "motion_reference": "motion_reference",
            "entity": "robot",
            "anchor_frame": "reference",
        },
    )

    torch.testing.assert_close(
        command.command,
        torch.tensor([[[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]]),
    )


def test_rotation_reference_uses_robot_frame_and_tangent_normal_schema() -> None:
    half_sqrt = 2.0**-0.5
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
    yaw_90 = torch.tensor([half_sqrt, 0.0, 0.0, half_sqrt])
    yaw_180 = torch.tensor([0.0, 0.0, 0.0, 1.0])
    data = SimpleNamespace(
        base_quat_w=torch.stack((yaw_180, identity)).reshape(1, 2, 4),
        validity=torch.ones(1, 2),
    )
    motion = SimpleNamespace(data=data, num_frames=2)
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_quat_w=yaw_90.unsqueeze(0),
            root_quat_w=identity.unsqueeze(0),
        )
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        scene={"motion_reference": motion, "robot": robot},
    )
    command = RotationReference(
        env,
        {
            "motion_reference": "motion_reference",
            "entity": "robot",
            "in_base_frame": True,
            "rotation_mode": "tannorm",
        },
    )

    torch.testing.assert_close(
        command.command,
        torch.tensor(
            [[[0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [0.0, -1.0, 0.0, 0.0, 0.0, 1.0]]]
        ),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_joint_position_reference_subtracts_frozen_defaults_by_joint_name() -> None:
    """A canonical reference must not subtract Isaac's native BFS defaults positionally."""

    native_names = ("left_hip", "right_hip", "waist")
    canonical_names = ("waist", "left_hip", "right_hip")
    default_native = torch.tensor([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]])
    reference = SimpleNamespace(
        joint_pos=torch.tensor(
            [
                [[100.0, 200.0, 300.0], [101.0, 201.0, 301.0]],
                [[110.0, 210.0, 310.0], [111.0, 211.0, 311.0]],
            ]
        ),
        validity=torch.ones(2, 2),
    )
    motion = SimpleNamespace(data=reference, joint_names=canonical_names)
    asset = SimpleNamespace(
        joint_names=native_names,
        data=SimpleNamespace(default_joint_pos=default_native),
    )
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene={"motion_reference": motion, "robot": asset},
    )
    command = JointPositionReference(
        env,
        {"motion_reference": "motion_reference", "entity": "robot"},
    )

    expected_default = torch.tensor([[30.0, 10.0, 20.0], [31.0, 11.0, 21.0]])
    torch.testing.assert_close(
        command.command, reference.joint_pos - expected_default.unsqueeze(1)
    )

    # Main snapshots the nominal defaults before the startup randomization event changes them.
    asset.data.default_joint_pos.add_(1000.0)
    command.reset(torch.tensor([0, 1]))
    torch.testing.assert_close(
        command.command, reference.joint_pos - expected_default.unsqueeze(1)
    )


def test_joint_velocity_reference_subtracts_frozen_defaults_by_joint_name() -> None:
    """Velocity references follow main's relative-default semantics in canonical DFS order."""

    native_names = ("left_hip", "right_hip", "waist")
    canonical_names = ("waist", "left_hip", "right_hip")
    default_native = torch.tensor([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])
    reference = SimpleNamespace(
        joint_vel=torch.tensor(
            [
                [[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]],
                [[12.0, 22.0, 32.0], [13.0, 23.0, 33.0]],
            ]
        ),
        validity=torch.ones(2, 2),
    )
    motion = SimpleNamespace(data=reference, joint_names=canonical_names)
    asset = SimpleNamespace(
        joint_names=native_names,
        data=SimpleNamespace(default_joint_vel=default_native),
    )
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene={"motion_reference": motion, "robot": asset},
    )
    command = JointVelocityReference(
        env,
        {"motion_reference": "motion_reference", "entity": "robot"},
    )

    expected_default = torch.tensor([[3.0, 1.0, 2.0], [3.5, 1.5, 2.5]])
    torch.testing.assert_close(
        command.command, reference.joint_vel - expected_default.unsqueeze(1)
    )

    asset.data.default_joint_vel.add_(1000.0)
    command.reset(torch.tensor([0, 1]))
    torch.testing.assert_close(
        command.command, reference.joint_vel - expected_default.unsqueeze(1)
    )


def test_joint_reference_refresh_uses_the_reference_update_window() -> None:
    """Match the source command's one-microsecond boundary around a sensor refresh."""

    reference = SimpleNamespace(
        joint_vel=torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]),
        validity=torch.ones(2, 2),
    )
    motion = SimpleNamespace(
        data=reference,
        joint_names=("joint",),
        time_passed_from_update=torch.tensor([0.0199995, 0.0]),
    )
    asset = SimpleNamespace(
        joint_names=("joint",),
        data=SimpleNamespace(default_joint_vel=torch.zeros(2, 1)),
    )
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        step_dt=0.02,
        scene={"motion_reference": motion, "robot": asset},
    )
    command = JointVelocityReference(
        env,
        {
            "motion_reference": "motion_reference",
            "entity": "robot",
            "realtime_mode": False,
        },
    )
    before = command.command.clone()

    reference.joint_vel.add_(10.0)
    command._update_command()

    torch.testing.assert_close(command.command[0], before[0])
    torch.testing.assert_close(command.command[1], reference.joint_vel[1])


def test_mjcf_natural_joint_order_is_the_policy_dfs_order() -> None:
    task = _task("Instinct-Shadowing-WholeBody-Plane-G1-v0")
    root = ElementTree.parse(task.robot.asset_for("mjlab").path).getroot()
    natural = tuple(
        joint.attrib["name"]
        for joint in root.iter("joint")
        if "name" in joint.attrib and joint.attrib.get("type", "hinge") != "free"
    )
    assert natural == G1_29DOF_DFS_JOINT_NAMES


def test_shadowing_checkpoint_rejects_a_joint_order_drift(tmp_path) -> None:
    task = _task("Instinct-Shadowing-WholeBody-Plane-G1-v0")
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    (tmp_path / "manifest.json").write_text(json.dumps(add_task_contract({}, task)))
    changed = replace(
        task,
        robot=replace(task.robot, joint_names=tuple(reversed(task.robot.joint_names))),
    )

    with pytest.raises(ValueError, match="canonical joint order mismatch"):
        validate_checkpoint_contract(checkpoint, changed)


def test_mjlab_builtin_pd_matches_the_shared_plant_without_delay() -> None:
    pytest.importorskip("mjlab")
    from instinctlab_engine_mjlab.assets import entity

    task = _task("Instinct-Shadowing-WholeBody-Plane-G1-v0")
    actuators = entity(task.robot).articulation.actuators
    pd_actuators = tuple(
        cfg for cfg in actuators if type(cfg).__name__ == "BuiltinPdActuatorCfg"
    )
    by_joint = {}
    for cfg in pd_actuators:
        matched = {
            name
            for pattern in cfg.target_names_expr
            for name in G1_29DOF_DFS_JOINT_NAMES
            if re.fullmatch(pattern, name)
        }
        for name in matched:
            assert name not in by_joint
            by_joint[name] = (
                cfg.stiffness,
                cfg.damping,
                cfg.armature,
                cfg.effort_limit,
            )

    assert len(pd_actuators) == 7
    assert set(by_joint) == set(G1_29DOF_DFS_JOINT_NAMES)
    assert all(cfg.delay_min_lag == cfg.delay_max_lag == 0 for cfg in pd_actuators)
    for joint in task.robot.joint_properties:
        assert by_joint[joint.name] == pytest.approx(
            (joint.stiffness, joint.damping, joint.armature, joint.effort_limit)
        )
