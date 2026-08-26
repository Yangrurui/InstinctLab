from __future__ import annotations

import json
import torch
from dataclasses import replace
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest

from instinctlab.assets.unitree_g1.isaacsim import G1_29DOF_DFS_JOINT_NAMES, G1_29DOF_ISAAC_BFS_JOINT_NAMES
from instinctlab.checkpoint import add_task_contract, validate_checkpoint_contract
from instinctlab.engines.shadowing_events import reset_robot_from_reference
from instinctlab.sim.backend import CanonicalIndexMap
from instinctlab.tasks import registry

SHADOWING_IDS = tuple(task_id for task_id in registry.ids() if "Shadowing" in task_id or "BeyondMimic" in task_id)


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
        pass

    def write_root_link_velocity_to_sim(self, velocity, env_ids=None):
        self.root_velocity_writer = "link"

    def write_root_com_velocity_to_sim(self, velocity, env_ids=None):
        self.root_velocity_writer = "com"

    def write_joint_state_to_sim(self, position, velocity, joint_ids=None, env_ids=None):
        self.joint_pos[env_ids[:, None], joint_ids[None, :]] = position
        self.joint_vel[env_ids[:, None], joint_ids[None, :]] = velocity


def _reset_env(canonical_joint_names: tuple[str, ...], native_joint_names: tuple[str, ...]):
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
    sensor = SimpleNamespace(init_reference_state=state, joint_names=canonical_joint_names)
    env = SimpleNamespace(device="cpu", scene={"robot": asset, "motion_reference": sensor})
    reset_robot_from_reference(env, torch.tensor([0]), randomize_joint_pos_range=(0.0, 0.0))
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
    env = SimpleNamespace(device="cpu", scene={"robot": asset, "motion_reference": sensor})
    reset_robot_from_reference(
        env,
        torch.tensor([0]),
        randomize_joint_pos_range=(0.0, 0.0),
        root_velocity_frame="com",
    )
    assert asset.root_velocity_writer == "com"


def test_shadowing_reset_rejects_an_incomplete_joint_mapping() -> None:
    with pytest.raises(ValueError, match="do not resolve one-to-one"):
        _reset_env(("waist_joint", "missing_joint", "right_joint"), ("right_joint", "waist_joint"))


@pytest.mark.parametrize("task_id", SHADOWING_IDS)
def test_every_shadowing_tensor_uses_the_canonical_joint_order(task_id: str) -> None:
    task = registry.spec(task_id)
    action = task.mdp.actions["joint_pos"]

    assert task.robot.joint_names == G1_29DOF_DFS_JOINT_NAMES
    assert task.robot.actuator_delay == (0, 0)
    assert action.target is not None
    assert action.target.joints == G1_29DOF_DFS_JOINT_NAMES
    assert action.target.preserve_order is True
    assert tuple(action.params["scale"]) == G1_29DOF_DFS_JOINT_NAMES
    assert tuple(action.params["scale"].values()) == tuple(joint.action_scale for joint in task.robot.joint_properties)
    assert task.scene.motion_references[0].joints == G1_29DOF_DFS_JOINT_NAMES

    for group in task.mdp.observations.values():
        for name in ("joint_pos", "joint_vel"):
            term = group.terms.get(name)
            if term is not None:
                selector = term.params["asset_cfg"]
                assert selector.joints == G1_29DOF_DFS_JOINT_NAMES
                assert selector.preserve_order is True


def test_isaac_bfs_to_policy_dfs_mapping_is_name_based_and_reversible() -> None:
    mapping = CanonicalIndexMap.build(G1_29DOF_DFS_JOINT_NAMES, G1_29DOF_ISAAC_BFS_JOINT_NAMES, device="cpu")
    native = torch.arange(len(G1_29DOF_ISAAC_BFS_JOINT_NAMES), dtype=torch.float32)
    canonical = mapping.to_canonical(native)
    rebuilt = torch.empty_like(native)
    mapping.copy_to_native(rebuilt, canonical)

    assert not mapping.is_identity
    torch.testing.assert_close(rebuilt, native)
    for name in ("waist_pitch_joint", "left_hip_pitch_joint", "right_wrist_yaw_joint"):
        assert canonical[G1_29DOF_DFS_JOINT_NAMES.index(name)] == G1_29DOF_ISAAC_BFS_JOINT_NAMES.index(name)


def test_mjcf_natural_joint_order_is_the_policy_dfs_order() -> None:
    task = registry.spec("Instinct-Shadowing-WholeBody-Plane-G1-v0")
    root = ElementTree.parse(task.robot.asset_for("mjlab").path).getroot()
    natural = tuple(
        joint.attrib["name"]
        for joint in root.iter("joint")
        if "name" in joint.attrib and joint.attrib.get("type", "hinge") != "free"
    )
    assert natural == G1_29DOF_DFS_JOINT_NAMES


def test_shadowing_checkpoint_does_not_reject_a_joint_order_hash_drift(tmp_path) -> None:
    task = registry.spec("Instinct-Shadowing-WholeBody-Plane-G1-v0")
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    (tmp_path / "manifest.json").write_text(json.dumps(add_task_contract({}, task)))
    changed = replace(task, robot=replace(task.robot, joint_names=tuple(reversed(task.robot.joint_names))))

    validate_checkpoint_contract(checkpoint, changed)


def test_mjlab_builtin_pd_matches_the_shared_plant_without_delay() -> None:
    pytest.importorskip("mjlab")
    from instinctlab.engines.mjlab.assets import entity

    task = registry.spec("Instinct-Shadowing-WholeBody-Plane-G1-v0")
    actuators = entity(task.robot).articulation.actuators
    pd_actuators = tuple(cfg for cfg in actuators if type(cfg).__name__ == "BuiltinPdActuatorCfg")
    by_joint = {
        name: (cfg.stiffness, cfg.damping, cfg.armature, cfg.effort_limit)
        for cfg in pd_actuators
        for name in cfg.target_names_expr
    }

    assert len(pd_actuators) == 7
    assert set(by_joint) == set(G1_29DOF_DFS_JOINT_NAMES)
    assert all(cfg.delay_min_lag == cfg.delay_max_lag == 0 for cfg in pd_actuators)
    for joint in task.robot.joint_properties:
        assert by_joint[joint.name] == pytest.approx(
            (joint.stiffness, joint.damping, joint.armature, joint.effort_limit)
        )
