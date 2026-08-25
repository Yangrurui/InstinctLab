from __future__ import annotations

import json
import torch
from dataclasses import replace
from xml.etree import ElementTree

import pytest

from instinctlab.assets.unitree_g1.isaacsim import G1_29DOF_DFS_JOINT_NAMES, G1_29DOF_ISAAC_BFS_JOINT_NAMES
from instinctlab.checkpoint import add_task_contract, validate_checkpoint_contract
from instinctlab.sim.backend import CanonicalIndexMap
from instinctlab.tasks import registry

SHADOWING_IDS = tuple(task_id for task_id in registry.ids() if "Shadowing" in task_id or "BeyondMimic" in task_id)


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


def test_shadowing_checkpoint_rejects_a_joint_order_change(tmp_path) -> None:
    task = registry.spec("Instinct-Shadowing-WholeBody-Plane-G1-v0")
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    (tmp_path / "manifest.json").write_text(json.dumps(add_task_contract({}, task)))
    changed = replace(task, robot=replace(task.robot, joint_names=tuple(reversed(task.robot.joint_names))))

    with pytest.raises(ValueError, match="Checkpoint task contract mismatch"):
        validate_checkpoint_contract(checkpoint, changed)


def test_mjlab_builtin_pd_matches_the_shared_plant_without_delay() -> None:
    pytest.importorskip("mjlab")
    from instinctlab.engines.mjlab.assets import entity

    task = registry.spec("Instinct-Shadowing-WholeBody-Plane-G1-v0")
    actuators = entity(task.robot).articulation.actuators
    by_joint = {
        name: (cfg.stiffness, cfg.damping, cfg.armature, cfg.effort_limit)
        for cfg in actuators
        for name in cfg.target_names_expr
    }

    assert len(actuators) == 7
    assert set(by_joint) == set(G1_29DOF_DFS_JOINT_NAMES)
    assert all(cfg.delay_min_lag == cfg.delay_max_lag == 0 for cfg in actuators)
    for joint in task.robot.joint_properties:
        assert by_joint[joint.name] == pytest.approx(
            (joint.stiffness, joint.damping, joint.armature, joint.effort_limit)
        )
