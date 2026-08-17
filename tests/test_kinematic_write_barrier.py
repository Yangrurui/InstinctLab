"""Guard: Manager / MDP must not assign ArticulationState kinematic fields."""

from __future__ import annotations

import ast
import torch
from pathlib import Path

import pytest

from instinctlab.sim.state import KINEMATIC_FIELD_NAMES, ArticulationState, freeze_kinematic_fields

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCAN_PATHS = (
    _REPO_ROOT / "source/instinctlab/instinctlab/tasks/locomotion/mdp/unified.py",
    _REPO_ROOT / "source/instinctlab/instinctlab/managers/unified.py",
)


def _assignment_target_name(node: ast.AST) -> str | None:
    target = node
    while isinstance(target, ast.Subscript):
        target = target.value
    if isinstance(target, ast.Attribute) and target.attr in KINEMATIC_FIELD_NAMES:
        return target.attr
    return None


def test_mdp_and_managers_do_not_assign_kinematic_fields() -> None:
    hits: list[str] = []
    for path in _SCAN_PATHS:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            targets: list[ast.AST] = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign) and node.target is not None:
                targets = [node.target]
            elif isinstance(node, ast.AugAssign):
                targets = [node.target]
            for target in targets:
                name = _assignment_target_name(target)
                if name is not None:
                    hits.append(f"{path}:{node.lineno} assigns {name}")
    assert not hits, "Manager/MDP must use backend.write_*; found:\n" + "\n".join(hits)


def test_write_barrier_blocks_item_assignment() -> None:
    state = ArticulationState.allocate(num_envs=2, num_joints=3, num_bodies=2, device="cpu")
    freeze_kinematic_fields(state)
    with pytest.raises(RuntimeError, match="backend.write_"):
        state.joint_pos[0] = 1.0
    with pytest.raises(RuntimeError, match="backend.write_"):
        state.body_pos_w[:, 0] = 0.0
    value = state.joint_pos[0, 0].item()
    assert value == 0.0
