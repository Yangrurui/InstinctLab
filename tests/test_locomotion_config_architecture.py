"""Structural guards for the robot-independent locomotion configuration boundary."""

from __future__ import annotations

import ast
from pathlib import Path

from tests.task_specs import task_spec

REPO = Path(__file__).resolve().parent.parent
CONFIG_ROOT = REPO / "source/instinctlab/instinctlab/tasks/locomotion/config"
BASE = CONFIG_ROOT / "locomotion_env_cfg.py"
FLAT = CONFIG_ROOT / "g1/flat_env_cfg.py"
ROUGH = CONFIG_ROOT / "g1/rough_env_cfg.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _class(path: Path, name: str) -> ast.ClassDef:
    for node in _tree(path).body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{path} does not declare {name}")


def _name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ast.unparse(node)


def test_concrete_env_configs_inherit_the_robot_independent_base_once() -> None:
    for path, class_name in (
        (FLAT, "G1LocomotionFlatEnvCfg"),
        (ROUGH, "G1LocomotionRoughEnvCfg"),
    ):
        declaration = _class(path, class_name)
        assert [_name(base) for base in declaration.bases] == ["LocomotionEnvCfg"]

    rough_imports = {
        node.module for node in _tree(ROUGH).body if isinstance(node, ast.ImportFrom)
    }
    assert "instinctlab.tasks.locomotion.config.g1.flat_env_cfg" not in rough_imports


def test_robot_independent_base_owns_no_concrete_selector() -> None:
    tree = _tree(BASE)
    calls = {_name(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "EntityRef" not in calls
    assert "ContactSensorRef" not in calls
    assert "RobotSpec" not in imports


def test_locomotion_configs_do_not_pass_or_cache_selector_aliases() -> None:
    forbidden_arguments = {
        "joints",
        "legs",
        "feet",
        "hip_joints",
        "upper_body_joints",
        "feet_contact",
    }
    selector_constructors = {"EntityRef", "ContactSensorRef"}

    for path in (BASE, FLAT, ROUGH):
        tree = _tree(path)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                arguments = {
                    argument.arg
                    for argument in (
                        *node.args.posonlyargs,
                        *node.args.args,
                        *node.args.kwonlyargs,
                    )
                }
                assert not arguments & forbidden_arguments, (
                    f"{path}:{node.lineno} passes selectors through "
                    f"{sorted(arguments & forbidden_arguments)}"
                )
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            if not isinstance(node.targets[0], ast.Name) or not isinstance(
                node.value, ast.Call
            ):
                continue
            constructor = _name(node.value.func)
            assert constructor not in selector_constructors, (
                f"{path}:{node.lineno} hides {constructor} behind "
                f"{node.targets[0].id!r}"
            )


def test_entity_refs_are_written_on_the_terms_that_consume_them() -> None:
    term_constructors = {
        "ActionTermSpec",
        "EventTermSpec",
        "ObsTermSpec",
        "RewardTermSpec",
    }
    for path in (FLAT, ROUGH):
        tree = _tree(path)
        parents: dict[ast.AST, ast.AST] = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        entity_refs = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _name(node.func) == "EntityRef"
        ]
        assert entity_refs
        for entity_ref in entity_refs:
            parent = parents[entity_ref]
            while not isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if (
                    isinstance(parent, ast.Call)
                    and _name(parent.func) in term_constructors
                ):
                    break
                parent = parents[parent]
            else:
                raise AssertionError(
                    f"{path}:{entity_ref.lineno} declares EntityRef away from its consuming term"
                )


def test_flat_and_rough_pin_complete_canonical_joint_and_body_axes() -> None:
    for task_id in ("Instinct-Velocity-Flat-G1", "Instinct-Velocity-Rough-G1"):
        task = task_spec(task_id)
        canonical_joints = tuple(task.robot.joint_names)
        canonical_bodies = tuple(task.robot.body_names)

        assert task.mdp.actions["joint_pos"].target.joints == canonical_joints
        for group in task.mdp.observations.values():
            for term_name in ("joint_pos", "joint_vel"):
                selector = group.terms[term_name].params["asset_cfg"]
                assert selector.joints == canonical_joints
                assert selector.preserve_order is True

        reset_target = task.mdp.events["reset_robot_joints"].target
        material_target = task.mdp.events["physics_material"].target
        assert reset_target is not None
        assert material_target is not None
        assert reset_target.joints == canonical_joints
        assert reset_target.preserve_order is True
        assert material_target.bodies == canonical_bodies
        assert material_target.preserve_order is True


def test_rough_preserves_flat_term_order_and_values() -> None:
    flat = task_spec("Instinct-Velocity-Flat-G1")
    rough = task_spec("Instinct-Velocity-Rough-G1")

    assert tuple(rough.mdp.observations) == tuple(flat.mdp.observations)
    assert tuple(rough.mdp.actions) == tuple(flat.mdp.actions)
    assert tuple(rough.mdp.commands) == tuple(flat.mdp.commands)
    assert tuple(rough.mdp.rewards["rewards"]) == tuple(flat.mdp.rewards["rewards"])
    assert {
        name: term.weight for name, term in rough.mdp.rewards["rewards"].items()
    } == {name: term.weight for name, term in flat.mdp.rewards["rewards"].items()}
    assert tuple(rough.mdp.terminations) == tuple(flat.mdp.terminations)
    assert tuple(rough.mdp.events) == tuple(flat.mdp.events)
