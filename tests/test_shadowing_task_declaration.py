"""Structure and contract tests for the engine-neutral shadowing declarations."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from instinctlab.tasks import registry
from tests.test_shadowing_reference_inventory import COMMON_IDS, MJ_ONLY_IDS

SHADOW_IDS = COMMON_IDS | MJ_ONLY_IDS
ROOT = Path("source/instinctlab/instinctlab/tasks/shadowing")


@pytest.mark.parametrize("task_id", sorted(SHADOW_IDS))
def test_every_reference_id_resolves_to_one_valid_shared_spec(task_id: str) -> None:
    task = registry.spec(task_id)
    assert task.task_id == task_id
    assert task.engines == ("isaacsim", "mjlab")
    assert tuple(task.mdp.actions) == ("joint_pos",)
    target = task.mdp.actions["joint_pos"].target
    assert target is not None
    assert target.preserve_order is True
    assert tuple(target.joints) == task.robot.joint_names


def test_registry_contains_exactly_the_reference_shadowing_surface() -> None:
    registered = {
        task_id for task_id in registry.ids() if any(word in task_id for word in ("Shadowing", "Mimic", "Vae"))
    }
    assert registered == SHADOW_IDS


def test_declaration_order_matches_effective_whole_body_factory() -> None:
    task = registry.spec("Instinct-Shadowing-WholeBody-Plane-G1-v0")
    assert tuple(task.mdp.commands) == (
        "position_ref_command",
        "position_b_ref_command",
        "rotation_ref_command",
        "joint_pos_ref_command",
        "joint_vel_ref_command",
    )
    assert tuple(task.mdp.observations["policy"].terms) == (
        "joint_pos_ref",
        "joint_vel_ref",
        "position_ref",
        "rotation_ref",
        "projected_gravity",
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "last_action",
    )
    assert tuple(task.mdp.rewards["rewards"]) == (
        "base_position_imitation_gauss",
        "base_rot_imitation_gauss",
        "link_pos_imitation_gauss",
        "link_rot_imitation_gauss",
        "link_lin_vel_imitation_gauss",
        "link_ang_vel_imitation_gauss",
        "action_rate_l2",
        "joint_limit",
        "undesired_contacts",
    )


def test_play_specs_are_explicit_contracts_not_gym_aliases() -> None:
    for task_id in sorted(SHADOW_IDS):
        task = registry.spec(task_id)
        play = "-Play-v0" in task_id
        assert task.engine_extras["isaacsim"]["play"] is play
        assert task.engine_extras["mjlab"]["play"] is play
        assert task.scene.env_spacing == (2.5 if play else 4.0)
        assert ("beyond_adaptive_sampling" in task.mdp.curriculum) is not play


def test_no_shadowing_module_registers_gym_or_imports_an_engine() -> None:
    engine_roots = {"isaaclab", "isaacsim", "mjlab", "mujoco", "omni", "pxr", "carb"}
    for path in ROOT.rglob("*.py"):
        if "agents" in path.parts:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module.split(".")[0])
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "register":
                pytest.fail(f"legacy Gym registration remains in {path}")
        assert not imported & engine_roots, f"{path} imports {sorted(imported & engine_roots)}"


def test_isaac_only_shadowing_surface_and_duplicate_mdp_are_removed() -> None:
    removed = (
        "play.py",
        "cli_args.py",
        "grid_search.sh",
        "mdp",
        "whole_body/shadowing_env_cfg.py",
        "perceptive/perceptive_env_cfg.py",
        "perceptive_hoi/perceptive_env_cfg.py",
        "beyondmimic/beyondmimic_env_cfg.py",
    )
    assert not [name for name in removed if (ROOT / name).is_file()]
    assert not list((ROOT / "mdp").glob("*.py"))
