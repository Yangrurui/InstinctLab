"""Structure and contract tests for the engine-neutral shadowing declarations."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from instinctlab.tasks import registry
from tests.test_shadowing_reference_inventory import COMMON_IDS, MJ_ONLY_IDS

SHADOW_IDS = COMMON_IDS | MJ_ONLY_IDS
ROOT = Path("source/instinctlab/instinctlab/tasks/shadowing")
MAIN_RESET_SOURCE = Path(
    "/root/InstinctLab-main/source/instinctlab/instinctlab/envs/mdp/events/motion_reference.py"
)
MJ_RESET_SOURCE = Path(
    "/root/InstinctMJ/src/instinct_mj/envs/mdp/events/motion_reference.py"
)


def _function_source(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(), filename=str(path))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    return ast.unparse(function)


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
    reset = task.mdp.events["reset_robot"]
    assert reset.resolved_params("isaacsim")["root_velocity_frame"] == "com"
    assert reset.resolved_params("mjlab").get("root_velocity_frame", "link") == "link"


def test_root_reset_velocity_point_tracks_each_reference_backend() -> None:
    main = _function_source(MAIN_RESET_SOURCE, "reset_robot_state_by_reference")
    mjlab = _function_source(MJ_RESET_SOURCE, "reset_robot_state_by_reference")
    assert "asset.write_root_velocity_to_sim" in main
    assert "asset.write_root_link_velocity_to_sim" not in main
    assert "asset.write_root_link_velocity_to_sim" in mjlab


def test_shadowing_physx_budgets_match_effective_main_configs() -> None:
    for task_id in sorted(SHADOW_IDS):
        profile = registry.spec(task_id).sim.profiles["isaacsim"]
        assert profile["use_terrain_physics_material"] is True
        assert profile["gpu_max_rigid_patch_count"] == 10 * 2**15
        if any(name in task_id for name in ("Perceptive", "Vae")):
            assert profile["gpu_max_rigid_contact_count"] == 2**27
            assert profile["gpu_collision_stack_size"] == 2**27
        else:
            assert "gpu_max_rigid_contact_count" not in profile
            assert "gpu_collision_stack_size" not in profile


def test_shadowing_isaac_merges_fixed_joints_like_effective_main() -> None:
    """The task overrides the catalog flag without changing MJLab's asset."""
    task = registry.spec("Instinct-Shadowing-WholeBody-Plane-G1-v0")
    assert task.robot.asset_for("isaacsim").import_options["merge_fixed_joints"] is True
    assert task.robot.asset_for("mjlab").path.endswith("g1_29dof_torsobase_popsicle.xml")


def test_shadowing_mjlab_budgets_match_effective_instinctmj_configs() -> None:
    expected = {
        "whole_body": (None, 1200, None, 500, None),
        "perceptive": (128, 700, 128, 128, "sparse"),
        "perceptive_vae": (128, 512, 128, 128, "sparse"),
        "perceptive_hoi": (256, 700, 256, 128, "sparse"),
        "beyondmimic": (100, 350, 100, 80, None),
    }
    for task_id in sorted(SHADOW_IDS):
        task = registry.spec(task_id)
        family = task.engine_extras["mjlab"]["shadowing_family"]
        nconmax, njmax, maxmatch, ccd, jacobian = expected[family]
        if family == "beyondmimic" and task.engine_extras["mjlab"]["play"]:
            nconmax, njmax, maxmatch = None, None, 500
        profile = task.sim.profiles["mjlab"]
        assert profile["nconmax"] == nconmax
        assert profile["njmax"] == njmax
        assert profile.get("contact_sensor_maxmatch") == maxmatch
        assert profile.get("ccd_iterations", 500) == ccd
        assert profile.get("jacobian") == jacobian


@pytest.mark.parametrize(
    ("task_id", "expected"),
    (
        ("Instinct-Shadowing-WholeBody-Plane-G1-v0", (None, 1200, 64, 500, "auto")),
        ("Instinct-Perceptive-Shadowing-G1-v0", (128, 700, 128, 128, "sparse")),
        ("Instinct-Perceptive-Vae-G1-v0", (128, 512, 128, 128, "sparse")),
        ("Instinct-Perceptive-HOI-Shadowing-G1-v0", (256, 700, 256, 128, "sparse")),
        ("Instinct-BeyondMimic-Plane-G1-v0", (100, 350, 100, 80, "auto")),
    ),
)
def test_compiled_mjlab_shadowing_capacities_match_instinctmj(task_id: str, expected: tuple) -> None:
    pytest.importorskip("mjlab")
    from instinctlab.engines.mjlab import MjlabAdapter

    cfg = MjlabAdapter().compile(registry.spec(task_id), num_envs=2, device="cpu").env_cfg
    actual = (
        cfg.sim.nconmax,
        cfg.sim.njmax,
        cfg.sim.contact_sensor_maxmatch,
        cfg.sim.mujoco.ccd_iterations,
        cfg.sim.mujoco.jacobian,
    )
    assert actual == expected


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
