"""Structure and contract tests for the engine-neutral shadowing declarations."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from instinctlab import engines
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


def _task(task_id: str, engine: str = "mjlab"):
    adapter = engines.adapter(engine)
    robot = adapter.robot_spec(registry.asset_id(task_id))
    return registry.spec(task_id, robot)


def _function_source(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return ast.unparse(function)


@pytest.mark.parametrize("task_id", sorted(SHADOW_IDS))
def test_every_reference_id_resolves_to_one_valid_shared_spec(task_id: str) -> None:
    task = _task(task_id)
    assert task.task_id == task_id
    assert task.engines == ("isaacsim", "mjlab")
    assert tuple(task.mdp.actions) == ("joint_pos",)
    target = task.mdp.actions["joint_pos"].target
    assert target is not None
    assert target.preserve_order is True
    assert tuple(target.joints) == task.robot.joint_names


def test_registry_contains_exactly_the_reference_shadowing_surface() -> None:
    registered = {
        task_id
        for task_id in registry.ids()
        if any(word in task_id for word in ("Shadowing", "Mimic", "Vae"))
    }
    assert registered == SHADOW_IDS


def test_declaration_order_matches_effective_whole_body_factory() -> None:
    task = _task("Instinct-Shadowing-WholeBody-Plane-G1-v0")
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
        profile = _task(task_id).sim.profiles["isaacsim"]
        assert profile["use_terrain_physics_material"] is True
        assert profile["gpu_max_rigid_patch_count"] == 10 * 2**15
        if any(name in task_id for name in ("Perceptive", "Vae")):
            assert profile["gpu_max_rigid_contact_count"] == 2**27
            assert profile["gpu_collision_stack_size"] == 2**27
        else:
            assert "gpu_max_rigid_contact_count" not in profile
            assert "gpu_collision_stack_size" not in profile


def test_shadowing_isaac_merges_fixed_joints_like_effective_main() -> None:
    """Each materialized task carries only its selected engine's native asset."""
    task_id = "Instinct-Shadowing-WholeBody-Plane-G1-v0"
    isaac = _task(task_id, "isaacsim")
    mjlab = _task(task_id, "mjlab")
    assert (
        isaac.robot.asset_for("isaacsim").import_options["merge_fixed_joints"] is True
    )
    assert mjlab.robot.asset_for("mjlab").path.endswith(
        "g1_29dof_torsobase_popsicle.xml"
    )


def test_shadowing_mjlab_budgets_match_reference_or_validated_safe_configs() -> None:
    expected = {
        "Instinct-Shadowing-WholeBody-Plane-G1-v0": (None, 1200, None, 500, None),
        "Instinct-Shadowing-WholeBody-Plane-G1-Play-v0": (None, 1200, None, 500, None),
        "Instinct-Perceptive-Shadowing-G1-v0": (None, 1200, None, 500, None),
        "Instinct-Perceptive-Shadowing-G1-Play-v0": (None, 1200, None, 500, None),
        "Instinct-Perceptive-Shadowing-G1-OneMotion-v0": (None, 1200, None, 500, None),
        "Instinct-Perceptive-Shadowing-G1-OneMotion-Play-v0": (
            None,
            1200,
            None,
            500,
            None,
        ),
        "Instinct-Perceptive-Vae-G1-v0": (128, 512, 128, 128, "sparse"),
        "Instinct-Perceptive-Vae-G1-Play-v0": (128, 512, 128, 128, "sparse"),
        "Instinct-Perceptive-HOI-Shadowing-G1-v0": (256, 700, 256, 128, "sparse"),
        "Instinct-Perceptive-HOI-Shadowing-G1-Play-v0": (256, 700, 256, 128, "sparse"),
        "Instinct-BeyondMimic-Plane-G1-v0": (100, 350, 100, 80, None),
        "Instinct-BeyondMimic-Plane-G1-Play-v0": (None, None, 500, 80, None),
    }
    for task_id in sorted(SHADOW_IDS):
        task = _task(task_id)
        nconmax, njmax, maxmatch, ccd, jacobian = expected[task_id]
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
        ("Instinct-Perceptive-Shadowing-G1-v0", (None, 1200, 64, 500, "auto")),
        ("Instinct-Perceptive-Vae-G1-v0", (128, 512, 128, 128, "sparse")),
        ("Instinct-Perceptive-HOI-Shadowing-G1-v0", (256, 700, 256, 128, "sparse")),
        ("Instinct-BeyondMimic-Plane-G1-v0", (100, 350, 100, 80, "auto")),
    ),
)
def test_compiled_mjlab_shadowing_capacities_match_instinctmj(
    task_id: str, expected: tuple
) -> None:
    pytest.importorskip("mjlab")
    from instinctlab.engines.mjlab import MjlabAdapter

    cfg = MjlabAdapter().compile(_task(task_id), num_envs=2, device="cpu").env_cfg
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
        task = _task(task_id)
        play = "-Play-v0" in task_id
        assert task.scene.env_spacing == (2.5 if play else 4.0)
        adaptive_sampling = (
            not play and task.scene.motion_references[0].motion_bin_length_s is not None
        )
        assert ("beyond_adaptive_sampling" in task.mdp.curriculum) is adaptive_sampling


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
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "register"
            ):
                pytest.fail(f"legacy Gym registration remains in {path}")
        assert not imported & engine_roots, (
            f"{path} imports {sorted(imported & engine_roots)}"
        )


def test_shadowing_owns_its_mdp_and_the_global_catalog_is_removed() -> None:
    removed = (
        "play.py",
        "cli_args.py",
        "grid_search.sh",
    )
    assert not [name for name in removed if (ROOT / name).is_file()]
    assert {path.name for path in (ROOT / "mdp").glob("*.py")} == {
        "__init__.py",
        "commands.py",
        "events.py",
        "observations.py",
        "rewards.py",
        "terminations.py",
        "terms.py",
    }
    assert not list(Path("source/instinctlab/instinctlab/mdp").glob("*.py"))


def test_shadowing_env_files_follow_the_reference_layout() -> None:
    expected = (
        "whole_body/shadowing_env_cfg.py",
        "perceptive/perceptive_env_cfg.py",
        "perceptive_hoi/perceptive_env_cfg.py",
        "beyondmimic/beyondmimic_env_cfg.py",
    )
    assert all((ROOT / name).is_file() for name in expected)


def test_shadowing_configs_have_no_dispatcher_or_task_spec_conversion_method() -> None:
    for path in ROOT.rglob("*.py"):
        if "agents" in path.parts:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert not node.name.startswith("make_")
                assert node.name != "to_task_spec"


def test_shadowing_term_configs_do_not_hide_selectors_behind_local_aliases() -> None:
    forbidden_constructors = {"EntityRef", "ContactSensorRef"}
    for path in ROOT.rglob("*_cfg.py"):
        if "agents" in path.parts:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            if not isinstance(node.targets[0], ast.Name) or not isinstance(
                node.value, ast.Call
            ):
                continue
            constructor = getattr(node.value.func, "id", None)
            assert constructor not in forbidden_constructors, (
                f"{path}:{node.lineno} hides {constructor} behind {node.targets[0].id!r}"
            )
