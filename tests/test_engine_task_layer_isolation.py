"""Architecture guards for the task/engine dependency boundary."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import torch

import instinctlab_engine as engines
from instinctlab_engine.registry import FAMILIES
from instinctlab_engine_isaacsim.terms import TERMS as ISAAC_TERMS
from instinctlab_engine_mjlab.terms import TERMS as MJLAB_TERMS
from instinctlab_engine.motion_reference import clip_frame, exhausted_envs
from instinctlab.tasks import registry
from tests.engine_packages import ENGINE_ROOTS as ENGINE_ROOTS_BY_NAME

ROOT = Path("source/instinctlab/instinctlab")
ENGINE_CORE_ROOT = Path(engines.__file__).parent
ENGINE_ROOTS = tuple(ENGINE_ROOTS_BY_NAME.values())
TASK_ROOT = ROOT / "tasks"
SHARED_ROOTS = (
    ROOT / "assets",
    ROOT / "tasks",
    ENGINE_CORE_ROOT / "bridge",
    ENGINE_CORE_ROOT / "motion_reference",
    ENGINE_CORE_ROOT / "spec",
)


def _imports(path: Path) -> tuple[str, ...]:
    names: list[str] = []
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return tuple(names)


def test_engine_core_owns_adapter_infrastructure() -> None:
    import instinctlab_engine

    backend_root_files = {path.name for path in (ROOT / "engines").glob("*.py")}
    core_root = Path(instinctlab_engine.__file__).parent
    core_files = {path.name for path in core_root.glob("*.py")}
    assert backend_root_files == set()
    assert {"__init__.py", "base.py", "compile.py", "registry.py"} <= core_files


def test_shared_packages_do_not_import_engine_implementations() -> None:
    offenders: list[str] = []
    for shared_root in SHARED_ROOTS:
        for path in shared_root.rglob("*.py"):
            forbidden = tuple(
                name
                for name in _imports(path)
                if name == "instinctlab_engine_isaacsim"
                or name.startswith("instinctlab_engine_isaacsim.")
                or name == "instinctlab_engine_mjlab"
                or name.startswith("instinctlab_engine_mjlab.")
            )
            if forbidden:
                offenders.append(f"{path}: {', '.join(forbidden)}")
    assert not offenders, f"shared packages import engine implementations: {sorted(offenders)}"


def test_engine_implementations_do_not_import_task_packages() -> None:
    offenders: list[str] = []
    for engine_root in ENGINE_ROOTS:
        for path in engine_root.rglob("*.py"):
            forbidden = tuple(
                name
                for name in _imports(path)
                if name == "instinctlab.tasks" or name.startswith("instinctlab.tasks.")
            )
            if forbidden:
                offenders.append(f"{path}: {', '.join(forbidden)}")
    assert not offenders, f"engine modules import task packages: {sorted(offenders)}"


def test_engine_implementations_do_not_import_playback_application() -> None:
    offenders: list[str] = []
    for engine_root in ENGINE_ROOTS:
        for path in engine_root.rglob("*.py"):
            forbidden = tuple(
                name
                for name in _imports(path)
                if name == "instinctlab.play" or name.startswith("instinctlab.play.")
            )
            if forbidden:
                offenders.append(f"{path}: {', '.join(forbidden)}")
    assert not offenders, f"engine modules import playback application: {sorted(offenders)}"


def test_backends_do_not_import_one_another() -> None:
    for engine, forbidden_engine in (("isaacsim", "mjlab"), ("mjlab", "isaacsim")):
        offenders: list[str] = []
        prefix = f"instinctlab_engine_{forbidden_engine}"
        for path in ENGINE_ROOTS_BY_NAME[engine].rglob("*.py"):
            forbidden = tuple(name for name in _imports(path) if name == prefix or name.startswith(prefix + "."))
            if forbidden:
                offenders.append(f"{path}: {', '.join(forbidden)}")
        assert not offenders, f"{engine} imports {forbidden_engine}: {sorted(offenders)}"


def test_sdk_imports_outside_engine_library_are_explicit_boundaries() -> None:
    sdk_roots = {"isaaclab", "isaacsim", "mjlab", "mujoco", "mujoco_warp", "omni", "pxr", "warp"}
    allowed = {
        ROOT / "assets" / "unitree_g1" / "isaacsim.py",
        ROOT / "assets" / "unitree_g1" / "mjlab.py",
        ROOT / "play" / "mjlab.py",
        ROOT / "play" / "viser.py",
    }
    offenders: list[str] = []
    for path in ROOT.rglob("*.py"):
        if path in allowed:
            continue
        forbidden = tuple(name for name in _imports(path) if name.split(".")[0] in sdk_roots)
        if forbidden:
            offenders.append(f"{path}: {', '.join(forbidden)}")
    assert not offenders, f"SDK imports leaked outside explicit native/application boundaries: {sorted(offenders)}"


def test_retired_runtime_packages_are_absent() -> None:
    retired = (
        ROOT / "envs",
        ROOT / "managers",
        ROOT / "monitors",
        ENGINE_ROOTS_BY_NAME["isaacsim"] / "legacy_motion_reference",
    )
    assert not [str(path) for path in retired if path.exists()]


def test_scene_modules_do_not_dispatch_concrete_terrain_kinds() -> None:
    concrete_kinds = (
        "motion_matched",
        "pyramid_stairs",
        "perlin_plane",
        "random_rough",
    )
    for engine_root in ENGINE_ROOTS:
        source = (engine_root / "scene.py").read_text()
        assert not any(kind in source for kind in concrete_kinds), engine_root


def test_isaac_engine_does_not_import_the_legacy_global_mdp_catalog() -> None:
    offenders: list[str] = []
    for path in ENGINE_ROOTS_BY_NAME["isaacsim"].rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = (node.module,)
            else:
                continue
            if any(name == "instinctlab.envs.mdp" or name.startswith("instinctlab.envs.mdp.") for name in names):
                offenders.append(str(path))
                break
    assert not offenders, f"Isaac engine imports the legacy global MDP catalog: {sorted(offenders)}"


def test_task_modules_do_not_import_engine_implementations() -> None:
    offenders: list[str] = []
    for path in TASK_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = (node.module,)
            else:
                continue
            forbidden = tuple(
                name
                for name in names
                if name == "instinctlab_engine_isaacsim"
                or name.startswith("instinctlab_engine_isaacsim.")
                or name == "instinctlab_engine_mjlab"
                or name.startswith("instinctlab_engine_mjlab.")
            )
            if forbidden:
                offenders.append(f"{path}: {', '.join(forbidden)}")
                break
    assert not offenders, f"task modules import engine implementations: {sorted(offenders)}"


def test_engine_registries_contain_only_generic_capability_names() -> None:
    task_vocabulary = ("shadow", "parkour", "perceptive", "mimic", "locomotion")
    for terms in (ISAAC_TERMS, MJLAB_TERMS):
        for family in FAMILIES:
            for kind in terms.kinds(family):
                assert not any(word in kind.lower() for word in task_vocabulary), (
                    f"{terms.engine} registers task-specific {family}/{kind}"
                )


def test_reward_and_termination_formulas_are_not_registered_by_engines() -> None:
    for terms in (ISAAC_TERMS, MJLAB_TERMS):
        assert terms.kinds("reward") == set()
        assert terms.kinds("termination") == set()

    assert not (ENGINE_ROOTS_BY_NAME["mjlab"] / "rewards.py").exists()


def test_task_owned_native_choices_are_explicit() -> None:
    required_params = {
        ("actions", "joint_position"): {"scale", "use_default_offset"},
        ("commands", "uniform_velocity"): {
            "entity",
            "resampling_time_range",
            "rel_standing_envs",
            "rel_heading_envs",
            "heading_command",
            "heading_control_stiffness",
            "debug_vis",
            "lin_vel_x",
            "lin_vel_y",
            "ang_vel_z",
            "heading",
        },
    }
    task_function_params = {
        "contact_slide": {"sensor_cfg", "asset_cfg", "threshold"},
        "undesired_contacts_by_force": {"sensor", "threshold"},
        "motors_power_square": {"asset_cfg", "normalize_by_stiffness"},
        "applied_torque_limits_by_ratio": {"asset_cfg", "limit_ratio"},
        "illegal_contact_by_force": {"sensor", "threshold"},
    }
    target_required = {
        ("events", "randomize_friction"),
        ("events", "randomize_body_mass"),
        ("events", "apply_external_force_torque"),
        ("events", "reset_root_state_uniform"),
        ("events", "reset_joints_by_scale"),
        ("events", "reset_joints_by_offset"),
        ("events", "push_by_setting_velocity"),
    }
    motion_commands = {
        "motion_reference_position",
        "motion_reference_rotation",
        "motion_reference_joint_position",
        "motion_reference_joint_velocity",
    }
    adapter = engines.adapter("mjlab")
    for task_id in registry.ids():
        task = registry.spec(task_id, adapter.robot_spec(registry.asset_id(task_id)))
        terms = list(_flat_terms(task))
        for family, name, term in terms:
            key = (family, term.kind)
            missing = required_params.get(key, set()) - set(term.params)
            assert not missing, f"{task_id} {family}/{name} omits task params {sorted(missing)}"
            function_name = getattr(term.func, "__name__", "")
            missing = task_function_params.get(function_name, set()) - set(term.params)
            assert not missing, f"{task_id} {family}/{name} omits task params {sorted(missing)}"
            if key in target_required:
                assert term.target is not None, f"{task_id} {family}/{name} relies on a default entity"
            if term.kind in motion_commands:
                missing = {"motion_reference", "entity", "current_state_command"} - set(term.params)
                assert not missing, f"{task_id} {family}/{name} omits command params {sorted(missing)}"
            if term.kind == "randomize_friction":
                for engine_name in task.engines:
                    effective = dict(term.params)
                    effective.update(term.engine_params.get(engine_name, {}))
                    if engine_name == "isaacsim":
                        required = {"static_friction_range", "dynamic_friction_range"}
                        missing = required - set(effective)
                    else:
                        has_native = "ranges" in effective
                        has_shared = {
                            "static_friction_range",
                            "dynamic_friction_range",
                        } <= set(effective)
                        missing = set() if has_native or has_shared else {"ranges"}
                    assert not missing, (
                        f"{task_id} friction omits {engine_name} params {sorted(missing)}"
                    )

        for sensor in task.scene.ray_casters:
            if sensor.pattern.kind != "pinhole":
                continue
            camera = task.sim.profiles.get("mjlab", {}).get("pinhole_cameras", {}).get(sensor.name, {})
            missing = {
                "include_geom_groups",
                "exclude_parent_body",
                "mesh_filter_max_hops",
                "mesh_filter_epsilon",
                "update_period",
            } - set(camera)
            assert not missing, f"{task_id} camera {sensor.name!r} omits MJLab settings {sorted(missing)}"


def _flat_terms(task):
    for family in ("actions", "commands", "terminations", "events", "curriculum"):
        collection = getattr(task.mdp, family)
        terms = collection.terms if hasattr(collection, "terms") else collection
        for name, term in terms.items():
            yield family, name, term
    for family in ("observations", "rewards"):
        for group_name, group in getattr(task.mdp, family).items():
            terms = group.terms if hasattr(group, "terms") else group
            for name, term in terms.items():
                yield family, f"{group_name}/{name}", term


def test_task_callables_compile_through_portable_families() -> None:
    portable_families = ("observations", "rewards", "terminations", "events", "curriculum")
    for engine_name in engines.names():
        adapter = engines.adapter(engine_name)
        for task_id in registry.ids():
            robot = adapter.robot_spec(registry.asset_id(task_id))
            task = registry.spec(task_id, robot)
            for family in portable_families:
                groups = getattr(task.mdp, family)
                collections = groups.values() if family in {"observations", "rewards"} else (groups,)
                for collection in collections:
                    terms = collection.terms if hasattr(collection, "terms") else collection
                    for term in terms.values():
                        module = getattr(term.func, "__module__", "")
                        if module.startswith("instinctlab.tasks."):
                            assert term.kind is None, (
                                f"{task_id} {family} routes task callable {module} "
                                f"through engine kind {term.kind!r}"
                            )


def test_motion_reference_reads_are_engine_neutral() -> None:
    buffers = SimpleNamespace(
        base_quat_w=torch.arange(16).reshape(2, 2, 4),
        base_lin_vel_w=torch.arange(12).reshape(2, 2, 3),
        base_ang_vel_w=torch.arange(12, 24).reshape(2, 2, 3),
        joint_pos=torch.arange(8).reshape(2, 2, 2),
        joint_vel=torch.arange(8, 16).reshape(2, 2, 2),
        validity=torch.tensor([[True, False], [False, True]]),
    )

    quat, lin_vel, ang_vel, joint_pos, joint_vel = clip_frame(buffers, frame=1)
    torch.testing.assert_close(quat, buffers.base_quat_w[:, 1])
    torch.testing.assert_close(lin_vel, buffers.base_lin_vel_w[:, 1])
    torch.testing.assert_close(ang_vel, buffers.base_ang_vel_w[:, 1])
    torch.testing.assert_close(joint_pos, buffers.joint_pos[:, 1])
    torch.testing.assert_close(joint_vel, buffers.joint_vel[:, 1])
    torch.testing.assert_close(
        exhausted_envs(buffers, torch.tensor([1, 0])),
        torch.ones(2, dtype=torch.bool),
    )
