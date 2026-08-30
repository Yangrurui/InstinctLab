"""Static URDF/MJCF and PD-gain parity for the unified G1 asset."""

from __future__ import annotations

import ast
import inspect
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from instinctlab.assets.unitree_g1.isaacsim import (
    G1_29DOF_DEFAULT_JOINT_POS,
    G1_29DOF_DFS_JOINT_NAMES,
    G1_29DOF_ISAAC_BFS_JOINT_NAMES,
    g1_symmetric_joint_augmentation,
)
from tests.g1_specs import paired_robot_spec

G1_ASSET_ID = "unitree_g1/popsicle_torsobase_v1"


def make_g1_29dof_robot_spec():
    """Test view of both independently owned native asset declarations."""
    return paired_robot_spec(G1_ASSET_ID)

_RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "source/instinctlab/instinctlab/assets/resources/unitree_g1"
_URDF_PATH = _RESOURCE_ROOT / "urdf" / "g1_29dof_torsobase_popsicle.urdf"
_MJCF_PATH = _RESOURCE_ROOT / "xml" / "g1_29dof_torsobase_popsicle.xml"
_MAIN_ASSET_CFG = Path("/root/InstinctLab-main/source/instinctlab/instinctlab/assets/unitree_g1.py")
_MJLAB_ASSET_CFG = Path("/root/InstinctMJ/src/instinct_mj/assets/unitree_g1.py")


def _floats(text: str | None) -> tuple[float, ...]:
    if not text:
        return ()
    return tuple(float(part) for part in text.split())


def _parse_urdf(path: Path) -> tuple[dict[str, tuple[float, tuple[float, float, float]]], dict[str, tuple[float, ...]]]:
    root = ET.parse(path).getroot()
    bodies: dict[str, tuple[float, tuple[float, float, float]]] = {}
    joints: dict[str, tuple[float, ...]] = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass = inertial.find("mass")
        origin = inertial.find("origin")
        if mass is None:
            continue
        com = _floats(origin.get("xyz") if origin is not None else None) or (0.0, 0.0, 0.0)
        bodies[link.get("name") or ""] = (float(mass.get("value", "0")), (com[0], com[1], com[2]))
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is None:
            continue
        joints[joint.get("name") or ""] = (
            float(limit.get("lower", "0")),
            float(limit.get("upper", "0")),
            float(limit.get("effort", "0")),
            float(limit.get("velocity", "0")),
        )
    return bodies, joints


def _parse_mjcf(path: Path) -> tuple[dict[str, tuple[float, tuple[float, float, float]]], dict[str, tuple[float, ...]]]:
    root = ET.parse(path).getroot()
    bodies: dict[str, tuple[float, tuple[float, float, float]]] = {}
    joints: dict[str, tuple[float, ...]] = {}
    for body in root.iter("body"):
        name = body.get("name")
        inertial = body.find("inertial")
        if name is None or inertial is None or inertial.get("mass") is None:
            continue
        com = _floats(inertial.get("pos")) or (0.0, 0.0, 0.0)
        bodies[name] = (float(inertial.get("mass", "0")), (com[0], com[1], com[2]))
    for joint in root.iter("joint"):
        name = joint.get("name")
        rng = _floats(joint.get("range"))
        force = _floats(joint.get("actuatorfrcrange"))
        if name is None or len(rng) != 2:
            continue
        effort = max(abs(force[0]), abs(force[1])) if len(force) == 2 else None
        joints[name] = (rng[0], rng[1], effort)
    return bodies, joints


def test_urdf_and_mjcf_mass_com_and_joint_limits_match() -> None:
    urdf_bodies, urdf_joints = _parse_urdf(_URDF_PATH)
    mjcf_bodies, mjcf_joints = _parse_mjcf(_MJCF_PATH)

    assert set(urdf_bodies) == set(mjcf_bodies)
    assert set(urdf_joints) == set(mjcf_joints)
    for name, (mass, com) in urdf_bodies.items():
        other_mass, other_com = mjcf_bodies[name]
        assert mass == pytest.approx(other_mass, abs=1e-6), name
        assert com == pytest.approx(other_com, abs=1e-6), name
    for name, (lower, upper, effort, _velocity) in urdf_joints.items():
        mj_lower, mj_upper, mj_effort = mjcf_joints[name]
        assert lower == pytest.approx(mj_lower, abs=1e-5), name
        assert upper == pytest.approx(mj_upper, abs=1e-5), name
        if mj_effort is not None:
            assert effort == pytest.approx(mj_effort, abs=1e-5), name


def _urdf_depth_first_joints(path: Path) -> list[str]:
    """The movable joints in a pre-order walk of the kinematic tree, children in file order."""
    root = ET.parse(path).getroot()
    children: dict[str, list[tuple[str, str, str]]] = {}
    linked: set[str] = set()
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_link = parent.get("link") or ""
        child_link = child.get("link") or ""
        children.setdefault(parent_link, []).append((joint.get("name") or "", child_link, joint.get("type") or ""))
        linked.add(child_link)
    base = next(link.get("name") or "" for link in root.findall("link") if (link.get("name") or "") not in linked)

    order: list[str] = []

    def walk(link: str) -> None:
        for name, child_link, kind in children.get(link, ()):
            if kind != "fixed":
                order.append(name)
            walk(child_link)

    walk(base)
    return order


def test_the_catalog_joint_order_is_the_depth_first_walk() -> None:
    """Decision D1's premise, which nothing else checks.

    Every other assertion about joints compares *sets*, so the canonical list could be reordered --
    or quietly regenerated from a tool that walks the tree breadth-first -- without a single test
    noticing. The order is the load-bearing part: it is what the action term and the joint
    observations are told to select by, and therefore what makes those vectors mean the same thing
    on two engines whose native orders are neither this one nor each other's.
    """
    robot = make_g1_29dof_robot_spec()
    assert list(robot.joint_names) == _urdf_depth_first_joints(_URDF_PATH)


def test_the_default_pose_is_written_for_every_joint_in_catalog_order() -> None:
    """A sparse dict that falls back to zero would make a forgotten hip look like a straight leg."""
    assert tuple(G1_29DOF_DEFAULT_JOINT_POS) == G1_29DOF_DFS_JOINT_NAMES
    robot = make_g1_29dof_robot_spec()
    assert tuple(item.default_pos for item in robot.joint_properties) == tuple(G1_29DOF_DEFAULT_JOINT_POS.values())


def test_isaac_bfs_is_the_same_joints_in_a_different_order() -> None:
    """Otherwise a rename in one list and not the other would look like a remap."""
    assert set(G1_29DOF_ISAAC_BFS_JOINT_NAMES) == set(G1_29DOF_DFS_JOINT_NAMES)
    assert G1_29DOF_ISAAC_BFS_JOINT_NAMES != G1_29DOF_DFS_JOINT_NAMES


def _mirrored_joint_name(name: str) -> str:
    if name.startswith("left_"):
        return "right_" + name[5:]
    if name.startswith("right_"):
        return "left_" + name[6:]
    return name


def test_symmetric_augmentation_follows_the_named_order() -> None:
    """The leftover comment in unitree_g1.py was PhysX BFS; training is DFS.

    Hardcoded indices against the wrong list swap the wrong joints and still look like a
    valid permutation. Building the tables from names is what makes a reordering fail here
    instead of in a training curve.
    """
    for names in (G1_29DOF_DFS_JOINT_NAMES, G1_29DOF_ISAAC_BFS_JOINT_NAMES):
        mapping, reverse = g1_symmetric_joint_augmentation(names)
        assert len(mapping) == len(names) == len(reverse)
        for i, name in enumerate(names):
            assert names[mapping[i]] == _mirrored_joint_name(name), name
            assert reverse[i] == (-1 if "roll" in name or "yaw" in name else 1), name
        twice = [mapping[j] for j in mapping]
        assert twice == list(range(len(names)))


def test_isaac_bfs_augmentation_matches_the_tables_main_shipped() -> None:
    """Parkour still indexes PhysX BFS. A helper that disagrees with main's lists would
    silently swap the wrong joints there, which is the failure this rewrite exists to stop.
    """
    mapping, reverse = g1_symmetric_joint_augmentation(G1_29DOF_ISAAC_BFS_JOINT_NAMES)
    assert mapping == (
        1,
        0,
        2,
        4,
        3,
        5,
        7,
        6,
        8,
        10,
        9,
        12,
        11,
        14,
        13,
        16,
        15,
        18,
        17,
        20,
        19,
        22,
        21,
        24,
        23,
        26,
        25,
        28,
        27,
    )
    assert reverse == (
        1,
        1,
        1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        1,
        1,
        1,
        1,
        -1,
        -1,
        -1,
        -1,
        1,
        1,
        -1,
        -1,
        -1,
        -1,
        1,
        1,
        1,
        1,
        -1,
        -1,
    )


def test_robot_spec_limits_match_urdf() -> None:
    _, urdf_joints = _parse_urdf(_URDF_PATH)
    robot = make_g1_29dof_robot_spec()
    assert set(urdf_joints) == set(robot.joint_names)
    for properties in robot.joint_properties:
        lower, upper, effort, velocity = urdf_joints[properties.name]
        del lower, upper
        assert properties.effort_limit == pytest.approx(effort)
        assert properties.velocity_limit == pytest.approx(velocity)


def _module_assignments(tree: ast.Module) -> dict[str, ast.AST]:
    assignments: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            assignments[node.targets[0].id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            assignments[node.target.id] = node.value
    return assignments


def _named_assignment(tree: ast.Module, name: str) -> ast.AST | None:
    """Find a named assignment even when a lazy native table lives in a loader."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return node.value
    return None


def _numeric_symbols(assignments: dict[str, ast.AST]) -> dict[str, float]:
    """Resolve the arithmetic constants used by both reference actuator tables."""
    symbols: dict[str, float] = {}
    pending = dict(assignments)
    changed = True
    while changed:
        changed = False
        for name, node in tuple(pending.items()):
            try:
                value = eval(compile(ast.Expression(node), "<actuator-reference>", "eval"), {"__builtins__": {}}, symbols)
            except (NameError, TypeError, ValueError):
                continue
            if not isinstance(value, (int, float)):
                continue
            symbols[name] = float(value)
            pending.pop(name)
            changed = True
    return symbols


def _reference_actuator_parameters(
    path: Path,
    table_name: str,
    *,
    target_field: str,
    effort_field: str,
    include_velocity: bool,
) -> dict[str, dict[str, float]]:
    """Expand a reference actuator table into exact per-joint numeric values."""
    tree = ast.parse(path.read_text())
    assignments = _module_assignments(tree)
    symbols = _numeric_symbols(assignments)
    table = assignments.get(table_name) or _named_assignment(tree, table_name)
    if isinstance(table, ast.Dict):
        calls = list(table.values)
    elif isinstance(table, (ast.Tuple, ast.List)):
        calls = [assignments.get(item.id) if isinstance(item, ast.Name) else item for item in table.elts]
    else:
        raise LookupError(f"{path} has no actuator table {table_name!r}")

    field_names = (effort_field, "stiffness", "damping", "armature")
    if include_velocity:
        field_names = (*field_names, "velocity_limit_sim")
    expanded: dict[str, dict[str, float]] = {}
    for call in calls:
        if not isinstance(call, ast.Call):
            raise LookupError(f"{table_name} contains a non-call entry")
        kwargs = {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg is not None}
        patterns = ast.literal_eval(kwargs[target_field])
        matched = [name for pattern in patterns for name in G1_29DOF_DFS_JOINT_NAMES if re.fullmatch(pattern, name)]
        if not matched:
            raise LookupError(f"{table_name} entry {patterns!r} matches no canonical joint")
        for joint_name in matched:
            if joint_name in expanded:
                raise LookupError(f"{table_name} assigns {joint_name!r} more than once")
            values: dict[str, float] = {}
            for field_name in field_names:
                field = kwargs.get(field_name)
                if field is None:
                    raise LookupError(f"{table_name} entry {patterns!r} has no {field_name}")
                if isinstance(field, ast.Dict):
                    candidates = [
                        value
                        for pattern, value in zip(field.keys, field.values, strict=True)
                        if pattern is not None and re.fullmatch(ast.literal_eval(pattern), joint_name)
                    ]
                    if len(candidates) != 1:
                        raise LookupError(
                            f"{table_name} field {field_name} matches {joint_name!r} {len(candidates)} times"
                        )
                    field = candidates[0]
                try:
                    value = eval(
                        compile(ast.Expression(field), "<actuator-reference>", "eval"),
                        {"__builtins__": {}},
                        symbols,
                    )
                except (NameError, TypeError, ValueError) as exc:
                    raise LookupError(f"cannot evaluate {table_name}.{field_name} for {joint_name!r}") from exc
                if not isinstance(value, (int, float)):
                    raise LookupError(f"{table_name}.{field_name} for {joint_name!r} is not numeric")
                values[field_name] = float(value)
            expanded[joint_name] = values
    missing = set(G1_29DOF_DFS_JOINT_NAMES) - set(expanded)
    if missing:
        raise LookupError(f"{table_name} leaves canonical joints unassigned: {sorted(missing)}")
    return expanded


def test_robot_spec_actuation_matches_both_reference_repositories() -> None:
    """The catalog is the bridge; compare its numbers to both sources, not to itself."""
    if not _MAIN_ASSET_CFG.is_file() or not _MJLAB_ASSET_CFG.is_file():
        pytest.skip("reference repositories are not checked out")

    main = _reference_actuator_parameters(
        _MAIN_ASSET_CFG,
        "beyondmimic_g1_29dof_actuators",
        target_field="joint_names_expr",
        effort_field="effort_limit_sim",
        include_velocity=True,
    )
    mjlab = _reference_actuator_parameters(
        _MJLAB_ASSET_CFG,
        "beyondmimic_g1_29dof_actuator_cfgs",
        target_field="target_names_expr",
        effort_field="effort_limit",
        include_velocity=False,
    )
    for joint in make_g1_29dof_robot_spec().joint_properties:
        main_values = main[joint.name]
        mjlab_values = mjlab[joint.name]
        for field_name in ("stiffness", "damping", "armature"):
            expected = getattr(joint, field_name)
            assert main_values[field_name] == pytest.approx(expected), (joint.name, field_name, "main")
            assert mjlab_values[field_name] == pytest.approx(expected), (joint.name, field_name, "InstinctMJ")
        assert main_values["effort_limit_sim"] == pytest.approx(joint.effort_limit), joint.name
        assert mjlab_values["effort_limit"] == pytest.approx(joint.effort_limit), joint.name
        assert main_values["velocity_limit_sim"] == pytest.approx(joint.velocity_limit), joint.name
        assert joint.action_scale == pytest.approx(0.25 * main_values["effort_limit_sim"] / main_values["stiffness"])
        assert joint.action_scale == pytest.approx(0.25 * mjlab_values["effort_limit"] / mjlab_values["stiffness"])


def test_reference_actuator_reader_observes_a_source_gain_mutation(tmp_path: Path) -> None:
    if not _MJLAB_ASSET_CFG.is_file():
        pytest.skip("InstinctMJ is not checked out")
    source = _MJLAB_ASSET_CFG.read_text()
    needle = "stiffness=STIFFNESS_7520_14,"
    assert source.count(needle) >= 1
    mutated_path = tmp_path / "unitree_g1.py"
    mutated_path.write_text(source.replace(needle, "stiffness=1.5 * STIFFNESS_7520_14,", 1))

    mutated = _reference_actuator_parameters(
        mutated_path,
        "beyondmimic_g1_29dof_actuator_cfgs",
        target_field="target_names_expr",
        effort_field="effort_limit",
        include_velocity=False,
    )

    nominal = next(
        joint.stiffness
        for joint in make_g1_29dof_robot_spec().joint_properties
        if joint.name == "left_hip_pitch_joint"
    )
    assert mutated["left_hip_pitch_joint"]["stiffness"] == pytest.approx(1.5 * nominal)


class _RecordedActuator:
    """SDK-free view of one native actuator constructor call."""

    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


def _field_for_joint(value, joint_name: str) -> float:
    if not isinstance(value, dict):
        return float(value)
    matched = [number for pattern, number in value.items() if re.fullmatch(pattern, joint_name)]
    assert len(matched) == 1, (joint_name, value)
    return float(matched[0])


def _native_actuator_parameters(
    actuators,
    *,
    target_field: str,
    effort_field: str,
) -> dict[str, dict[str, float]]:
    expanded: dict[str, dict[str, float]] = {}
    for actuator in actuators:
        patterns = getattr(actuator, target_field)
        matched = [name for pattern in patterns for name in G1_29DOF_DFS_JOINT_NAMES if re.fullmatch(pattern, name)]
        assert matched, patterns
        for joint_name in matched:
            assert joint_name not in expanded
            expanded[joint_name] = {
                "effort_limit": _field_for_joint(getattr(actuator, effort_field), joint_name),
                "stiffness": _field_for_joint(actuator.stiffness, joint_name),
                "damping": _field_for_joint(actuator.damping, joint_name),
                "armature": _field_for_joint(actuator.armature, joint_name),
            }
            velocity = getattr(actuator, "velocity_limit_sim", None)
            if velocity is not None:
                expanded[joint_name]["velocity_limit"] = _field_for_joint(velocity, joint_name)
    return expanded


def test_both_native_actuator_tables_are_explicit_and_cover_the_robot_spec() -> None:
    """Native configs state their own groups; neither builder derives them from RobotSpec."""
    from instinctlab.assets.unitree_g1.mjlab import beyondmimic_actuator_cfgs

    robot = make_g1_29dof_robot_spec()
    mjlab_cfgs = beyondmimic_actuator_cfgs(_RecordedActuator)
    assert len(mjlab_cfgs) == 7

    isaac = _reference_actuator_parameters(
        _ASSETS,
        "beyondmimic_g1_29dof_actuators",
        target_field="joint_names_expr",
        effort_field="effort_limit_sim",
        include_velocity=True,
    )
    for values in isaac.values():
        values["effort_limit"] = values.pop("effort_limit_sim")
        values["velocity_limit"] = values.pop("velocity_limit_sim")
    mjlab = _native_actuator_parameters(
        mjlab_cfgs,
        target_field="target_names_expr",
        effort_field="effort_limit",
    )
    assert set(isaac) == set(mjlab) == set(robot.joint_names)
    for joint in robot.joint_properties:
        for native in (isaac[joint.name], mjlab[joint.name]):
            assert native["effort_limit"] == pytest.approx(joint.effort_limit)
            assert native["stiffness"] == pytest.approx(joint.stiffness)
            assert native["damping"] == pytest.approx(joint.damping)
            assert native["armature"] == pytest.approx(joint.armature)
        assert isaac[joint.name]["velocity_limit"] == pytest.approx(joint.velocity_limit)


def test_native_actuator_builders_do_not_infer_groups_from_the_shared_contract() -> None:
    """Guard the boundary: native tables may be validated against RobotSpec, not generated from it."""
    from instinctlab.assets.unitree_g1.mjlab import beyondmimic_actuator_cfgs
    from instinctlab.engines.mjlab import assets as mjlab_assets

    source = inspect.getsource(beyondmimic_actuator_cfgs)
    assert "joint_properties" not in source
    assert "actuator_group" not in source
    assert not any(isinstance(node, (ast.If, ast.IfExp)) for node in ast.walk(ast.parse(source)))

    isaac_tree = ast.parse(_ASSETS.read_text())
    isaac_table = _named_assignment(isaac_tree, "beyondmimic_g1_29dof_actuators")
    assert isinstance(isaac_table, ast.Dict)
    assert "joint_properties" not in ast.unparse(isaac_table)
    assert not any(isinstance(node, (ast.If, ast.IfExp)) for node in ast.walk(isaac_table))

    adapter_source = inspect.getsource(mjlab_assets)
    # The adapter may normalize explicit joint metadata, but it must not synthesize
    # native actuator groups from that metadata.
    assert "actuator_group" not in adapter_source
    assert "grouped_actuators" not in adapter_source


# --- the sim2sim scene restates the Isaac spawn, so the restatement has to be checked -------------

_ASSETS = Path(__file__).resolve().parents[1] / "source/instinctlab/instinctlab/assets/unitree_g1/isaacsim.py"


def _popsicle_spawn() -> dict[str, object]:
    """Read the PhysX spawn properties off ``G1_29DOF_TORSOBASE_POPSICLE_CFG`` without Isaac Sim.

    Importing the module pulls in ``omni``, so the config is read as source. Only literal keywords
    are collected; the ones that reference the catalog are checked elsewhere.
    """
    import ast

    tree = ast.parse(_ASSETS.read_text())
    for node in ast.walk(tree):
        targets = getattr(node, "targets", [])
        if not (targets and isinstance(targets[0], ast.Name) and targets[0].id == "G1_29DOF_TORSOBASE_POPSICLE_CFG"):
            continue
        spawn = next(
            keyword.value for keyword in node.value.keywords if keyword.arg == "spawn"  # type: ignore[union-attr]
        )
        collected: dict[str, object] = {}
        for keyword in spawn.keywords:  # type: ignore[union-attr]
            if isinstance(keyword.value, ast.Constant):
                collected[str(keyword.arg)] = keyword.value.value
            elif isinstance(keyword.value, ast.Call) and keyword.arg in ("rigid_props", "articulation_props"):
                collected[str(keyword.arg)] = {
                    str(inner.arg): inner.value.value
                    for inner in keyword.value.keywords
                    if isinstance(inner.value, ast.Constant)
                }
        return collected
    raise AssertionError("G1_29DOF_TORSOBASE_POPSICLE_CFG is no longer assigned")


def test_the_isaac_config_starts_the_robot_where_the_catalog_says() -> None:
    """The POPSICLE config must take the catalog pose, not restate the numbers."""
    import ast

    for node in ast.walk(ast.parse(_ASSETS.read_text())):
        if not (
            isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "G1_29DOF_TORSOBASE_POPSICLE_CFG"
        ):
            continue
        init = next(k.value for k in node.value.keywords if k.arg == "init_state")  # type: ignore[union-attr]
        stated = next(k.value for k in init.keywords if k.arg == "joint_pos")  # type: ignore[union-attr]
        assert isinstance(stated, ast.Call) and getattr(stated.func, "id", None) == "dict"
        assert getattr(stated.args[0], "id", None) == "G1_29DOF_DEFAULT_JOINT_POS"
        break
    else:
        raise AssertionError("G1_29DOF_TORSOBASE_POPSICLE_CFG no longer states init_state.joint_pos")
