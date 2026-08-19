"""Static URDF/MJCF and PD-gain parity for the unified G1 asset."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from instinctlab.assets.unitree_g1.isaacsim import (
    G1_29DOF_DEFAULT_JOINT_POS,
    G1_29DOF_DFS_JOINT_NAMES,
    G1_29DOF_ISAAC_BFS_JOINT_NAMES,
    g1_symmetric_joint_augmentation,
    make_g1_29dof_robot_spec,
)

_RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "source/instinctlab/instinctlab/assets/resources/unitree_g1"
_URDF_PATH = _RESOURCE_ROOT / "urdf" / "g1_29dof_torsobase_popsicle.urdf"
_MJCF_PATH = _RESOURCE_ROOT / "xml" / "g1_29dof_torsobase_popsicle.xml"


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


def test_beyondmimic_groups_cover_the_robot_spec() -> None:
    """Isaac actuator tables are filled from ``RobotSpec``; every joint must have a group."""
    from instinctlab.assets.unitree_g1.isaacsim import _BEYONDMIMIC_JOINT_GROUPS, _pattern_fields

    robot = make_g1_29dof_robot_spec()
    covered: set[str] = set()
    for patterns in _BEYONDMIMIC_JOINT_GROUPS.values():
        fields = _pattern_fields(patterns)
        for pattern in patterns:
            matched = [joint for joint in robot.joint_properties if re.fullmatch(pattern, joint.name)]
            assert matched, pattern
            for joint in matched:
                covered.add(joint.name)
                assert fields["stiffness"][pattern] == pytest.approx(joint.stiffness)
                assert fields["damping"][pattern] == pytest.approx(joint.damping)
                assert fields["armature"][pattern] == pytest.approx(joint.armature)
                assert fields["effort_limit_sim"][pattern] == pytest.approx(joint.effort_limit)
                assert fields["velocity_limit_sim"][pattern] == pytest.approx(joint.velocity_limit)
    assert covered == set(robot.joint_names)


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


def test_the_verification_scene_spawns_the_robot_the_task_trains() -> None:
    """sim2sim builds its spawn from the RobotSpec, so it restates what the Isaac config declares.

    A restatement drifts. This one carried ``self_collision=True`` while the config sets none, which
    meant the engines were being compared on a robot neither of them trains, and nothing said so.
    """
    from instinctlab.verify.scene import locomotion_flat_scene

    scene = locomotion_flat_scene(num_envs=2).scene
    stated = dict(scene.backend_options_for("isaacsim")["robot_spawn"])
    declared = _popsicle_spawn()

    for group in ("rigid_props", "articulation_props"):
        assert stated[group] == declared[group], (
            f"the sim2sim scene's {group} no longer matches G1_29DOF_TORSOBASE_POPSICLE_CFG; "
            "sim2sim would be comparing the engines on a robot the task does not train"
        )

    # The backend computes this one from the sensor list rather than reading it, which is the right
    # way round -- but then it has to agree with what the config states.
    assert bool(scene.contact_sensors) == declared["activate_contact_sensors"]

    rest = {"rigid_props", "articulation_props", "activate_contact_sensors"}
    assert set(stated) - rest == set(declared) - rest, (
        "the sim2sim scene and the Isaac config disagree on which spawn keys exist: "
        f"{sorted(set(stated) - rest)} vs {sorted(set(declared) - rest)}"
    )


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
