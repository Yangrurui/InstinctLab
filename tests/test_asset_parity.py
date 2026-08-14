"""Static URDF/MJCF and PD-gain parity for the unified G1 asset."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from instinctlab.assets.unitree_g1 import make_g1_29dof_robot_spec

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


def test_robot_spec_limits_match_urdf() -> None:
    _, urdf_joints = _parse_urdf(_URDF_PATH)
    robot = make_g1_29dof_robot_spec()
    assert set(urdf_joints) == set(robot.joint_names)
    for properties in robot.joint_properties:
        lower, upper, effort, velocity = urdf_joints[properties.name]
        del lower, upper
        assert properties.effort_limit == pytest.approx(effort)
        assert properties.velocity_limit == pytest.approx(velocity)


def _beyondmimic_actuator_block() -> str:
    text = (
        Path(__file__).resolve().parents[1] / "source/instinctlab/instinctlab/assets/unitree_g1_isaac.py"
    ).read_text()
    start = text.index("beyondmimic_g1_29dof_actuators = {")
    end = text.index("beyondmimic_g1_29dof_delayed_actuators")
    return text[start:end]


def _resolve_gain(token: str) -> float:
    from instinctlab.assets import unitree_g1 as catalog

    namespace = {
        name: getattr(catalog, name)
        for name in (
            "ARMATURE_4010",
            "ARMATURE_5020",
            "ARMATURE_7520_14",
            "ARMATURE_7520_22",
            "DAMPING_4010",
            "DAMPING_5020",
            "DAMPING_7520_14",
            "DAMPING_7520_22",
            "STIFFNESS_4010",
            "STIFFNESS_5020",
            "STIFFNESS_7520_14",
            "STIFFNESS_7520_22",
        )
    }
    return float(eval(token, {"__builtins__": {}}, namespace))


def test_beyondmimic_pd_gains_match_robot_spec() -> None:
    robot = make_g1_29dof_robot_spec()
    block = _beyondmimic_actuator_block()
    groups = re.split(r'\n    "[^"]+": ImplicitActuatorCfg\(', block)[1:]
    expected: dict[str, dict[str, float]] = {}
    for group in groups:
        patterns = re.findall(r'"(?:\.\*)?[^"]+_joint"', group)
        patterns = [pattern.strip('"') for pattern in patterns if "joint" in pattern]

        def _field_map(name: str) -> dict[str, str]:
            match = re.search(rf"{name}=(\{{.*?}}|[^\n,]+)", group, flags=re.S)
            assert match is not None, name
            raw = match.group(1).strip()
            if raw.startswith("{"):
                return dict(re.findall(r'"(.*?)":\s*([A-Za-z0-9_.*+\- ]+)', raw))
            return {pattern: raw for pattern in patterns}

        effort = _field_map("effort_limit_sim")
        velocity = _field_map("velocity_limit_sim")
        stiffness = _field_map("stiffness")
        damping = _field_map("damping")
        armature = _field_map("armature")
        for pattern in patterns:
            if pattern not in effort:
                continue
            for name in robot.joint_names:
                if re.fullmatch(pattern, name) is None:
                    continue
                kp = _resolve_gain(stiffness[pattern])
                expected[name] = {
                    "stiffness": kp,
                    "damping": _resolve_gain(damping[pattern]),
                    "armature": _resolve_gain(armature[pattern]),
                    "effort_limit": _resolve_gain(effort[pattern]),
                    "velocity_limit": _resolve_gain(velocity[pattern]),
                    "action_scale": 0.25 * _resolve_gain(effort[pattern]) / kp,
                }
    assert set(expected) == set(robot.joint_names)
    for properties in robot.joint_properties:
        for field, value in expected[properties.name].items():
            assert getattr(properties, field) == pytest.approx(value), f"{properties.name}.{field}"
