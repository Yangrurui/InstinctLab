"""MJLab assets and actuator configurations for Unitree G1."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from instinctlab.sim.robot_spec import RobotSpec

from . import (
    ARMATURE_4010,
    ARMATURE_5020,
    ARMATURE_7520_14,
    ARMATURE_7520_22,
    DAMPING_4010,
    DAMPING_5020,
    DAMPING_7520_14,
    DAMPING_7520_22,
    STIFFNESS_4010,
    STIFFNESS_5020,
    STIFFNESS_7520_14,
    STIFFNESS_7520_22,
)

DELAY_RESET_ONLY_PERIOD = 1_000_000
ENTITIES = frozenset(
    {
        "popsicle_torsobase_v1",
        "popsicle_torsobase_shadowing_v1",
        "popsicle_torsobase_parkour_v1",
    }
)


def _delay(delay: tuple[int, int], group_offset: int) -> dict[str, int | bool]:
    if delay == (0, 0):
        return {}
    return {
        "delay_min_lag": delay[0],
        "delay_max_lag": delay[1],
        "delay_update_period": DELAY_RESET_ONLY_PERIOD + group_offset,
        "delay_per_env_phase": False,
    }


def beyondmimic_actuator_cfgs(
    actuator_cfg_type,
    *,
    delay: tuple[int, int] = (0, 0),
) -> tuple[object, ...]:
    """Build MJLab's seven explicit native groups.

    The two gain splits in ``legs`` and ``arms`` share a delay period so each
    physical motor group draws one lag per environment and episode.
    """
    return (
        actuator_cfg_type(
            target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint"),
            effort_limit=88.0,
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
            **_delay(delay, 0),
        ),
        actuator_cfg_type(
            target_names_expr=("waist_yaw_joint",),
            effort_limit=88.0,
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
            **_delay(delay, 3),
        ),
        actuator_cfg_type(
            target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
            effort_limit=139.0,
            stiffness=STIFFNESS_7520_22,
            damping=DAMPING_7520_22,
            armature=ARMATURE_7520_22,
            **_delay(delay, 0),
        ),
        actuator_cfg_type(
            target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
            effort_limit=50.0,
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
            **_delay(delay, 1),
        ),
        actuator_cfg_type(
            target_names_expr=("waist_roll_joint", "waist_pitch_joint"),
            effort_limit=50.0,
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
            **_delay(delay, 2),
        ),
        actuator_cfg_type(
            target_names_expr=(
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
            ),
            effort_limit=25.0,
            stiffness=STIFFNESS_5020,
            damping=DAMPING_5020,
            armature=ARMATURE_5020,
            **_delay(delay, 4),
        ),
        actuator_cfg_type(
            target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
            effort_limit=5.0,
            stiffness=STIFFNESS_4010,
            damping=DAMPING_4010,
            armature=ARMATURE_4010,
            **_delay(delay, 4),
        ),
    )


def _without_visual_meshes(xml: str) -> str:
    root = ElementTree.fromstring(xml)
    for asset in root.findall("asset"):
        for mesh in tuple(asset.findall("mesh")):
            asset.remove(mesh)
    for parent in root.iter():
        for geom in tuple(parent.findall("geom")):
            if geom.get("type") == "mesh" or geom.get("mesh"):
                parent.remove(geom)
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
    return ElementTree.tostring(root, encoding="unicode")


def _load_spec(path: Path, load_mode: str) -> Any:
    import mujoco

    if load_mode == "default":
        return mujoco.MjSpec.from_file(str(path))
    if load_mode == "strip_visual_meshes":
        try:
            return mujoco.MjSpec.from_file(str(path))
        except (ValueError, OSError):
            return mujoco.MjSpec.from_string(_without_visual_meshes(path.read_text()))
    raise NotImplementedError(f"Unitree G1 has no MJLab loader for {load_mode!r}")


def entity(variant: str, robot: RobotSpec, *, actuator_order=None) -> Any:
    """Build one registered G1 variant as an MJLab entity."""
    del actuator_order
    if variant not in ENTITIES:
        raise KeyError(f"Unknown MJLab Unitree G1 variant {variant!r}; registered: {sorted(ENTITIES)}")

    from mjlab.actuator import BuiltinPdActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

    asset = robot.asset_for("mjlab")
    path = Path(asset.path)
    if not path.is_file():
        raise FileNotFoundError(f"The MJLab asset for {robot.name!r} is missing: {path}")
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=robot.default_root_pos,
            rot=robot.default_root_quat_wxyz,
            joint_pos={joint.name: joint.default_pos for joint in robot.joint_properties},
            joint_vel={".*": 0.0},
        ),
        spec_fn=lambda: _load_spec(path, asset.load_mode),
        articulation=EntityArticulationInfoCfg(
            actuators=beyondmimic_actuator_cfgs(BuiltinPdActuatorCfg, delay=robot.actuator_delay),
            soft_joint_pos_limit_factor=robot.soft_joint_pos_limit_factor,
        ),
    )


__all__ = ["DELAY_RESET_ONLY_PERIOD", "ENTITIES", "beyondmimic_actuator_cfgs", "entity"]
