"""mjlab entities, generated from the robot catalog rather than looked up.

The opposite of the Isaac Sim side, and the difference is worth noting because it is the asset
pipeline of decision D5 in miniature. mjlab entities are cheap to derive: the MJCF already carries
the geometry, so an :class:`EntityCfg` is the file plus actuator gains, and every number it needs is
on :class:`RobotSpec`. Isaac Lab's ``ArticulationCfg`` additionally carries USD authoring choices
that nothing in the catalog records, which is why that side is still a lookup table.

So adding a robot costs an entry there and nothing here, and the pipeline's job is to make the
first one cost nothing either.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from instinctlab.sim.robot_spec import JointProperties, RobotSpec

__all__ = ["entity", "grouped_actuators"]


def grouped_actuators(joints: Iterable[JointProperties]) -> tuple[tuple[tuple[str, ...], JointProperties], ...]:
    """Collapse joints sharing PD gains into one actuator config each, in declaration order.

    Grouping rather than one actuator per joint: mjlab does not fuse identical actuator configs, so
    a per-joint config multiplies the control writes for no behavioural difference. The gains stay
    per-joint identical -- only the batching changes -- and the first joint of each group carries
    them.
    """
    groups: dict[tuple[float, float, float, float], list[str]] = {}
    heads: dict[tuple[float, float, float, float], JointProperties] = {}
    order: list[tuple[float, float, float, float]] = []
    for joint in joints:
        key = (float(joint.stiffness), float(joint.damping), float(joint.effort_limit), float(joint.armature))
        if key not in groups:
            groups[key], heads[key] = [], joint
            order.append(key)
        groups[key].append(joint.name)
    return tuple((tuple(groups[key]), heads[key]) for key in order)


def _without_visual_meshes(xml: str) -> str:
    """Drop mesh assets and mesh geoms from an MJCF document."""
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
    """Load an MJCF the way the catalog says to.

    ``strip_visual_meshes`` tries the file as written and falls back to a mesh-free version. The
    fallback exists because visual meshes are often shipped separately from the model, and a model
    that cannot be loaded for want of decoration is not a physics problem -- collision geometry is
    unaffected, so the simulation is the same one.
    """
    import mujoco

    if load_mode == "default":
        return mujoco.MjSpec.from_file(str(path))
    if load_mode == "strip_visual_meshes":
        try:
            return mujoco.MjSpec.from_file(str(path))
        except (ValueError, OSError):
            return mujoco.MjSpec.from_string(_without_visual_meshes(path.read_text()))
    raise NotImplementedError(f"The mjlab backend has no loader for asset load mode {load_mode!r}.")


def entity(robot: RobotSpec, *, actuator_order: Sequence[str] | None = None) -> Any:
    """The mjlab entity for ``robot``.

    Args:
        robot: Robot from the catalog.
        actuator_order: Ignored except as documentation of intent -- mjlab resolves actuator
            selections against the model, and joint ordering is settled by ``preserve_order`` on the
            action term rather than here.
    """
    from mjlab.actuator import BuiltinPdActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

    asset = robot.asset_for("mjlab")
    asset.verify()
    path = Path(asset.path)
    if not path.is_file():
        raise FileNotFoundError(f"The mjlab asset for {robot.name!r} is missing: {path}")

    actuators = tuple(
        BuiltinPdActuatorCfg(
            target_names_expr=names,
            stiffness=head.stiffness,
            damping=head.damping,
            effort_limit=head.effort_limit,
            armature=head.armature,
        )
        for names, head in grouped_actuators(robot.joint_properties)
    )
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=robot.default_root_pos,
            rot=robot.default_root_quat_wxyz,
            joint_pos={joint.name: joint.default_pos for joint in robot.joint_properties},
            joint_vel={".*": 0.0},
        ),
        spec_fn=lambda: _load_spec(path, asset.load_mode),
        articulation=EntityArticulationInfoCfg(
            actuators=actuators,
            soft_joint_pos_limit_factor=robot.soft_joint_pos_limit_factor,
        ),
    )
