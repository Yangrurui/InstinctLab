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

__all__ = ["DELAY_RESET_ONLY_PERIOD", "entity", "grouped_actuators"]

# Isaac DelayedPD draws one lag in reset and holds it for the episode. mjlab's
# DelayBuffer resamples every physics step when delay_update_period == 0 — a
# same-name-different-meaning trap, see compat/denylist.py ``actuator_delay``.
# A period larger than any episode's step count recovers the hub (per-episode)
# draw; distinct periods keep mjlab from fusing groups onto one shared lag.
# delay_per_env_phase must be False so the draw lands on the first post-reset
# step (a random phase would skip step 0 and leave the lag at its reset value 0).
DELAY_RESET_ONLY_PERIOD = 1_000_000


def _torque_delay_kwargs(robot: RobotSpec, group_index: int) -> dict[str, int | bool]:
    """Hub torque delay: physics steps, inclusive, one draw per episode per group."""
    lo, hi = robot.actuator_delay
    if lo == 0 and hi == 0:
        return {}
    return {
        "delay_min_lag": lo,
        "delay_max_lag": hi,
        "delay_update_period": DELAY_RESET_ONLY_PERIOD + group_index,
        "delay_per_env_phase": False,
    }


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
        robot: Robot the task holds — catalog or a ``RobotSpec.overridden`` copy.
        actuator_order: Ignored except as documentation of intent -- mjlab resolves actuator
            selections against the model, and joint ordering is settled by ``preserve_order`` on the
            action term rather than here.
    """
    from mjlab.actuator import BuiltinPdActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

    asset = robot.asset_for("mjlab")
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
            **_torque_delay_kwargs(robot, group_index),
        )
        for group_index, (names, head) in enumerate(grouped_actuators(robot.joint_properties))
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
