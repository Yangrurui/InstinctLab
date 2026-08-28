"""Isaac Lab articulation skeletons driven by :class:`RobotSpec`.

The registry supplies the simulator-native URDF and solver skeleton. Joint
defaults and every actuator field are rebuilt from ``RobotSpec`` so Isaac Sim
and mjlab consume the same physical declaration.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from instinctlab.sim.robot_spec import RobotSpec

__all__ = ["ARTICULATIONS", "actuators_from_robot_spec", "apply_robot_spec", "articulation"]

ARTICULATIONS: dict[str, str] = {
    "adam_sp_23dof": "instinctlab.assets.adam_sp.isaacsim:ADAM_SP_23DOF_CFG",
    "popsicle_torsobase_v1": "instinctlab.assets.unitree_g1.isaacsim:G1_29DOF_TORSOBASE_POPSICLE_CFG",
}
""":attr:`RobotSpec.asset_id` -> dotted path of the ``ArticulationCfg`` describing it."""

_SPAWN_OPTION_KEYS = ("merge_fixed_joints", "fix_base", "replace_cylinders_with_capsules")


def actuators_from_robot_spec(robot: RobotSpec) -> dict[str, Any]:
    """Build one Isaac actuator per declared motor group."""
    from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg

    properties = {joint.name: joint for joint in robot.joint_properties}
    min_delay, max_delay = robot.actuator_delay
    actuator_type = DelayedPDActuatorCfg if max_delay > 0 else ImplicitActuatorCfg

    actuators: dict[str, Any] = {}
    for group, names in robot.actuator_groups():
        group_name = group or "all"
        fields: dict[str, Any] = {
            "joint_names_expr": list(names),
            "effort_limit_sim": {name: properties[name].effort_limit for name in names},
            "velocity_limit_sim": {name: properties[name].velocity_limit for name in names},
            "stiffness": {name: properties[name].stiffness for name in names},
            "damping": {name: properties[name].damping for name in names},
            "armature": {name: properties[name].armature for name in names},
        }
        if max_delay > 0:
            fields["min_delay"] = min_delay
            fields["max_delay"] = max_delay
        actuators[group_name] = actuator_type(**fields)
    return actuators


def apply_robot_spec(cfg: Any, robot: RobotSpec) -> Any:
    """Apply the complete ``RobotSpec`` plant to a copied articulation skeleton."""
    asset = robot.asset_for("isaacsim")
    spawn_updates: dict[str, Any] = {"asset_path": asset.path}
    for key in _SPAWN_OPTION_KEYS:
        if key in asset.import_options:
            spawn_updates[key] = asset.import_options[key]
    cfg.spawn = cfg.spawn.replace(**spawn_updates)
    cfg.init_state = cfg.init_state.replace(
        pos=robot.default_root_pos,
        rot=robot.default_root_quat_wxyz,
        joint_pos={joint.name: joint.default_pos for joint in robot.joint_properties},
        joint_vel={".*": 0.0},
    )
    cfg.soft_joint_pos_limit_factor = robot.soft_joint_pos_limit_factor
    cfg.actuators = actuators_from_robot_spec(robot)
    return cfg


def articulation(robot: RobotSpec) -> Any:
    """The Isaac Lab articulation for ``robot``, copied so callers can retarget it safely."""
    try:
        path = ARTICULATIONS[robot.asset_id]
    except KeyError:
        have = ", ".join(sorted(ARTICULATIONS)) or "none"
        raise KeyError(
            f"No Isaac Lab articulation is registered for asset id {robot.asset_id!r} "
            f"(robot {robot.name!r}). Registered: {have}."
        ) from None
    module_path, _, attribute = path.partition(":")
    cfg = getattr(import_module(module_path), attribute).copy()
    return apply_robot_spec(cfg, robot)
