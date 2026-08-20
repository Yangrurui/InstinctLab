"""Isaac Lab articulations for the robots in the catalog.

A lookup from a robot's ``asset_id`` to the hand-written ``ArticulationCfg`` that already describes
it, not a conversion. Decision D5 calls for generating this from :class:`RobotSpec` with numerical
validation, and until that pipeline exists the honest thing is to reuse what main already ships
rather than to hand-write a second description that can disagree with it.

What that costs is visible: adding a robot means adding an entry here *and* one in the mjlab
adapter, which is the ``N x M`` growth the rest of the design avoids. The asset pipeline is what
removes it, and this module is the place it will land.

The lookup is the USD / authoring skeleton. Path, spawn height, ``merge_fixed_joints``, and
torque delay live on :class:`RobotSpec` and are applied here — otherwise a task-level
``RobotSpec.overridden`` would be silently ignored on Isaac and the catalog plant would train.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from instinctlab.sim.robot_spec import RobotSpec

__all__ = ["ARTICULATIONS", "apply_robot_spec", "articulation"]

ARTICULATIONS: dict[str, str] = {
    "popsicle_torsobase_v1": "instinctlab.assets.unitree_g1.isaacsim:G1_29DOF_TORSOBASE_POPSICLE_CFG",
}
""":attr:`RobotSpec.asset_id` -> dotted path of the ``ArticulationCfg`` describing it."""

_SPAWN_OPTION_KEYS = ("merge_fixed_joints", "fix_base", "replace_cylinders_with_capsules")
_DELAY_COPY_FIELDS = (
    "joint_names_expr",
    "effort_limit_sim",
    "velocity_limit_sim",
    "effort_limit",
    "velocity_limit",
    "stiffness",
    "damping",
    "armature",
)


def apply_robot_spec(cfg: Any, robot: RobotSpec) -> Any:
    """Overlay ``robot`` onto a copied ArticulationCfg. Duck-typed for tests without Kit.

    Catalog values applied onto the catalog cfg are a no-op. A task that replaced the
    path, spawn, merge flag, or delay is the whole reason this function exists: the
    lookup table does not see those fields.
    """
    asset = robot.asset_for("isaacsim")
    spawn_updates: dict[str, Any] = {"asset_path": asset.path}
    for key in _SPAWN_OPTION_KEYS:
        if key in asset.import_options:
            spawn_updates[key] = asset.import_options[key]
    cfg.spawn = cfg.spawn.replace(**spawn_updates)
    cfg.init_state = cfg.init_state.replace(pos=robot.default_root_pos)
    if robot.actuator_delay != (0, 0):
        cfg.actuators = delayed_actuators(cfg.actuators, robot.actuator_delay)
    return cfg


def delayed_actuators(actuators: dict[str, Any], delay: tuple[int, int]) -> dict[str, Any]:
    """Same PD numbers, DelayedPD command lag. ``delay`` is hub physics-step inclusive bounds."""
    min_delay, max_delay = delay
    out: dict[str, Any] = {}
    factory = None
    for name, cfg in actuators.items():
        if type(cfg).__name__ == "DelayedPDActuatorCfg":
            out[name] = cfg.replace(min_delay=min_delay, max_delay=max_delay)
            continue
        if factory is None:
            from isaaclab.actuators import DelayedPDActuatorCfg

            factory = DelayedPDActuatorCfg
        kwargs = {field: getattr(cfg, field) for field in _DELAY_COPY_FIELDS if getattr(cfg, field, None) is not None}
        kwargs["min_delay"] = min_delay
        kwargs["max_delay"] = max_delay
        out[name] = factory(**kwargs)
    return out


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
