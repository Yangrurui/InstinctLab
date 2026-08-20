"""Parkour MDP terms, resolved lazily so a name's binding is visible.

This package used to be seven star imports. Star imports bind eagerly and the later
import silently wins, which is how ``joint_torques_l2`` — defined in both
``isaaclab.envs.mdp`` and ``instinctlab.envs.mdp`` — ran as the InstinctLab
implementation while a reader of the config would reasonably assume Isaac Lab's.

Sources are searched in the order below. A name that exists in more than one
source raises rather than returning the first hit, except for the one already-
measured collision, which is pinned to the winner the old last-wins order had
so the legacy Isaac parkour task keeps resolving the same object.
"""

from __future__ import annotations

from importlib import import_module

# Local parkour modules first, then InstinctLab's Isaac extensions, then Isaac Lab.
# That is the old star-import last-wins order (isaaclab, instinctlab, then five
# local modules) expressed as a first-wins search. Collisions still raise.
_SOURCES: tuple[str, ...] = (
    "instinctlab.tasks.parkour.mdp.commands",
    "instinctlab.tasks.parkour.mdp.curriculums",
    "instinctlab.tasks.parkour.mdp.events",
    "instinctlab.tasks.parkour.mdp.rewards",
    "instinctlab.tasks.parkour.mdp.terminations",
    "instinctlab.envs.mdp",
    "isaaclab.envs.mdp",
)

# The AST collision statistic across the three layers is one name. Pinning the
# winner here is what keeps ``mdp.joint_torques_l2`` resolving; a new collision
# is not added here — it fails at lookup and in tests/test_parkour_mdp_imports.py.
_COLLISION_WINNERS: dict[str, str] = {
    "joint_torques_l2": "instinctlab.envs.mdp",
}

# Every ``mdp.<name>`` the parkour config files reference, plus the local public
# names the configs do not currently call. Built from the config AST and the
# local FunctionDef/ClassDef set so a rename in either place is a missing-name
# failure rather than a silent AttributeError at env construction.
__all__ = [
    "JointPositionActionCfg",
    "PoseVelocityCommand",
    "PoseVelocityCommandCfg",
    "action_rate_l2",
    "ang_vel_xy_l2",
    "applied_torque_limits_by_ratio",
    "bad_orientation",
    "base_ang_vel",
    "base_ang_vel_reference_as_state",
    "base_lin_vel",
    "base_lin_vel_reference_as_state",
    "contact_slide",
    "delayed_visualizable_image",
    "dont_wait",
    "feet_air_time",
    "feet_at_plane",
    "feet_close_xy_gauss",
    "feet_orientation_contact",
    "flat_orientation_l2",
    "generated_commands",
    "heading_error",
    "illegal_contact",
    "is_alive",
    "joint_acc_l2",
    "joint_deviation_l1",
    "joint_deviation_square",
    "joint_pos_limits",
    "joint_pos_rel",
    "joint_pos_rel_reference_as_state",
    "joint_torques_l2",
    "joint_vel_l2",
    "joint_vel_limits",
    "joint_vel_rel",
    "joint_vel_rel_reference_as_state",
    "last_action",
    "link_orientation",
    "motors_power_square",
    "projected_gravity",
    "projected_gravity_reference_as_state",
    "push_by_setting_velocity_without_stand",
    "randomize_rigid_body_material",
    "reset_joints_by_offset",
    "reset_root_state_uniform",
    "root_height_below_env_origin_minimum",
    "stand_still",
    "sub_terrain_out_of_bounds",
    "terrain_out_of_bounds",
    "time_out",
    "track_ang_vel_z_exp",
    "track_lin_vel_xy_exp",
    "tracking_exp_vel",
    "undesired_contacts",
    "volume_points_penetration",
]


def _defines(module: object, name: str) -> bool:
    """Whether ``module`` itself bound ``name``, without triggering a nested ``__getattr__``."""
    namespace = vars(module)
    if name in namespace:
        return True
    exported = namespace.get("__all__")
    return isinstance(exported, (list, tuple)) and name in exported


def __getattr__(name: str):
    found: list[tuple[str, object]] = []
    for module_name in _SOURCES:
        module = import_module(module_name)
        if _defines(module, name):
            found.append((module_name, module))
    if not found:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if len(found) == 1:
        return getattr(found[0][1], name)
    winner = _COLLISION_WINNERS.get(name)
    modules = [module_name for module_name, _ in found]
    if winner is None or winner not in modules:
        raise AttributeError(
            f"{name!r} is defined in {modules}; refusing to silently pick one. "
            "The old star imports last-won; name that collision or qualify the import."
        )
    return getattr(dict(found)[winner], name)


def __dir__() -> list[str]:
    return sorted(__all__)
