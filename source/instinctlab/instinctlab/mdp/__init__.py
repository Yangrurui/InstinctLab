"""Portable MDP terms: one implementation, run natively by either engine's manager.

A task declares an observation as ``ObsTermSpec(mdp.base_ang_vel)`` and the compiler hands that
function straight to Isaac Lab's ``ObservationManager`` or mjlab's, unwrapped. Nothing here is
called through an adapter at runtime, which is why a migrated task keeps performing the way it did
and why the migration is mostly an import change.

The terms are ports of the Isaac Lab and mjlab originals with the attribute names moved onto the
hub vocabulary in :mod:`instinctlab.compat.vocab`. Where a port changes a number, or where a term
turned out not to be portable at all, the reason is written at the point it applies -- see the
module docstrings of :mod:`~instinctlab.mdp.rewards` and :mod:`~instinctlab.mdp.terminations` for
the four cases in the flat-locomotion task.

Like ``spec/``, this package imports no engine. It reads ``env`` and ``asset.data`` attributes that
both engines spell identically, which is a property maintained by ``compat/`` and checked by tests,
not a coincidence to be relied on.
"""

from __future__ import annotations

from .amp import (
    base_ang_vel_from_reference,
    base_lin_vel_from_reference,
    joint_pos_rel_from_reference,
    joint_vel_rel_from_reference,
    projected_gravity_from_reference,
)
from .curriculums import terrain_levels_vel, tracking_exp_vel
from .events import register_virtual_obstacles
from .observations import (
    DelayedDepthImage,
    base_ang_vel,
    base_lin_vel,
    clear_delayed_depth_history,
    generated_commands,
    joint_pos_rel,
    joint_vel,
    joint_vel_rel,
    last_action,
    projected_gravity,
)
from .rewards import (
    action_rate_l2,
    ang_vel_xy_l2,
    dont_wait,
    feet_air_time,
    feet_air_time_positive_biped,
    feet_at_plane,
    feet_close_xy_gauss,
    feet_orientation_contact,
    flat_orientation_l2,
    heading_error,
    is_alive,
    is_terminated,
    joint_deviation_l1,
    joint_deviation_square,
    joint_pos_limits,
    joint_vel_l2,
    joint_vel_limits,
    lin_vel_z_l2,
    link_orientation,
    stand_still,
    stand_still_when_idle,
    track_ang_vel_z_exp,
    track_ang_vel_z_world_exp,
    track_lin_vel_xy_exp,
    track_lin_vel_xy_yaw_frame_exp,
    undesired_contacts,
    volume_points_penetration,
)
from .terminations import (
    bad_orientation,
    illegal_contact,
    root_height_below_env_origin_minimum,
    terrain_out_of_bounds,
    time_out,
)

__all__ = [
    "action_rate_l2",
    "ang_vel_xy_l2",
    "bad_orientation",
    "DelayedDepthImage",
    "clear_delayed_depth_history",
    "base_ang_vel",
    "base_ang_vel_from_reference",
    "base_lin_vel",
    "base_lin_vel_from_reference",
    "dont_wait",
    "feet_air_time",
    "feet_air_time_positive_biped",
    "feet_at_plane",
    "feet_close_xy_gauss",
    "feet_orientation_contact",
    "flat_orientation_l2",
    "generated_commands",
    "heading_error",
    "illegal_contact",
    "is_alive",
    "is_terminated",
    "joint_deviation_l1",
    "joint_deviation_square",
    "joint_pos_limits",
    "joint_pos_rel",
    "joint_pos_rel_from_reference",
    "joint_vel",
    "joint_vel_rel_from_reference",
    "joint_vel_l2",
    "joint_vel_limits",
    "joint_vel_rel",
    "last_action",
    "lin_vel_z_l2",
    "link_orientation",
    "projected_gravity",
    "projected_gravity_from_reference",
    "register_virtual_obstacles",
    "root_height_below_env_origin_minimum",
    "stand_still",
    "stand_still_when_idle",
    "terrain_levels_vel",
    "terrain_out_of_bounds",
    "time_out",
    "track_ang_vel_z_exp",
    "track_ang_vel_z_world_exp",
    "track_lin_vel_xy_exp",
    "track_lin_vel_xy_yaw_frame_exp",
    "tracking_exp_vel",
    "undesired_contacts",
    "volume_points_penetration",
]
