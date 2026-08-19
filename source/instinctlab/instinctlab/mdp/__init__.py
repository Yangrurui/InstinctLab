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

from .curriculums import terrain_levels_vel
from .observations import (
    base_ang_vel,
    base_lin_vel,
    generated_commands,
    joint_pos_rel,
    joint_vel,
    joint_vel_rel,
    last_action,
    projected_gravity,
)
from .rewards import (
    action_rate_l2,
    feet_air_time_positive_biped,
    flat_orientation_l2,
    is_terminated,
    joint_deviation_l1,
    joint_pos_limits,
    lin_vel_z_l2,
    stand_still,
    track_ang_vel_z_world_exp,
    track_lin_vel_xy_yaw_frame_exp,
)
from .terminations import illegal_contact, time_out

__all__ = [
    "action_rate_l2",
    "base_ang_vel",
    "base_lin_vel",
    "feet_air_time_positive_biped",
    "flat_orientation_l2",
    "generated_commands",
    "illegal_contact",
    "is_terminated",
    "joint_deviation_l1",
    "joint_pos_limits",
    "joint_pos_rel",
    "joint_vel",
    "joint_vel_rel",
    "last_action",
    "lin_vel_z_l2",
    "projected_gravity",
    "stand_still",
    "terrain_levels_vel",
    "time_out",
    "track_ang_vel_z_world_exp",
    "track_lin_vel_xy_yaw_frame_exp",
]
