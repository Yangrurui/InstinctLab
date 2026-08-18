"""Reward terms that run unmodified under either engine's native manager.

Most of these are the Isaac Lab bodies with hub attribute names substituted. Three of the golden
task's rewards are **not** here, and their absence is the useful part of this module, because each
one looks portable and is not:

===================  ===============================================================================
term                 why it cannot be a portable term
===================  ===============================================================================
``dof_acc_l2``       Reads ``joint_acc``. Isaac Lab finite-differences joint velocity across steps;
                     mjlab reads MuJoCo's analytic ``qacc``. Near a contact these disagree by more
                     than the reward's own scale, and the weight is ``-2e-7`` -- small enough that
                     the disagreement shows up as a slightly different gait rather than as anything
                     one would investigate.
``dof_torques_l2``   Reads ``applied_torque``, which mjlab does not have. Its joint-space
                     equivalent is ``qfrc_actuator``; ``actuator_force`` is the false friend, being
                     actuation-space (nu) rather than joint-space (nv).
``contact_slide``    Reads a newton threshold on contact force *and* per-body linear velocity.
                     Neither ports: the two engines report different force quantities (normal-only
                     against full 3-D), and the hub deliberately carries no per-body velocity,
                     because Isaac Lab offsets each body's velocity to its own centre of mass while
                     mjlab reports about the root's subtree centre.
===================  ===============================================================================

All three are declared with ``kind=`` and implemented per engine, with a stated tolerance. That is
not a workaround -- it is the design working: the alternative is a term that produces plausible
numbers on both engines and quietly optimises something different on each.

Contact rewards go through :mod:`instinctlab.compat.sensors` and take a
:class:`~instinctlab.spec.sensor.ContactSensorRef` in place of Isaac Lab's ``sensor_cfg``. That is
the one signature change a migrated task has to make, and it is unavoidable: Isaac Lab declares one
broad sensor that terms slice, mjlab declares narrow sensors that terms read whole, so there is no
single native object to pass.
"""

from __future__ import annotations

import torch
from typing import Any

from instinctlab.compat import math as math_utils
from instinctlab.compat import sensors as compat_sensors
from instinctlab.compat.env import RlEnv, get_command
from instinctlab.spec.sensor import ContactSensorRef

from .observations import _joint_ids, _name

__all__ = [
    "action_rate_l2",
    "feet_air_time_positive_biped",
    "flat_orientation_l2",
    "is_terminated",
    "joint_deviation_l1",
    "joint_pos_limits",
    "lin_vel_z_l2",
    "stand_still",
    "track_ang_vel_z_world_exp",
    "track_lin_vel_xy_yaw_frame_exp",
]


def is_terminated(env: RlEnv) -> torch.Tensor:
    """Whether the episode ended for a reason other than the time limit."""
    return env.termination_manager.terminated.float()


def track_lin_vel_xy_yaw_frame_exp(env: RlEnv, std: float, command_name: str, asset_cfg: Any = None) -> torch.Tensor:
    """Track the commanded planar velocity, measured in the gravity-aligned robot frame.

    Uses the root **link** velocity, where Isaac Lab's original reads the centre-of-mass alias
    ``root_lin_vel_w``. The two differ by ``ω × R(−com_pos_b)``, so this reward's error term is not
    bitwise identical to the golden's; the difference belongs in the whitelist with that reason.
    Unlike the critic observation, this one does shape the policy, so the whitelist entry is the
    place a reviewer gets to object.
    """
    asset = env.scene[_name(asset_cfg)]
    yaw_frame = math_utils.yaw_quat(asset.data.root_link_quat_w)
    vel_yaw = math_utils.quat_apply_inverse(yaw_frame, asset.data.root_link_lin_vel_w[:, :3])
    error = torch.sum(torch.square(get_command(env, command_name)[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-error / std**2)


def track_ang_vel_z_world_exp(env: RlEnv, command_name: str, std: float, asset_cfg: Any = None) -> torch.Tensor:
    """Track the commanded yaw rate, in the world frame.

    Value-for-value identical to the golden despite reading the link spelling, for the reason set
    out in :mod:`instinctlab.mdp.observations`: Isaac Lab copies the angular rows between its
    centre-of-mass and link velocity buffers untouched.
    """
    asset = env.scene[_name(asset_cfg)]
    error = torch.square(get_command(env, command_name)[:, 2] - asset.data.root_link_ang_vel_w[:, 2])
    return torch.exp(-error / std**2)


def feet_air_time_positive_biped(
    env: RlEnv, command_name: str, threshold: float, sensor: ContactSensorRef
) -> torch.Tensor:
    """Reward long single-stance steps, up to ``threshold`` seconds each.

    The step-timing signals this reads are the portable part of contact sensing. Each engine
    computes air and contact durations inside its own sensor, from its own solver's forces and its
    own contact criterion, so by the time a duration in seconds reaches this function the two
    engines have already reconciled. A force threshold would not have that property.
    """
    air_time = compat_sensors.air_time(env.scene.sensors[sensor.name], sensor)
    contact_time = compat_sensors.contact_time(env.scene.sensors[sensor.name], sensor)
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    return reward * (torch.norm(get_command(env, command_name)[:, :2], dim=1) > 0.1)


def flat_orientation_l2(env: RlEnv, asset_cfg: Any = None) -> torch.Tensor:
    """Penalise a non-upright base. Identical to Isaac Lab's, which mjlab also copied verbatim."""
    asset = env.scene[_name(asset_cfg)]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def stand_still(env: RlEnv, command_name: str, asset_cfg: Any = None) -> torch.Tensor:
    """Penalise joint deviation from the default pose while the command is near zero."""
    asset = env.scene[_name(asset_cfg)]
    command = get_command(env, command_name)
    deviation = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    standing = (torch.norm(command[:, :2], dim=1) < 0.1) * (torch.abs(command[:, 2]) < 0.1)
    return deviation * standing


def joint_pos_limits(env: RlEnv, asset_cfg: Any = None) -> torch.Tensor:
    """Penalise joint positions past their soft limits."""
    asset = env.scene[_name(asset_cfg)]
    joint_ids = _joint_ids(asset_cfg)
    joint_pos = asset.data.joint_pos[:, joint_ids]
    limits = asset.data.soft_joint_pos_limits[:, joint_ids]
    out_of_limits = -(joint_pos - limits[..., 0]).clip(max=0.0)
    out_of_limits += (joint_pos - limits[..., 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def joint_deviation_l1(env: RlEnv, asset_cfg: Any = None) -> torch.Tensor:
    """Penalise joint positions that drift from the default pose."""
    asset = env.scene[_name(asset_cfg)]
    joint_ids = _joint_ids(asset_cfg)
    angle = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def lin_vel_z_l2(env: RlEnv, asset_cfg: Any = None) -> torch.Tensor:
    """Penalise vertical base velocity. Reads the link quantity; see :func:`base_lin_vel`."""
    asset = env.scene[_name(asset_cfg)]
    return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def action_rate_l2(env: RlEnv) -> torch.Tensor:
    """Penalise fast changes in the action. Identical on both engines."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
