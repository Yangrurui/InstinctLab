"""Reward terms that run unmodified under either engine's native manager.

Most of these are the Isaac Lab bodies with hub attribute names substituted. Three rewards that
look portable are **not** here, and their absence is the useful part of this module:

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

``joint_vel_limits`` used to sit in that table. Isaac Lab reads ``soft_joint_vel_limits`` from
engine data and mjlab's ``EntityData`` has no equivalent field, so a port that asked either
engine for the cap would not be portable. The limits now come from the task declaration, which
reads them off :class:`~instinctlab.sim.robot_spec.RobotSpec` -- the catalog is this repo's
single source of truth for the robot. That is what makes the term portable, and it also removes
a dependency on an engine-derived value that the two engines compute differently.

Each is declared with ``kind=`` and implemented per engine, with a stated tolerance. That is
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
from collections.abc import Sequence
from typing import Any

from instinctlab.compat import math as math_utils
from instinctlab.compat import sensors as compat_sensors
from instinctlab.compat.env import RlEnv, get_command
from instinctlab.spec.sensor import ContactSensorRef, RayCasterRef, VolumePointsRef

from .observations import _body_index_list, _joint_ids, _name

__all__ = [
    "action_rate_l2",
    "ang_vel_xy_l2",
    "dont_wait",
    "feet_air_time",
    "feet_air_time_positive_biped",
    "feet_at_plane",
    "feet_close_xy_gauss",
    "feet_orientation_contact",
    "flat_orientation_l2",
    "heading_error",
    "is_alive",
    "is_terminated",
    "joint_deviation_l1",
    "joint_deviation_square",
    "joint_pos_limits",
    "joint_vel_l2",
    "joint_vel_limits",
    "lin_vel_z_l2",
    "link_orientation",
    "stand_still",
    "stand_still_when_idle",
    "track_ang_vel_z_exp",
    "track_ang_vel_z_world_exp",
    "track_lin_vel_xy_exp",
    "track_lin_vel_xy_yaw_frame_exp",
    "undesired_contacts",
    "volume_points_penetration",
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


def joint_vel_limits(
    env: RlEnv,
    soft_ratio: float,
    limits: Sequence[float],
    asset_cfg: Any = None,
) -> torch.Tensor:
    """Penalise joint velocities past a catalog-stated soft cap.

    Limits come from the task declaration, which reads them off :class:`~instinctlab.sim.robot_spec.RobotSpec`.
    Reading the catalog is what makes this portable: Isaac Lab's original reads
    ``soft_joint_vel_limits`` from engine data, and mjlab's ``EntityData`` has no equivalent
    field. It also removes a dependency on an engine-derived value that the two engines
    compute differently.

    The arithmetic is Isaac Lab's: absolute velocity minus ``limits * soft_ratio``, clipped
    to ``[0, 1]`` per joint and summed. ``limits`` must be listed in the same order as the
    selected joints -- a length mismatch raises rather than broadcasting a wrong cap.
    """
    asset = env.scene[_name(asset_cfg)]
    vel = torch.abs(asset.data.joint_vel[:, _joint_ids(asset_cfg)])
    cap = torch.as_tensor(limits, device=vel.device, dtype=vel.dtype)
    if cap.ndim != 1 or cap.shape[0] != vel.shape[-1]:
        raise RuntimeError(
            f"joint_vel_limits got {tuple(cap.shape)} limits for {vel.shape[-1]} selected joints. "
            "The catalog limits must be listed in the same order as the joints the term selects."
        )
    return torch.sum((vel - cap * soft_ratio).clamp(min=0.0, max=1.0), dim=1)


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


def is_alive(env: RlEnv) -> torch.Tensor:
    """Whether the episode is still running. The complement of :func:`is_terminated`."""
    return 1.0 - env.termination_manager.terminated.float()


def track_lin_vel_xy_exp(env: RlEnv, command_name: str, std: float, asset_cfg: Any = None) -> torch.Tensor:
    """Track the commanded planar velocity, measured in the base frame.

    Uses the root **link** velocity, where Isaac Lab's original reads the centre-of-mass alias
    ``root_lin_vel_b``. The two differ by ``ω × R(−com_pos_b)``, so this reward's error term is not
    bitwise identical to Isaac Lab's; the same call was already made for :func:`base_lin_vel` and
    :func:`track_lin_vel_xy_yaw_frame_exp`, and the reason is the same — the hub carries the
    quantity both engines express. mjlab's parkour term already reads the link spelling.

    Measured rather than assumed, since the difference grows with ``ω`` and parkour is where ``ω``
    is large: ``scripts/probe_velocity_frame.py`` scores both frames on the same rollout, so policy
    quality cancels. The two velocities differ by 0.067 m/s on average (the lever is 0.185 m, not
    the 0.076 m of the bare pelvis — ``merge_fixed_joints`` folds the torso into the root body) and
    the reward moves 0.2%. Mean tracking error is 0.41 m/s in both frames, and the exp kernel at
    ``std=0.5`` does not resolve a shift that small on top of it.

    That 0.2% did **not** by itself clear the frame of the gap against main, and reading it that way
    was a mistake worth naming: rescoring a fixed rollout answers "how much does this number move",
    while the gap asks "how much worse does a policy trained under this signal track", and a bias
    small at a point can still steer what gets learned. Attributing the return term by term, this
    reward carries 30% of the Isaac-vs-main gap with delayed actuators and 60% without, and
    ``dont_wait`` — the other term on this frame — carries 14-23%.

    So it was settled the only way it could be, by training: the whole cluster (both rewards, the
    command metrics, the observation) was flipped to main's COM spelling and retrained at
    256/seed42/700. It does not explain the gap. 85% of this term's shortfall survived the flip
    (-0.0345 -> -0.0292) and measured tracking error moved 0.2682 -> 0.2634 against main's 0.2435.
    Return went 0.847 -> 0.905, which is inside the 12% seed noise floor and so is not a claim
    either. We track worse than main for a reason that is not this.
    """
    asset = env.scene[_name(asset_cfg)]
    error = torch.sum(
        torch.square(get_command(env, command_name)[:, :2] - asset.data.root_link_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-error / std**2)


def track_ang_vel_z_exp(env: RlEnv, command_name: str, std: float, asset_cfg: Any = None) -> torch.Tensor:
    """Track the commanded yaw rate, in the base frame.

    Value-for-value identical to Isaac Lab's despite reading the link spelling, for the reason set
    out in :mod:`instinctlab.mdp.observations`: Isaac Lab copies the angular rows between its
    centre-of-mass and link velocity buffers untouched. Already measured and pinned.
    """
    asset = env.scene[_name(asset_cfg)]
    error = torch.square(get_command(env, command_name)[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
    return torch.exp(-error / std**2)


def ang_vel_xy_l2(env: RlEnv, asset_cfg: Any = None) -> torch.Tensor:
    """Penalise xy body angular velocity. Angular link and COM spellings are identical here."""
    asset = env.scene[_name(asset_cfg)]
    return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def joint_vel_l2(env: RlEnv, asset_cfg: Any = None) -> torch.Tensor:
    """Penalise squared joint velocity over the selected joints."""
    asset = env.scene[_name(asset_cfg)]
    return torch.sum(torch.square(asset.data.joint_vel[:, _joint_ids(asset_cfg)]), dim=1)


def stand_still_when_idle(
    env: RlEnv,
    command_name: str,
    asset_cfg: Any = None,
    threshold: float = 0.15,
    offset: float = 1.0,
) -> torch.Tensor:
    """Parkour's idle-pose penalty: joint deviation minus ``offset``, gated on a near-zero command.

    :func:`stand_still` is the flat/rough G1 term — L1 deviation, no offset, a 0.1 command gate.
    Changing that function would move those tasks' reward and trip the declaration snapshot.
    Parkour subtracts ``offset`` and uses a 0.15 gate; the two are not the same function.
    """
    asset = env.scene[_name(asset_cfg)]
    command = get_command(env, command_name)
    deviation = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    standing = (torch.norm(command[:, :2], dim=1) < threshold) * (torch.abs(command[:, 2]) < threshold)
    return (deviation - offset) * standing


def heading_error(env: RlEnv, command_name: str) -> torch.Tensor:
    """Magnitude of the commanded yaw rate. Both parkour references compute exactly this."""
    return torch.abs(get_command(env, command_name)[:, 2])


def dont_wait(env: RlEnv, command_name: str, asset_cfg: Any = None) -> torch.Tensor:
    """Penalise standing still when there is a forward velocity command.

    Reads the hub spelling ``root_link_lin_vel_b``. Isaac Lab's parkour term reads the
    centre-of-mass alias ``root_lin_vel_b``; see :func:`track_lin_vel_xy_exp`.
    """
    asset = env.scene[_name(asset_cfg)]
    lin_vel_cmd_x = get_command(env, command_name)[:, 0]
    lin_vel_x = asset.data.root_link_lin_vel_b[:, 0]
    return (lin_vel_cmd_x > 0.3) * (
        (lin_vel_x < 0.15).float() + (lin_vel_x < 0.0).float() + (lin_vel_x < -0.15).float()
    )


def feet_air_time(
    env: RlEnv,
    command_name: str,
    sensor: ContactSensorRef,
    vel_threshold: float,
    threshold: float | None = None,
) -> torch.Tensor:
    """Reward single-stance step timing. Parkour's variant, not :func:`feet_air_time_positive_biped`.

    The biped term clamps each step at ``threshold`` seconds and gates on a hardcoded 0.1 m/s
    planar command. Parkour's references do not clamp, and they gate on ``vel_threshold`` against
    either the planar command or the yaw command. ``threshold`` is optional so a caller can add
    the biped cap without changing the parkour default.
    """
    air_time = compat_sensors.air_time(env.scene.sensors[sensor.name], sensor)
    contact_time = compat_sensors.contact_time(env.scene.sensors[sensor.name], sensor)
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    if threshold is not None:
        reward = torch.clamp(reward, max=threshold)
    command = get_command(env, command_name)
    reward = reward * torch.logical_or(
        torch.norm(command[:, :2], dim=1) > vel_threshold,
        torch.abs(command[:, 2]) > vel_threshold,
    )
    return reward


def joint_deviation_square(env: RlEnv, asset_cfg: Any = None) -> torch.Tensor:
    """Penalise squared joint deviation from the default pose."""
    asset = env.scene[_name(asset_cfg)]
    joint_ids = _joint_ids(asset_cfg)
    angle = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    return torch.sum(torch.square(angle), dim=1)


def link_orientation(env: RlEnv, asset_cfg: Any = None) -> torch.Tensor:
    """Penalise tilt of the first selected link.

    Both parkour references project world gravity through that link's quaternion. Isaac Lab reads
    ``GRAVITY_VEC_W`` and the legacy ``body_quat_w``; mjlab reads ``gravity_vec_w`` and
    ``body_link_quat_w``. The raw gravity vectors are denylisted — different spelling and different
    behaviour under non-default gravity — so this reconstructs world gravity from the hub pair
    ``root_link_quat_w`` / ``projected_gravity_b`` and rotates it into the link with
    :func:`instinctlab.compat.math.quat_apply_inverse`.
    """
    asset = env.scene[_name(asset_cfg)]
    ids = _body_index_list(asset_cfg, asset.data.body_link_quat_w.shape[1])
    if not ids:
        raise RuntimeError("link_orientation needs a body to penalise; the selector matched none.")
    gravity_w = math_utils.quat_apply(asset.data.root_link_quat_w, asset.data.projected_gravity_b)
    link_quat = asset.data.body_link_quat_w[:, ids[0], :]
    link_projected_gravity = math_utils.quat_apply_inverse(link_quat, gravity_w)
    return torch.sum(torch.square(link_projected_gravity[:, :2]), dim=1)


def feet_orientation_contact(env: RlEnv, sensor: ContactSensorRef, asset_cfg: Any = None) -> torch.Tensor:
    """Penalise foot tilt, only while that foot is in contact.

    Isaac Lab gated on ``‖net_forces_w‖ > 1`` N (world-frame normal only). That threshold is
    forbidden here: mjlab's ``force`` is a different physical quantity. Contact is
    :func:`instinctlab.compat.sensors.in_contact`, so a light touch now counts where the 1 N
    threshold would not. Gravity for each foot is reconstructed the same way as
    :func:`link_orientation`.
    """
    asset = env.scene[_name(asset_cfg)]
    ids = _body_index_list(asset_cfg, asset.data.body_link_quat_w.shape[1])
    if not ids:
        raise RuntimeError("feet_orientation_contact needs at least one body; the selector matched none.")
    quats = asset.data.body_link_quat_w[:, ids, :]
    n_envs, n_feet = quats.shape[:2]
    gravity_w = math_utils.quat_apply(asset.data.root_link_quat_w, asset.data.projected_gravity_b)
    gravity_w = gravity_w.unsqueeze(1).expand(-1, n_feet, -1)
    projected = math_utils.quat_apply_inverse(quats.reshape(-1, 4), gravity_w.reshape(-1, 3)).reshape(n_envs, n_feet, 3)
    tilt = torch.linalg.vector_norm(projected[:, :, :2], dim=-1)
    touching = compat_sensors.in_contact(env.scene.sensors[sensor.name], sensor)
    return torch.sum(tilt * touching.float(), dim=1)


def feet_at_plane(
    env: RlEnv,
    sensor: ContactSensorRef,
    left_scanner: RayCasterRef,
    right_scanner: RayCasterRef,
    asset_cfg: Any = None,
    height_offset: float = 0.035,
) -> torch.Tensor:
    """Penalise a stance foot sitting higher than the scanned terrain, plus ``height_offset``.

    Both parkour references converted a ray miss into ``hit_z = 0``. That is the silent
    failure this term exists to refuse: a miss then means "ground at world height zero",
    which is a number the clamp happily eats. The IR's ``miss="infinity"`` comes through
    :func:`instinctlab.compat.sensors.ray_hits_w` as ``+inf``; ``foot_z - inf`` clamps to
    zero, so a gap contributes nothing instead of a fake plane.

    Contact is :func:`instinctlab.compat.sensors.in_contact`, not a newton threshold.
    Foot height is ``body_link_pos_w`` of the first two selected bodies (left, then right).
    """
    asset = env.scene[_name(asset_cfg)]
    ids = _body_index_list(asset_cfg, asset.data.body_link_pos_w.shape[1])
    if len(ids) < 2:
        raise RuntimeError(f"feet_at_plane needs two bodies, got {len(ids)}.")
    touching = compat_sensors.in_contact(env.scene.sensors[sensor.name], sensor)
    left_hit_z = compat_sensors.ray_hits_w(env.scene.sensors[left_scanner.name])[..., 2]
    right_hit_z = compat_sensors.ray_hits_w(env.scene.sensors[right_scanner.name])[..., 2]
    left_height = asset.data.body_link_pos_w[:, ids[0], 2].unsqueeze(-1)
    right_height = asset.data.body_link_pos_w[:, ids[1], 2].unsqueeze(-1)
    left_reward = torch.clamp(left_height - left_hit_z - height_offset, min=0.0, max=0.3) * touching[:, 0:1].float()
    right_reward = torch.clamp(right_height - right_hit_z - height_offset, min=0.0, max=0.3) * touching[:, 1:2].float()
    return torch.sum(left_reward, dim=-1) + torch.sum(right_reward, dim=-1)


def feet_close_xy_gauss(env: RlEnv, threshold: float, std: float = 0.1, asset_cfg: Any = None) -> torch.Tensor:
    """Reward the first two selected bodies staying at least ``threshold`` apart in body-frame y.

    Reads ``body_link_pos_w`` and ``heading_w``. Isaac Lab's parkour term used the legacy
    ``body_pos_w`` alias, which is the same link quantity.
    """
    asset = env.scene[_name(asset_cfg)]
    ids = _body_index_list(asset_cfg, asset.data.body_link_pos_w.shape[1])
    if len(ids) < 2:
        raise RuntimeError(f"feet_close_xy_gauss needs two bodies, got {len(ids)}.")
    left_xy = asset.data.body_link_pos_w[:, ids[0], :2]
    right_xy = asset.data.body_link_pos_w[:, ids[1], :2]
    heading_w = asset.data.heading_w
    cos_heading = torch.cos(heading_w)
    sin_heading = torch.sin(heading_w)
    left_y = -sin_heading * left_xy[:, 0] + cos_heading * left_xy[:, 1]
    right_y = -sin_heading * right_xy[:, 0] + cos_heading * right_xy[:, 1]
    feet_distance_y = torch.abs(left_y - right_y)
    return torch.exp(-torch.clamp(threshold - feet_distance_y, min=0.0) / std**2) - 1


def undesired_contacts(env: RlEnv, sensor: ContactSensorRef) -> torch.Tensor:
    """Count referenced elements that are in contact.

    Isaac Lab's original thresholds ``‖net_forces_w‖`` at 1 N. That is forbidden here: Isaac Lab's
    ``net_forces_w`` is world-frame normal-only, mjlab's ``force`` is full 3-D in the contact frame.
    Contact is :func:`instinctlab.compat.sensors.in_contact`. A light touch now counts as contact
    where the newton threshold would not; a task that needs the threshold should declare this
    reward per engine.
    """
    touching = compat_sensors.in_contact(env.scene.sensors[sensor.name], sensor)
    return torch.sum(touching, dim=1)


def volume_points_penetration(env: RlEnv, sensor: VolumePointsRef, tolerance: float = 0.0) -> torch.Tensor:
    """Penalise points inside a registered virtual obstacle, weighted by speed.

    Both parkour sources: ``sum (in_obstacle * (|v| + 1e-6) * depth)``. Depth is
    ``‖penetration_offset‖``. The offset is world-frame, surface → point; velocity
    is the attach-body link velocity plus ``ω × r``. An unregistered sensor is
    refused by the compat reader — it is not a well-behaved zero.
    """
    volume = env.scene.sensors[sensor.name]
    penetration = compat_sensors.volume_points_penetration_offset(volume).flatten(1, 2)
    depth = torch.linalg.vector_norm(penetration, dim=-1)
    in_obstacle = (depth > tolerance).float()
    speed = torch.linalg.vector_norm(compat_sensors.volume_points_vel_w(volume).flatten(1, 2), dim=-1)
    return torch.sum(in_obstacle * (speed + 1e-6) * depth, dim=-1)
