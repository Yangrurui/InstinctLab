"""Locomotion rewards called directly by both engines."""

from __future__ import annotations

from typing import Any

import torch

from instinctlab.compat import math as math_utils
from instinctlab.compat import robot as compat_robot
from instinctlab.compat import sensors as compat_sensors
from instinctlab.compat.env import RlEnv, get_command
from instinctlab.spec.sensor import ContactSensorRef

from .observations import _joint_ids, _name


def _body_ids(asset_cfg: Any) -> Any:
    if asset_cfg is None:
        return slice(None)
    ids = getattr(asset_cfg, "body_ids", slice(None))
    return slice(None) if ids is None else ids


def is_terminated(env: RlEnv) -> torch.Tensor:
    return env.termination_manager.terminated.float()


def contact_slide(
    env: RlEnv,
    sensor_cfg: ContactSensorRef,
    asset_cfg: Any = None,
    ang_vel_penalty: bool = False,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize native body motion while the selected bodies are in contact."""
    sensor = env.scene.sensors[sensor_cfg.name]
    history = compat_sensors.contact_force_history(sensor, sensor_cfg)
    touching = torch.linalg.vector_norm(history, dim=-1).amax(dim=1) > threshold

    asset = env.scene[_name(asset_cfg)]
    body_ids = _body_ids(asset_cfg)
    linear = compat_robot.body_linear_velocity_w(env, asset)[:, body_ids, :2]
    penalty = torch.sum(torch.linalg.vector_norm(linear, dim=-1) * touching, dim=1)
    if ang_vel_penalty:
        angular = compat_robot.body_angular_velocity_w(env, asset)[:, body_ids, :2]
        penalty += torch.sum(
            torch.linalg.vector_norm(angular, dim=-1) * touching, dim=1
        )
    return penalty


def joint_acc_l2(env: RlEnv, asset_cfg: Any = None) -> torch.Tensor:
    """Penalize each engine's native joint-acceleration quantity."""
    asset = env.scene[_name(asset_cfg)]
    acceleration = compat_robot.joint_acceleration(env, asset)
    return torch.sum(torch.square(acceleration[:, _joint_ids(asset_cfg)]), dim=1)


def joint_torques_l2(env: RlEnv, asset_cfg: Any = None) -> torch.Tensor:
    """Penalize native joint-space actuator effort for the selected joints."""
    asset = env.scene[_name(asset_cfg)]
    torque = compat_robot.joint_applied_torque(env, asset)[:, _joint_ids(asset_cfg)]
    return torch.sum(torch.square(torque), dim=1)


def track_lin_vel_xy_yaw_frame_exp(
    env: RlEnv,
    std: float,
    command_name: str,
    asset_cfg: Any = None,
) -> torch.Tensor:
    asset = env.scene[_name(asset_cfg)]
    yaw_frame = math_utils.yaw_quat(asset.data.root_link_quat_w)
    vel_yaw = math_utils.quat_apply_inverse(
        yaw_frame, asset.data.root_link_lin_vel_w[:, :3]
    )
    error = torch.sum(
        torch.square(get_command(env, command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-error / std**2)


def track_ang_vel_z_world_exp(
    env: RlEnv,
    command_name: str,
    std: float,
    asset_cfg: Any = None,
) -> torch.Tensor:
    asset = env.scene[_name(asset_cfg)]
    error = torch.square(
        get_command(env, command_name)[:, 2] - asset.data.root_link_ang_vel_w[:, 2]
    )
    return torch.exp(-error / std**2)


def feet_air_time_positive_biped(
    env: RlEnv,
    command_name: str,
    threshold: float,
    sensor: ContactSensorRef,
) -> torch.Tensor:
    air_time = compat_sensors.air_time(env.scene.sensors[sensor.name], sensor)
    contact_time = compat_sensors.contact_time(env.scene.sensors[sensor.name], sensor)
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(
        torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1
    )[0]
    reward = torch.clamp(reward, max=threshold)
    return reward * (torch.norm(get_command(env, command_name)[:, :2], dim=1) > 0.1)


def flat_orientation_l2(env: RlEnv, asset_cfg: Any = None) -> torch.Tensor:
    asset = env.scene[_name(asset_cfg)]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def stand_still(env: RlEnv, command_name: str, asset_cfg: Any = None) -> torch.Tensor:
    asset = env.scene[_name(asset_cfg)]
    command = get_command(env, command_name)
    deviation = torch.sum(
        torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1
    )
    standing = (torch.norm(command[:, :2], dim=1) < 0.1) * (
        torch.abs(command[:, 2]) < 0.1
    )
    return deviation * standing


def joint_pos_limits(env: RlEnv, asset_cfg: Any = None) -> torch.Tensor:
    asset = env.scene[_name(asset_cfg)]
    joint_ids = _joint_ids(asset_cfg)
    joint_pos = asset.data.joint_pos[:, joint_ids]
    limits = asset.data.soft_joint_pos_limits[:, joint_ids]
    out_of_limits = -(joint_pos - limits[..., 0]).clip(max=0.0)
    out_of_limits += (joint_pos - limits[..., 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def joint_deviation_l1(env: RlEnv, asset_cfg: Any = None) -> torch.Tensor:
    asset = env.scene[_name(asset_cfg)]
    joint_ids = _joint_ids(asset_cfg)
    angle = (
        asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    )
    return torch.sum(torch.abs(angle), dim=1)


def lin_vel_z_l2(env: RlEnv, asset_cfg: Any = None) -> torch.Tensor:
    return torch.square(env.scene[_name(asset_cfg)].data.root_link_lin_vel_b[:, 2])


def action_rate_l2(env: RlEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
    )
