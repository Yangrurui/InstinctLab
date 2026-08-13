"""Engine-neutral Locomotion MDP terms operating on canonical scene views."""

from __future__ import annotations

import torch
from functools import lru_cache

from instinctlab.sim.backend import MaterialProperties
from instinctlab.sim.math import quat_apply_inverse, yaw_quaternion


def _robot(env):
    return env.scene.articulations[env.cfg.scene.primary_entity]


def _entity_name(env) -> str:
    return env.cfg.scene.primary_entity


@lru_cache(maxsize=None)
def _cached_ids(
    names: tuple[str, ...],
    selected: tuple[str, ...],
    device: str,
) -> torch.Tensor:
    return torch.tensor([names.index(name) for name in selected], device=device, dtype=torch.int64)


def _ids(names: tuple[str, ...], selected: tuple[str, ...], device: torch.device) -> torch.Tensor:
    return _cached_ids(names, selected, str(device))


# Observations.


def base_ang_vel(env) -> torch.Tensor:
    data = _robot(env).data
    return quat_apply_inverse(data.root_quat_w, data.root_ang_vel_w)


def base_lin_vel(env) -> torch.Tensor:
    data = _robot(env).data
    return quat_apply_inverse(data.root_quat_w, data.root_lin_vel_w)


def projected_gravity(env) -> torch.Tensor:
    data = _robot(env).data
    gravity = torch.zeros_like(data.root_lin_vel_w)
    gravity[:, 2] = -1.0
    return quat_apply_inverse(data.root_quat_w, gravity)


def velocity_commands(env, command_name: str = "base_velocity") -> torch.Tensor:
    return env.command_manager.get_command(command_name)


def joint_pos_rel(env) -> torch.Tensor:
    data = _robot(env).data
    return data.joint_pos - data.default_joint_pos


def joint_vel(env) -> torch.Tensor:
    return _robot(env).data.joint_vel


def last_action(env) -> torch.Tensor:
    return env.action_manager.action


# Rewards.


def is_terminated(env) -> torch.Tensor:
    return env.termination_manager.terminated.float()


def track_lin_vel_xy_yaw_frame_exp(env, std: float, command_name: str) -> torch.Tensor:
    data = _robot(env).data
    velocity_yaw = quat_apply_inverse(yaw_quaternion(data.root_quat_w), data.root_lin_vel_w)
    error = torch.sum(torch.square(env.command_manager.get_command(command_name)[:, :2] - velocity_yaw[:, :2]), dim=1)
    return torch.exp(-error / std**2)


def track_ang_vel_z_world_exp(env, std: float, command_name: str) -> torch.Tensor:
    error = torch.square(env.command_manager.get_command(command_name)[:, 2] - _robot(env).data.root_ang_vel_w[:, 2])
    return torch.exp(-error / std**2)


def feet_air_time_positive_biped(
    env,
    command_name: str,
    threshold: float,
    sensor_name: str,
    body_names: tuple[str, ...],
) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_name]
    body_ids = _ids(sensor.body_names, body_names, env.device)
    air_time = sensor.current_air_time[:, body_ids]
    contact_time = sensor.current_contact_time[:, body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1).values
    reward = torch.clamp(reward, max=threshold)
    return reward * (torch.linalg.vector_norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1)


def contact_slide(env, sensor_name: str, body_names: tuple[str, ...]) -> torch.Tensor:
    robot = _robot(env)
    sensor = env.scene.sensors[sensor_name]
    sensor_ids = _ids(sensor.body_names, body_names, env.device)
    body_ids = _ids(robot.body_names, body_names, env.device)
    contact = sensor.contact_active[:, sensor_ids]
    speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, body_ids, :2], dim=-1)
    return torch.sum(contact * speed, dim=1)


def flat_orientation_l2(env) -> torch.Tensor:
    return torch.sum(torch.square(projected_gravity(env)[:, :2]), dim=1)


def stand_still(env, command_name: str) -> torch.Tensor:
    data = _robot(env).data
    command = env.command_manager.get_command(command_name)
    still = (torch.linalg.vector_norm(command[:, :2], dim=1) < 0.1) & (torch.abs(command[:, 2]) < 0.1)
    return torch.sum(torch.abs(data.joint_pos - data.default_joint_pos), dim=1) * still


def joint_pos_limits(env, joint_names: tuple[str, ...]) -> torch.Tensor:
    robot = _robot(env)
    joint_ids = _ids(robot.joint_names, joint_names, env.device)
    position = robot.data.joint_pos[:, joint_ids]
    limits = robot.data.soft_joint_pos_limits[:, joint_ids]
    below = torch.clamp(limits[..., 0] - position, min=0.0)
    above = torch.clamp(position - limits[..., 1], min=0.0)
    return torch.sum(below + above, dim=1)


def joint_deviation_l1(env, joint_names: tuple[str, ...]) -> torch.Tensor:
    robot = _robot(env)
    joint_ids = _ids(robot.joint_names, joint_names, env.device)
    return torch.sum(
        torch.abs(robot.data.joint_pos[:, joint_ids] - robot.data.default_joint_pos[:, joint_ids]),
        dim=1,
    )


def lin_vel_z_l2(env) -> torch.Tensor:
    return torch.square(base_lin_vel(env)[:, 2])


def action_rate_l2(env) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def joint_acc_l2(env, joint_names: tuple[str, ...]) -> torch.Tensor:
    robot = _robot(env)
    joint_ids = _ids(robot.joint_names, joint_names, env.device)
    return torch.sum(torch.square(robot.data.joint_acc[:, joint_ids]), dim=1)


def joint_torques_l2(env, joint_names: tuple[str, ...]) -> torch.Tensor:
    robot = _robot(env)
    joint_ids = _ids(robot.joint_names, joint_names, env.device)
    return torch.sum(torch.square(robot.data.applied_joint_effort[:, joint_ids]), dim=1)


# Terminations.


def time_out(env) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


def illegal_contact(env, sensor_name: str, body_names: tuple[str, ...]) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_name]
    body_ids = _ids(sensor.body_names, body_names, env.device)
    return torch.any(sensor.contact_active_history[:, :, body_ids], dim=(1, 2))


# Events.


def reset_root_state_uniform(
    env,
    env_ids: torch.Tensor,
    *,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
) -> bool:
    robot = _robot(env)
    count = int(env_ids.numel())
    position = torch.tensor(env.cfg.scene.robot.default_root_pos, device=env.device).repeat(count, 1)
    position += env.scene.env_origins[env_ids]
    position[:, 0] += env.rng.uniform("reset.root.x", *pose_range.get("x", (0.0, 0.0)), (count,))
    position[:, 1] += env.rng.uniform("reset.root.y", *pose_range.get("y", (0.0, 0.0)), (count,))
    position[:, 2] += env.rng.uniform("reset.root.z", *pose_range.get("z", (0.0, 0.0)), (count,))
    yaw = env.rng.uniform("reset.root.yaw", *pose_range.get("yaw", (0.0, 0.0)), (count,))
    quaternion = torch.zeros((count, 4), device=env.device)
    quaternion[:, 0] = torch.cos(0.5 * yaw)
    quaternion[:, 3] = torch.sin(0.5 * yaw)
    velocity = torch.stack(
        [
            env.rng.uniform(f"reset.root.velocity.{axis}", *velocity_range.get(axis, (0.0, 0.0)), (count,))
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        ],
        dim=1,
    )
    env.backend.write_root_state(_entity_name(env), torch.cat((position, quaternion, velocity), dim=1), env_ids)
    return True


def reset_joints_by_scale(
    env,
    env_ids: torch.Tensor,
    *,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
) -> bool:
    data = _robot(env).data
    count = int(env_ids.numel())
    position = data.default_joint_pos[env_ids] * env.rng.uniform(
        "reset.joints.position", *position_range, (count, data.num_joints)
    )
    position = torch.maximum(
        torch.minimum(position, data.soft_joint_pos_limits[env_ids, :, 1]),
        data.soft_joint_pos_limits[env_ids, :, 0],
    )
    # Existing Isaac/MJLab ``reset_joints_by_scale`` scales the nominal joint
    # velocity (typically zero at default pose); keep that behavior instead of
    # sampling absolute velocity.
    velocity_scale = env.rng.uniform("reset.joints.velocity", *velocity_range, (count, data.num_joints))
    velocity = torch.zeros_like(position) * velocity_scale
    env.backend.write_joint_state(_entity_name(env), position, velocity, env_ids)
    return True


def randomize_sliding_friction(
    env,
    env_ids: torch.Tensor,
    *,
    body_names: tuple[str, ...],
    friction_range: tuple[float, float],
) -> bool:
    robot = _robot(env)
    body_ids = _ids(robot.body_names, body_names, env.device)
    values = env.rng.uniform(
        "event.material.sliding_friction",
        *friction_range,
        (int(env_ids.numel()), int(body_ids.numel())),
    )
    env.backend.set_body_material(
        MaterialProperties(
            entity_name=_entity_name(env),
            body_ids=body_ids,
            env_ids=env_ids,
            sliding_friction=values,
            restitution=None,
        )
    )
    return False


def push_by_setting_velocity(
    env,
    env_ids: torch.Tensor,
    *,
    velocity_range: dict[str, tuple[float, float]],
) -> bool:
    data = _robot(env).data
    state = torch.cat(
        (
            data.root_pos_w[env_ids],
            data.root_quat_w[env_ids],
            data.root_lin_vel_w[env_ids],
            data.root_ang_vel_w[env_ids],
        ),
        dim=1,
    ).clone()
    for index, axis in enumerate(("x", "y", "z")):
        if axis in velocity_range:
            state[:, 7 + index] += env.rng.uniform(f"event.push.{axis}", *velocity_range[axis], (int(env_ids.numel()),))
    env.backend.write_root_state(_entity_name(env), state, env_ids)
    return True


__all__ = [
    "action_rate_l2",
    "base_ang_vel",
    "base_lin_vel",
    "contact_slide",
    "feet_air_time_positive_biped",
    "flat_orientation_l2",
    "illegal_contact",
    "is_terminated",
    "joint_acc_l2",
    "joint_deviation_l1",
    "joint_pos_limits",
    "joint_pos_rel",
    "joint_torques_l2",
    "joint_vel",
    "last_action",
    "lin_vel_z_l2",
    "projected_gravity",
    "push_by_setting_velocity",
    "randomize_sliding_friction",
    "reset_joints_by_scale",
    "reset_root_state_uniform",
    "stand_still",
    "time_out",
    "track_ang_vel_z_world_exp",
    "track_lin_vel_xy_yaw_frame_exp",
    "velocity_commands",
]
