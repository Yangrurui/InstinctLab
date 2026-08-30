"""Shadowing reset, object-update, and adaptive-sampling events."""

from __future__ import annotations

import torch

from instinctlab_engine.bridge import math as math_utils


def _method(asset, *names):
    for name in names:
        method = getattr(asset, name, None)
        if callable(method):
            return method
    raise AttributeError(
        f"{type(asset).__name__} provides none of the required methods {names!r}"
    )


def _root_velocity_writer(asset, frame: str):
    """Resolve the reference backend's explicitly chosen root-velocity point."""
    if frame == "link":
        return _method(asset, "write_root_link_velocity_to_sim")
    if frame == "com":
        return _method(
            asset, "write_root_com_velocity_to_sim", "write_root_velocity_to_sim"
        )
    raise ValueError(f"root_velocity_frame must be 'link' or 'com', got {frame!r}")


def reference_joint_ids(asset, joint_names, *, device):
    """Resolve canonical motion joints to entity-local indices without losing their order."""
    expected = tuple(joint_names)
    joint_ids, resolved_names = asset.find_joints(expected, preserve_order=True)
    resolved = tuple(resolved_names)
    if resolved != expected:
        missing = tuple(name for name in expected if name not in resolved)
        unexpected = tuple(name for name in resolved if name not in expected)
        raise ValueError(
            "Motion-reference joints do not resolve one-to-one on the target articulation: "
            f"expected {expected!r}, resolved {resolved!r}, missing {missing!r}, unexpected {unexpected!r}."
        )
    return torch.as_tensor(joint_ids, dtype=torch.long, device=device)


def match_reference_origin(env, env_ids, motion_reference="motion_reference"):
    del env_ids
    sensor = env.scene[motion_reference]
    match_scene = getattr(sensor, "match_scene", None)
    if callable(match_scene):
        match_scene(env.scene)
    else:
        sensor.bind_origins(env.scene.env_origins)


def reset_robot_from_reference(
    env,
    env_ids,
    motion_reference="motion_reference",
    entity_name="robot",
    position_offset=(0.0, 0.0, 0.0),
    dof_vel_ratio=1.0,
    base_lin_vel_ratio=1.0,
    base_ang_vel_ratio=1.0,
    randomize_pose_range=None,
    randomize_velocity_range=None,
    randomize_joint_pos_range=(0.0, 0.0),
    root_velocity_frame="link",
):
    """Write the separately floor-indexed reset sample exactly once, in canonical order."""
    sensor = env.scene[motion_reference]
    state = sensor.init_reference_state
    asset = env.scene[entity_name]
    pos = state.base_pos_w[env_ids, 0].clone() + torch.as_tensor(
        position_offset, device=env.device
    )
    quat = state.base_quat_w[env_ids, 0].clone()
    if randomize_pose_range:
        keys = ("x", "y", "z", "roll", "pitch", "yaw")
        ranges = torch.tensor(
            [randomize_pose_range.get(key, (0.0, 0.0)) for key in keys],
            device=env.device,
        )
        sample = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device
        )
        pos += sample[:, :3]
        quat = math_utils.quat_mul(
            math_utils.quat_from_euler_xyz(*sample[:, 3:].T), quat
        )
    lin = state.base_lin_vel_w[env_ids, 0].clone() * base_lin_vel_ratio
    ang = state.base_ang_vel_w[env_ids, 0].clone() * base_ang_vel_ratio
    if randomize_velocity_range:
        keys = ("x", "y", "z", "roll", "pitch", "yaw")
        ranges = torch.tensor(
            [randomize_velocity_range.get(key, (0.0, 0.0)) for key in keys],
            device=env.device,
        )
        sample = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device
        )
        lin += sample[:, :3]
        ang += sample[:, 3:]
    pose = torch.cat((pos, quat), dim=-1)
    velocity = torch.cat((lin, ang), dim=-1)
    write_pose = _method(asset, "write_root_link_pose_to_sim", "write_root_pose_to_sim")
    write_velocity = _root_velocity_writer(asset, root_velocity_frame)
    write_pose(pose, env_ids=env_ids)
    write_velocity(velocity, env_ids=env_ids)
    joint_pos = state.joint_pos[env_ids, 0].clone()
    if randomize_joint_pos_range != (0.0, 0.0):
        joint_pos += math_utils.sample_uniform(
            *randomize_joint_pos_range, joint_pos.shape, device=env.device
        )
    joint_ids = reference_joint_ids(asset, sensor.joint_names, device=env.device)
    asset.write_joint_state_to_sim(
        joint_pos,
        state.joint_vel[env_ids, 0] * dof_vel_ratio,
        joint_ids=joint_ids,
        env_ids=env_ids,
    )


def smooth_bin_failures(
    env,
    env_ids,
    curriculum_name="beyond_adaptive_sampling",
    motion_reference="motion_reference",
):
    """Apply the same per-step EMA used by both reference implementations."""
    del env_ids, curriculum_name
    env.scene[motion_reference]._runtime.smooth_failures(alpha=0.001)


def adaptive_sampling(env, env_ids=None):
    """Record failed reset bins and update BeyondMimic sampling probabilities."""
    sensor = env.scene["motion_reference"]
    runtime = sensor._runtime
    if env_ids is None:
        env_ids = torch.arange(sensor.data.timestamp.shape[0], device=env.device)
    else:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    if env_ids.numel():
        failed = env.termination_manager.terminated[env_ids]
        elapsed = env.episode_length_buf[env_ids] * env.step_dt
        runtime.record_failures(env_ids, failed, elapsed)
    return runtime.update_adaptive_weights()


def _write_objects(
    env, env_ids, state, frame: int, invalid_object_pos=None, root_velocity_frame="link"
):
    names = state.scene_object_names
    validity = state.object_validity[env_ids, frame]
    for object_id, name in enumerate(names):
        asset = env.scene[name]
        valid = validity[:, object_id]
        selected = env_ids[valid]
        if selected.numel():
            pose = torch.cat(
                (
                    state.object_pos_w[env_ids, frame, object_id][valid],
                    state.object_quat_w[env_ids, frame, object_id][valid],
                ),
                dim=-1,
            )
            velocity = torch.cat(
                (
                    state.object_lin_vel_w[env_ids, frame, object_id][valid],
                    state.object_ang_vel_w[env_ids, frame, object_id][valid],
                ),
                dim=-1,
            )
            mocap = getattr(asset, "write_mocap_pose_to_sim", None)
            write_pose = (
                mocap
                if callable(mocap)
                else _method(
                    asset, "write_root_link_pose_to_sim", "write_root_pose_to_sim"
                )
            )
            write_pose(pose, env_ids=selected)
            if not callable(mocap):
                write_velocity = _root_velocity_writer(asset, root_velocity_frame)
                write_velocity(velocity, env_ids=selected)
        invalid = ~valid
        if invalid_object_pos is not None and invalid.any():
            selected = env_ids[invalid]
            pose = torch.zeros(len(selected), 7, device=env.device)
            pose[:, :3] = torch.as_tensor(invalid_object_pos, device=env.device)
            pose[:, 3] = 1.0
            mocap = getattr(asset, "write_mocap_pose_to_sim", None)
            writer = (
                mocap
                if callable(mocap)
                else _method(
                    asset, "write_root_link_pose_to_sim", "write_root_pose_to_sim"
                )
            )
            writer(pose, env_ids=selected)


def reset_objects_from_reference(
    env, env_ids, motion_reference="motion_reference", root_velocity_frame="link"
):
    sensor = env.scene[motion_reference]
    _write_objects(
        env,
        env_ids,
        sensor.init_reference_state,
        0,
        root_velocity_frame=root_velocity_frame,
    )


def update_objects_from_reference(
    env,
    env_ids,
    motion_reference="motion_reference",
    invalid_object_pos=None,
    root_velocity_frame="link",
):
    del env_ids
    sensor = env.scene[motion_reference]
    all_ids = torch.arange(sensor.data.timestamp.shape[0], device=env.device)
    _write_objects(
        env,
        all_ids,
        sensor.data,
        0,
        invalid_object_pos,
        root_velocity_frame=root_velocity_frame,
    )
