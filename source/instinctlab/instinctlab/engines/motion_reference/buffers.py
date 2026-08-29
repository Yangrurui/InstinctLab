"""Per-environment motion-reference buffers and look-ahead sampling."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Literal

from instinctlab.compat.motion_reference import clip_frame, exhausted_envs

from .clip import MotionSample


@dataclass
class MotionReferenceBuffers:
    """Per-env output the sensors expose. Exhaustion is a counter, not a silent restart."""

    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    base_pos_w: torch.Tensor
    base_quat_w: torch.Tensor
    base_lin_vel_w: torch.Tensor
    base_ang_vel_w: torch.Tensor
    link_pos_b: torch.Tensor
    link_quat_b: torch.Tensor
    link_pos_w: torch.Tensor
    link_quat_w: torch.Tensor
    link_lin_vel_b: torch.Tensor
    link_ang_vel_b: torch.Tensor
    link_lin_vel_w: torch.Tensor
    link_ang_vel_w: torch.Tensor
    validity: torch.Tensor
    time_to_target_frame: torch.Tensor
    exhausted_count: torch.Tensor
    ever_exhausted: torch.Tensor
    timestamp: torch.Tensor
    start_s: torch.Tensor
    motion_id: torch.Tensor
    frame_index: torch.Tensor
    scene_object_names: tuple[str, ...]
    object_pos_w: torch.Tensor
    object_quat_w: torch.Tensor
    object_lin_vel_w: torch.Tensor
    object_ang_vel_w: torch.Tensor
    object_validity: torch.Tensor


def make_buffers(
    num_envs: int,
    num_frames: int,
    num_joints: int,
    num_links: int,
    scene_object_names: tuple[str, ...] = (),
    device: torch.device | str = "cpu",
) -> MotionReferenceBuffers:
    zeros_pose = torch.zeros(num_envs, num_frames, 3, device=device)
    zeros_quat = torch.zeros(num_envs, num_frames, 4, device=device)
    zeros_quat[..., 0] = 1.0
    num_objects = len(scene_object_names)
    object_quat = torch.zeros(num_envs, num_frames, num_objects, 4, device=device)
    object_quat[..., 0] = 1.0
    return MotionReferenceBuffers(
        joint_pos=torch.zeros(num_envs, num_frames, num_joints, device=device),
        joint_vel=torch.zeros(num_envs, num_frames, num_joints, device=device),
        base_pos_w=zeros_pose.clone(),
        base_quat_w=zeros_quat.clone(),
        base_lin_vel_w=zeros_pose.clone(),
        base_ang_vel_w=zeros_pose.clone(),
        link_pos_b=torch.zeros(num_envs, num_frames, num_links, 3, device=device),
        link_quat_b=torch.zeros(num_envs, num_frames, num_links, 4, device=device),
        link_pos_w=torch.zeros(num_envs, num_frames, num_links, 3, device=device),
        link_quat_w=torch.zeros(num_envs, num_frames, num_links, 4, device=device),
        link_lin_vel_b=torch.zeros(num_envs, num_frames, num_links, 3, device=device),
        link_ang_vel_b=torch.zeros(num_envs, num_frames, num_links, 3, device=device),
        link_lin_vel_w=torch.zeros(num_envs, num_frames, num_links, 3, device=device),
        link_ang_vel_w=torch.zeros(num_envs, num_frames, num_links, 3, device=device),
        validity=torch.zeros(num_envs, num_frames, dtype=torch.bool, device=device),
        time_to_target_frame=torch.zeros(num_envs, num_frames, device=device),
        exhausted_count=torch.zeros(num_envs, dtype=torch.long, device=device),
        ever_exhausted=torch.zeros(num_envs, dtype=torch.bool, device=device),
        timestamp=torch.zeros(num_envs, device=device),
        start_s=torch.zeros(num_envs, device=device),
        motion_id=torch.zeros(num_envs, dtype=torch.long, device=device),
        frame_index=torch.zeros(num_envs, num_frames, dtype=torch.long, device=device),
        scene_object_names=scene_object_names,
        object_pos_w=torch.zeros(num_envs, num_frames, num_objects, 3, device=device),
        object_quat_w=object_quat,
        object_lin_vel_w=torch.zeros(
            num_envs, num_frames, num_objects, 3, device=device
        ),
        object_ang_vel_w=torch.zeros(
            num_envs, num_frames, num_objects, 3, device=device
        ),
        object_validity=torch.zeros(
            num_envs, num_frames, num_objects, dtype=torch.bool, device=device
        ),
    )


def envs_due_for_update(
    timestamp: torch.Tensor,
    last_update: torch.Tensor,
    update_period: float,
) -> torch.Tensor:
    """Env ids whose clip clock has reached ``update_period``.

    Both engines advance a per-env timestamp by the physics dt. Refreshing on
    every substep would make ``exhausted_count`` grow with ``decimation`` on one
    side only. Period 0 means every call is due (Isaac's SensorBase convention).
    """
    if timestamp.shape != last_update.shape:
        raise ValueError("timestamp and last_update must have the same shape.")
    if update_period <= 0.0:
        return torch.arange(timestamp.shape[0], device=timestamp.device)
    due = timestamp - last_update + 1e-6 >= update_period
    return due.nonzero(as_tuple=False).flatten()


def lookahead_times(
    timestamp: torch.Tensor,
    start_s: torch.Tensor,
    num_frames: int,
    frame_interval_s: float,
    data_start_from: Literal["one_frame_interval", "current_time"],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample times and time-to-target, matching both source managers."""
    offsets = torch.arange(num_frames, device=timestamp.device, dtype=timestamp.dtype)
    if data_start_from == "one_frame_interval":
        offsets = offsets + 1
    time_to = offsets.unsqueeze(0) * frame_interval_s
    time_to = time_to.expand(timestamp.shape[0], -1)
    sample_time = start_s.unsqueeze(-1) + timestamp.unsqueeze(-1) + time_to
    return sample_time, time_to


def fill_buffers(
    buffers: MotionReferenceBuffers,
    env_ids: torch.Tensor,
    sample: MotionSample,
    time_to_target: torch.Tensor,
) -> None:
    """Write a clip-local sample into ``buffers[env_ids]``."""
    for name in (
        "joint_pos",
        "joint_vel",
        "base_pos_w",
        "base_quat_w",
        "base_lin_vel_w",
        "base_ang_vel_w",
        "link_pos_b",
        "link_quat_b",
        "link_pos_w",
        "link_quat_w",
        "link_lin_vel_b",
        "link_ang_vel_b",
        "link_lin_vel_w",
        "link_ang_vel_w",
        "validity",
        "frame_index",
    ):
        getattr(buffers, name)[env_ids] = getattr(sample, name)
    buffers.time_to_target_frame[env_ids] = time_to_target
    if buffers.scene_object_names:
        buffers.object_pos_w[env_ids] = 0.0
        buffers.object_quat_w[env_ids] = 0.0
        buffers.object_quat_w[env_ids, :, :, 0] = 1.0
        buffers.object_lin_vel_w[env_ids] = 0.0
        buffers.object_ang_vel_w[env_ids] = 0.0
        buffers.object_validity[env_ids] = False
        if sample.object_name is not None:
            try:
                object_id = buffers.scene_object_names.index(sample.object_name)
            except ValueError as exc:
                raise KeyError(
                    f"motion object {sample.object_name!r} is absent from scene objects {buffers.scene_object_names}"
                ) from exc
            buffers.object_pos_w[env_ids, :, object_id] = sample.object_pos_w
            buffers.object_quat_w[env_ids, :, object_id] = sample.object_quat_w
            buffers.object_lin_vel_w[env_ids, :, object_id] = sample.object_lin_vel_w
            buffers.object_ang_vel_w[env_ids, :, object_id] = sample.object_ang_vel_w
            buffers.object_validity[env_ids, :, object_id] = sample.validity
    buffers.time_to_target_frame[env_ids] = torch.where(
        buffers.validity[env_ids],
        buffers.time_to_target_frame[env_ids],
        torch.full_like(buffers.time_to_target_frame[env_ids], -1.0),
    )
    newly = (~sample.validity).any(dim=-1)
    buffers.exhausted_count[env_ids] += newly.to(torch.long)
    buffers.ever_exhausted[env_ids] |= newly


def translate_world_positions(
    buffers: MotionReferenceBuffers,
    env_ids: torch.Tensor,
    env_origins: torch.Tensor,
) -> None:
    """Translate clip positions after augmentation into each environment's world frame."""
    origin = env_origins[env_ids]
    buffers.base_pos_w[env_ids] += origin.unsqueeze(1)
    buffers.link_pos_w[env_ids] += origin.unsqueeze(1).unsqueeze(1)
    buffers.object_pos_w[env_ids] += origin.unsqueeze(1).unsqueeze(
        1
    ) * buffers.object_validity[env_ids].unsqueeze(-1)
