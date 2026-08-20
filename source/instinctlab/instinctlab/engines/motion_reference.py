"""Clip load, name remap, FK and exhaustion — the motion-reference algorithm.

Both source managers (Isaac ``motion_reference/`` and InstinctMJ) do the same
four things at load: read a retargetted ``.npz``, remap joints by name, run
``pytorch_kinematics`` on that engine's robot description, and finite-difference
velocities. Duplicating that would let the copies drift the way a positional
fallback already would: training still converges, the reference is a different
robot. A third engine therefore pays for a description path, not another loader.

This module is engine-free. URDF vs MJCF is a file extension, not an engine
name. ``spec/`` stays a declaration; the numbers live here so a comparison can
run without a simulator.
"""

from __future__ import annotations

import numpy as np
import os
import torch
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from instinctlab.compat.math import combine_frame_transforms, quat_from_matrix
from instinctlab.utils.math import quat_angular_velocity

__all__ = [
    "ChainInventory",
    "MotionClip",
    "MotionReferenceBuffers",
    "MotionSample",
    "build_kinematics_chain",
    "chain_inventory",
    "envs_due_for_update",
    "estimate_angular_velocity",
    "estimate_velocity",
    "fill_buffers",
    "forward_kinematics_links",
    "index_at_time",
    "lookahead_times",
    "load_retargetted_clip",
    "make_buffers",
    "clip_frame",
    "pack_motion_clip",
    "remap_by_name",
    "resolve_clip_path",
    "sample_clip",
]


class JointNameMappingError(ValueError):
    """A name-based remap failed. Length equality is not a substitute."""


def resolve_clip_path(path: str) -> str:
    """Expand ``~`` and follow symlinks. Fail if the result is not a real file."""
    expanded = os.path.expanduser(path)
    real = os.path.realpath(expanded)
    if not os.path.isfile(real):
        raise FileNotFoundError(
            f"Motion clip {path!r} resolves to {real!r}, which is not a file. A dangling symlink is the usual cause."
        )
    return real


def remap_by_name(
    values: torch.Tensor,
    source_names: Sequence[str],
    target_names: Sequence[str],
    *,
    what: str = "joint",
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Reorder the last axis of ``values`` from ``source_names`` to ``target_names``.

    Missing names on either side fail. Equal lengths without equal names fail.
    The index map is returned so a test can prove it is not the identity.
    """
    source = tuple(source_names)
    target = tuple(target_names)
    if len(source) != len(set(source)):
        raise JointNameMappingError(f"Source {what} names are not unique: {source}.")
    if len(target) != len(set(target)):
        raise JointNameMappingError(f"Target {what} names are not unique: {target}.")
    source_set, target_set = set(source), set(target)
    missing_in_source = [name for name in target if name not in source_set]
    missing_in_target = [name for name in source if name not in target_set]
    if missing_in_source or missing_in_target:
        raise JointNameMappingError(
            f"Name-based {what} remap failed. Missing in source (needed by target): "
            f"{missing_in_source}. Missing in target (present in source): {missing_in_target}."
        )
    index = tuple(source.index(name) for name in target)
    if values.shape[-1] != len(source):
        raise JointNameMappingError(
            f"{what} tensor last axis is {values.shape[-1]}, source names are {len(source)}. "
            "A length check on the tensor is not a name check."
        )
    return values[..., list(index)], index


def estimate_velocity(
    positions: torch.Tensor,
    dt: float,
    estimation_type: Literal["frontward", "backward", "frontbackward"] = "frontward",
) -> torch.Tensor:
    """Finite-difference linear / joint velocity. Ported from both source utils."""
    if positions.ndim != 3:
        raise ValueError("positions must be (batch, frames, dim).")
    if estimation_type == "frontward":
        nxt = torch.roll(positions, -1, dims=1)
        nxt[:, -1] = positions[:, -1]
        return (nxt - positions) / dt
    if estimation_type == "backward":
        prev = torch.roll(positions, 1, dims=1)
        prev[:, 0] = positions[:, 0]
        return (positions - prev) / dt
    if estimation_type == "frontbackward":
        prev = torch.roll(positions, 1, dims=1)
        prev[:, 0] = positions[:, 0]
        nxt = torch.roll(positions, -1, dims=1)
        nxt[:, -1] = positions[:, -1]
        return (nxt - prev) / (2 * dt)
    raise ValueError(f"Unknown estimation type: {estimation_type}.")


def estimate_angular_velocity(
    quaternions: torch.Tensor,
    dt: float,
    estimation_type: Literal["frontward", "backward", "frontbackward"] = "frontward",
) -> torch.Tensor:
    """Finite-difference angular velocity from wxyz quaternions. Ported from both sources."""
    if quaternions.shape[-1] != 4 or quaternions.ndim != 3:
        raise ValueError("quaternions must be (batch, frames, 4) wxyz.")
    if estimation_type == "frontward":
        nxt = torch.roll(quaternions, -1, dims=1)
        nxt[:, -1] = quaternions[:, -1]
        prev = quaternions
        span = dt
    elif estimation_type == "backward":
        prev = torch.roll(quaternions, 1, dims=1)
        prev[:, 0] = quaternions[:, 0]
        nxt = quaternions
        span = dt
    elif estimation_type == "frontbackward":
        prev = torch.roll(quaternions, 1, dims=1)
        prev[:, 0] = quaternions[:, 0]
        nxt = torch.roll(quaternions, -1, dims=1)
        nxt[:, -1] = quaternions[:, -1]
        span = 2 * dt
    else:
        raise ValueError(f"Unknown estimation type: {estimation_type}.")
    return quat_angular_velocity(prev.reshape(-1, 4), nxt.reshape(-1, 4), span).reshape(
        quaternions.shape[0], quaternions.shape[1], 3
    )


def _walk_frame_names(frame: Any) -> list[str]:
    names = [getattr(frame, "name", None) or str(frame)]
    children = getattr(frame, "children", None) or []
    if not isinstance(children, (list, tuple)):
        children = [children] if children else []
    for child in children:
        names.extend(_walk_frame_names(child))
    return names


@dataclass(frozen=True)
class ChainInventory:
    """What a kinematic description actually contains, not what the outputs look like."""

    model_path: str
    root: str
    joint_names: tuple[str, ...]
    link_names: tuple[str, ...]
    kind: Literal["urdf", "mjcf"]


def build_kinematics_chain(model_path: str, device: torch.device | str = "cpu") -> Any:
    """Parse a robot description the way InstinctMJ / Isaac do.

    MJCF: strip free/ball joints (pytorch_kinematics only does hinge/slide) and
    zero the root-body pose so FK is in the base-body frame, matching URDF.
    Relative mesh paths are resolved from the file's directory.
    """
    import pytorch_kinematics as pk

    path = os.path.abspath(os.path.expanduser(model_path))
    with open(path) as handle:
        content = handle.read()
    if path.endswith(".xml"):
        mjcf_root = ET.fromstring(content)
        for parent in mjcf_root.iter():
            for child in list(parent):
                tag = child.tag.split("}")[-1] if isinstance(child.tag, str) else child.tag
                if tag == "freejoint":
                    parent.remove(child)
                    continue
                if tag == "joint" and child.attrib.get("type", "hinge").lower() in {"free", "ball"}:
                    parent.remove(child)
        worldbody = mjcf_root.find("worldbody")
        if worldbody is not None:
            root_body = worldbody.find("body")
            if root_body is not None:
                root_body.set("pos", "0 0 0")
                if "quat" in root_body.attrib:
                    root_body.set("quat", "1 0 0 0")
                if "euler" in root_body.attrib:
                    root_body.set("euler", "0 0 0")
                if "axisangle" in root_body.attrib:
                    root_body.set("axisangle", "0 0 1 0")
        content = ET.tostring(mjcf_root, encoding="unicode")
        previous = os.getcwd()
        os.chdir(os.path.dirname(path))
        try:
            chain = pk.build_chain_from_mjcf(content)
        finally:
            os.chdir(previous)
    else:
        chain = pk.build_chain_from_urdf(content)
    return chain.to(dtype=torch.float, device=device)


def chain_inventory(chain: Any, model_path: str) -> ChainInventory:
    root = getattr(chain, "_root", None)
    if root is None:
        raise RuntimeError(f"Kinematic chain from {model_path!r} has no root frame.")
    kind: Literal["urdf", "mjcf"] = "mjcf" if model_path.endswith(".xml") else "urdf"
    return ChainInventory(
        model_path=os.path.abspath(os.path.expanduser(model_path)),
        root=getattr(root, "name", "") or "",
        joint_names=tuple(chain.get_joint_parameter_names()),
        link_names=tuple(_walk_frame_names(root)),
        kind=kind,
    )


def forward_kinematics_links(
    chain: Any,
    joint_pos: torch.Tensor,
    joint_names: Sequence[str],
    link_names: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Link poses in the chain root frame. ``joint_pos`` is in ``joint_names`` order.

    The chain has its own joint order. The remap onto that order is by name.
    A missing link name fails; we do not skip it and leave a zero pose.
    """
    chain_joints = tuple(chain.get_joint_parameter_names())
    ordered, _ = remap_by_name(joint_pos, joint_names, chain_joints, what="kinematics-joint")
    poses = chain.forward_kinematics(ordered)
    pos = torch.zeros(joint_pos.shape[0], len(link_names), 3, device=joint_pos.device, dtype=joint_pos.dtype)
    quat = torch.zeros(joint_pos.shape[0], len(link_names), 4, device=joint_pos.device, dtype=joint_pos.dtype)
    quat[..., 0] = 1.0
    missing = [name for name in link_names if name not in poses]
    if missing:
        have = sorted(poses)
        raise KeyError(
            f"FK chain is missing links {missing}. The chain has {have}. "
            "Joint angles can agree while link poses do not; this is that failure."
        )
    for index, name in enumerate(link_names):
        matrix = poses[name].get_matrix().reshape(-1, 4, 4)
        pos[:, index] = matrix[:, :3, 3]
        quat[:, index] = quat_from_matrix(matrix[:, :3, :3])
    return pos, quat


@dataclass
class MotionClip:
    """One retargetted clip, remapped and FK'd. Ready to index by frame."""

    path: str
    source_joint_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    joint_index_map: tuple[int, ...]
    link_names: tuple[str, ...]
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
    framerate: float
    inventory: ChainInventory

    @property
    def dt(self) -> float:
        return 1.0 / float(self.framerate)

    @property
    def nframes(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def duration_s(self) -> float:
        return (self.nframes - 1) * self.dt


def load_retargetted_clip(path: str, device: torch.device | str = "cpu") -> dict[str, Any]:
    """Read a ``*_retargetted.npz``. Does not remap; that happens in :func:`pack_motion_clip`."""
    real = resolve_clip_path(path)
    raw = np.load(real, mmap_mode="r", allow_pickle=True)
    required = ("framerate", "joint_names", "joint_pos", "base_pos_w", "base_quat_w")
    missing = [key for key in required if key not in raw]
    if missing:
        raise KeyError(f"Clip {real} is missing {missing}; have {list(raw.keys())}.")
    names = raw["joint_names"]
    joint_names = tuple(names.tolist() if hasattr(names, "tolist") else list(names))
    return {
        "path": real,
        "framerate": float(np.asarray(raw["framerate"]).item()),
        "joint_names": joint_names,
        "joint_pos": torch.as_tensor(np.asarray(raw["joint_pos"]), device=device, dtype=torch.float32),
        "base_pos_w": torch.as_tensor(np.asarray(raw["base_pos_w"]), device=device, dtype=torch.float32),
        "base_quat_w": torch.as_tensor(np.asarray(raw["base_quat_w"]), device=device, dtype=torch.float32),
    }


def pack_motion_clip(
    raw: dict[str, Any],
    *,
    joint_names: Sequence[str],
    link_names: Sequence[str],
    model_path: str,
    velocity_method: Literal["frontward"] = "frontward",
    target_fps: float | None = None,
    device: torch.device | str = "cpu",
) -> MotionClip:
    """Remap by name, FK, finite-difference. This is the load-time path both sources share."""
    target = tuple(joint_names)
    links = tuple(link_names)
    joint_pos, index_map = remap_by_name(raw["joint_pos"], raw["joint_names"], target, what="joint")
    root_pos = raw["base_pos_w"]
    root_quat = raw["base_quat_w"]
    framerate = float(raw["framerate"])
    if target_fps is not None and abs(target_fps - framerate) > 1e-9:
        raise ValueError(
            f"Clip fps is {framerate}, target is {target_fps}. Interpolation is the parkour "
            "source's bilinear path; this increment only packs a clip already at the target rate "
            "so a fps mismatch cannot silently change the time base."
        )
    chain = build_kinematics_chain(model_path, device=device)
    inventory = chain_inventory(chain, model_path)
    missing_links = [name for name in links if name not in inventory.link_names]
    if missing_links:
        raise KeyError(
            f"Robot description {inventory.model_path} is missing links {missing_links}. "
            f"Chain links: {list(inventory.link_names)}."
        )
    link_pos_b, link_quat_b = forward_kinematics_links(chain, joint_pos, target, links)
    n_links = link_pos_b.shape[1]
    link_pos_w, link_quat_w = combine_frame_transforms(
        root_pos.unsqueeze(1).expand(-1, n_links, -1).reshape(-1, 3),
        root_quat.unsqueeze(1).expand(-1, n_links, -1).reshape(-1, 4),
        link_pos_b.reshape(-1, 3),
        link_quat_b.reshape(-1, 4),
    )
    link_pos_w = link_pos_w.reshape(-1, n_links, 3)
    link_quat_w = link_quat_w.reshape(-1, n_links, 4)
    dt = 1.0 / framerate
    joint_vel = estimate_velocity(joint_pos.unsqueeze(0), dt, velocity_method).squeeze(0)
    base_lin_vel_w = estimate_velocity(root_pos.unsqueeze(0), dt, velocity_method).squeeze(0)
    base_ang_vel_w = estimate_angular_velocity(root_quat.unsqueeze(0), dt, velocity_method).squeeze(0)
    link_lin_vel_b = estimate_velocity(link_pos_b.permute(1, 0, 2), dt, velocity_method).permute(1, 0, 2)
    link_ang_vel_b = estimate_angular_velocity(link_quat_b.permute(1, 0, 2), dt, velocity_method).permute(1, 0, 2)
    link_lin_vel_w = estimate_velocity(link_pos_w.permute(1, 0, 2), dt, velocity_method).permute(1, 0, 2)
    link_ang_vel_w = estimate_angular_velocity(link_quat_w.permute(1, 0, 2), dt, velocity_method).permute(1, 0, 2)
    return MotionClip(
        path=raw["path"],
        source_joint_names=tuple(raw["joint_names"]),
        joint_names=target,
        joint_index_map=index_map,
        link_names=links,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        base_pos_w=root_pos,
        base_quat_w=root_quat,
        base_lin_vel_w=base_lin_vel_w,
        base_ang_vel_w=base_ang_vel_w,
        link_pos_b=link_pos_b,
        link_quat_b=link_quat_b,
        link_pos_w=link_pos_w,
        link_quat_w=link_quat_w,
        link_lin_vel_b=link_lin_vel_b,
        link_ang_vel_b=link_ang_vel_b,
        link_lin_vel_w=link_lin_vel_w,
        link_ang_vel_w=link_ang_vel_w,
        framerate=framerate,
        inventory=inventory,
    )


@dataclass
class MotionSample:
    """One gather from a packed clip, including the exhaustion flag the sources hid."""

    frame_index: torch.Tensor
    validity: torch.Tensor
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


def index_at_time(
    time_s: torch.Tensor,
    framerate: float,
    nframes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip time → frame index. Past the end: freeze on the last frame, validity false.

    Both sources ``torch.round`` time × fps and clamp with ``validity``. Restarting
    or wrapping is refused here; that is how exhaustion became invisible.
    """
    frames = torch.round(time_s * framerate)
    valid = frames < nframes
    clamped = torch.where(valid, frames, torch.full_like(frames, nframes - 1)).to(torch.long)
    return clamped, valid.to(torch.bool)


def sample_clip(clip: MotionClip, time_s: torch.Tensor) -> MotionSample:
    """Gather ``time_s`` (any leading shape) from ``clip``. Exhaustion is in ``validity``."""
    frames, valid = index_at_time(time_s, clip.framerate, clip.nframes)
    flat = frames.reshape(-1)

    def _take(tensor: torch.Tensor) -> torch.Tensor:
        return tensor[flat].reshape(*frames.shape, *tensor.shape[1:])

    return MotionSample(
        frame_index=frames,
        validity=valid,
        joint_pos=_take(clip.joint_pos),
        joint_vel=_take(clip.joint_vel),
        base_pos_w=_take(clip.base_pos_w),
        base_quat_w=_take(clip.base_quat_w),
        base_lin_vel_w=_take(clip.base_lin_vel_w),
        base_ang_vel_w=_take(clip.base_ang_vel_w),
        link_pos_b=_take(clip.link_pos_b),
        link_quat_b=_take(clip.link_quat_b),
        link_pos_w=_take(clip.link_pos_w),
        link_quat_w=_take(clip.link_quat_w),
        link_lin_vel_b=_take(clip.link_lin_vel_b),
        link_ang_vel_b=_take(clip.link_ang_vel_b),
        link_lin_vel_w=_take(clip.link_lin_vel_w),
        link_ang_vel_w=_take(clip.link_ang_vel_w),
    )


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
    frame_index: torch.Tensor


def make_buffers(
    num_envs: int,
    num_frames: int,
    num_joints: int,
    num_links: int,
    device: torch.device | str = "cpu",
) -> MotionReferenceBuffers:
    zeros_pose = torch.zeros(num_envs, num_frames, 3, device=device)
    zeros_quat = torch.zeros(num_envs, num_frames, 4, device=device)
    zeros_quat[..., 0] = 1.0
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
        frame_index=torch.zeros(num_envs, num_frames, dtype=torch.long, device=device),
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
    env_origins: torch.Tensor | None = None,
) -> None:
    """Write a sample into ``buffers[env_ids]``. Exhaustion increments, it does not wrap."""
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
    if env_origins is not None:
        origin = env_origins[env_ids]
        buffers.base_pos_w[env_ids] = buffers.base_pos_w[env_ids] + origin.unsqueeze(1)
        buffers.link_pos_w[env_ids] = buffers.link_pos_w[env_ids] + origin.unsqueeze(1).unsqueeze(1)
    buffers.time_to_target_frame[env_ids] = time_to_target
    buffers.time_to_target_frame[env_ids] = torch.where(
        buffers.validity[env_ids],
        buffers.time_to_target_frame[env_ids],
        torch.full_like(buffers.time_to_target_frame[env_ids], -1.0),
    )
    newly = (~sample.validity).any(dim=-1)
    buffers.exhausted_count[env_ids] += newly.to(torch.long)
    buffers.ever_exhausted[env_ids] |= newly


def clip_frame(
    buffers: MotionReferenceBuffers, frame: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Root orientation, world velocities and joints at one look-ahead frame.

    Field names of the clip buffers live here so portable AMP terms never write
    ``*.data.base_quat_w``. Frame 0 is the first look-ahead sample, matching both
    source managers' ``[:, 0]`` AMP path — not the 10-frame imitation horizon.
    """
    return (
        buffers.base_quat_w[:, frame],
        buffers.base_lin_vel_w[:, frame],
        buffers.base_ang_vel_w[:, frame],
        buffers.joint_pos[:, frame],
        buffers.joint_vel[:, frame],
    )
