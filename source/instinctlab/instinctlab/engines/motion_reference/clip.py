"""Motion-clip loading, name remapping, kinematics and sampling."""

from __future__ import annotations

import numpy as np
import os
import torch
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from instinctlab.compat.math import combine_frame_transforms, quat_from_matrix
from instinctlab.utils.math import quat_angular_velocity, quat_slerp_batch
from instinctlab.utils.name_order import NameOrderError, resolve_name_indices


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
    try:
        index = resolve_name_indices(source, target, require_exact=True)
    except NameOrderError as exc:
        raise JointNameMappingError(f"Name-based {what} remap failed. {exc}") from exc
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

    @property
    def sampling_length_s(self) -> float:
        """Length used by the upstream managers when sampling episode starts."""
        return self.nframes * self.dt


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
    if target_fps is not None:
        root_pos, root_quat, joint_pos = interpolate_motion(
            root_pos,
            root_quat,
            joint_pos,
            source_fps=framerate,
            target_fps=float(target_fps),
        )
        framerate = float(target_fps)
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


def interpolate_motion(
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    joint_pos: torch.Tensor,
    *,
    source_fps: float,
    target_fps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resample exactly as InstinctMJ/main's ``motion_interpolate_bilinear``.

    Their half-open timeline deliberately omits the raw endpoint. This also
    happens when source and target rates are equal, so bypassing the operation
    would leave one extra valid frame in the unified runtime.
    """
    if source_fps <= 0.0 or target_fps <= 0.0:
        raise ValueError(f"motion frame rates must be positive, got {source_fps=} and {target_fps=}.")
    if root_pos.shape[0] < 2:
        raise ValueError("motion interpolation needs at least two source frames.")
    if root_pos.shape[0] != root_quat.shape[0] or root_pos.shape[0] != joint_pos.shape[0]:
        raise ValueError("motion position, orientation and joint arrays must have the same frame count.")

    duration = (root_pos.shape[0] - 1) / source_fps
    count = int(np.ceil(duration * target_fps))
    frame = torch.arange(count, device=root_pos.device, dtype=root_pos.dtype) * (source_fps / target_fps)
    frame = frame[frame <= root_pos.shape[0] - 1]
    before = torch.floor(frame).to(torch.long)
    after = torch.ceil(frame).to(torch.long)
    ratio = frame - before.to(frame.dtype)
    root_pos_out = torch.lerp(root_pos[before], root_pos[after], ratio.unsqueeze(-1))
    root_quat_out = quat_slerp_batch(root_quat[before], root_quat[after], ratio)
    joint_pos_out = torch.lerp(joint_pos[before], joint_pos[after], ratio.unsqueeze(-1))
    return root_pos_out, root_quat_out, joint_pos_out


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
