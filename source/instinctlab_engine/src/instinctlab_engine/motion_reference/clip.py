"""Motion-clip loading, name remapping, kinematics and sampling."""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from instinctlab_engine.bridge.math import combine_frame_transforms, quat_from_matrix
from instinctlab_engine.data import resolve_data_path
from instinctlab_engine.math import quat_angular_velocity, quat_slerp_batch
from instinctlab_engine.name_order import NameOrderError, resolve_name_indices


class JointNameMappingError(ValueError):
    """A name-based remap failed. Length equality is not a substitute."""


def resolve_clip_path(path: str) -> str:
    """Resolve dataset URIs/``~``/symlinks and require a real clip file."""
    real = resolve_data_path(path)
    if not real.is_file():
        raise FileNotFoundError(
            f"Motion clip {path!r} resolves to {str(real)!r}, which is not a file. "
            "Check INSTINCTLAB_DATA_ROOT or a legacy compatibility symlink."
        )
    return str(real)


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


@dataclass
class _KinematicJoint:
    name: str
    kind: Literal["fixed", "revolute", "prismatic"]
    axis: torch.Tensor
    origin: torch.Tensor


@dataclass
class _KinematicFrame:
    name: str
    joint: _KinematicJoint
    children: list["_KinematicFrame"]


class _KinematicPose:
    def __init__(self, matrix: torch.Tensor) -> None:
        self._matrix = matrix

    def get_matrix(self) -> torch.Tensor:
        return self._matrix


class _UrdfKinematicsChain:
    """Small batched URDF FK chain used by the shared motion runtime.

    Importing :mod:`pytorch_kinematics` imports its MJCF module and therefore
    MuJoCo, even when the caller only asks for URDF FK. That contaminates an
    Isaac process with the unselected SDK and, under Kit, can load a second
    incompatible CFFI runtime. The motion runtime needs only this narrow
    fixed/revolute/prismatic tree contract, so URDF uses a direct parser.
    """

    def __init__(self, root: _KinematicFrame) -> None:
        self._root = root
        self._dtype = torch.float32
        self._device = torch.device("cpu")
        self._joint_names = tuple(self._walk_joint_names(root))

    @classmethod
    def from_xml(cls, content: str) -> "_UrdfKinematicsChain":
        xml_root = ET.fromstring(content)
        link_names = [
            element.attrib["name"]
            for element in xml_root.iter()
            if _xml_tag(element) == "link" and element.attrib.get("name")
        ]
        if not link_names:
            raise ValueError("URDF contains no named links.")
        if len(set(link_names)) != len(link_names):
            raise ValueError("URDF contains duplicate link names.")

        child_joints: dict[str, list[tuple[str, str, str, torch.Tensor, torch.Tensor]]] = {}
        child_links: set[str] = set()
        joint_names: set[str] = set()
        for element in xml_root.iter():
            if _xml_tag(element) != "joint":
                continue
            name = element.attrib.get("name", "")
            if not name or name in joint_names:
                raise ValueError(f"URDF has an invalid or duplicate joint name {name!r}.")
            joint_names.add(name)
            raw_kind = element.attrib.get("type", "fixed")
            kind = "revolute" if raw_kind == "continuous" else raw_kind
            if kind not in {"fixed", "revolute", "prismatic"}:
                raise ValueError(
                    f"URDF joint {name!r} has unsupported type {raw_kind!r}; "
                    "motion-reference FK supports fixed, revolute/continuous, and prismatic joints."
                )
            parent_element = _xml_child(element, "parent")
            child_element = _xml_child(element, "child")
            if parent_element is None or child_element is None:
                raise ValueError(f"URDF joint {name!r} has no parent/child link.")
            parent = parent_element.attrib.get("link", "")
            child = child_element.attrib.get("link", "")
            if parent not in link_names or child not in link_names:
                raise ValueError(
                    f"URDF joint {name!r} references unknown links {parent!r} -> {child!r}."
                )
            if child in child_links:
                raise ValueError(f"URDF link {child!r} has more than one parent joint.")
            child_links.add(child)
            origin = _urdf_origin_matrix(_xml_child(element, "origin"))
            axis_element = _xml_child(element, "axis")
            # Preserve pytorch_kinematics' historical default used by the
            # accepted motion baselines when an axis is omitted.
            axis = _parse_vector(
                None if axis_element is None else axis_element.attrib.get("xyz"),
                default=(0.0, 0.0, 1.0),
            )
            child_joints.setdefault(parent, []).append(
                (child, name, kind, axis, origin)
            )

        roots = [name for name in link_names if name not in child_links]
        if len(roots) != 1:
            raise ValueError(f"URDF must contain one kinematic root, got {roots}.")

        def build(link_name: str) -> _KinematicFrame:
            children = []
            for child, name, kind, axis, origin in child_joints.get(link_name, ()):
                frame = build(child)
                frame.joint = _KinematicJoint(name, kind, axis, origin)
                children.append(frame)
            return _KinematicFrame(
                name=link_name,
                joint=_KinematicJoint(
                    name="",
                    kind="fixed",
                    axis=torch.tensor((0.0, 0.0, 1.0)),
                    origin=torch.eye(4),
                ),
                children=children,
            )

        return cls(build(roots[0]))

    @staticmethod
    def _walk_joint_names(frame: _KinematicFrame) -> list[str]:
        names = [] if frame.joint.kind == "fixed" else [frame.joint.name]
        for child in frame.children:
            names.extend(_UrdfKinematicsChain._walk_joint_names(child))
        return names

    def get_joint_parameter_names(self) -> list[str]:
        return list(self._joint_names)

    def to(self, *, dtype: torch.dtype, device: torch.device | str) -> "_UrdfKinematicsChain":
        self._dtype = dtype
        self._device = torch.device(device)

        def move(frame: _KinematicFrame) -> None:
            frame.joint.axis = frame.joint.axis.to(dtype=dtype, device=device)
            frame.joint.origin = frame.joint.origin.to(dtype=dtype, device=device)
            for child in frame.children:
                move(child)

        move(self._root)
        return self

    def forward_kinematics(self, joint_pos: torch.Tensor) -> dict[str, _KinematicPose]:
        if joint_pos.ndim != 2 or joint_pos.shape[1] != len(self._joint_names):
            raise ValueError(
                f"URDF FK expected (batch, {len(self._joint_names)}) joint positions, "
                f"got {tuple(joint_pos.shape)}."
            )
        batch = joint_pos.shape[0]
        index = {name: position for position, name in enumerate(self._joint_names)}
        identity = torch.eye(
            4, dtype=joint_pos.dtype, device=joint_pos.device
        ).expand(batch, -1, -1)
        poses: dict[str, _KinematicPose] = {}

        def visit(frame: _KinematicFrame, parent: torch.Tensor) -> None:
            local = frame.joint.origin.expand(batch, -1, -1)
            if frame.joint.kind == "revolute":
                motion = _axis_angle_matrix(
                    frame.joint.axis, joint_pos[:, index[frame.joint.name]]
                )
                local = local @ motion
            elif frame.joint.kind == "prismatic":
                motion = identity.clone()
                motion[:, :3, 3] = (
                    joint_pos[:, index[frame.joint.name]].unsqueeze(-1)
                    * frame.joint.axis
                )
                local = local @ motion
            world = parent @ local
            poses[frame.name] = _KinematicPose(world)
            for child in frame.children:
                visit(child, world)

        visit(self._root, identity)
        return poses


def _xml_tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _xml_child(element: ET.Element, name: str) -> ET.Element | None:
    return next((child for child in element if _xml_tag(child) == name), None)


def _parse_vector(value: str | None, *, default: tuple[float, float, float]) -> torch.Tensor:
    if value is None:
        values = default
    else:
        values = tuple(float(item) for item in value.split())
        if len(values) != 3:
            raise ValueError(f"expected a three-vector, got {value!r}.")
    return torch.tensor(values, dtype=torch.float32)


def _urdf_origin_matrix(origin: ET.Element | None) -> torch.Tensor:
    xyz = _parse_vector(
        None if origin is None else origin.attrib.get("xyz"),
        default=(0.0, 0.0, 0.0),
    )
    rpy = _parse_vector(
        None if origin is None else origin.attrib.get("rpy"),
        default=(0.0, 0.0, 0.0),
    )
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = torch.tensor(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=torch.float32,
    )
    matrix = torch.eye(4, dtype=torch.float32)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = xyz
    return matrix


def _axis_angle_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / torch.linalg.vector_norm(axis)
    x, y, z = axis.unbind()
    zero = torch.zeros((), dtype=axis.dtype, device=axis.device)
    skew = torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        )
    )
    identity = torch.eye(3, dtype=axis.dtype, device=axis.device)
    sine = torch.sin(angle).reshape(-1, 1, 1)
    cosine = torch.cos(angle).reshape(-1, 1, 1)
    rotation = identity + sine * skew + (1.0 - cosine) * (skew @ skew)
    matrix = torch.eye(4, dtype=axis.dtype, device=axis.device).expand(
        angle.shape[0], -1, -1
    ).clone()
    matrix[:, :3, :3] = rotation
    return matrix


def build_kinematics_chain(model_path: str, device: torch.device | str = "cpu") -> Any:
    """Parse a robot description the way InstinctMJ / Isaac do.

    URDF uses a small direct batched parser so an Isaac process does not import
    MuJoCo through pytorch_kinematics' eager MJCF re-export. MJCF retains the
    accepted pytorch_kinematics/MuJoCo path: strip free/ball joints and zero the
    root-body pose so FK is in the base-body frame, matching URDF. Relative
    mesh paths are resolved from the file's directory.
    """
    path = os.path.abspath(os.path.expanduser(model_path))
    with open(path) as handle:
        content = handle.read()
    if path.endswith(".xml"):
        import pytorch_kinematics as pk

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
        chain = _UrdfKinematicsChain.from_xml(content)
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
    object_name: str | None = None
    object_pos_w: torch.Tensor | None = None
    object_quat_w: torch.Tensor | None = None
    object_lin_vel_w: torch.Tensor | None = None
    object_ang_vel_w: torch.Tensor | None = None

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
    framerate = float(np.asarray(raw["framerate"]).item())
    if not math.isfinite(framerate) or framerate <= 0.0:
        raise ValueError(f"Clip {real} has invalid framerate {framerate!r}.")
    names = raw["joint_names"]
    joint_names = tuple(names.tolist() if hasattr(names, "tolist") else list(names))
    if any(not isinstance(name, str) or not name for name in joint_names):
        raise ValueError(f"Clip {real} has an invalid joint name.")
    duplicates = sorted({name for name in joint_names if joint_names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Clip {real} repeats joint names: {duplicates}.")
    joint_pos = torch.as_tensor(np.asarray(raw["joint_pos"]), device=device, dtype=torch.float32)
    base_pos_w = torch.as_tensor(np.asarray(raw["base_pos_w"]), device=device, dtype=torch.float32)
    base_quat_w = torch.as_tensor(np.asarray(raw["base_quat_w"]), device=device, dtype=torch.float32)
    object_pos_w = object_quat_w = None
    object_name = None
    if "object_pos_w" in raw or "object_quat_w" in raw:
        if "object_pos_w" not in raw or "object_quat_w" not in raw:
            raise KeyError(f"Clip {real} must carry object_pos_w and object_quat_w together.")
        object_pos_w = torch.as_tensor(np.asarray(raw["object_pos_w"]), device=device, dtype=torch.float32)
        object_quat_w = torch.as_tensor(np.asarray(raw["object_quat_w"]), device=device, dtype=torch.float32)
        object_name = Path(real).name.split("_")[1]
    if joint_pos.ndim != 2 or joint_pos.shape[1] != len(joint_names):
        raise ValueError(
            f"Clip {real} joint_pos must have shape (frames, {len(joint_names)}), got {tuple(joint_pos.shape)}."
        )
    frame_count = joint_pos.shape[0]
    if frame_count < 2 or base_pos_w.shape != (frame_count, 3) or base_quat_w.shape != (frame_count, 4):
        raise ValueError(
            f"Clip {real} arrays must have at least two aligned frames: joint_pos={tuple(joint_pos.shape)}, "
            f"base_pos_w={tuple(base_pos_w.shape)}, base_quat_w={tuple(base_quat_w.shape)}."
        )
    if (
        not torch.isfinite(joint_pos).all()
        or not torch.isfinite(base_pos_w).all()
        or not torch.isfinite(base_quat_w).all()
    ):
        raise ValueError(f"Clip {real} contains non-finite motion values.")
    quaternion_norm = torch.linalg.vector_norm(base_quat_w, dim=-1)
    if not torch.allclose(quaternion_norm, torch.ones_like(quaternion_norm), atol=1e-3, rtol=1e-3):
        raise ValueError(f"Clip {real} contains non-unit base quaternions.")
    if object_pos_w is not None:
        if object_pos_w.shape != (frame_count, 3) or object_quat_w.shape != (frame_count, 4):
            raise ValueError(f"Clip {real} object arrays are not aligned with its {frame_count} frames.")
        object_norm = torch.linalg.vector_norm(object_quat_w, dim=-1)
        if not torch.allclose(object_norm, torch.ones_like(object_norm), atol=1e-3, rtol=1e-3):
            raise ValueError(f"Clip {real} contains non-unit object quaternions.")
    return {
        "path": real,
        "framerate": framerate,
        "joint_names": joint_names,
        "joint_pos": joint_pos,
        "base_pos_w": base_pos_w,
        "base_quat_w": base_quat_w,
        "object_name": object_name,
        "object_pos_w": object_pos_w,
        "object_quat_w": object_quat_w,
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
    object_pos = raw.get("object_pos_w")
    object_quat = raw.get("object_quat_w")
    if target_fps is not None:
        root_pos, root_quat, joint_pos = interpolate_motion(
            root_pos,
            root_quat,
            joint_pos,
            source_fps=framerate,
            target_fps=float(target_fps),
        )
        if object_pos is not None:
            object_pos, object_quat, _ = interpolate_motion(
                object_pos,
                object_quat,
                raw["joint_pos"][:, :1],
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
    object_lin_vel_w = (
        estimate_velocity(object_pos.unsqueeze(0), dt, velocity_method).squeeze(0) if object_pos is not None else None
    )
    object_ang_vel_w = (
        estimate_angular_velocity(object_quat.unsqueeze(0), dt, velocity_method).squeeze(0)
        if object_quat is not None
        else None
    )
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
        object_name=raw.get("object_name"),
        object_pos_w=object_pos,
        object_quat_w=object_quat,
        object_lin_vel_w=object_lin_vel_w,
        object_ang_vel_w=object_ang_vel_w,
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
    object_name: str | None = None
    object_pos_w: torch.Tensor | None = None
    object_quat_w: torch.Tensor | None = None
    object_lin_vel_w: torch.Tensor | None = None
    object_ang_vel_w: torch.Tensor | None = None


_PACKED_SAMPLE_FIELDS = (
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
)


@dataclass(frozen=True)
class PackedMotionClips:
    """Several compatible clips concatenated for one device-side gather."""

    framerate: float
    frame_offsets: torch.Tensor
    frame_counts: torch.Tensor
    fields: dict[str, torch.Tensor]


def pack_motion_clips_for_sampling(clips: Sequence[MotionClip]) -> PackedMotionClips:
    """Pack object-free clips that share one sampling rate.

    The reference dataset manager indexes all trajectories in one batched call. Keeping one
    :class:`MotionClip` per inventory entry is useful for validation and adaptive sampling, but
    looping over those entries on every environment step launches the same small gather once per
    clip and once per field. Concatenated frame storage preserves clip-local indices while making
    the hot-path gather independent of the number of trajectories.
    """
    if not clips:
        raise ValueError("at least one motion clip is required")
    framerate = float(clips[0].framerate)
    if any(not math.isclose(float(clip.framerate), framerate) for clip in clips):
        raise ValueError("packed motion clips must share one framerate")
    if any(clip.object_name is not None for clip in clips):
        raise ValueError("object motion clips need scene-object-aware sampling")
    device = clips[0].joint_pos.device
    counts = tuple(clip.nframes for clip in clips)
    offsets = [0]
    for count in counts[:-1]:
        offsets.append(offsets[-1] + count)
    return PackedMotionClips(
        framerate=framerate,
        frame_offsets=torch.tensor(offsets, dtype=torch.long, device=device),
        frame_counts=torch.tensor(counts, dtype=torch.long, device=device),
        fields={
            name: torch.cat(tuple(getattr(clip, name) for clip in clips), dim=0)
            for name in _PACKED_SAMPLE_FIELDS
        },
    )


def sample_packed_motion_clips(
    packed: PackedMotionClips,
    motion_ids: torch.Tensor,
    time_s: torch.Tensor,
) -> MotionSample:
    """Gather per-environment times from concatenated clips without host polling."""
    if time_s.shape[0] != motion_ids.shape[0]:
        raise ValueError("motion_ids must select the first dimension of time_s")
    frames = torch.round(time_s * packed.framerate)
    counts = packed.frame_counts[motion_ids]
    while counts.ndim < frames.ndim:
        counts = counts.unsqueeze(-1)
    valid = frames < counts
    clamped = torch.where(valid, frames, counts - 1).to(torch.long)
    offsets = packed.frame_offsets[motion_ids]
    while offsets.ndim < clamped.ndim:
        offsets = offsets.unsqueeze(-1)
    flat = offsets + clamped

    def _take(name: str) -> torch.Tensor:
        return packed.fields[name][flat]

    return MotionSample(
        frame_index=clamped,
        validity=valid.to(torch.bool),
        joint_pos=_take("joint_pos"),
        joint_vel=_take("joint_vel"),
        base_pos_w=_take("base_pos_w"),
        base_quat_w=_take("base_quat_w"),
        base_lin_vel_w=_take("base_lin_vel_w"),
        base_ang_vel_w=_take("base_ang_vel_w"),
        link_pos_b=_take("link_pos_b"),
        link_quat_b=_take("link_quat_b"),
        link_pos_w=_take("link_pos_w"),
        link_quat_w=_take("link_quat_w"),
        link_lin_vel_b=_take("link_lin_vel_b"),
        link_ang_vel_b=_take("link_ang_vel_b"),
        link_lin_vel_w=_take("link_lin_vel_w"),
        link_ang_vel_w=_take("link_ang_vel_w"),
    )


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
        object_name=clip.object_name,
        object_pos_w=_take(clip.object_pos_w) if clip.object_pos_w is not None else None,
        object_quat_w=_take(clip.object_quat_w) if clip.object_quat_w is not None else None,
        object_lin_vel_w=_take(clip.object_lin_vel_w) if clip.object_lin_vel_w is not None else None,
        object_ang_vel_w=_take(clip.object_ang_vel_w) if clip.object_ang_vel_w is not None else None,
    )
