"""Motion-reference and bilateral-augmentation declarations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from ._names import as_name_tuple


def _left_right_name(name: str) -> str:
    """Return a bilateral counterpart, leaving midline names unchanged."""
    if name.startswith("left_"):
        return "right_" + name[5:]
    if name.startswith("right_"):
        return "left_" + name[6:]
    return name


def _left_right_swaps(names: Sequence[str]) -> dict[str, str]:
    """Build ``destination -> source`` swaps for an ordered collection of names."""
    unique = tuple(names)
    if len(set(unique)) != len(unique):
        raise ValueError(f"Names for a left/right swap must be unique: {unique}.")
    available = set(unique)
    swaps: dict[str, str] = {}
    for name in unique:
        other = _left_right_name(name)
        if other not in available:
            raise ValueError(f"{name!r} mirrors to {other!r}, which is not in the name list.")
        swaps[name] = other
    return swaps


def _roll_yaw_signs(names: Sequence[str]) -> dict[str, int]:
    """Return the sagittal-plane signs used by both source G1 tables."""
    return {name: -1 if "roll" in name or "yaw" in name else 1 for name in names}


def _require_involution(kind: str, swaps: Mapping[str, str]) -> None:
    if not swaps:
        raise ValueError(
            f"{kind}_swaps is empty. Pass None on MotionReferenceRef.symmetric_augmentation to disable mirroring."
        )
    for name, other in swaps.items():
        if not name or not other:
            raise ValueError(f"{kind} swap has an empty name: {name!r} -> {other!r}.")
        back = swaps.get(other)
        if back != name:
            raise ValueError(
                f"{kind} swap {name!r} -> {other!r} is not an involution (reverse is {back!r}). "
                "An unpaired left/right name is the usual cause."
            )


@dataclass(frozen=True)
class SymmetricAugmentationSpec:
    """Left-right mirror of an AMP reference, declared by name.

    Indices never live in this declaration. The runtime resolves names against the sensor's
    canonical joint and link order, avoiding accidental reuse of a source repository's integer
    table with a different order. ``joint_swaps[name]`` names the source copied onto ``name``;
    signs are applied at the destination.
    """

    joint_swaps: Mapping[str, str]
    joint_signs: Mapping[str, int]
    link_swaps: Mapping[str, str]

    def __post_init__(self) -> None:
        joint_swaps = dict(self.joint_swaps)
        joint_signs = {name: int(sign) for name, sign in self.joint_signs.items()}
        link_swaps = dict(self.link_swaps)
        object.__setattr__(self, "joint_swaps", joint_swaps)
        object.__setattr__(self, "joint_signs", joint_signs)
        object.__setattr__(self, "link_swaps", link_swaps)
        _require_involution("joint", joint_swaps)
        _require_involution("link", link_swaps)
        if set(joint_signs) != set(joint_swaps):
            extra = sorted(set(joint_signs) - set(joint_swaps))
            missing = sorted(set(joint_swaps) - set(joint_signs))
            raise ValueError(
                "joint_signs must cover exactly the joints in joint_swaps. "
                f"Missing signs: {missing}. Extra signs: {extra}."
            )
        bad = sorted(name for name, sign in joint_signs.items() if sign not in (-1, 1))
        if bad:
            raise ValueError(f"joint_signs must be +1 or -1; bad: {bad}.")

    @classmethod
    def from_left_right(cls, joints: Sequence[str], links: Sequence[str]) -> SymmetricAugmentationSpec:
        """Build the name pairing and roll/yaw signs used by both G1 sources."""
        return cls(
            joint_swaps=_left_right_swaps(joints),
            joint_signs=_roll_yaw_signs(joints),
            link_swaps=_left_right_swaps(links),
        )


@dataclass(frozen=True)
class MotionReferenceRef:
    """A clip-backed motion reference shared by both simulation engines.

    Quaternions use ``wxyz``. Joints leave the file in its published order and are remapped by
    name onto :attr:`joints`. Exhaustion holds the last frame and raises a runtime flag; it never
    silently restarts. Symmetric augmentation is disabled unless explicitly declared by name.
    """

    name: str
    clip: str
    joints: str | Sequence[str]
    links: str | Sequence[str]
    entity: str = "robot"
    num_frames: int = 1
    frame_interval_s: float = 0.02
    update_period: float = 0.02
    data_start_from: Literal["one_frame_interval", "current_time"] = "one_frame_interval"
    clip_target_fps: float = 50.0
    velocity_method: Literal["frontward"] = "frontward"
    start_range: tuple[float, float] = (0.0, 0.0)
    exhaustion: Literal["freeze_last_and_flag"] = "freeze_last_and_flag"
    quaternion: Literal["wxyz"] = "wxyz"
    symmetric_augmentation: SymmetricAugmentationSpec | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "joints", as_name_tuple(self.joints))
        object.__setattr__(self, "links", as_name_tuple(self.links))
        if not self.name:
            raise ValueError("Motion reference has no name.")
        if not self.clip:
            raise ValueError(f"Motion reference {self.name!r} has no clip path.")
        if not self.joints:
            raise ValueError(f"Motion reference {self.name!r} was given no joint names.")
        if not self.links:
            raise ValueError(f"Motion reference {self.name!r} was given no link names.")
        if len(set(self.joints)) != len(self.joints):
            raise ValueError(f"Motion reference {self.name!r} repeats a joint name.")
        if len(set(self.links)) != len(self.links):
            raise ValueError(f"Motion reference {self.name!r} repeats a link name.")
        if self.num_frames < 1:
            raise ValueError(f"Motion reference {self.name!r} has num_frames={self.num_frames}.")
        if not math.isfinite(self.frame_interval_s) or self.frame_interval_s <= 0.0:
            raise ValueError(f"Motion reference {self.name!r} has a non-positive frame_interval_s.")
        if not math.isfinite(self.update_period) or self.update_period <= 0.0:
            raise ValueError(f"Motion reference {self.name!r} has a non-positive update_period.")
        if not math.isfinite(self.clip_target_fps) or self.clip_target_fps <= 0.0:
            raise ValueError(f"Motion reference {self.name!r} has a non-positive clip_target_fps.")
        lo, hi = self.start_range
        if not math.isfinite(lo) or not math.isfinite(hi) or lo < 0.0 or hi < lo or hi > 1.0:
            raise ValueError(
                f"Motion reference {self.name!r} has start_range={self.start_range}; "
                "it must satisfy 0 <= lo <= hi <= 1 (a fraction of clip duration)."
            )
        if self.data_start_from not in {"one_frame_interval", "current_time"}:
            raise ValueError(f"Motion reference {self.name!r} has data_start_from={self.data_start_from!r}.")
        if self.velocity_method != "frontward":
            raise ValueError(
                f"Motion reference {self.name!r} has velocity_method={self.velocity_method!r}; "
                "parkour's sources finite-difference frontward."
            )
        if self.exhaustion != "freeze_last_and_flag":
            raise ValueError(
                f"Motion reference {self.name!r} has exhaustion={self.exhaustion!r}; "
                "silent restart is refused because it hides dataset exhaustion."
            )
        if self.quaternion != "wxyz":
            raise ValueError(
                f"Motion reference {self.name!r} has quaternion={self.quaternion!r}; hub convention is wxyz."
            )
        if self.symmetric_augmentation is not None:
            spec_joints = set(self.symmetric_augmentation.joint_swaps)
            have_joints = set(self.joints)
            if spec_joints != have_joints:
                raise ValueError(
                    f"Motion reference {self.name!r} symmetric_augmentation joints "
                    f"{sorted(spec_joints)} do not match sensor joints {sorted(have_joints)}. "
                    "The maps are names; a leftover integer table from another order is the usual cause."
                )
            spec_links = set(self.symmetric_augmentation.link_swaps)
            have_links = set(self.links)
            if spec_links != have_links:
                raise ValueError(
                    f"Motion reference {self.name!r} symmetric_augmentation links "
                    f"{sorted(spec_links)} do not match sensor links {sorted(have_links)}."
                )


__all__ = ["MotionReferenceRef", "SymmetricAugmentationSpec"]
