"""Engine-neutral sensor declarations.

These dataclasses describe what a task measures: contact subsets, ray patterns, motion-reference
clips and body-local point clouds. Engine adapters own native sensor construction; shared MDP terms
read normalized outputs through :mod:`instinctlab.compat.sensors`.

The contract intentionally stops where physical meanings diverge. Contact timing is portable, for
example, while raw solver force is not. Unsupported semantics are rejected during compilation
instead of being silently approximated.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ._names import as_name_tuple
from .motion_reference import MotionReferenceRef, SymmetricAugmentationSpec
from .volume import Grid3dPointsRef, VirtualObstacleRef, VolumePointsRef

__all__ = [
    "ContactSensorRef",
    "Grid3dPointsRef",
    "MotionReferenceRef",
    "RayCasterRef",
    "RayPatternRef",
    "SymmetricAugmentationSpec",
    "VirtualObstacleRef",
    "VolumePointsRef",
]

_TERRAIN_HIT = "terrain"


def _normalise(patterns: str | Sequence[str]) -> tuple[str, ...]:
    return as_name_tuple(patterns)


@dataclass(frozen=True)
class ContactSensorRef:
    """A contact measurement on part of an entity.

    Args:
        name: Key this sensor is registered under in the scene. Terms use it to find the sensor,
            and the backend uses it when it has to create one.
        elements: Name patterns for the elements whose contacts are tracked, e.g. the feet.
            Matched against the entity's own names by the same helper both engines use.
        entity: Key of the entity the elements belong to.
        against: Optional counterpart restriction -- ``"terrain"`` to count only contacts with the
            ground, ``None`` to count any contact. mjlab honours this as a ``secondary`` match on
            the contact sensor itself. Isaac Lab's ``filter_prim_paths_expr`` does **not** change
            ``net_forces_w`` or air-time, which is what portable terms read, and it is documented
            not to work on a multi-body ``prim_path``. The Isaac backend therefore refuses a
            reference that sets this field rather than emitting a sensor that looks filtered and
            is not.
        track_air_time: Ask the engine to accumulate air and contact durations. Both engines gate
            this behind a config flag because it costs per-step bookkeeping, and both return
            ``None`` for the corresponding tensors when it is off.
        air_time_force_threshold: Newtons of net contact force a body must carry before the clock
            calls it touchdown. Declared here rather than left to the engines because their
            defaults are what disagree: Isaac Lab thresholds at 1 N, mjlab counts any contact the
            solver reports at any force. Both produce same-named, same-shaped, plausible-looking
            duration tensors, so a task that leaves this to the engine is scoring two different
            gaits without saying so. Only air and contact timing use it; the force tensors terms
            read are untouched.
        history_length: Number of past substeps of force data to retain. ``0`` disables it. Both
            engines order the history newest-first.
        preserve_order: Whether the element order follows the patterns rather than the entity's.

    Note:
        ``elements`` are *bodies* on the Isaac Lab side, because its contact sensor attaches to
        rigid-body prims. mjlab can additionally scope a contact to a geom or to a whole subtree.
        Which of those a backend picks is its own decision, recorded in the manifest; the reference
        deliberately does not force the finer distinction, because a task that demands geom-level
        contact is not portable to Isaac Lab in the first place and should say so through the
        capability mechanism instead.
    """

    name: str
    elements: str | Sequence[str]
    entity: str = "robot"
    against: str | None = None
    track_air_time: bool = False
    air_time_force_threshold: float = 1.0
    history_length: int = 0
    preserve_order: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "elements", _normalise(self.elements))
        if not self.name or not self.entity:
            raise ValueError("Contact sensor name and entity must be non-empty.")
        if not self.elements:
            raise ValueError(f"Contact sensor {self.name!r} was given no element patterns.")
        if self.against == "":
            raise ValueError(f"Contact sensor {self.name!r} has an empty against pattern.")
        if not math.isfinite(self.air_time_force_threshold) or self.air_time_force_threshold < 0.0:
            raise ValueError(f"Contact sensor {self.name!r} has an invalid air_time_force_threshold.")
        if self.history_length < 0:
            raise ValueError(f"Contact sensor {self.name!r} has a negative history_length.")


@dataclass(frozen=True)
class RayPatternRef:
    """How rays are laid out in the sensor frame.

    ``kind="grid"`` is the foot-height scanner. ``kind="pinhole"`` is the depth
    camera. Grid fields are ignored for a pinhole and pinhole fields are ignored
    for a grid -- a later kind can be added the same way.
    """

    kind: Literal["grid", "pinhole"] = "grid"
    resolution: float = 0.12
    size: tuple[float, float] = (0.12, 0.0)
    width: int = 64
    height: int = 36
    horizontal_fov_deg: float = 89.51
    vertical_fov_deg: float = 58.29
    focal_length: float = 1.0

    def __post_init__(self) -> None:
        if self.kind == "grid":
            if not math.isfinite(self.resolution) or self.resolution <= 0.0:
                raise ValueError(f"Ray pattern resolution must be positive, got {self.resolution}.")
            if not all(math.isfinite(value) for value in self.size) or self.size[0] < 0.0 or self.size[1] < 0.0:
                raise ValueError(f"Ray pattern size must be non-negative, got {self.size}.")
            return
        if self.kind == "pinhole":
            if self.width < 1 or self.height < 1:
                raise ValueError(f"Pinhole resolution must be at least 1x1, got {self.width}x{self.height}.")
            if not 0.0 < self.horizontal_fov_deg < 180.0 or not 0.0 < self.vertical_fov_deg < 180.0:
                raise ValueError(
                    "Pinhole FOV must be between 0 and 180 degrees, got "
                    f"horizontal={self.horizontal_fov_deg}, vertical={self.vertical_fov_deg}."
                )
            if not math.isfinite(self.focal_length) or self.focal_length <= 0.0:
                raise ValueError(f"Pinhole focal_length must be positive, got {self.focal_length}.")
            return
        raise ValueError(f"RayPatternRef.kind must be 'grid' or 'pinhole', got {self.kind!r}.")

    @property
    def horizontal_aperture(self) -> float:
        """Film-plane width at ``focal_length``, matching both parkour sources."""
        return 2.0 * math.tan(math.radians(self.horizontal_fov_deg) / 2.0) * self.focal_length

    @property
    def vertical_aperture(self) -> float:
        """Film-plane height at ``focal_length``, matching both parkour sources."""
        return 2.0 * math.tan(math.radians(self.vertical_fov_deg) / 2.0) * self.focal_length


@dataclass(frozen=True)
class RayCasterRef:
    """A ray sample attached to a body -- grid scanner or pinhole camera.

    The field set is what both parkour sensors share (attach, offset, hit, miss,
    max distance) plus the extras a pinhole needs (orientation, crop, min
    distance). VolumePoints is a different measurement and is a different
    reference.

    Args:
        name: Scene key. Terms find the sensor by this.
        mode: ``"ray"`` preserves the declared ray origin and range. ``"terrain_height"``
            asks for terrain hit positions beneath the attached sample grid and lets each backend
            use its native equivalent implementation. The latter is the Parkour foot scanner.
        attach: Body name on ``entity`` the rays are hung from. One body per
            reference: Isaac Lab's ``RayCaster`` is a single prim path.
        entity: Entity the attach body belongs to.
        offset: Translation of the sensor frame from the attach body, metres,
            expressed in the attach body frame. The parkour scanner is
            ``(0.04, 0.0, 20.0)``; the depth camera is the head offset.
        offset_rot: Sensor-frame quaternion ``(w, x, y, z)`` in
            ``offset_convention``. Identity for the scanner. The parkour camera
            is a world-convention pitch of about 48 degrees.
        offset_convention: How ``offset_rot`` orients the optical frame.
            ``"world"`` is +X forward, +Z up, +Y left -- both parkour sources
            write this. ROS / OpenGL are refused until a task asserts them.
        direction: Ray direction in the alignment frame. Used by the grid.
            ``(0, 0, -1)`` is straight down.
        pattern: Sample layout -- grid or pinhole.
        hit: What the rays may strike. The string ``"terrain"`` is the scanner:
            the ground mesh / terrain body, not "geom groups 0, 1, 2". A tuple
            is an explicit list; the reserved name ``"terrain"`` means the
            ground and every other entry is a body / link name. A group mask
            is not expressible here on purpose -- on the G1, group 2 is the
            visual shoe and group 3 is the collision capsule.
        ray_alignment: How the sensor frame tracks the attach body.
            ``"yaw"`` ignores roll and pitch (the scanner). ``"base"`` uses the
            full body rotation (the camera -- both source *codes* do this,
            even though both source *configs* write ``ray_alignment="yaw"``).
        miss: What a ray that hits nothing reports. ``"infinity"`` is +inf.
            A depth camera that clipped a miss to ``max_distance`` would feed
            the policy a plausible far wall; the raw sensor keeps +inf so that
            failure is visible. The observation term then clips for the net.
        max_distance: Longest ray, metres. The scanner uses ``1e6`` so a 20 m
            sky origin reaches the ground. The camera uses ``2.5``, the same
            number the parkour sources normalise against.
        min_distance: Hits closer than this are ignored, metres. The camera
            uses ``0.1`` so the torso it hangs from is not a wall.
        crop: ``(top, bottom, left, right)`` pixels discarded from a pinhole
            image before the observation term stores history. ``None`` keeps
            the full resolution. Parkour is ``(18, 0, 16, 16)`` on 36x64,
            which is an 18x32 crop of the lower half.
        update_period: Seconds between sensor updates. ``None`` lets the
            backend use the physics step.
    """

    name: str
    attach: str
    mode: Literal["ray", "terrain_height"] = "ray"
    entity: str = "robot"
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    offset_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    offset_convention: Literal["world"] = "world"
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
    pattern: RayPatternRef = RayPatternRef()
    hit: str | tuple[str, ...] = _TERRAIN_HIT
    ray_alignment: Literal["base", "yaw", "world"] = "yaw"
    miss: Literal["infinity"] = "infinity"
    max_distance: float = 1e6
    min_distance: float = 0.0
    crop: tuple[int, int, int, int] | None = None
    update_period: float | None = 0.02

    def __post_init__(self) -> None:
        if not self.name or not self.entity:
            raise ValueError("Ray caster name and entity must be non-empty.")
        if not self.attach:
            raise ValueError(f"Ray caster {self.name!r} has no attach body.")
        if self.mode not in {"ray", "terrain_height"}:
            raise ValueError(f"Ray caster {self.name!r} has unsupported mode={self.mode!r}.")
        if isinstance(self.hit, str):
            if self.hit != _TERRAIN_HIT:
                raise ValueError(
                    f"Ray caster {self.name!r} has hit={self.hit!r}; the only string "
                    "form is 'terrain'. A camera lists terrain plus body names as a tuple."
                )
        else:
            names = tuple(self.hit)
            if not names:
                raise ValueError(f"Ray caster {self.name!r} was given an empty hit list.")
            object.__setattr__(self, "hit", names)
        if self.pattern.kind == "grid" and self.hit != _TERRAIN_HIT:
            raise ValueError(
                f"Ray caster {self.name!r} is a grid and has hit={self.hit!r}; "
                "the scanner increment only implements terrain-only hits."
            )
        if self.mode == "terrain_height" and (self.pattern.kind != "grid" or self.hit != _TERRAIN_HIT):
            raise ValueError(
                f"Ray caster {self.name!r} uses mode='terrain_height', which requires a terrain-only grid."
            )
        if self.miss != "infinity":
            raise ValueError(
                f"Ray caster {self.name!r} has miss={self.miss!r}; the portable contract "
                "is +inf so a miss cannot be mistaken for a hit at a plausible depth."
            )
        if self.offset_convention != "world":
            raise ValueError(
                f"Ray caster {self.name!r} has offset_convention={self.offset_convention!r}; "
                "only 'world' (+X forward, +Z up) is implemented."
            )
        if self.ray_alignment not in {"base", "yaw", "world"}:
            raise ValueError(f"Ray caster {self.name!r} has unsupported ray_alignment={self.ray_alignment!r}.")
        if not math.isfinite(self.max_distance) or self.max_distance <= 0.0:
            raise ValueError(f"Ray caster {self.name!r} has a non-positive max_distance.")
        if not math.isfinite(self.min_distance) or self.min_distance < 0.0:
            raise ValueError(f"Ray caster {self.name!r} has a negative min_distance.")
        if self.min_distance >= self.max_distance:
            raise ValueError(
                f"Ray caster {self.name!r} has min_distance={self.min_distance} >= max_distance={self.max_distance}."
            )
        if self.update_period is not None and (not math.isfinite(self.update_period) or self.update_period <= 0.0):
            raise ValueError(f"Ray caster {self.name!r} has a non-positive update_period.")
        if not all(math.isfinite(value) for value in (*self.offset, *self.direction, *self.offset_rot)):
            raise ValueError(f"Ray caster {self.name!r} has non-finite pose or direction values.")
        direction_norm = math.sqrt(sum(value * value for value in self.direction))
        if not math.isclose(direction_norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"Ray caster {self.name!r} direction must be unit length, got norm={direction_norm}.")
        quaternion_norm = math.sqrt(sum(value * value for value in self.offset_rot))
        if not math.isclose(quaternion_norm, 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError(f"Ray caster {self.name!r} offset_rot must be unit length, got norm={quaternion_norm}.")
        if self.crop is not None and self.pattern.kind != "pinhole":
            raise ValueError(f"Ray caster {self.name!r} sets crop on a non-pinhole pattern.")
        if self.crop is not None:
            if any(v < 0 for v in self.crop):
                raise ValueError(f"Ray caster {self.name!r} has a negative crop {self.crop}.")
            height, width = self.cropped_hw()
            if height < 1 or width < 1:
                raise ValueError(
                    f"Ray caster {self.name!r} crop {self.crop} leaves a "
                    f"{height}x{width} image on {self.pattern.height}x{self.pattern.width}."
                )

    def hit_names(self) -> tuple[str, ...]:
        """Everything this sensor may strike, including the ``terrain`` token."""
        if isinstance(self.hit, str):
            return (self.hit,)
        return self.hit

    def hits_terrain(self) -> bool:
        return _TERRAIN_HIT in self.hit_names()

    def hit_bodies(self) -> tuple[str, ...]:
        """Body / link names only -- not the reserved ``terrain`` token."""
        return tuple(name for name in self.hit_names() if name != _TERRAIN_HIT)

    def cropped_hw(self) -> tuple[int, int]:
        """Image height and width after ``crop``, or the pinhole resolution."""
        height, width = self.pattern.height, self.pattern.width
        if self.crop is None:
            return height, width
        top, bottom, left, right = self.crop
        return height - top - bottom, width - left - right
