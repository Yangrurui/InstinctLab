"""References to contact measurements, stated without naming an engine.

The two engines take opposite approaches to contact sensing, and the difference is structural
rather than cosmetic:

* Isaac Lab declares **one broad sensor** over a prim-path pattern -- typically every body of the
  robot -- and each term slices out the bodies it cares about with a ``SceneEntityCfg``.
* mjlab declares **many narrow sensors**, each one already scoped to its elements by a ``primary``
  pattern, and terms read the whole sensor.

Neither is more correct, and a portable term cannot be written against either shape directly. So a
:class:`ContactSensorRef` says only *what is being measured* -- these elements of this entity,
optionally only against that counterpart -- and each backend decides whether that becomes a slice
of a broad sensor or a sensor of its own. This is the same move the rest of the design makes: state
the intent in the IR, let the backend pick the idiom.

What can be read back portably is narrower than it looks. See
:mod:`~instinctlab.compat.sensors`: the air/contact-time signals line up across engines, raw
contact force does not.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "ContactSensorRef",
    "Grid3dPointsRef",
    "MotionReferenceRef",
    "RayCasterRef",
    "RayPatternRef",
    "VirtualObstacleRef",
    "VolumePointsRef",
]

_TERRAIN_HIT = "terrain"


def _normalise(patterns: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(patterns, str):
        return (patterns,)
    return tuple(patterns)


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
    history_length: int = 0
    preserve_order: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "elements", _normalise(self.elements))
        if not self.elements:
            raise ValueError(f"Contact sensor {self.name!r} was given no element patterns.")
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
            if self.resolution <= 0.0:
                raise ValueError(f"Ray pattern resolution must be positive, got {self.resolution}.")
            if self.size[0] < 0.0 or self.size[1] < 0.0:
                raise ValueError(f"Ray pattern size must be non-negative, got {self.size}.")
            return
        if self.kind == "pinhole":
            if self.width < 1 or self.height < 1:
                raise ValueError(f"Pinhole resolution must be at least 1x1, got {self.width}x{self.height}.")
            if self.horizontal_fov_deg <= 0.0 or self.vertical_fov_deg <= 0.0:
                raise ValueError(
                    "Pinhole FOV must be positive, got "
                    f"horizontal={self.horizontal_fov_deg}, vertical={self.vertical_fov_deg}."
                )
            if self.focal_length <= 0.0:
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
        if not self.attach:
            raise ValueError(f"Ray caster {self.name!r} has no attach body.")
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
        if self.max_distance <= 0.0:
            raise ValueError(f"Ray caster {self.name!r} has a non-positive max_distance.")
        if self.min_distance < 0.0:
            raise ValueError(f"Ray caster {self.name!r} has a negative min_distance.")
        if self.min_distance >= self.max_distance:
            raise ValueError(
                f"Ray caster {self.name!r} has min_distance={self.min_distance} >= max_distance={self.max_distance}."
            )
        if self.update_period is not None and self.update_period < 0.0:
            raise ValueError(f"Ray caster {self.name!r} has a negative update_period.")
        if abs(self.direction[0]) + abs(self.direction[1]) + abs(self.direction[2]) <= 0.0:
            raise ValueError(f"Ray caster {self.name!r} has a zero direction.")
        if sum(c * c for c in self.offset_rot) <= 0.0:
            raise ValueError(f"Ray caster {self.name!r} has a zero offset_rot.")
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


@dataclass(frozen=True)
class MotionReferenceRef:
    """A clip-backed motion reference. These numbers are not simulation quantities.

    Both source implementations (Isaac ``motion_reference/`` and InstinctMJ) load a
    retargetted clip, remap joints **by name**, run ``pytorch_kinematics`` on that
    engine's robot description, and finite-difference velocities at load. This
    declaration says that contract once. The backend picks the description (URDF
    or MJCF) and attaches a sensor; it does not invent a second clip format.

    Frames, written here so a mix-up cannot hide behind a plausible pose:

    * Quaternions are hub ``wxyz``. ``xyzw`` is refused.
    * ``base_*_w`` is the clip root in the clip world.
    * ``link_*_b`` is root-relative (the clip root, not the live robot).
    * ``link_*_w`` is the same pose composed into the clip world.
    * Velocities are finite-differenced from the clip at ``1 / clip_fps``.

    Joints leave the file in the published order (parkour: legs-first) and are
    remapped by name onto :attr:`joints`. A length-only check is not a remap:
    missing names fail loudly on either side.

    Exhaustion is ``freeze_last_and_flag``: the last frame is held, ``validity``
    goes false, and ``exhausted_count`` increments. The clip does not silently
    restart. ``reset_without_notice`` on the AMP termination is a later increment
    and is what made ``dataset_exhausted`` read as 0; this sensor keeps the
    counter even if that term is added later.

    Left out of this increment, on purpose: AMP observation groups, a
    discriminator / WASABI reward, MoE policy, visualization, symmetric
    augmentation, multi-process clip splits, HOI objects, SMPL retargetting.
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "joints", _normalise(self.joints))
        object.__setattr__(self, "links", _normalise(self.links))
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
        if self.frame_interval_s <= 0.0:
            raise ValueError(f"Motion reference {self.name!r} has a non-positive frame_interval_s.")
        if self.update_period < 0.0:
            raise ValueError(f"Motion reference {self.name!r} has a negative update_period.")
        if self.clip_target_fps <= 0.0:
            raise ValueError(f"Motion reference {self.name!r} has a non-positive clip_target_fps.")
        lo, hi = self.start_range
        if lo < 0.0 or hi < lo or hi > 1.0:
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


@dataclass(frozen=True)
class Grid3dPointsRef:
    """A regular grid in the attach-body local frame.

    ``z_min`` / ``z_max`` are body-local, not world. The G1 shoe sole is
    ``z_min=-0.063``, ``z_max=-0.023`` — below the ankle origin. A sign flip
    here puts the cloud in the shin and the penalty still looks smooth.
    """

    x_min: float
    x_max: float
    x_num: int
    y_min: float
    y_max: float
    y_num: int
    z_min: float
    z_max: float
    z_num: int

    def __post_init__(self) -> None:
        for axis, lo, hi, count in (
            ("x", self.x_min, self.x_max, self.x_num),
            ("y", self.y_min, self.y_max, self.y_num),
            ("z", self.z_min, self.z_max, self.z_num),
        ):
            if count < 1:
                raise ValueError(f"Grid3dPointsRef.{axis}_num must be at least 1, got {count}.")
            if lo > hi:
                raise ValueError(f"Grid3dPointsRef.{axis} has min={lo} above max={hi}.")

    @property
    def count(self) -> int:
        return self.x_num * self.y_num * self.z_num

    def points(self) -> tuple[tuple[float, float, float], ...]:
        """Local points, ``ij`` linspace order, matching both source generators."""
        xs = _linspace(self.x_min, self.x_max, self.x_num)
        ys = _linspace(self.y_min, self.y_max, self.y_num)
        zs = _linspace(self.z_min, self.z_max, self.z_num)
        return tuple((x, y, z) for x in xs for y in ys for z in zs)


def _linspace(lo: float, hi: float, count: int) -> tuple[float, ...]:
    if count == 1:
        return (lo,)
    step = (hi - lo) / (count - 1)
    return tuple(lo + i * step for i in range(count))


@dataclass(frozen=True)
class VolumePointsRef:
    """A cloud of points hung from one or more bodies, in each body's local frame.

    This is not a ray caster. Each point is transformed by the attach body's
    pose; penetration is a vector from the obstacle surface toward the point
    (world frame); point velocity is the attach-body **link** velocity plus
    ``ω × (p_w - origin_w)``. Quaternions are hub ``wxyz``.

    Left out of this increment, on purpose: arm clouds, ``volume_points_step_safety``,
    visualisation, and any detector other than greedy-concat edge cylinders.
    """

    name: str
    attach: str | Sequence[str]
    entity: str = "robot"
    grid: Grid3dPointsRef = Grid3dPointsRef(
        x_min=-0.025,
        x_max=0.12,
        x_num=10,
        y_min=-0.03,
        y_max=0.03,
        y_num=5,
        z_min=-0.04,
        z_max=0.0,
        z_num=2,
    )
    update_period: float | None = None
    frame: Literal["attach"] = "attach"
    quaternion: Literal["wxyz"] = "wxyz"
    velocity: Literal["attach_link"] = "attach_link"

    def __post_init__(self) -> None:
        object.__setattr__(self, "attach", _normalise(self.attach))
        if not self.name:
            raise ValueError("Volume points sensor has no name.")
        if not self.attach:
            raise ValueError(f"Volume points {self.name!r} was given no attach bodies.")
        if self.frame != "attach":
            raise ValueError(
                f"Volume points {self.name!r} has frame={self.frame!r}; the grid is in the attach-body local frame."
            )
        if self.quaternion != "wxyz":
            raise ValueError(f"Volume points {self.name!r} has quaternion={self.quaternion!r}; hub convention is wxyz.")
        if self.velocity != "attach_link":
            raise ValueError(
                f"Volume points {self.name!r} has velocity={self.velocity!r}; "
                "point speed is link-origin velocity plus ω × r."
            )
        if self.update_period is not None and self.update_period < 0.0:
            raise ValueError(f"Volume points {self.name!r} has a negative update_period.")

    @property
    def bodies(self) -> tuple[str, ...]:
        return tuple(self.attach)


@dataclass(frozen=True)
class VirtualObstacleRef:
    """An abstract obstacle generated from terrain geometry at import time.

    Not a raycast. Edges are detected on the terrain mesh (Isaac) or the
    reconstructed / repaired height-field surface (mjlab), then fattened into
    cylinders. ``kind`` is the detector; extras only one engine has stay in
    that engine's builder, not here.
    """

    name: str
    kind: Literal["greedy_edge_cylinder"] = "greedy_edge_cylinder"
    cylinder_radius: float = 0.05
    min_points: int = 2
    angle_threshold: float = 70.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Virtual obstacle has no name.")
        if self.kind != "greedy_edge_cylinder":
            raise ValueError(
                f"Virtual obstacle {self.name!r} has kind={self.kind!r}; "
                "only greedy-concat edge cylinders are implemented."
            )
        if self.cylinder_radius <= 0.0:
            raise ValueError(f"Virtual obstacle {self.name!r} has a non-positive cylinder_radius.")
        if self.min_points < 2:
            raise ValueError(f"Virtual obstacle {self.name!r} has min_points={self.min_points}.")
        if self.angle_threshold <= 0.0:
            raise ValueError(f"Virtual obstacle {self.name!r} has a non-positive angle_threshold.")
