"""Body-local volume-point and terrain-obstacle declarations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ._names import as_name_tuple


def _linspace(lo: float, hi: float, count: int) -> tuple[float, ...]:
    if count == 1:
        return (lo,)
    step = (hi - lo) / (count - 1)
    return tuple(lo + index * step for index in range(count))


@dataclass(frozen=True)
class Grid3dPointsRef:
    """A regular grid in the attach-body local frame."""

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
            if not math.isfinite(lo) or not math.isfinite(hi):
                raise ValueError(f"Grid3dPointsRef.{axis} bounds must be finite.")
            if lo > hi:
                raise ValueError(f"Grid3dPointsRef.{axis} has min={lo} above max={hi}.")

    @property
    def count(self) -> int:
        return self.x_num * self.y_num * self.z_num

    def points(self) -> tuple[tuple[float, float, float], ...]:
        """Return local points in the ``ij`` linspace order used by both sources."""
        xs = _linspace(self.x_min, self.x_max, self.x_num)
        ys = _linspace(self.y_min, self.y_max, self.y_num)
        zs = _linspace(self.z_min, self.z_max, self.z_num)
        return tuple((x, y, z) for x in xs for y in ys for z in zs)


@dataclass(frozen=True)
class VolumePointsRef:
    """A body-local point cloud used to measure obstacle penetration and point speed."""

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
        object.__setattr__(self, "attach", as_name_tuple(self.attach))
        if not self.name or not self.entity:
            raise ValueError("Volume points sensor name and entity must be non-empty.")
        if not self.attach:
            raise ValueError(f"Volume points {self.name!r} was given no attach bodies.")
        if len(set(self.attach)) != len(self.attach):
            raise ValueError(f"Volume points {self.name!r} repeats an attach body.")
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
        if self.update_period is not None and (not math.isfinite(self.update_period) or self.update_period <= 0.0):
            raise ValueError(f"Volume points {self.name!r} has a non-positive update_period.")

    @property
    def bodies(self) -> tuple[str, ...]:
        return tuple(self.attach)


@dataclass(frozen=True)
class VirtualObstacleRef:
    """An abstract obstacle generated from terrain geometry at import time."""

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
        if not math.isfinite(self.cylinder_radius) or self.cylinder_radius <= 0.0:
            raise ValueError(f"Virtual obstacle {self.name!r} has a non-positive cylinder_radius.")
        if self.min_points < 2:
            raise ValueError(f"Virtual obstacle {self.name!r} has min_points={self.min_points}.")
        if not math.isfinite(self.angle_threshold) or not 0.0 < self.angle_threshold <= 180.0:
            raise ValueError(f"Virtual obstacle {self.name!r} angle_threshold must be in (0, 180].")


__all__ = ["Grid3dPointsRef", "VirtualObstacleRef", "VolumePointsRef"]
