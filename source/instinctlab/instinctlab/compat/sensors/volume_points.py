"""Portable volume-point registration, penetration and velocity reads."""

from __future__ import annotations

import math
from typing import Any

import torch

from instinctlab.spec.sensor import Grid3dPointsRef

from ..denylist import PortabilityError


def grid3d_points(grid: Grid3dPointsRef) -> tuple[tuple[float, float, float], ...]:
    """Return local points in the order used by both native generators."""
    return grid.points()


def cylinder_penetration_offset(
    point: tuple[float, float, float],
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    radius: float,
) -> tuple[float, float, float]:
    """Return the world-frame surface-to-point offset for a finite cylinder."""
    px, py, pz = point
    ax, ay, az = start
    bx, by, bz = end
    abx, aby, abz = bx - ax, by - ay, bz - az
    ab_len = math.sqrt(abx * abx + aby * aby + abz * abz)
    if ab_len <= 0.0:
        raise ValueError("cylinder segment has zero length.")
    ux, uy, uz = abx / ab_len, aby / ab_len, abz / ab_len
    t = (px - ax) * ux + (py - ay) * uy + (pz - az) * uz
    if t < 0.0 or t > ab_len:
        return (0.0, 0.0, 0.0)
    projx, projy, projz = ax + t * ux, ay + t * uy, az + t * uz
    dx, dy, dz = px - projx, py - projy, pz - projz
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    if dist >= radius:
        return (0.0, 0.0, 0.0)
    if dist <= 0.0:
        nx, ny, nz = uy, -ux, 0.0
        nlen = math.sqrt(nx * nx + ny * ny + nz * nz)
        if nlen <= 1e-8:
            nx, ny, nz = 0.0, uz, -uy
            nlen = math.sqrt(nx * nx + ny * ny + nz * nz)
        nx, ny, nz = nx / nlen, ny / nlen, nz / nlen
        return (-nx * radius, -ny * radius, -nz * radius)
    scale = (radius - dist) / dist
    return ((projx - px) * scale, (projy - py) * scale, (projz - pz) * scale)


def point_velocity_from_link(
    link_lin_vel: tuple[float, float, float],
    link_ang_vel: tuple[float, float, float],
    link_pos: tuple[float, float, float],
    point_pos: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Apply ``v_point = v_link + omega x (point - link_origin)``."""
    rx = point_pos[0] - link_pos[0]
    ry = point_pos[1] - link_pos[1]
    rz = point_pos[2] - link_pos[2]
    wx, wy, wz = link_ang_vel
    return (
        link_lin_vel[0] + wy * rz - wz * ry,
        link_lin_vel[1] + wz * rx - wx * rz,
        link_lin_vel[2] + wx * ry - wy * rx,
    )


def link_linear_velocity_from_com(
    com_lin_vel: tuple[float, float, float],
    ang_vel: tuple[float, float, float],
    com_offset_w: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Convert COM linear velocity to the link-origin linear velocity."""
    wx, wy, wz = ang_vel
    ox, oy, oz = com_offset_w
    return (
        com_lin_vel[0] + (wy * -oz - wz * -oy),
        com_lin_vel[1] + (wz * -ox - wx * -oz),
        com_lin_vel[2] + (wx * -oy - wy * -ox),
    )


def registered_cylinder_count(sensor: Any) -> int:
    """Return the number of edge cylinders registered with a volume-point sensor."""
    count = getattr(sensor, "registered_cylinder_count", None)
    if count is not None:
        return int(count)
    obstacles = getattr(sensor, "_virtual_obstacles", None)
    if not obstacles:
        return 0
    total = 0
    for obstacle in obstacles.values():
        edges = getattr(obstacle, "edges_pyt", None)
        if edges is not None:
            total += int(edges.shape[0])
            continue
        cylinders = getattr(obstacle, "cylinders", None)
        if cylinders is not None:
            total += int(getattr(cylinders, "num_cylinders", 0))
    return total


def require_volume_points_registered(sensor: Any) -> None:
    """Reject the silent all-zero output produced by an unregistered obstacle set."""
    cfg = getattr(sensor, "cfg", None)
    name = getattr(cfg, "name", None) or type(sensor).__name__
    obstacles = getattr(sensor, "_virtual_obstacles", None)
    registered = bool(getattr(sensor, "virtual_obstacles_registered", False))
    if not registered and not obstacles:
        raise RuntimeError(
            f"Volume-points sensor {name!r} has no registered virtual obstacles; "
            "its penetration offset would be identically zero."
        )
    if registered_cylinder_count(sensor) <= 0:
        raise RuntimeError(
            f"Volume-points sensor {name!r} registered 0 cylinders; its penetration penalty would remain zero."
        )


def _require_link_velocity(sensor: Any) -> None:
    cfg = getattr(sensor, "cfg", None)
    velocity = getattr(cfg, "velocity", None)
    if velocity is None:
        velocity = getattr(sensor, "velocity", None)
    if velocity is None or velocity == "attach_link":
        return
    name = getattr(cfg, "name", None) or type(sensor).__name__
    raise PortabilityError(
        f"Volume-points sensor {name!r} uses velocity={velocity!r}; the portable convention is "
        "the attach-body link origin velocity plus angular transport."
    )


def volume_points_penetration_offset(sensor: Any) -> torch.Tensor:
    """Return surface-to-point penetration offsets as ``(env, body, point, 3)``."""
    require_volume_points_registered(sensor)
    offset = getattr(sensor.data, "penetration_offset", None)
    if offset is None:
        raise PortabilityError(f"{type(sensor).__name__} has no data.penetration_offset.")
    return offset


def volume_points_vel_w(sensor: Any) -> torch.Tensor:
    """Return world-frame point velocity as ``(env, body, point, 3)``."""
    require_volume_points_registered(sensor)
    _require_link_velocity(sensor)
    velocity = getattr(sensor.data, "points_vel_w", None)
    if velocity is None:
        raise PortabilityError(f"{type(sensor).__name__} has no data.points_vel_w.")
    return velocity
