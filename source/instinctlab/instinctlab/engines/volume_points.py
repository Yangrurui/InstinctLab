"""Shared VolumePoints geometry. No engine SDK.

The local grid is what both source generators emit (``ij`` linspace). The
cylinder offset is the warp kernel written out so a test can compute the
number by hand without launching Warp.
"""

from __future__ import annotations

import math

from instinctlab.spec.sensor import Grid3dPointsRef

__all__ = [
    "cylinder_penetration_offset",
    "grid3d_points",
    "link_linear_velocity_from_com",
    "penetration_reward",
    "point_velocity_from_link",
]


def grid3d_points(grid: Grid3dPointsRef) -> tuple[tuple[float, float, float], ...]:
    """Local points in the attach-body frame. Same order as both source generators."""
    return grid.points()


def cylinder_penetration_offset(
    point: tuple[float, float, float],
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    radius: float,
) -> tuple[float, float, float]:
    """World-frame offset from the cylinder surface toward ``point``.

    Matches ``points_penetrate_cylinder_kernel``: project onto the finite
    segment; if the radial distance is inside ``radius``, the offset is
    ``(proj - p) * (radius - dist) / dist``. A point on the axis has depth
    ``radius``; the direction is an arbitrary stable perpendicular (world-z
    cross axis, else world-x), pointing from a surface point toward the axis.
    """
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
        nx, ny, nz = uy * 1.0 - uz * 0.0, uz * 0.0 - ux * 1.0, ux * 0.0 - uy * 0.0
        nlen = math.sqrt(nx * nx + ny * ny + nz * nz)
        if nlen <= 1e-8:
            nx, ny, nz = uy * 0.0 - uz * 0.0, uz * 1.0 - ux * 0.0, ux * 0.0 - uy * 1.0
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
    """Hub point velocity: ``v_link + ω × (p_w - origin_w)``.

    ``v_link`` and ``ω`` are about the attach-body **link origin**, not the
    centre of mass. PhysX ``get_velocities()`` linear is COM; pairing that
    with a link origin as the lever arm is a different number whenever the
    foot COM is offset and ω ≠ 0.
    """
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
    """``v_link = v_com + ω × (origin - com)`` = ``v_com + ω × (-com_offset_w)``.

    ``com_offset_w`` is the world vector from the link origin to the COM
    (Isaac Lab: ``quat_apply(link_quat, com_pos_b)``).
    """
    wx, wy, wz = ang_vel
    ox, oy, oz = com_offset_w
    return (
        com_lin_vel[0] + (wy * -oz - wz * -oy),
        com_lin_vel[1] + (wz * -ox - wx * -oz),
        com_lin_vel[2] + (wx * -oy - wy * -ox),
    )


def penetration_reward(depth: float, speed: float, *, tolerance: float = 0.0) -> float:
    """One-point form of ``volume_points_penetration``."""
    if depth <= tolerance:
        return 0.0
    return (speed + 1e-6) * depth
