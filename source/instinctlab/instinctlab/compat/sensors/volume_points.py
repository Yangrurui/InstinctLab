"""Portable volume-point registration, penetration and velocity reads."""

from __future__ import annotations

import torch
from typing import Any

from ..denylist import PortabilityError


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
