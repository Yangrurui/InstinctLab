"""Portable event terms. One implementation, run by either engine's EventManager.

Events are usually per-engine (``kind=``). Registration is the exception: both
sensors expose ``register_virtual_obstacles`` and both terrains expose
``virtual_obstacles``. The function still has to fail loudly on an empty set —
that is the silent-zero path this increment exists to close.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from instinctlab.compat.sensors import registered_cylinder_count, require_volume_points_registered
from instinctlab.spec.sensor import VolumePointsRef

__all__ = ["register_virtual_obstacles"]


def _sensor_name(sensor: VolumePointsRef | str) -> str:
    return sensor.name if isinstance(sensor, VolumePointsRef) else sensor


def register_virtual_obstacles(
    env: Any,
    env_ids: Any,
    sensor: VolumePointsRef | str | Sequence[VolumePointsRef | str],
) -> None:
    """Wire terrain edge cylinders onto each volume-points sensor.

    An empty ``terrain.virtual_obstacles`` is a construction bug, not a well-behaved
    robot. The penalty would then exist, appear in the reward table, and stay 0.0.
    """
    del env_ids
    names: tuple[str, ...]
    if isinstance(sensor, (str, VolumePointsRef)):
        names = (_sensor_name(sensor),)
    else:
        names = tuple(_sensor_name(item) for item in sensor)

    terrain = env.scene.terrain
    obstacles = getattr(terrain, "virtual_obstacles", None)
    if not obstacles:
        raise RuntimeError(
            "register_virtual_obstacles: terrain.virtual_obstacles is empty. "
            "The volume-points penalty would be identically zero. The parkour "
            "importer must generate edge cylinders before this startup event."
        )

    for name in names:
        volume = env.scene.sensors[name]
        if not hasattr(volume, "register_virtual_obstacles"):
            raise RuntimeError(f"sensor {name!r} has no register_virtual_obstacles; it is not a volume-points sensor.")
        volume.register_virtual_obstacles(obstacles)
        require_volume_points_registered(volume)
        count = registered_cylinder_count(volume)
        print(f"[{name}] registered {len(obstacles)} virtual obstacle set(s), {count} cylinders.")
