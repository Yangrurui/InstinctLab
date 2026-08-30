"""Current registry-boundary task materialization for tests."""

from __future__ import annotations

from instinctlab_engine_isaacsim.assets import robot_spec as isaac_robot_spec
from instinctlab_engine_mjlab.assets import robot_spec as mjlab_robot_spec
from instinctlab.tasks import registry


def task_spec(task_id: str, engine: str = "mjlab"):
    """Materialize a task with the one native asset selected by ``engine``."""
    asset_id = registry.asset_id(task_id)
    if engine == "mjlab":
        robot = mjlab_robot_spec(asset_id)
    elif engine == "isaacsim":
        robot = isaac_robot_spec(asset_id)
    else:
        raise ValueError(f"Unknown test engine {engine!r}.")
    return registry.spec(task_id, robot)
