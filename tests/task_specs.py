"""Current registry-boundary task materialization for tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from instinctlab_engine_isaacsim.assets import robot_spec as isaac_robot_spec
from instinctlab_engine_mjlab.assets import robot_spec as mjlab_robot_spec
from instinctlab.tasks import registry


RIGID_OBJECT_FIXTURE = Path(__file__).parent / "fixtures" / "rigid_object" / "cube.obj"


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


def with_rigid_object_fixture(spec):
    """Replace optional external object data while preserving object semantics."""
    objects = tuple(
        replace(obj, mesh=str(RIGID_OBJECT_FIXTURE), engine_meshes={})
        for obj in spec.scene.rigid_objects
    )
    return replace(spec, scene=replace(spec.scene, rigid_objects=objects))
