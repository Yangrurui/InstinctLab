"""Where a task id resolves to a :class:`TaskSpec`, for every engine.

Gym's registry cannot serve this purpose. Registering ``Instinct-Parkour-Flat-G1-v0`` requires
importing the Isaac Lab env config that the id points at, so the act of listing what tasks exist
demands Isaac Sim -- which is exactly wrong for a task that is supposed to be engine-independent.

So task ids live here, mapped to factories rather than to specs. A factory is imported only when
its task is asked for, which keeps ``instinctlab.tasks.registry`` importable in a bare interpreter
and keeps listing cheap. There is no second Gym registry to keep synchronized.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from instinctlab.spec import TaskSpec

TASKS: dict[str, str] = {
    "Instinct-Velocity-Flat-G1": "instinctlab.tasks.locomotion.config.g1:flat_g1",
    "Instinct-Velocity-Rough-G1": "instinctlab.tasks.locomotion.config.g1:rough_g1",
    "Instinct-Parkour-Target-G1": "instinctlab.tasks.parkour.config.g1:parkour_target_g1",
    "Instinct-Shadowing-WholeBody-Plane-G1-v0": "instinctlab.tasks.shadowing.config:g1_plane_shadowing",
    "Instinct-Shadowing-WholeBody-Plane-G1-Play-v0": (
        "instinctlab.tasks.shadowing.config:g1_plane_shadowing_play"
    ),
    "Instinct-Perceptive-Shadowing-G1-v0": "instinctlab.tasks.shadowing.config:g1_perceptive_shadowing",
    "Instinct-Perceptive-Shadowing-G1-Play-v0": (
        "instinctlab.tasks.shadowing.config:g1_perceptive_shadowing_play"
    ),
    "Instinct-Perceptive-Shadowing-G1-OneMotion-v0": (
        "instinctlab.tasks.shadowing.config:g1_perceptive_shadowing_one_motion"
    ),
    "Instinct-Perceptive-Shadowing-G1-OneMotion-Play-v0": (
        "instinctlab.tasks.shadowing.config:g1_perceptive_shadowing_one_motion_play"
    ),
    "Instinct-Perceptive-Vae-G1-v0": "instinctlab.tasks.shadowing.config:g1_perceptive_vae",
    "Instinct-Perceptive-Vae-G1-Play-v0": "instinctlab.tasks.shadowing.config:g1_perceptive_vae_play",
    "Instinct-Perceptive-HOI-Shadowing-G1-v0": (
        "instinctlab.tasks.shadowing.config:g1_perceptive_hoi_shadowing"
    ),
    "Instinct-Perceptive-HOI-Shadowing-G1-Play-v0": (
        "instinctlab.tasks.shadowing.config:g1_perceptive_hoi_shadowing_play"
    ),
    "Instinct-BeyondMimic-Plane-G1-v0": "instinctlab.tasks.shadowing.config:g1_beyondmimic_plane",
    "Instinct-BeyondMimic-Plane-G1-Play-v0": (
        "instinctlab.tasks.shadowing.config:g1_beyondmimic_plane_play"
    ),
}
"""Task id -> dotted path of the factory returning its :class:`TaskSpec`.

The id is the one the spec declares. Keep the two in step; :func:`spec` checks.
"""


def ids() -> tuple[str, ...]:
    """Every declared task id, without importing any of them."""
    return tuple(sorted(TASKS))


def factory(task_id: str) -> Callable[[], TaskSpec]:
    """Import and return the factory for ``task_id``."""
    try:
        path = TASKS[task_id]
    except KeyError:
        raise KeyError(f"unknown task {task_id!r}; declared tasks are {', '.join(ids())}") from None
    module_path, _, attr = path.partition(":")
    return getattr(import_module(module_path), attr)


def spec(task_id: str) -> TaskSpec:
    """Build the spec for ``task_id``, checking that it agrees about its own name."""
    built = factory(task_id)()
    if built.task_id != task_id:
        raise ValueError(
            f"the registry calls this task {task_id!r} but the spec calls itself {built.task_id!r}; "
            "a task known by two names cannot be compared across engines"
        )
    built.validate()
    return built
