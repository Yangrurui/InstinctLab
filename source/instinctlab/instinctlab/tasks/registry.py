"""Where a task id resolves to a :class:`TaskSpec`, for every engine.

Gym's registry cannot serve this purpose. Registering ``Instinct-Parkour-Flat-G1-v0`` requires
importing the Isaac Lab env config that the id points at, so the act of listing what tasks exist
demands Isaac Sim -- which is exactly wrong for a task that is supposed to be engine-independent.

So task ids live here, mapped to factories rather than to specs. A factory is imported only when
its task is asked for, which keeps ``instinctlab.tasks.registry`` importable in a bare interpreter
and keeps listing cheap. The Isaac-only tasks still register with Gym, in
``register_legacy_isaac_tasks()``, which is where anything needing those ids has to ask for them.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from instinctlab.spec import TaskSpec

TASKS: dict[str, str] = {
    "Instinct-Velocity-Flat-G1": "instinctlab.tasks.locomotion.config.flat_g1:flat_g1",
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
    return built
