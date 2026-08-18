"""Task registrations.

Two registries exist here for a reason that is temporary. :mod:`instinctlab.tasks.registry` maps a
task id to the engine-independent :class:`~instinctlab.spec.TaskSpec` that both engines compile;
the Gym registrations under ``config/`` remain the entry point for main's Isaac-only training path
and are loaded only after ``AppLauncher`` has started, since importing them requires Isaac Sim.

Importing this package must stay cheap and engine-free -- it is what a launcher reads to find out
what can be trained, before it knows which engine will train it.
"""

from .registry import TASKS, factory, ids, spec


def register_legacy_isaac_tasks() -> None:
    """Import the Gym registrations. Requires Isaac Sim; call only after it has been launched."""
    from isaaclab_tasks.utils import import_packages

    import_packages(__name__, ["utils", "registry", "agents"])


__all__ = ["TASKS", "factory", "ids", "register_legacy_isaac_tasks", "spec"]
