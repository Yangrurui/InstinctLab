"""Task registrations.

The unified registry is engine-neutral. Legacy Isaac Gym registrations can be
loaded explicitly after ``AppLauncher`` has started.
"""

from .registry import TASKS, TaskRegistration, TaskRegistry


def register_legacy_isaac_tasks() -> None:
    from isaaclab_tasks.utils import import_packages

    import_packages(__name__, ["utils", "registry"])


__all__ = ["TASKS", "TaskRegistration", "TaskRegistry", "register_legacy_isaac_tasks"]
