"""InstinctLab.

Importing the package is intentionally engine-neutral. Applications explicitly
import :mod:`instinctlab.tasks` after the selected backend has bootstrapped.
"""

from __future__ import annotations


def register_tasks() -> None:
    """Import task registrations after an engine has been selected."""
    from . import tasks  # noqa: F401


def register_ui_extensions() -> None:
    """Load Isaac Sim UI extensions on demand."""
    from . import ui_extension_example  # noqa: F401


__all__ = ["register_tasks", "register_ui_extensions"]
