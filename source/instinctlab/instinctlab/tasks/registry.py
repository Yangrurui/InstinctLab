"""Engine-neutral task registry."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable


def _load(path: str) -> Any:
    module_name, attribute = path.split(":", maxsplit=1)
    return getattr(importlib.import_module(module_name), attribute)


@dataclass(frozen=True)
class TaskRegistration:
    task_id: str
    env_cfg_entry_point: str
    agent_cfg_entry_point: str
    supported_backends: frozenset[str]
    schema_entry_point: str | None = None

    def make_env_cfg(self, **kwargs: Any) -> Any:
        factory: Callable[..., Any] = _load(self.env_cfg_entry_point)
        return factory(**kwargs)

    def make_agent_cfg(self, **kwargs: Any) -> Any:
        factory: Callable[..., Any] = _load(self.agent_cfg_entry_point)
        return factory(**kwargs)

    def make_schema(self, **kwargs: Any) -> Any:
        if self.schema_entry_point is None:
            raise ValueError(f"task {self.task_id!r} does not declare a schema_entry_point")
        factory: Callable[..., Any] = _load(self.schema_entry_point)
        return factory(**kwargs)


class TaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, TaskRegistration] = {}

    def register(self, registration: TaskRegistration) -> None:
        previous = self._tasks.get(registration.task_id)
        if previous is not None and previous != registration:
            raise ValueError(f"task {registration.task_id!r} is already registered")
        self._tasks[registration.task_id] = registration

    def get(self, task_id: str) -> TaskRegistration:
        try:
            return self._tasks[task_id]
        except KeyError as error:
            raise KeyError(f"unknown task {task_id!r}; available: {', '.join(self.ids())}") from error

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._tasks))


TASKS = TaskRegistry()
TASKS.register(
    TaskRegistration(
        task_id="Instinct-Locomotion-Flat-G1-v0",
        env_cfg_entry_point="instinctlab.tasks.locomotion.unified_flat_env_cfg:locomotion_flat_env_cfg",
        agent_cfg_entry_point="instinctlab.tasks.locomotion.unified_flat_env_cfg:locomotion_flat_agent_cfg",
        supported_backends=frozenset({"mock", "isaacsim", "mjlab"}),
        schema_entry_point="instinctlab.tasks.locomotion.unified_flat_env_cfg:locomotion_flat_env_schema",
    )
)


__all__ = ["TASKS", "TaskRegistration", "TaskRegistry"]
