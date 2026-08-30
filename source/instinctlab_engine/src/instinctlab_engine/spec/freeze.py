"""Create the immutable declaration snapshot consumed by compilers and contracts."""

from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from dataclasses import fields, is_dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .task import TaskSpec

_IN_PROGRESS = object()


def _freeze(value: Any, memo: dict[int, Any]) -> Any:
    """Copy one declaration value while replacing mutable containers."""
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return value

    value_id = id(value)
    previous = memo.get(value_id)
    if previous is _IN_PROGRESS:
        raise ValueError("TaskSpec declarations must not contain container cycles")
    if previous is not None:
        return previous

    if isinstance(value, Mapping):
        memo[value_id] = _IN_PROGRESS
        backing: dict[Any, Any] = {}
        for key, item in value.items():
            backing[_freeze(key, memo)] = _freeze(item, memo)
        frozen = MappingProxyType(backing)
        memo[value_id] = frozen
        return frozen

    if is_dataclass(value) and not isinstance(value, type):
        parameters = getattr(type(value), "__dataclass_params__", None)
        if parameters is None or not parameters.frozen:
            raise TypeError(
                "TaskSpec declarations may contain only frozen dataclasses; "
                f"got {type(value).__module__}.{type(value).__qualname__}"
            )
        memo[value_id] = _IN_PROGRESS
        frozen = copy(value)
        for declaration_field in fields(value):
            object.__setattr__(
                frozen,
                declaration_field.name,
                _freeze(getattr(value, declaration_field.name), memo),
            )
        memo[value_id] = frozen
        return frozen

    if isinstance(value, (tuple, list)):
        memo[value_id] = _IN_PROGRESS
        frozen = tuple(_freeze(item, memo) for item in value)
        memo[value_id] = frozen
        return frozen

    if isinstance(value, (set, frozenset)):
        memo[value_id] = _IN_PROGRESS
        frozen = frozenset(_freeze(item, memo) for item in value)
        memo[value_id] = frozen
        return frozen

    # Functions, enums, paths, scalar library values, and other already-immutable
    # leaves are retained. Contract serialization remains the authority that
    # rejects values which cannot be represented reproducibly.
    return value


def freeze_task_spec(spec: TaskSpec) -> TaskSpec:
    """Return a deeply immutable copy of a validated task declaration.

    Task configuration objects remain mutable while a family assembles them.
    The application registry calls this once, after initial validation, and
    validates the returned snapshot again. Hashing and compilation therefore
    consume exactly the same object and cannot observe later source mutations.
    """
    from .task import TaskSpec

    if not isinstance(spec, TaskSpec):
        raise TypeError(f"freeze_task_spec expects TaskSpec, got {type(spec).__name__}")
    frozen = _freeze(spec, {})
    if not isinstance(frozen, TaskSpec):  # pragma: no cover - guarded above
        raise TypeError("TaskSpec freezing produced an invalid root object")
    return frozen


__all__ = ["freeze_task_spec"]
