"""Recoverable state contracts for controllers and other runtime components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypeAlias, runtime_checkable

EnvIds: TypeAlias = Sequence[int] | Any | None
ComponentState: TypeAlias = Mapping[str, Any]


class ComponentContractError(RuntimeError):
    """A runtime component does not implement its declared lifecycle."""


@runtime_checkable
class StatefulComponent(Protocol):
    """A partially resettable component whose mutable state is recoverable.

    State mappings must contain detached values owned by the caller. Restoring
    may update only ``env_ids``; implementations must reject incompatible shape
    or schema rather than partially applying a snapshot.
    """

    def reset(self, env_ids: EnvIds = None) -> None: ...

    def snapshot_state(self, env_ids: EnvIds = None) -> ComponentState: ...

    def restore_state(
        self, state: ComponentState, env_ids: EnvIds = None
    ) -> None: ...


@runtime_checkable
class StatefulController(StatefulComponent, Protocol):
    """The 1.0 contract for a controller with history, delay, or recurrent state.

    ``control_dt`` names the cadence at which the controller consumes a new
    command. A backend may apply its held output on every physics tick, but it
    must not pretend the command clock itself runs at the physics rate.
    """

    @property
    def control_dt(self) -> float: ...


def validate_stateful_component(name: str, component: object) -> None:
    """Fail early when a declared stateful component lacks a required hook."""
    missing = [
        method
        for method in ("reset", "snapshot_state", "restore_state")
        if not callable(getattr(component, method, None))
    ]
    if missing:
        raise ComponentContractError(
            f"Lifecycle component {name!r} declares recoverable state but "
            f"{type(component).__name__} lacks: {', '.join(missing)}."
        )


__all__ = [
    "ComponentContractError",
    "ComponentState",
    "EnvIds",
    "StatefulComponent",
    "StatefulController",
    "validate_stateful_component",
]
