"""Runtime lifecycle services shared by every engine adapter."""

from .component import (
    ComponentContractError,
    StatefulComponent,
    StatefulController,
)
from .runtime import ClockReading, LifecycleRuntime, attach_lifecycle
from .snapshot import EnvironmentSnapshot, SnapshotError, SnapshotProvider

__all__ = [
    "ClockReading",
    "ComponentContractError",
    "EnvironmentSnapshot",
    "LifecycleRuntime",
    "SnapshotError",
    "SnapshotProvider",
    "StatefulComponent",
    "StatefulController",
    "attach_lifecycle",
]
