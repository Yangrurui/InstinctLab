"""Runtime lifecycle services shared by every engine adapter."""

from .component import (
    ComponentContractError,
    StatefulComponent,
    StatefulController,
)
from .runtime import ClockReading, LifecycleRuntime, attach_lifecycle
from .snapshot import EnvironmentSnapshot, SnapshotError, SnapshotProvider
from .trace import (
    EpisodeTrace,
    ReplayMismatch,
    ReplayReport,
    TraceError,
    replay_trace,
)

__all__ = [
    "ClockReading",
    "ComponentContractError",
    "EnvironmentSnapshot",
    "EpisodeTrace",
    "LifecycleRuntime",
    "ReplayMismatch",
    "ReplayReport",
    "SnapshotError",
    "SnapshotProvider",
    "StatefulComponent",
    "StatefulController",
    "TraceError",
    "attach_lifecycle",
    "replay_trace",
]
