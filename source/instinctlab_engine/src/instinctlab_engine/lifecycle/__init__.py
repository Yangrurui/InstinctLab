"""Runtime lifecycle services shared by every engine adapter."""

from .benchmark import (
    BenchmarkThresholdError,
    duration_statistics,
    evaluate_thresholds,
    require_thresholds,
)
from .component import (
    ComponentContractError,
    StatefulComponent,
    StatefulController,
    validate_stateful_controller,
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
    "BenchmarkThresholdError",
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
    "duration_statistics",
    "evaluate_thresholds",
    "replay_trace",
    "require_thresholds",
    "validate_stateful_controller",
]
