"""Engine-agnostic declaration layer.

Nothing here may import a physics engine, directly or transitively. A task is declared once in
these types and compiled to a native environment by the backend for whichever engine is running,
so anything that reaches for an engine at this level has already lost the property that makes the
declaration portable. ``tests/test_spec_isolation.py`` enforces this.

:class:`~instinctlab.spec.task.TaskSpec` is the entry point: a whole task, produced by a frontend
that read some project's native definition and consumed by the backend for the engine in use.
"""

from __future__ import annotations

from .capability import Requirement
from .entity import UNIVERSAL_KINDS, EntityRef
from .mdp import (
    ActionTermSpec,
    CommandTermSpec,
    CurriculumTermSpec,
    DoneTermSpec,
    EventTermSpec,
    MdpSpec,
    NoiseSpec,
    ObsGroupSpec,
    ObsTermSpec,
    RewardTermSpec,
    TermSpec,
)
from .sensor import (
    ContactSensorRef,
    Grid3dPointsRef,
    MotionReferenceRef,
    RayCasterRef,
    RayPatternRef,
    SymmetricAugmentationSpec,
    VirtualObstacleRef,
    VolumePointsRef,
)
from .task import AgentSpec, SceneSpec, SimSpec, SubTerrainSpec, TaskSpec, TerrainGeneratorSpec, TerrainSpec

__all__ = [
    "UNIVERSAL_KINDS",
    "ActionTermSpec",
    "AgentSpec",
    "CommandTermSpec",
    "ContactSensorRef",
    "Grid3dPointsRef",
    "MotionReferenceRef",
    "RayCasterRef",
    "RayPatternRef",
    "SymmetricAugmentationSpec",
    "VirtualObstacleRef",
    "VolumePointsRef",
    "CurriculumTermSpec",
    "DoneTermSpec",
    "EntityRef",
    "EventTermSpec",
    "MdpSpec",
    "NoiseSpec",
    "ObsGroupSpec",
    "ObsTermSpec",
    "Requirement",
    "RewardTermSpec",
    "SceneSpec",
    "SimSpec",
    "SubTerrainSpec",
    "TaskSpec",
    "TerrainGeneratorSpec",
    "TerrainSpec",
    "TermSpec",
]
