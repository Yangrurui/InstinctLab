"""Engine-agnostic declaration layer.

Nothing here may import a physics engine, directly or transitively. A task is declared once in
these types and compiled to a native environment by the backend for whichever engine is running,
so anything that reaches for an engine at this level has already lost the property that makes the
declaration portable. Engine isolation is an explicit package boundary.

:class:`~instinctlab_engine.spec.task.TaskSpec` is the entry point: a whole task, produced by a frontend
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
from .rigid_object import RigidObjectRef
from .robot import BackendAsset, JointProperties, RobotSpec
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
from .task import (
    AgentSpec,
    SceneSpec,
    SimSpec,
    SubTerrainSpec,
    TaskSpec,
    TerrainGeneratorSpec,
    TerrainSpec,
)

__all__ = [
    "UNIVERSAL_KINDS",
    "ActionTermSpec",
    "AgentSpec",
    "BackendAsset",
    "CommandTermSpec",
    "ContactSensorRef",
    "CurriculumTermSpec",
    "DoneTermSpec",
    "EntityRef",
    "EventTermSpec",
    "Grid3dPointsRef",
    "JointProperties",
    "MdpSpec",
    "MotionReferenceRef",
    "NoiseSpec",
    "ObsGroupSpec",
    "ObsTermSpec",
    "RayCasterRef",
    "RayPatternRef",
    "Requirement",
    "RewardTermSpec",
    "RigidObjectRef",
    "RobotSpec",
    "SceneSpec",
    "SimSpec",
    "SubTerrainSpec",
    "SymmetricAugmentationSpec",
    "TaskSpec",
    "TermSpec",
    "TerrainGeneratorSpec",
    "TerrainSpec",
    "VirtualObstacleRef",
    "VolumePointsRef",
]
