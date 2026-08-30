"""Engine-agnostic declaration layer.

Nothing here may import a physics engine, directly or transitively. A task is declared once in
these types and compiled to a native environment by the backend for whichever engine is running,
so anything that reaches for an engine at this level has already lost the property that makes the
declaration portable. Engine isolation is an explicit package boundary.

:class:`~instinctlab_engine.spec.task.TaskSpec` is the entry point: a whole task, produced by a frontend
that read some project's native definition and consumed by the backend for the engine in use.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "Requirement": ("capability", "Requirement"),
    "UNIVERSAL_KINDS": ("entity", "UNIVERSAL_KINDS"),
    "EntityRef": ("entity", "EntityRef"),
    "freeze_task_spec": ("freeze", "freeze_task_spec"),
    "ActionTermSpec": ("mdp", "ActionTermSpec"),
    "CommandTermSpec": ("mdp", "CommandTermSpec"),
    "CurriculumTermSpec": ("mdp", "CurriculumTermSpec"),
    "DoneTermSpec": ("mdp", "DoneTermSpec"),
    "EventTermSpec": ("mdp", "EventTermSpec"),
    "MdpSpec": ("mdp", "MdpSpec"),
    "NoiseSpec": ("mdp", "NoiseSpec"),
    "ObsGroupSpec": ("mdp", "ObsGroupSpec"),
    "ObsTermSpec": ("mdp", "ObsTermSpec"),
    "RewardTermSpec": ("mdp", "RewardTermSpec"),
    "TermSpec": ("mdp", "TermSpec"),
    "portability_report": ("portability", "portability_report"),
    "RigidObjectRef": ("rigid_object", "RigidObjectRef"),
    "BackendAsset": ("robot", "BackendAsset"),
    "JointProperties": ("robot", "JointProperties"),
    "RobotSpec": ("robot", "RobotSpec"),
    "ContactSensorRef": ("sensor", "ContactSensorRef"),
    "Grid3dPointsRef": ("sensor", "Grid3dPointsRef"),
    "MotionReferenceRef": ("sensor", "MotionReferenceRef"),
    "NativeSensorRef": ("sensor", "NativeSensorRef"),
    "RayCasterRef": ("sensor", "RayCasterRef"),
    "RayPatternRef": ("sensor", "RayPatternRef"),
    "SymmetricAugmentationSpec": ("sensor", "SymmetricAugmentationSpec"),
    "VirtualObstacleRef": ("sensor", "VirtualObstacleRef"),
    "VolumePointsRef": ("sensor", "VolumePointsRef"),
    "AgentSpec": ("task", "AgentSpec"),
    "SceneSpec": ("task", "SceneSpec"),
    "SimSpec": ("task", "SimSpec"),
    "SubTerrainSpec": ("task", "SubTerrainSpec"),
    "TaskSpec": ("task", "TaskSpec"),
    "TerrainGeneratorSpec": ("task", "TerrainGeneratorSpec"),
    "TerrainSpec": ("task", "TerrainSpec"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value

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
    "NativeSensorRef",
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
    "freeze_task_spec",
    "portability_report",
]
