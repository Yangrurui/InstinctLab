# Copyright (c) 2024, Instinct Lab.
# SPDX-License-Identifier: MIT

"""Engine-neutral simulation contract.

Legacy Isaac spawners are loaded lazily so importing ``instinctlab.sim`` does
not require Isaac Sim in an MJLab process.
"""

from .backend import (
    BACKENDS,
    BackendMetadata,
    CanonicalIndexMap,
    MassProperties,
    MaterialProperties,
    RuntimeRequirements,
    SensorReadPhase,
    SimulatorBackend,
)
from .capabilities import Capability, CapabilitySet
from .control import ControlMode, ControlSemantics, JointControlTarget
from .robot_spec import RobotSpec, make_g1_29dof_robot_spec
from .scene import ContactSensorSpec, SceneSpec, SceneView, SimulationSpec, TerrainSpec
from .schema import CheckpointManifest, EnvSchema, locomotion_flat_schema
from .state import ArticulationState, ContactState

_LEGACY_SPAWNER_EXPORTS = {"MeshFileCfg", "spawn_from_mesh"}


def __getattr__(name: str):
    if name in _LEGACY_SPAWNER_EXPORTS:
        from . import spawners

        return getattr(spawners, name)
    raise AttributeError(name)


__all__ = [
    "BACKENDS",
    "ArticulationState",
    "BackendMetadata",
    "CanonicalIndexMap",
    "Capability",
    "CapabilitySet",
    "CheckpointManifest",
    "ContactSensorSpec",
    "ContactState",
    "ControlMode",
    "ControlSemantics",
    "EnvSchema",
    "JointControlTarget",
    "MassProperties",
    "MaterialProperties",
    "RobotSpec",
    "RuntimeRequirements",
    "SceneSpec",
    "SceneView",
    "SensorReadPhase",
    "SimulationSpec",
    "SimulatorBackend",
    "TerrainSpec",
    "locomotion_flat_schema",
    "make_g1_29dof_robot_spec",
]
