"""Guard: a task must declare every backend capability it actually exercises.

``RuntimeRequirements.capabilities`` is hand-written per task. If a task starts
using a backend write/domain-randomization/contact feature without declaring it,
the mock backend (which supports everything) hides the mistake and it only
surfaces as a runtime failure on Isaac Sim or MJLab. This test records the
capabilities exercised during a real rollout and asserts they are a subset of
the declared requirements.
"""

from __future__ import annotations

import torch

from instinctlab.backends.mock import MockSimulatorBackend
from instinctlab.envs.unified_manager_based_rl_env import UnifiedManagerBasedRLEnv
from instinctlab.sim.backend import MassProperties, MaterialProperties
from instinctlab.sim.capabilities import Capability
from instinctlab.sim.control import ControlMode, JointControlTarget
from instinctlab.tasks.locomotion.unified_flat_env_cfg import locomotion_flat_env_cfg


class CapabilityRecordingBackend(MockSimulatorBackend):
    """Mock backend that records the capabilities each contract call implies."""

    def __init__(self) -> None:
        super().__init__(device="cpu")
        self.observed: set[Capability] = set()

    def initialize(self, scene_spec, simulation_spec, requirements) -> None:
        super().initialize(scene_spec, simulation_spec, requirements)
        for sensor in scene_spec.contact_sensors:
            self.observed.add(Capability.CONTACT_ACTIVE)
            self.observed.add(Capability.CONTACT_FORCE_VECTOR)
            if sensor.history_length:
                self.observed.add(Capability.CONTACT_HISTORY)
            if sensor.track_air_time:
                self.observed.add(Capability.CONTACT_AIR_TIME)

    def write_root_state(self, entity_name, state_wxyz, env_ids) -> None:
        self.observed.add(Capability.ROOT_STATE)
        if torch.any(state_wxyz[:, 7:13] != 0.0):
            self.observed.add(Capability.ROOT_VELOCITY_WRITE)
        super().write_root_state(entity_name, state_wxyz, env_ids)

    def write_joint_state(self, entity_name, position, velocity, env_ids, joint_ids=None) -> None:
        self.observed.add(Capability.JOINT_STATE)
        super().write_joint_state(entity_name, position, velocity, env_ids, joint_ids)

    def set_joint_control_target(self, entity_name, target: JointControlTarget, env_ids=None) -> None:
        if target.mode is ControlMode.POSITION:
            self.observed.add(Capability.IMPLICIT_POSITION_CONTROL)
        elif target.mode is ControlMode.EFFORT:
            self.observed.add(Capability.EFFORT_CONTROL)
        super().set_joint_control_target(entity_name, target, env_ids)

    def set_external_wrench(self, entity_name, body_ids, force_w, torque_w, env_ids) -> None:
        self.observed.add(Capability.EXTERNAL_WRENCH)
        super().set_external_wrench(entity_name, body_ids, force_w, torque_w, env_ids)

    def set_body_material(self, values: MaterialProperties) -> None:
        self.observed.add(Capability.DR_SLIDING_FRICTION)
        if values.restitution is not None:
            self.observed.add(Capability.DR_RESTITUTION)
        super().set_body_material(values)

    def set_body_mass_properties(self, values: MassProperties) -> None:
        self.observed.add(Capability.BODY_MASS_PROPERTIES)
        super().set_body_mass_properties(values)


def test_locomotion_flat_declares_every_exercised_capability() -> None:
    cfg = locomotion_flat_env_cfg(num_envs=4)
    backend = CapabilityRecordingBackend()
    env = UnifiedManagerBasedRLEnv(cfg, backend)
    try:
        zero_action = torch.zeros((env.num_envs, len(cfg.scene.robot.joint_names)), device=env.device)
        for _ in range(4):
            env.step(zero_action)
        # Force a reset to exercise state-writing reset events.
        env._reset_idx(torch.arange(env.num_envs, device=env.device, dtype=torch.int64))
    finally:
        env.close()

    declared = cfg.requirements.capabilities
    undeclared = backend.observed - declared
    assert not undeclared, (
        "task exercises capabilities that are not declared in RuntimeRequirements: "
        f"{sorted(c.value for c in undeclared)}"
    )


def test_locomotion_flat_requirements_supported_by_mock() -> None:
    cfg = locomotion_flat_env_cfg(num_envs=4)
    # The mock backend advertises every capability, so the declared set must
    # be a valid subset (this also exercises RuntimeRequirements construction).
    MockSimulatorBackend().capabilities.require(cfg.requirements.capabilities, context="capability contract test")
