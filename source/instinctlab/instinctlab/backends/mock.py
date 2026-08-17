"""Deterministic backend used for contract and manager tests."""

from __future__ import annotations

import torch
from typing import Any

from instinctlab.sim.backend import (
    BackendMetadata,
    MassProperties,
    MaterialProperties,
    RuntimeRequirements,
    SensorReadPhase,
)
from instinctlab.sim.capabilities import Capability, CapabilitySet
from instinctlab.sim.control import ControlMode, JointControlTarget
from instinctlab.sim.scene import ArticulationView, SceneSpec, SceneView, SimulationSpec
from instinctlab.sim.state import ArticulationState, ContactState


class MockSimulatorBackend:
    """Small tensor-only simulator with the real backend contract."""

    capabilities = CapabilitySet.of(Capability)
    metadata = BackendMetadata(
        name="mock",
        version="1",
        engine_version="tensor",
        control_semantics="native_implicit_v1",
        contact_force_semantics="net_resultant_v1",
        joint_acc_source="fd_v1",
    )

    def __init__(self, *, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.scene: SceneView
        self.num_envs = 0
        self.sim_dt = 0.0
        self._scene_spec: SceneSpec | None = None
        self._control_target: JointControlTarget | None = None
        self._step_count = 0
        self.material_properties: MaterialProperties | None = None
        self.mass_properties: MassProperties | None = None
        self._body_mass: torch.Tensor | None = None
        self._body_inertia: torch.Tensor | None = None
        self._body_com: torch.Tensor | None = None

    def initialize(
        self,
        scene_spec: SceneSpec,
        simulation_spec: SimulationSpec,
        requirements: RuntimeRequirements,
    ) -> None:
        scene_spec.validate()
        simulation_spec.validate()
        self.capabilities.require(requirements.capabilities, context="mock runtime")
        self._scene_spec = scene_spec
        self.num_envs = scene_spec.num_envs
        self.sim_dt = simulation_spec.sim_dt
        robot = scene_spec.robot
        entity_name = scene_spec.primary_entity
        state = ArticulationState.allocate(
            num_envs=self.num_envs,
            num_joints=len(robot.joint_names),
            num_bodies=len(robot.body_names),
            device=self.device,
        )
        tensors = robot.materialize(device=self.device)
        state.default_joint_pos[:] = tensors["default_pos"]
        state.joint_pos[:] = tensors["default_pos"]
        state.joint_velocity_limits[:] = tensors["velocity_limit"]
        state.joint_effort_limits[:] = tensors["effort_limit"]
        state.root_pos_w[:] = torch.tensor(robot.default_root_pos, device=self.device)
        state.root_quat_w[:] = torch.tensor(robot.default_root_quat_wxyz, device=self.device)
        state.soft_joint_pos_limits[..., 0] = -torch.pi
        state.soft_joint_pos_limits[..., 1] = torch.pi
        articulation = ArticulationView(
            name=entity_name,
            joint_names=robot.joint_names,
            body_names=robot.body_names,
            data=state,
        )
        env_origins = torch.zeros((self.num_envs, 3), device=self.device)
        columns = max(1, int(self.num_envs**0.5))
        ids = torch.arange(self.num_envs, device=self.device)
        env_origins[:, 0] = torch.remainder(ids, columns) * scene_spec.env_spacing
        env_origins[:, 1] = torch.div(ids, columns, rounding_mode="floor") * scene_spec.env_spacing
        sensors = {
            spec.name: ContactState.allocate(
                num_envs=self.num_envs,
                body_names=spec.body_names,
                history_length=spec.history_length,
                device=self.device,
            )
            for spec in scene_spec.contact_sensors
        }
        self.scene = SceneView(
            env_origins=env_origins,
            articulations={entity_name: articulation},
            sensors=sensors,
        )
        num_bodies = len(robot.body_names)
        self._body_mass = torch.full((self.num_envs, num_bodies), 10.0, device=self.device)
        self._body_inertia = torch.ones((self.num_envs, num_bodies, 3), device=self.device)
        self._body_com = torch.zeros((self.num_envs, num_bodies, 3), device=self.device)
        self.synchronize(SensorReadPhase.POST_RESET)

    def _primary(self) -> ArticulationView:
        assert self._scene_spec is not None
        return self.scene.articulations[self._scene_spec.primary_entity]

    def reset(self, env_ids: torch.Tensor) -> None:
        state = self._primary().data
        state.joint_pos[env_ids] = state.default_joint_pos[env_ids]
        state.joint_vel[env_ids] = 0.0
        state.joint_acc[env_ids] = 0.0
        state.applied_joint_effort[env_ids] = 0.0
        self.scene.reset(env_ids)

    def write_root_state(self, entity_name: str, state_wxyz: torch.Tensor, env_ids: torch.Tensor) -> None:
        if tuple(state_wxyz.shape) != (env_ids.numel(), 13):
            raise ValueError("root state must have shape [len(env_ids), 13]")
        state = self.scene.articulations[entity_name].data
        state.root_pos_w[env_ids] = state_wxyz[:, :3]
        state.root_quat_w[env_ids] = state_wxyz[:, 3:7]
        state.root_lin_vel_w[env_ids] = state_wxyz[:, 7:10]
        state.root_ang_vel_w[env_ids] = state_wxyz[:, 10:13]

    def write_joint_state(
        self,
        entity_name: str,
        position: torch.Tensor,
        velocity: torch.Tensor,
        env_ids: torch.Tensor,
        joint_ids: torch.Tensor | None = None,
    ) -> None:
        state = self.scene.articulations[entity_name].data
        if joint_ids is None:
            state.joint_pos[env_ids] = position
            state.joint_vel[env_ids] = velocity
        else:
            row_ids = env_ids[:, None]
            state.joint_pos[row_ids, joint_ids] = position
            state.joint_vel[row_ids, joint_ids] = velocity

    def set_joint_control_target(
        self,
        entity_name: str,
        target: JointControlTarget,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        del entity_name, env_ids
        target.validate(
            num_envs=self.num_envs,
            num_joints=self._primary().data.num_joints,
        )
        self._control_target = target

    def set_external_wrench(
        self,
        entity_name: str,
        body_ids: torch.Tensor,
        force_w: torch.Tensor,
        torque_w: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        del entity_name, body_ids, torque_w
        self._primary().data.root_lin_vel_w[env_ids] += force_w.sum(dim=1) * self.sim_dt

    def material_shape_counts(self, entity_name: str, body_ids: torch.Tensor) -> torch.Tensor:
        del entity_name
        if self._scene_spec is None:
            raise RuntimeError("mock backend is not initialized")
        frames = frozenset(self._scene_spec.robot.frame_names)
        counts = [0 if self._scene_spec.robot.body_names[int(index)] in frames else 1 for index in body_ids.tolist()]
        return torch.tensor(counts, device=body_ids.device, dtype=torch.int64)

    def set_body_material(self, values: MaterialProperties) -> None:
        self.material_properties = values

    def set_body_mass_properties(self, values: MassProperties) -> None:
        self.mass_properties = values
        if self._body_mass is None or self._body_inertia is None or self._body_com is None:
            return
        rows = values.env_ids[:, None]
        cols = values.body_ids
        self._body_mass[rows, cols] = values.mass
        if values.inertia.ndim == 3:
            self._body_inertia[rows, cols] = values.inertia
        else:
            self._body_inertia[rows, cols] = torch.diagonal(values.inertia, dim1=-2, dim2=-1)
        self._body_com[rows, cols] = values.center_of_mass

    def get_body_mass_properties(
        self,
        entity_name: str,
        env_ids: torch.Tensor,
        body_ids: torch.Tensor,
    ) -> MassProperties:
        del entity_name
        if self._body_mass is None or self._body_inertia is None or self._body_com is None:
            raise RuntimeError("mock backend is not initialized")
        return MassProperties(
            entity_name=self._scene_spec.primary_entity if self._scene_spec is not None else "robot",
            body_ids=body_ids,
            env_ids=env_ids,
            mass=self._body_mass[env_ids][:, body_ids].clone(),
            inertia=self._body_inertia[env_ids][:, body_ids].clone(),
            center_of_mass=self._body_com[env_ids][:, body_ids].clone(),
        )

    def step(self) -> None:
        state = self._primary().data
        previous_velocity = state.joint_vel.clone()
        if self._control_target is not None:
            target = self._control_target
            joint_ids = target.joint_ids
            if joint_ids is None:
                joint_ids = torch.arange(state.num_joints, device=self.device)
            if target.mode is ControlMode.POSITION:
                velocity_target = 0.0 if target.velocity is None else target.velocity
                properties = self._scene_spec.robot.materialize(device=self.device)  # type: ignore[union-attr]
                error = target.value - state.joint_pos[:, joint_ids]
                effort = properties["stiffness"][joint_ids] * error + properties["damping"][joint_ids] * (
                    velocity_target - state.joint_vel[:, joint_ids]
                )
                limit = properties["effort_limit"][joint_ids]
                effort = effort.clamp(-limit, limit)
            elif target.mode is ControlMode.VELOCITY:
                effort = target.value - state.joint_vel[:, joint_ids]
            else:
                effort = target.value
            state.applied_joint_effort[:, joint_ids] = effort
            state.joint_vel[:, joint_ids] += effort * self.sim_dt
            state.joint_pos[:, joint_ids] += state.joint_vel[:, joint_ids] * self.sim_dt
        state.update_joint_acceleration(previous_velocity, sim_dt=self.sim_dt)
        for sensor in self.scene.sensors.values():
            if sensor.history_length:
                sensor.net_forces_w_history[:, 1:] = sensor.net_forces_w_history[:, :-1].clone()
                sensor.net_forces_w_history[:, 0] = sensor.net_forces_w
                sensor.contact_active_history[:, 1:] = sensor.contact_active_history[:, :-1].clone()
                sensor.contact_active_history[:, 0] = sensor.contact_active
        self._step_count += 1

    def synchronize(self, phase: SensorReadPhase) -> None:
        del phase
        state = self._primary().data
        state.body_pos_w[:] = state.root_pos_w[:, None, :]
        state.body_quat_w[:] = state.root_quat_w[:, None, :]
        state.body_lin_vel_w[:] = state.root_lin_vel_w[:, None, :]
        state.body_ang_vel_w[:] = state.root_ang_vel_w[:, None, :]
        if self._scene_spec is not None:
            specs = {spec.name: spec for spec in self._scene_spec.contact_sensors}
            for name, sensor in self.scene.sensors.items():
                sensor.update_active(specs[name].force_threshold)

    def render(self, mode: str) -> object | None:
        if mode in {"human", "none"}:
            return None
        if mode == "rgb_array":
            return torch.zeros((64, 64, 3), dtype=torch.uint8).numpy()
        raise ValueError(f"unsupported mock render mode: {mode}")

    def close(self) -> None:
        self._control_target = None


class MockBackendProvider:
    @staticmethod
    def add_cli_args(parser: Any) -> None:
        del parser

    @staticmethod
    def bootstrap(args: Any) -> object | None:
        del args
        return None

    @staticmethod
    def create(*, device: str, bootstrap_context: object | None = None) -> MockSimulatorBackend:
        del bootstrap_context
        return MockSimulatorBackend(device=device)


__all__ = ["MockBackendProvider", "MockSimulatorBackend"]
