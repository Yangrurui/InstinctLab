"""Isaac Sim adapter for the engine-neutral simulator contract.

Isaac Lab and Isaac Sim imports intentionally live inside methods.  Importing
this module is therefore safe before :class:`isaaclab.app.AppLauncher` starts
the Kit application.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Mapping
from typing import Any

import torch

from instinctlab.sim.backend import (
    BackendMetadata,
    CanonicalIndexMap,
    MassProperties,
    MaterialProperties,
    RuntimeRequirements,
    SensorReadPhase,
)
from instinctlab.sim.capabilities import Capability, CapabilitySet
from instinctlab.sim.control import ControlMode, JointControlTarget
from instinctlab.sim.scene import ArticulationView, SceneSpec, SceneView, SimulationSpec
from instinctlab.sim.state import ArticulationState, ContactState


_BASE_CAPABILITIES = frozenset(
    {
        Capability.BATCHED_SIMULATION,
        Capability.PLANE_TERRAIN,
        Capability.ROOT_STATE,
        Capability.JOINT_STATE,
        Capability.BODY_STATE,
        Capability.IMPLICIT_POSITION_CONTROL,
        Capability.EFFORT_CONTROL,
        Capability.CONTACT_ACTIVE,
        Capability.CONTACT_HISTORY,
        Capability.CONTACT_AIR_TIME,
        Capability.CONTACT_FORCE_VECTOR,
        Capability.DR_SLIDING_FRICTION,
        Capability.DR_RESTITUTION,
        Capability.BODY_MASS_PROPERTIES,
        Capability.EXTERNAL_WRENCH,
        Capability.ROOT_VELOCITY_WRITE,
    }
)
_RESERVED_SCENE_NAMES = frozenset(
    {
        "terrain",
        "robot",
        "num_envs",
        "env_spacing",
        "lazy_sensor_update",
        "replicate_physics",
        "filter_collisions",
        "clone_in_fabric",
    }
)


class IsaacSimBackend:
    """Isaac Lab direct-workflow implementation of ``SimulatorBackend``."""

    def __init__(self, *, device: str, bootstrap_context: object) -> None:
        if bootstrap_context is None or not hasattr(bootstrap_context, "app"):
            raise RuntimeError(
                "IsaacSimBackend requires the AppLauncher returned by "
                "IsaacSimBackendProvider.bootstrap(); Isaac imports are only legal after bootstrap"
            )

        self.device = torch.device(device)
        self._launcher = bootstrap_context
        capabilities = set(_BASE_CAPABILITIES)
        if self.device.type == "cuda":
            capabilities.add(Capability.GPU_SIMULATION)
        if (
            not bool(getattr(bootstrap_context, "_headless", False))
            or int(getattr(bootstrap_context, "_livestream", 0)) in {1, 2}
        ):
            capabilities.add(Capability.HUMAN_VIEWER)
        self.capabilities = CapabilitySet.of(capabilities)
        self.metadata = BackendMetadata(
            name="isaacsim",
            version=self._adapter_version(),
            engine_version="uninitialized",
            control_semantics="native_implicit_v1",
            contact_force_semantics="physx_net_normal_resultant_v1",
            physics={
                "canonical_order": "dfs_v1",
                "quaternion_order": "wxyz",
                "rigid_body_frame": "link",
                "friction_mapping": "static_equals_dynamic_v1",
                "mass_property_semantics": "absolute_mass_inertia_com_v1",
            },
        )

        self.num_envs = 0
        self.sim_dt = 0.0
        self.scene: SceneView
        self._sim: Any | None = None
        self._native_scene: Any | None = None
        self._robot: Any | None = None
        self._scene_spec: SceneSpec | None = None
        self._joint_map: CanonicalIndexMap | None = None
        self._body_map: CanonicalIndexMap | None = None
        self._contact_maps: dict[str, CanonicalIndexMap] = {}
        self._shape_counts_by_native_body: tuple[int, ...] = ()
        self._joint_properties: dict[str, torch.Tensor] = {}
        self._closed = False

    @staticmethod
    def _adapter_version() -> str:
        try:
            return importlib.metadata.version("instinctlab")
        except importlib.metadata.PackageNotFoundError:
            return "0+local"

    def initialize(
        self,
        scene_spec: SceneSpec,
        simulation_spec: SimulationSpec,
        requirements: RuntimeRequirements,
    ) -> None:
        if self._sim is not None:
            raise RuntimeError("IsaacSimBackend is already initialized")
        scene_spec.validate()
        simulation_spec.validate()
        if scene_spec.terrain.terrain_type != "plane":
            raise ValueError(
                f"IsaacSimBackend currently supports only plane terrain, got {scene_spec.terrain.terrain_type!r}"
            )
        if scene_spec.terrain.sliding_friction < 0.0:
            raise ValueError("plane sliding friction must be non-negative")
        if not 0.0 <= scene_spec.terrain.restitution <= 1.0:
            raise ValueError("plane restitution must be within [0, 1]")
        for sensor in scene_spec.contact_sensors:
            if sensor.entity_name != "robot":
                raise ValueError(
                    f"Isaac Sim contact sensor {sensor.name!r} targets {sensor.entity_name!r}; "
                    "the SceneSpec robot entity is named 'robot'"
                )
            if sensor.name in _RESERVED_SCENE_NAMES:
                raise ValueError(
                    f"Isaac Sim contact sensor name {sensor.name!r} conflicts with a native scene field"
                )
        self.capabilities.require(requirements.capabilities, context="Isaac Sim runtime")

        # These imports must remain after AppLauncher construction.
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg, AssetBaseCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sensors import ContactSensorCfg

        sim_cfg = sim_utils.SimulationCfg(
            device=str(self.device),
            dt=simulation_spec.sim_dt,
            render_interval=simulation_spec.decimation,
            gravity=simulation_spec.gravity,
        )
        self._apply_engine_options(sim_cfg, simulation_spec.engine_options)
        self._sim = sim_utils.SimulationContext(sim_cfg)

        native_scene_cfg = InteractiveSceneCfg(
            num_envs=scene_spec.num_envs,
            env_spacing=scene_spec.env_spacing,
            lazy_sensor_update=False,
            replicate_physics=True,
            filter_collisions=True,
        )
        material_cfg = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=scene_spec.terrain.sliding_friction,
            dynamic_friction=scene_spec.terrain.sliding_friction,
            restitution=scene_spec.terrain.restitution,
        )
        native_scene_cfg.terrain = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(physics_material=material_cfg),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, scene_spec.terrain.height)),
            collision_group=-1,
        )
        native_scene_cfg.robot = self._make_robot_cfg(
            scene_spec,
            sim_utils=sim_utils,
            articulation_cfg_type=ArticulationCfg,
            actuator_cfg_type=ImplicitActuatorCfg,
        )
        for sensor_spec in scene_spec.contact_sensors:
            setattr(
                native_scene_cfg,
                sensor_spec.name,
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/.*",
                    update_period=simulation_spec.sim_dt,
                    history_length=sensor_spec.history_length,
                    track_air_time=sensor_spec.track_air_time,
                    force_threshold=sensor_spec.force_threshold,
                ),
            )

        self._native_scene = InteractiveScene(native_scene_cfg)
        self._sim.reset()
        self._native_scene.update(simulation_spec.sim_dt)

        self._scene_spec = scene_spec
        self.num_envs = scene_spec.num_envs
        self.sim_dt = simulation_spec.sim_dt
        self._robot = self._native_scene.articulations["robot"]
        self._build_runtime_views(scene_spec)
        self.capabilities.require(requirements.capabilities, context="Isaac Sim runtime")
        self._update_metadata(simulation_spec)
        self.synchronize(SensorReadPhase.POST_RESET)

    def _make_robot_cfg(
        self,
        scene_spec: SceneSpec,
        *,
        sim_utils: Any,
        articulation_cfg_type: Any,
        actuator_cfg_type: Any,
    ) -> Any:
        robot = scene_spec.robot
        asset = robot.asset_for("isaacsim")
        properties = robot.materialize(device="cpu")
        by_name = {
            field: {
                name: float(properties[field][index])
                for index, name in enumerate(robot.joint_names)
            }
            for field in ("stiffness", "damping", "armature", "effort_limit", "velocity_limit")
        }
        default_joint_pos = {
            item.name: float(item.default_pos)
            for item in robot.joint_properties
        }
        return articulation_cfg_type(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UrdfFileCfg(
                asset_path=asset.path,
                fix_base=False,
                merge_fixed_joints=False,
                replace_cylinders_with_capsules=True,
                self_collision=True,
                activate_contact_sensors=True,
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                        stiffness=None,
                        damping=None,
                    )
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=4,
                ),
            ),
            init_state=articulation_cfg_type.InitialStateCfg(
                pos=robot.default_root_pos,
                rot=robot.default_root_quat_wxyz,
                joint_pos=default_joint_pos,
                joint_vel={".*": 0.0},
            ),
            soft_joint_pos_limit_factor=robot.soft_joint_pos_limit_factor,
            actuators={
                "canonical": actuator_cfg_type(
                    joint_names_expr=list(robot.joint_names),
                    stiffness=by_name["stiffness"],
                    damping=by_name["damping"],
                    armature=by_name["armature"],
                    effort_limit_sim=by_name["effort_limit"],
                    velocity_limit_sim=by_name["velocity_limit"],
                )
            },
        )

    @classmethod
    def _apply_engine_options(cls, target: Any, options: Mapping[str, Any], prefix: str = "") -> None:
        for name, value in options.items():
            qualified = f"{prefix}.{name}" if prefix else name
            if not hasattr(target, name):
                raise ValueError(f"unknown Isaac Sim engine option: {qualified}")
            current = getattr(target, name)
            if isinstance(value, Mapping):
                cls._apply_engine_options(current, value, qualified)
            else:
                setattr(target, name, value)

    def _build_runtime_views(self, scene_spec: SceneSpec) -> None:
        assert self._robot is not None
        assert self._native_scene is not None
        native_joint_names = tuple(self._robot.joint_names)
        native_body_names = tuple(self._robot.body_names)
        if not native_body_names or native_body_names[0] != scene_spec.robot.root_body:
            native_root = native_body_names[0] if native_body_names else None
            raise ValueError(
                f"Isaac articulation root body {native_root!r} does not match "
                f"canonical root {scene_spec.robot.root_body!r}"
            )
        extra_joints = tuple(name for name in native_joint_names if name not in scene_spec.robot.joint_names)
        if extra_joints:
            raise ValueError(f"Isaac articulation contains non-canonical controlled joints: {extra_joints}")

        self._joint_map = CanonicalIndexMap.build(
            scene_spec.robot.joint_names,
            native_joint_names,
            device=self.device,
        )
        self._body_map = CanonicalIndexMap.build(
            scene_spec.robot.body_names,
            native_body_names,
            device=self.device,
        )
        self._joint_properties = scene_spec.robot.materialize(device=self.device)

        state = ArticulationState.allocate(
            num_envs=self.num_envs,
            num_joints=len(scene_spec.robot.joint_names),
            num_bodies=len(scene_spec.robot.body_names),
            device=self.device,
        )
        native_data = self._robot.data
        state.default_joint_pos.copy_(self._joint_map.to_canonical(native_data.default_joint_pos, dim=1))
        state.soft_joint_pos_limits.copy_(
            self._joint_map.to_canonical(native_data.soft_joint_pos_limits, dim=1)
        )
        state.joint_velocity_limits.copy_(
            self._joint_map.to_canonical(native_data.joint_vel_limits, dim=1)
        )
        state.joint_effort_limits.copy_(
            self._joint_map.to_canonical(native_data.joint_effort_limits, dim=1)
        )
        articulation = ArticulationView(
            name="robot",
            joint_names=scene_spec.robot.joint_names,
            body_names=scene_spec.robot.body_names,
            data=state,
        )

        sensors: dict[str, ContactState] = {}
        for sensor_spec in scene_spec.contact_sensors:
            native_sensor = self._native_scene.sensors[sensor_spec.name]
            contact_map = CanonicalIndexMap.build(
                sensor_spec.body_names,
                tuple(native_sensor.body_names),
                device=self.device,
            )
            self._contact_maps[sensor_spec.name] = contact_map
            sensors[sensor_spec.name] = ContactState.allocate(
                num_envs=self.num_envs,
                body_names=sensor_spec.body_names,
                history_length=sensor_spec.history_length,
                device=self.device,
            )

        self.scene = SceneView(
            env_origins=self._native_scene.env_origins,
            articulations={"robot": articulation},
            sensors=sensors,
        )
        self._shape_counts_by_native_body = self._query_shape_counts()
        material_width = int(self._robot.root_physx_view.get_material_properties().shape[-1])
        if material_width < 3:
            values = set(self.capabilities.values)
            values.discard(Capability.DR_RESTITUTION)
            self.capabilities = CapabilitySet.of(values)

    def _query_shape_counts(self) -> tuple[int, ...]:
        assert self._robot is not None
        counts: list[int] = []
        for link_path in self._robot.root_physx_view.link_paths[0]:
            link_view = self._robot._physics_sim_view.create_rigid_body_view(link_path)
            counts.append(int(link_view.max_shapes))
        expected = int(self._robot.root_physx_view.max_shapes)
        if sum(counts) != expected:
            raise RuntimeError(
                "failed to map Isaac collision shapes to bodies: "
                f"resolved {sum(counts)} shapes, expected {expected}"
            )
        return tuple(counts)

    def _update_metadata(self, simulation_spec: SimulationSpec) -> None:
        assert self._sim is not None
        engine_version = ".".join(str(value) for value in self._sim.get_version())
        physics = dict(self.metadata.physics)
        physics.update(
            {
                "gravity": simulation_spec.gravity,
                "sim_dt": simulation_spec.sim_dt,
                "solver": "physx",
                "friction_combine_mode": "multiply",
                "restitution_combine_mode": "multiply",
                "joint_effort_observation": "isaaclab_implicit_pd_estimate_clipped_v1",
            }
        )
        self.metadata = BackendMetadata(
            name="isaacsim",
            version=self.metadata.version,
            engine_version=engine_version,
            control_semantics=self.metadata.control_semantics,
            contact_force_semantics=self.metadata.contact_force_semantics,
            physics=physics,
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        self._require_initialized()
        env_ids = self._validate_ids("env_ids", env_ids, self.num_envs)
        self._native_scene.reset(env_ids)
        self.scene.reset(env_ids)
        native = self._robot.data
        root_state = native.default_root_state[env_ids].clone()
        root_state[:, :3] += self._native_scene.env_origins[env_ids]
        self._robot.write_root_link_state_to_sim(root_state, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(
            native.default_joint_pos[env_ids],
            native.default_joint_vel[env_ids],
            env_ids=env_ids,
        )
        state = self.scene.articulations["robot"].data
        state.joint_acc[env_ids] = 0.0
        self.set_joint_control_target(
            "robot",
            JointControlTarget(
                mode=ControlMode.POSITION,
                value=state.default_joint_pos,
                velocity=torch.zeros_like(state.default_joint_pos),
            ),
            env_ids=env_ids,
        )

    def write_root_state(self, entity_name: str, state_wxyz: torch.Tensor, env_ids: torch.Tensor) -> None:
        robot = self._entity(entity_name)
        env_ids = self._validate_ids("env_ids", env_ids, self.num_envs)
        expected = (env_ids.numel(), 13)
        if tuple(state_wxyz.shape) != expected:
            raise ValueError(f"root state has shape {tuple(state_wxyz.shape)}, expected {expected}")
        self._validate_float_tensor("root state", state_wxyz)
        quat_norm = torch.linalg.vector_norm(state_wxyz[:, 3:7], dim=-1)
        if torch.any(torch.abs(quat_norm - 1.0) > 1.0e-4):
            raise ValueError("root state contains a non-unit WXYZ quaternion")
        robot.write_root_link_state_to_sim(state_wxyz, env_ids=env_ids)

    def write_joint_state(
        self,
        entity_name: str,
        position: torch.Tensor,
        velocity: torch.Tensor,
        env_ids: torch.Tensor,
        joint_ids: torch.Tensor | None = None,
    ) -> None:
        robot = self._entity(entity_name)
        env_ids = self._validate_ids("env_ids", env_ids, self.num_envs)
        canonical_ids = self._canonical_joint_ids(joint_ids)
        expected = (env_ids.numel(), canonical_ids.numel())
        if tuple(position.shape) != expected or tuple(velocity.shape) != expected:
            raise ValueError(
                "joint state position/velocity must both have shape "
                f"{expected}, received {tuple(position.shape)} and {tuple(velocity.shape)}"
            )
        self._validate_float_tensor("joint position", position)
        self._validate_float_tensor("joint velocity", velocity)
        native_ids = self._joint_map.native_ids(canonical_ids)
        robot.write_joint_state_to_sim(
            position,
            velocity,
            joint_ids=native_ids,
            env_ids=env_ids,
        )

    def set_joint_control_target(
        self,
        entity_name: str,
        target: JointControlTarget,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        robot = self._entity(entity_name)
        if env_ids is None:
            selected_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int64)
            native_env_ids = None
        else:
            selected_env_ids = self._validate_ids("env_ids", env_ids, self.num_envs)
            native_env_ids = selected_env_ids
        target.validate(
            num_envs=self.num_envs,
            num_joints=len(self._joint_map.canonical_names),
        )
        self._validate_float_tensor("control target", target.value)
        if target.velocity is not None:
            self._validate_float_tensor("control velocity target", target.velocity)
        if target.joint_ids is not None and target.joint_ids.device != self.device:
            raise ValueError("control target joint_ids must be on the backend device")
        canonical_ids = self._canonical_joint_ids(target.joint_ids)
        native_ids = self._joint_map.native_ids(canonical_ids)
        value = target.value[selected_env_ids]
        zeros = torch.zeros_like(value)

        if target.mode is ControlMode.POSITION:
            stiffness = self._joint_properties["stiffness"][canonical_ids]
            damping = self._joint_properties["damping"][canonical_ids]
            velocity = zeros if target.velocity is None else target.velocity[selected_env_ids]
            self._write_drive_gains(robot, stiffness, damping, native_ids, selected_env_ids, native_env_ids)
            robot.set_joint_position_target(value, joint_ids=native_ids, env_ids=native_env_ids)
            robot.set_joint_velocity_target(velocity, joint_ids=native_ids, env_ids=native_env_ids)
            robot.set_joint_effort_target(zeros, joint_ids=native_ids, env_ids=native_env_ids)
        elif target.mode is ControlMode.VELOCITY:
            damping = self._joint_properties["damping"][canonical_ids]
            self._write_drive_gains(robot, torch.zeros_like(damping), damping, native_ids, selected_env_ids, native_env_ids)
            current_position = robot.data.joint_pos[selected_env_ids[:, None], native_ids]
            robot.set_joint_position_target(current_position, joint_ids=native_ids, env_ids=native_env_ids)
            robot.set_joint_velocity_target(value, joint_ids=native_ids, env_ids=native_env_ids)
            robot.set_joint_effort_target(zeros, joint_ids=native_ids, env_ids=native_env_ids)
        elif target.mode is ControlMode.EFFORT:
            limits = self.scene.articulations[entity_name].data.joint_effort_limits[
                selected_env_ids[:, None], canonical_ids
            ]
            if torch.any(torch.abs(value) > limits + 1.0e-6):
                raise ValueError("effort target exceeds a canonical joint effort limit")
            zero_gain = torch.zeros(canonical_ids.numel(), device=self.device)
            self._write_drive_gains(robot, zero_gain, zero_gain, native_ids, selected_env_ids, native_env_ids)
            current_position = robot.data.joint_pos[selected_env_ids[:, None], native_ids]
            robot.set_joint_position_target(current_position, joint_ids=native_ids, env_ids=native_env_ids)
            robot.set_joint_velocity_target(zeros, joint_ids=native_ids, env_ids=native_env_ids)
            robot.set_joint_effort_target(value, joint_ids=native_ids, env_ids=native_env_ids)
        else:
            raise ValueError(f"unsupported control mode: {target.mode}")

    def _write_drive_gains(
        self,
        robot: Any,
        stiffness: torch.Tensor,
        damping: torch.Tensor,
        native_ids: torch.Tensor,
        selected_env_ids: torch.Tensor,
        native_env_ids: torch.Tensor | None,
    ) -> None:
        rows = int(selected_env_ids.numel())
        stiffness_batch = stiffness.expand(rows, -1)
        damping_batch = damping.expand(rows, -1)
        robot.write_joint_stiffness_to_sim(
            stiffness_batch,
            joint_ids=native_ids,
            env_ids=native_env_ids,
        )
        robot.write_joint_damping_to_sim(
            damping_batch,
            joint_ids=native_ids,
            env_ids=native_env_ids,
        )
        # Isaac Lab keeps a parallel estimate for implicit-actuator torque.
        actuator = robot.actuators["canonical"]
        actuator.stiffness[selected_env_ids[:, None], native_ids] = stiffness_batch
        actuator.damping[selected_env_ids[:, None], native_ids] = damping_batch

    def set_external_wrench(
        self,
        entity_name: str,
        body_ids: torch.Tensor,
        force_w: torch.Tensor,
        torque_w: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        robot = self._entity(entity_name)
        env_ids = self._validate_ids("env_ids", env_ids, self.num_envs)
        body_ids = self._validate_ids("body_ids", body_ids, len(self._body_map.canonical_names))
        expected = (env_ids.numel(), body_ids.numel(), 3)
        if tuple(force_w.shape) != expected or tuple(torque_w.shape) != expected:
            raise ValueError(
                f"external force and torque must both have shape {expected}, "
                f"received {tuple(force_w.shape)} and {tuple(torque_w.shape)}"
            )
        self._validate_float_tensor("external force", force_w)
        self._validate_float_tensor("external torque", torque_w)
        robot.set_external_force_and_torque(
            forces=force_w,
            torques=torque_w,
            body_ids=self._body_map.native_ids(body_ids),
            env_ids=env_ids,
            is_global=True,
        )

    def set_body_material(self, values: MaterialProperties) -> None:
        robot = self._entity(values.entity_name)
        env_ids = self._validate_ids("env_ids", values.env_ids, self.num_envs)
        body_ids = self._validate_ids("body_ids", values.body_ids, len(self._body_map.canonical_names))
        expected = (env_ids.numel(), body_ids.numel())
        self._validate_float_tensor("sliding friction", values.sliding_friction)
        if tuple(values.sliding_friction.shape) != expected:
            raise ValueError(
                f"sliding_friction has shape {tuple(values.sliding_friction.shape)}, expected {expected}"
            )
        if torch.any(values.sliding_friction < 0.0):
            raise ValueError("sliding friction must be non-negative")
        if values.restitution is not None:
            self._validate_float_tensor("restitution", values.restitution)
            if tuple(values.restitution.shape) != expected:
                raise ValueError(f"restitution has shape {tuple(values.restitution.shape)}, expected {expected}")
            if torch.any(values.restitution < 0.0) or torch.any(values.restitution > 1.0):
                raise ValueError("restitution must be within [0, 1]")

        materials = robot.root_physx_view.get_material_properties()
        if values.restitution is not None and materials.shape[-1] < 3:
            raise NotImplementedError("this Isaac Sim runtime cannot write per-shape restitution")
        cpu_env_ids = env_ids.cpu()
        native_body_ids = self._body_map.native_ids(body_ids).cpu()
        friction = values.sliding_friction.detach().cpu()
        restitution = None if values.restitution is None else values.restitution.detach().cpu()
        for column, native_body_id in enumerate(native_body_ids.tolist()):
            start = sum(self._shape_counts_by_native_body[:native_body_id])
            stop = start + self._shape_counts_by_native_body[native_body_id]
            if start == stop:
                # Fixed visual/sensor links can be part of the canonical body
                # order without owning a collision shape. Material assignment
                # is semantically a no-op for those links.
                continue
            materials[cpu_env_ids, start:stop, 0] = friction[:, column, None]
            materials[cpu_env_ids, start:stop, 1] = friction[:, column, None]
            if restitution is not None:
                materials[cpu_env_ids, start:stop, 2] = restitution[:, column, None]
        robot.root_physx_view.set_material_properties(materials, cpu_env_ids)

    def set_body_mass_properties(self, values: MassProperties) -> None:
        robot = self._entity(values.entity_name)
        env_ids = self._validate_ids("env_ids", values.env_ids, self.num_envs)
        body_ids = self._validate_ids("body_ids", values.body_ids, len(self._body_map.canonical_names))
        expected_prefix = (env_ids.numel(), body_ids.numel())
        for name, tensor in (
            ("mass", values.mass),
            ("inertia", values.inertia),
            ("center_of_mass", values.center_of_mass),
        ):
            self._validate_float_tensor(name, tensor)
        if tuple(values.mass.shape) != expected_prefix:
            raise ValueError(f"mass has shape {tuple(values.mass.shape)}, expected {expected_prefix}")
        if tuple(values.center_of_mass.shape) != (*expected_prefix, 3):
            raise ValueError(
                f"center_of_mass has shape {tuple(values.center_of_mass.shape)}, "
                f"expected {(*expected_prefix, 3)}"
            )
        if tuple(values.inertia.shape) == (*expected_prefix, 3):
            inertia_matrix = torch.diag_embed(values.inertia)
        elif tuple(values.inertia.shape) == (*expected_prefix, 3, 3):
            inertia_matrix = 0.5 * (values.inertia + values.inertia.transpose(-1, -2))
            if torch.any(torch.abs(values.inertia - inertia_matrix) > 1.0e-5):
                raise ValueError("inertia tensor must be symmetric")
        else:
            raise ValueError(
                f"inertia must have shape {(*expected_prefix, 3)} (principal moments) or "
                f"{(*expected_prefix, 3, 3)} (full tensor)"
            )
        if torch.any(values.mass <= 0.0) or torch.any(torch.linalg.eigvalsh(inertia_matrix) <= 0.0):
            raise ValueError("mass and principal inertia moments must be positive")

        cpu_env_ids = env_ids.cpu()
        native_body_ids = self._body_map.native_ids(body_ids).cpu()
        row_ids = cpu_env_ids[:, None]
        masses = robot.root_physx_view.get_masses()
        inertias = robot.root_physx_view.get_inertias()
        coms = robot.root_physx_view.get_coms()
        masses[row_ids, native_body_ids] = values.mass.detach().cpu()
        inertias[row_ids, native_body_ids] = inertia_matrix.reshape(*expected_prefix, 9).detach().cpu()
        coms[row_ids, native_body_ids, :3] = values.center_of_mass.detach().cpu()
        robot.root_physx_view.set_masses(masses, cpu_env_ids)
        robot.root_physx_view.set_inertias(inertias, cpu_env_ids)
        robot.root_physx_view.set_coms(coms, cpu_env_ids)

    def step(self) -> None:
        self._require_initialized()
        self._native_scene.write_data_to_sim()
        self._sim.step(render=False)
        self._native_scene.update(self.sim_dt)

    def synchronize(self, phase: SensorReadPhase) -> None:
        self._require_initialized()
        if not isinstance(phase, SensorReadPhase):
            raise ValueError(f"invalid sensor read phase: {phase!r}")
        if phase in {SensorReadPhase.POST_RESET, SensorReadPhase.POST_EVENT}:
            self._sim.forward()

        native = self._robot.data
        state = self.scene.articulations["robot"].data
        root = native.root_link_state_w
        body = self._body_map.to_canonical(native.body_link_state_w, dim=1)
        state.root_pos_w.copy_(root[:, :3])
        state.root_quat_w.copy_(root[:, 3:7])
        state.root_lin_vel_w.copy_(root[:, 7:10])
        state.root_ang_vel_w.copy_(root[:, 10:13])
        state.body_pos_w.copy_(body[..., :3])
        state.body_quat_w.copy_(body[..., 3:7])
        state.body_lin_vel_w.copy_(body[..., 7:10])
        state.body_ang_vel_w.copy_(body[..., 10:13])
        state.joint_pos.copy_(self._joint_map.to_canonical(native.joint_pos, dim=1))
        state.joint_vel.copy_(self._joint_map.to_canonical(native.joint_vel, dim=1))
        state.joint_acc.copy_(self._joint_map.to_canonical(native.joint_acc, dim=1))
        state.applied_joint_effort.copy_(
            self._joint_map.to_canonical(native.applied_torque, dim=1)
        )

        specs = {item.name: item for item in self._scene_spec.contact_sensors}
        for name, canonical in self.scene.sensors.items():
            sensor_spec = specs[name]
            native_sensor = self._native_scene.sensors[name]
            native_data = native_sensor.data
            contact_map = self._contact_maps[name]
            canonical.net_forces_w.copy_(
                contact_map.to_canonical(native_data.net_forces_w, dim=1)
            )
            if canonical.history_length:
                canonical.net_forces_w_history.copy_(
                    contact_map.to_canonical(
                        native_data.net_forces_w_history[:, : canonical.history_length],
                        dim=2,
                    )
                )
            if sensor_spec.track_air_time:
                canonical.current_air_time.copy_(
                    contact_map.to_canonical(native_data.current_air_time, dim=1)
                )
                canonical.current_contact_time.copy_(
                    contact_map.to_canonical(native_data.current_contact_time, dim=1)
                )
                canonical.last_air_time.copy_(
                    contact_map.to_canonical(native_data.last_air_time, dim=1)
                )
                canonical.last_contact_time.copy_(
                    contact_map.to_canonical(native_data.last_contact_time, dim=1)
                )
            canonical.update_active(sensor_spec.force_threshold)

    def render(self, mode: str) -> object | None:
        self._require_initialized()
        if mode == "none":
            return None
        if mode == "human":
            if not self.capabilities.supports(Capability.HUMAN_VIEWER):
                raise RuntimeError("human rendering is unavailable because AppLauncher is headless")
            self._sim.render()
            return None
        if mode == "rgb_array":
            raise NotImplementedError(
                "IsaacSimBackend does not create a camera and does not advertise RGB_ARRAY"
            )
        raise ValueError(f"unsupported Isaac Sim render mode: {mode!r}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._robot = None
        self._native_scene = None
        if self._sim is not None:
            self._sim.clear_all_callbacks()
            self._sim.clear_instance()
            self._sim = None
        app = getattr(self._launcher, "app", None)
        if app is not None:
            app.close()

    def _entity(self, entity_name: str) -> Any:
        self._require_initialized()
        if entity_name != "robot":
            raise KeyError(f"unknown Isaac Sim articulation {entity_name!r}; available: 'robot'")
        return self._robot

    def _require_initialized(self) -> None:
        if self._sim is None or self._native_scene is None or self._robot is None:
            raise RuntimeError("IsaacSimBackend is not initialized or has been closed")

    def _canonical_joint_ids(self, joint_ids: torch.Tensor | None) -> torch.Tensor:
        if joint_ids is None:
            return torch.arange(
                len(self._joint_map.canonical_names),
                device=self.device,
                dtype=torch.int64,
            )
        return self._validate_ids("joint_ids", joint_ids, len(self._joint_map.canonical_names))

    def _validate_ids(self, name: str, ids: torch.Tensor, upper_bound: int) -> torch.Tensor:
        if ids.dtype != torch.int64 or ids.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional int64 tensor")
        if ids.device != self.device:
            raise ValueError(f"{name} must be on backend device {self.device}, received {ids.device}")
        if torch.any(ids < 0) or torch.any(ids >= upper_bound):
            raise ValueError(f"{name} contains an index outside [0, {upper_bound})")
        return ids

    def _validate_float_tensor(self, name: str, value: torch.Tensor) -> None:
        if value.dtype != torch.float32:
            raise ValueError(f"{name} must use float32, received {value.dtype}")
        if value.device != self.device:
            raise ValueError(
                f"{name} must be on backend device {self.device}, received {value.device}"
            )


class IsaacSimBackendProvider:
    """Launch Isaac Sim and construct its backend without eager engine imports."""

    @staticmethod
    def add_cli_args(parser: Any) -> None:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)

    @staticmethod
    def bootstrap(args: Any) -> object:
        from isaaclab.app import AppLauncher

        return AppLauncher(args)

    @staticmethod
    def create(*, device: str, bootstrap_context: object | None = None) -> IsaacSimBackend:
        if bootstrap_context is None:
            raise RuntimeError(
                "Isaac Sim must be bootstrapped with IsaacSimBackendProvider.bootstrap() before create()"
            )
        return IsaacSimBackend(device=device, bootstrap_context=bootstrap_context)


__all__ = ["IsaacSimBackend", "IsaacSimBackendProvider"]
