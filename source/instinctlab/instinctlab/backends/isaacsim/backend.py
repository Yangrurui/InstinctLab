"""Isaac Sim adapter for the engine-neutral simulator contract.

Isaac Lab and Isaac Sim imports intentionally live inside methods.  Importing
this module is therefore safe before :class:`isaaclab.app.AppLauncher` starts
the Kit application.
"""

from __future__ import annotations

import importlib.metadata
import re
import torch
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from instinctlab.sim.backend import (
    MATERIAL_LAYOUTS,
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
_SCENE_FLAG_KEYS = (
    "lazy_sensor_update",
    "replicate_physics",
    "filter_collisions",
    "clone_in_fabric",
)


def _contact_prim_path(
    body_names: tuple[str, ...],
    *,
    aliases: Mapping[str, str],
    prim_path: str,
) -> str:
    native_names = tuple(aliases.get(name, name) for name in body_names)
    leaf_pattern = "|".join(re.escape(name) for name in native_names)
    return f"{prim_path}/({leaf_pattern})"


def _typed_cfg(cfg_type: Any, value: Any) -> Any:
    if value is None or not isinstance(value, Mapping):
        return value
    return cfg_type(**value)


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
        if not bool(getattr(bootstrap_context, "_headless", False)) or int(
            getattr(bootstrap_context, "_livestream", 0)
        ) in {1, 2}:
            capabilities.add(Capability.HUMAN_VIEWER)
        self.capabilities = CapabilitySet.of(capabilities)
        self.metadata = BackendMetadata(
            name="isaacsim",
            version=self._adapter_version(),
            engine_version="uninitialized",
            control_semantics="native_implicit_v1",
            contact_force_semantics="physx_net_normal_resultant_v1",
            joint_acc_source="isaaclab_lazy_fd_v1",
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
        self._entity_name = "robot"
        self._scene_spec: SceneSpec | None = None
        self._joint_map: CanonicalIndexMap | None = None
        self._body_map: CanonicalIndexMap | None = None
        self._contact_maps: dict[str, CanonicalIndexMap] = {}
        self._shape_counts_by_native_body: tuple[int, ...] = ()
        self._joint_properties: dict[str, torch.Tensor] = {}
        self._all_env_ids: torch.Tensor | None = None
        self._all_joint_ids: torch.Tensor | None = None
        self._global_control_mode: ControlMode | None = None
        self._position_velocity_is_zero: bool | None = None
        self._last_position_target: torch.Tensor | None = None
        self._last_position_target_version = -1
        self._last_position_velocity: torch.Tensor | None = None
        self._last_position_velocity_version = -1
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
        scene_spec.robot.verify_assets("isaacsim")
        if scene_spec.terrain.terrain_type != "plane":
            raise ValueError(
                f"IsaacSimBackend currently supports only plane terrain, got {scene_spec.terrain.terrain_type!r}"
            )
        if scene_spec.terrain.sliding_friction < 0.0:
            raise ValueError("plane sliding friction must be non-negative")
        if not 0.0 <= scene_spec.terrain.restitution <= 1.0:
            raise ValueError("plane restitution must be within [0, 1]")
        for sensor in scene_spec.contact_sensors:
            if sensor.name in _RESERVED_SCENE_NAMES:
                raise ValueError(f"Isaac Sim contact sensor name {sensor.name!r} conflicts with a native scene field")

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
        self._apply_engine_options(
            sim_cfg,
            simulation_spec.engine_options_for("isaacsim"),
        )
        self._sim = sim_utils.SimulationContext(sim_cfg)

        isaac_options = scene_spec.backend_options_for("isaacsim")
        scene_flags = {
            key: isaac_options["scene"][key]
            for key in _SCENE_FLAG_KEYS
            if isinstance(isaac_options.get("scene"), Mapping) and key in isaac_options["scene"]
        }
        native_scene_cfg = InteractiveSceneCfg(
            num_envs=scene_spec.num_envs,
            env_spacing=scene_spec.env_spacing,
            **scene_flags,
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
        setattr(
            native_scene_cfg,
            scene_spec.primary_entity,
            self._make_robot_cfg(
                scene_spec,
                sim_utils=sim_utils,
                articulation_cfg_type=ArticulationCfg,
                actuator_cfg_type=ImplicitActuatorCfg,
            ),
        )
        asset = scene_spec.robot.asset_for("isaacsim")
        prim_path = str(asset.import_options.get("prim_path", "{ENV_REGEX_NS}/Robot"))
        for sensor_spec in scene_spec.contact_sensors:
            setattr(
                native_scene_cfg,
                sensor_spec.name,
                ContactSensorCfg(
                    prim_path=_contact_prim_path(
                        sensor_spec.body_names,
                        aliases=asset.contact_body_aliases,
                        prim_path=prim_path,
                    ),
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
        self._entity_name = scene_spec.primary_entity
        self.num_envs = scene_spec.num_envs
        self.sim_dt = simulation_spec.sim_dt
        self._robot = self._native_scene.articulations[self._entity_name]
        # The articulation is configured with canonical position-control gains.
        # Keep this mode cached so static gains are not rewritten every substep.
        self._global_control_mode = ControlMode.POSITION
        self._position_velocity_is_zero = True
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
            field: {name: float(properties[field][index]) for index, name in enumerate(robot.joint_names)}
            for field in ("stiffness", "damping", "armature", "effort_limit", "velocity_limit")
        }
        default_joint_pos = {item.name: float(item.default_pos) for item in robot.joint_properties}
        import_options = dict(asset.import_options)
        prim_path = str(import_options.pop("prim_path", "{ENV_REGEX_NS}/Robot"))
        spawn_profile = dict(scene_spec.backend_options_for("isaacsim").get("robot_spawn", {}))
        rigid_props = spawn_profile.pop("rigid_props", None)
        articulation_props = spawn_profile.pop("articulation_props", None)
        spawn_kwargs: dict[str, Any] = {
            "asset_path": asset.path,
            "activate_contact_sensors": bool(scene_spec.contact_sensors),
            **import_options,
            **spawn_profile,
            "joint_drive": sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=None,
                    damping=None,
                )
            ),
        }
        if rigid_props is not None:
            spawn_kwargs["rigid_props"] = _typed_cfg(sim_utils.RigidBodyPropertiesCfg, rigid_props)
        if articulation_props is not None:
            spawn_kwargs["articulation_props"] = _typed_cfg(
                sim_utils.ArticulationRootPropertiesCfg,
                articulation_props,
            )
        return articulation_cfg_type(
            prim_path=prim_path,
            spawn=sim_utils.UrdfFileCfg(**spawn_kwargs),
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
        self._all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int64)
        self._all_joint_ids = torch.arange(
            len(scene_spec.robot.joint_names),
            device=self.device,
            dtype=torch.int64,
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
        state.soft_joint_pos_limits.copy_(self._joint_map.to_canonical(native_data.soft_joint_pos_limits, dim=1))
        state.joint_velocity_limits.copy_(self._joint_map.to_canonical(native_data.joint_vel_limits, dim=1))
        state.joint_effort_limits.copy_(self._joint_map.to_canonical(native_data.joint_effort_limits, dim=1))
        articulation = ArticulationView(
            name=scene_spec.primary_entity,
            joint_names=scene_spec.robot.joint_names,
            body_names=scene_spec.robot.body_names,
            data=state,
        )

        asset = scene_spec.robot.asset_for("isaacsim")
        sensors: dict[str, ContactState] = {}
        for sensor_spec in scene_spec.contact_sensors:
            native_sensor = self._native_scene.sensors[sensor_spec.name]
            contact_map = CanonicalIndexMap.build(
                asset.resolve_contact_body_names(sensor_spec.body_names),
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
            articulations={scene_spec.primary_entity: articulation},
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
                f"failed to map Isaac collision shapes to bodies: resolved {sum(counts)} shapes, expected {expected}"
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
        self.metadata = replace(
            self.metadata,
            engine_version=engine_version,
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
        state = self.scene.articulations[self._entity_name].data
        state.joint_acc[env_ids] = 0.0
        self.set_joint_control_target(
            self._entity_name,
            JointControlTarget(
                mode=ControlMode.POSITION,
                value=state.default_joint_pos,
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
            assert self._all_env_ids is not None
            selected_env_ids = self._all_env_ids
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
        mode_changed = target.mode is not self._global_control_mode
        full_control_write = env_ids is None and target.joint_ids is None

        if target.mode is ControlMode.POSITION:
            if mode_changed:
                stiffness = self._joint_properties["stiffness"][canonical_ids]
                damping = self._joint_properties["damping"][canonical_ids]
                self._write_drive_gains(robot, stiffness, damping, native_ids, selected_env_ids, native_env_ids)
                robot.set_joint_effort_target(
                    torch.zeros_like(value),
                    joint_ids=native_ids,
                    env_ids=native_env_ids,
                )
            position_changed = (
                mode_changed
                or not full_control_write
                or target.value is not self._last_position_target
                or target.value._version != self._last_position_target_version
            )
            if position_changed:
                robot.set_joint_position_target(
                    value,
                    joint_ids=native_ids,
                    env_ids=native_env_ids,
                )
            if target.velocity is not None:
                velocity_changed = (
                    mode_changed
                    or not full_control_write
                    or target.velocity is not self._last_position_velocity
                    or target.velocity._version != self._last_position_velocity_version
                )
                if velocity_changed:
                    robot.set_joint_velocity_target(
                        target.velocity[selected_env_ids],
                        joint_ids=native_ids,
                        env_ids=native_env_ids,
                    )
                self._position_velocity_is_zero = False if full_control_write else None
            elif mode_changed or self._position_velocity_is_zero is not True:
                robot.set_joint_velocity_target(
                    torch.zeros_like(value),
                    joint_ids=native_ids,
                    env_ids=native_env_ids,
                )
                self._position_velocity_is_zero = True if full_control_write else None
            if full_control_write:
                self._last_position_target = target.value
                self._last_position_target_version = target.value._version
                self._last_position_velocity = target.velocity
                self._last_position_velocity_version = -1 if target.velocity is None else target.velocity._version
            else:
                self._clear_position_target_cache()
        elif target.mode is ControlMode.VELOCITY:
            self._clear_position_target_cache()
            damping = self._joint_properties["damping"][canonical_ids]
            if mode_changed:
                self._write_drive_gains(
                    robot,
                    torch.zeros_like(damping),
                    damping,
                    native_ids,
                    selected_env_ids,
                    native_env_ids,
                )
            current_position = robot.data.joint_pos[selected_env_ids[:, None], native_ids]
            robot.set_joint_position_target(current_position, joint_ids=native_ids, env_ids=native_env_ids)
            robot.set_joint_velocity_target(value, joint_ids=native_ids, env_ids=native_env_ids)
            self._position_velocity_is_zero = False if full_control_write else None
            if mode_changed:
                robot.set_joint_effort_target(
                    torch.zeros_like(value),
                    joint_ids=native_ids,
                    env_ids=native_env_ids,
                )
        elif target.mode is ControlMode.EFFORT:
            self._clear_position_target_cache()
            limits = self.scene.articulations[entity_name].data.joint_effort_limits[
                selected_env_ids[:, None], canonical_ids
            ]
            if torch.any(torch.abs(value) > limits + 1.0e-6):
                raise ValueError("effort target exceeds a canonical joint effort limit")
            if mode_changed:
                zero_gain = torch.zeros(canonical_ids.numel(), device=self.device)
                self._write_drive_gains(
                    robot,
                    zero_gain,
                    zero_gain,
                    native_ids,
                    selected_env_ids,
                    native_env_ids,
                )
                current_position = robot.data.joint_pos[selected_env_ids[:, None], native_ids]
                robot.set_joint_position_target(current_position, joint_ids=native_ids, env_ids=native_env_ids)
                robot.set_joint_velocity_target(
                    torch.zeros_like(value),
                    joint_ids=native_ids,
                    env_ids=native_env_ids,
                )
                self._position_velocity_is_zero = True if full_control_write else None
            robot.set_joint_effort_target(value, joint_ids=native_ids, env_ids=native_env_ids)
        else:
            raise ValueError(f"unsupported control mode: {target.mode}")

        if mode_changed:
            # A full articulation write establishes one global mode. A partial
            # mode switch makes the cache conservative until the next full write.
            self._global_control_mode = target.mode if full_control_write else None

    def _clear_position_target_cache(self) -> None:
        self._last_position_target = None
        self._last_position_target_version = -1
        self._last_position_velocity = None
        self._last_position_velocity_version = -1

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

    def material_shape_counts(self, entity_name: str, body_ids: torch.Tensor) -> torch.Tensor:
        self._entity(entity_name)
        body_ids = self._validate_ids("body_ids", body_ids, len(self._body_map.canonical_names))
        native_ids = self._body_map.native_ids(body_ids)
        counts = torch.tensor(self._shape_counts_by_native_body, device=body_ids.device, dtype=torch.int64)
        return counts[native_ids]

    def set_body_material(self, values: MaterialProperties) -> None:
        robot = self._entity(values.entity_name)
        env_ids = self._validate_ids("env_ids", values.env_ids, self.num_envs)
        body_ids = self._validate_ids("body_ids", values.body_ids, len(self._body_map.canonical_names))
        if values.layout not in MATERIAL_LAYOUTS:
            raise ValueError(f"unsupported material layout: {values.layout!r}")
        n_columns = (
            int(body_ids.numel())
            if values.layout == "body"
            else int(self.material_shape_counts(values.entity_name, body_ids).sum().item())
        )
        expected = (env_ids.numel(), n_columns)
        self._validate_float_tensor("sliding friction", values.sliding_friction)
        if tuple(values.sliding_friction.shape) != expected:
            raise ValueError(f"sliding_friction has shape {tuple(values.sliding_friction.shape)}, expected {expected}")
        if torch.any(values.sliding_friction < 0.0):
            raise ValueError("sliding friction must be non-negative")
        if values.dynamic_friction is not None:
            self._validate_float_tensor("dynamic friction", values.dynamic_friction)
            if tuple(values.dynamic_friction.shape) != expected:
                raise ValueError(
                    f"dynamic_friction has shape {tuple(values.dynamic_friction.shape)}, expected {expected}"
                )
            if torch.any(values.dynamic_friction < 0.0):
                raise ValueError("dynamic friction must be non-negative")
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
        dynamic = friction if values.dynamic_friction is None else values.dynamic_friction.detach().cpu()
        restitution = None if values.restitution is None else values.restitution.detach().cpu()
        shape_offset = 0
        for column, native_body_id in enumerate(native_body_ids.tolist()):
            start = sum(self._shape_counts_by_native_body[:native_body_id])
            stop = start + self._shape_counts_by_native_body[native_body_id]
            n_shapes = stop - start
            if n_shapes == 0:
                canonical_name = self._body_map.canonical_names[int(body_ids[column])]
                raise ValueError(
                    f"Isaac Sim body {canonical_name!r} has no collision shapes; "
                    "material writes must target RobotSpec.material_body_names"
                )
            if values.layout == "body":
                static_vals = friction[:, column, None]
                dynamic_vals = dynamic[:, column, None]
                restitution_vals = None if restitution is None else restitution[:, column, None]
            else:
                sl = slice(shape_offset, shape_offset + n_shapes)
                static_vals = friction[:, sl]
                dynamic_vals = dynamic[:, sl]
                restitution_vals = None if restitution is None else restitution[:, sl]
                shape_offset += n_shapes
            materials[cpu_env_ids, start:stop, 0] = static_vals
            materials[cpu_env_ids, start:stop, 1] = dynamic_vals
            if restitution_vals is not None:
                materials[cpu_env_ids, start:stop, 2] = restitution_vals
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
                f"center_of_mass has shape {tuple(values.center_of_mass.shape)}, expected {(*expected_prefix, 3)}"
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

    def get_body_mass_properties(
        self,
        entity_name: str,
        env_ids: torch.Tensor,
        body_ids: torch.Tensor,
    ) -> MassProperties:
        robot = self._entity(entity_name)
        env_ids = self._validate_ids("env_ids", env_ids, self.num_envs)
        body_ids = self._validate_ids("body_ids", body_ids, len(self._body_map.canonical_names))
        cpu_env_ids = env_ids.cpu()
        native_body_ids = self._body_map.native_ids(body_ids).cpu()
        row_ids = cpu_env_ids[:, None]
        prefix = (env_ids.numel(), body_ids.numel())
        masses = robot.root_physx_view.get_masses()[row_ids, native_body_ids]
        inertias = robot.root_physx_view.get_inertias()[row_ids, native_body_ids]
        coms = robot.root_physx_view.get_coms()[row_ids, native_body_ids, :3]
        return MassProperties(
            entity_name=entity_name,
            body_ids=body_ids,
            env_ids=env_ids,
            mass=masses.to(device=self.device, dtype=torch.float32),
            inertia=inertias.to(device=self.device, dtype=torch.float32).reshape(*prefix, 3, 3),
            center_of_mass=coms.to(device=self.device, dtype=torch.float32),
        )

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
        state = self.scene.articulations[self._entity_name].data
        root_pose = native.root_link_pose_w
        root_vel = native.root_link_vel_w
        body_pose = native.body_link_pose_w
        body_vel = native.body_link_vel_w
        state.root_pos_w.copy_(root_pose[:, :3])
        state.root_quat_w.copy_(root_pose[:, 3:7])
        state.root_lin_vel_w.copy_(root_vel[:, :3])
        state.root_ang_vel_w.copy_(root_vel[:, 3:6])
        self._body_map.copy_to_canonical(body_pose[..., :3], state.body_pos_w, dim=1)
        self._body_map.copy_to_canonical(body_pose[..., 3:7], state.body_quat_w, dim=1)
        self._body_map.copy_to_canonical(body_vel[..., :3], state.body_lin_vel_w, dim=1)
        self._body_map.copy_to_canonical(body_vel[..., 3:6], state.body_ang_vel_w, dim=1)
        self._joint_map.copy_to_canonical(native.joint_pos, state.joint_pos, dim=1)
        self._joint_map.copy_to_canonical(native.joint_vel, state.joint_vel, dim=1)
        self._joint_map.copy_to_canonical(native.joint_acc, state.joint_acc, dim=1)
        self._joint_map.copy_to_canonical(native.applied_torque, state.applied_joint_effort, dim=1)

        specs = {item.name: item for item in self._scene_spec.contact_sensors}
        for name, canonical in self.scene.sensors.items():
            sensor_spec = specs[name]
            native_sensor = self._native_scene.sensors[name]
            native_data = native_sensor.data
            contact_map = self._contact_maps[name]
            contact_map.copy_to_canonical(native_data.net_forces_w, canonical.net_forces_w, dim=1)
            if canonical.history_length:
                contact_map.copy_to_canonical(
                    native_data.net_forces_w_history[:, : canonical.history_length],
                    canonical.net_forces_w_history,
                    dim=2,
                )
            if sensor_spec.track_air_time:
                contact_map.copy_to_canonical(native_data.current_air_time, canonical.current_air_time, dim=1)
                contact_map.copy_to_canonical(native_data.current_contact_time, canonical.current_contact_time, dim=1)
                contact_map.copy_to_canonical(native_data.last_air_time, canonical.last_air_time, dim=1)
                contact_map.copy_to_canonical(native_data.last_contact_time, canonical.last_contact_time, dim=1)
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
            raise NotImplementedError("IsaacSimBackend does not create a camera and does not advertise RGB_ARRAY")
        raise ValueError(f"unsupported Isaac Sim render mode: {mode!r}")

    def close(self, *, shutdown_app: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        self._robot = None
        self._native_scene = None
        if self._sim is not None:
            self._sim.clear_all_callbacks()
            self._sim.clear_instance()
            self._sim = None
        # Kit's SimulationApp.close() terminates the process. Live pytest
        # cells must pass shutdown_app=False so the runner can emit results.
        if shutdown_app:
            app = getattr(self._launcher, "app", None)
            if app is not None:
                app.close()

    def _entity(self, entity_name: str) -> Any:
        self._require_initialized()
        if entity_name != self._entity_name:
            raise KeyError(f"unknown Isaac Sim articulation {entity_name!r}; available: {self._entity_name!r}")
        return self._robot

    def _require_initialized(self) -> None:
        if self._sim is None or self._native_scene is None or self._robot is None:
            raise RuntimeError("IsaacSimBackend is not initialized or has been closed")

    def _canonical_joint_ids(self, joint_ids: torch.Tensor | None) -> torch.Tensor:
        if joint_ids is None:
            assert self._all_joint_ids is not None
            return self._all_joint_ids
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
            raise ValueError(f"{name} must be on backend device {self.device}, received {value.device}")


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
