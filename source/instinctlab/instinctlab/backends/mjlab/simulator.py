"""MJLab implementation of the backend-independent simulator contract."""

from __future__ import annotations

import importlib.metadata
import math
import torch
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import tomllib

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

_MASS_MODEL_FIELDS = (
    "body_mass",
    "body_ipos",
    "body_inertia",
    "body_iquat",
    # Derived fields written by MJWarp set_const.
    "body_subtreemass",
    "dof_invweight0",
    "body_invweight0",
    "tendon_length0",
    "tendon_invweight0",
)
_RANDOMIZATION_FIELD_ALIASES = {
    "sliding_friction": ("geom_friction",),
    "restitution": ("geom_solref",),
    "mass": ("body_mass",),
    "center_of_mass": ("body_ipos",),
    "inertia": ("body_inertia", "body_iquat"),
    # These randomizations write data/state buffers, not replicated model
    # fields, so they require no MJWarp model expansion.
    "root_pose": (),
    "root_velocity": (),
    "joint_state": (),
}


@dataclass
class _ContactBinding:
    native_sensor: Any
    index_map: CanonicalIndexMap
    force_threshold: float
    track_air_time: bool


def _strip_visual_meshes_xml(xml: str) -> str:
    """Remove mesh assets and mesh geoms from an MJCF document."""

    root = ET.fromstring(xml)
    for asset in root.findall("asset"):
        for mesh in tuple(asset.findall("mesh")):
            asset.remove(mesh)
    for parent in root.iter():
        for geom in tuple(parent.findall("geom")):
            if geom.get("type") == "mesh" or geom.get("mesh"):
                parent.remove(geom)
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
    return ET.tostring(root, encoding="unicode")


def _load_mjcf(path: Path, mujoco: Any, load_mode: str) -> Any:
    """Dispatch MJCF loading according to the asset's declared load mode."""

    if load_mode == "default":
        return mujoco.MjSpec.from_file(str(path))
    if load_mode == "strip_visual_meshes":
        try:
            return mujoco.MjSpec.from_file(str(path))
        except (ValueError, OSError):
            return mujoco.MjSpec.from_string(_strip_visual_meshes_xml(path.read_text()))
    raise ValueError(f"unsupported MJLab asset load_mode: {load_mode!r}")


def _expanded_randomization_fields(requirements: RuntimeRequirements) -> tuple[str, ...]:
    result: set[str] = set()
    for field_name in requirements.randomization_fields:
        result.update(_RANDOMIZATION_FIELD_ALIASES.get(field_name, (field_name,)))
    requested = requirements.capabilities | requirements.optional_capabilities
    if Capability.DR_SLIDING_FRICTION in requested:
        result.add("geom_friction")
    if Capability.DR_RESTITUTION in requested:
        result.add("geom_solref")
    if Capability.BODY_MASS_PROPERTIES in requested:
        result.update(_MASS_MODEL_FIELDS)
    return tuple(sorted(result))


def _solref_dampratio_from_restitution(restitution: torch.Tensor) -> torch.Tensor:
    """Map a PhysX-style coefficient of restitution onto MuJoCo ``solref[1]``.

    ``e = 0`` keeps the default critically-damped contact; ``e → 1`` removes
    damping so the constraint can ring. The time constant ``solref[0]`` is
    left unchanged by the caller.
    """
    return 1.0 - restitution


def _enable_effort_actuator(
    supports_effort_control: bool,
    requirements: RuntimeRequirements,
) -> bool:
    return supports_effort_control and Capability.EFFORT_CONTROL in requirements.capabilities


def _active_mjlab_version(mjlab: Any) -> str:
    """Prefer the active source tree version over an unrelated installed wheel."""

    module_file = getattr(mjlab, "__file__", None)
    if module_file is not None:
        project_file = Path(module_file).resolve().parents[2] / "pyproject.toml"
        if project_file.is_file():
            with project_file.open("rb") as stream:
                version = tomllib.load(stream).get("project", {}).get("version")
            if isinstance(version, str):
                return version
    try:
        return importlib.metadata.version("mjlab")
    except importlib.metadata.PackageNotFoundError:
        return str(getattr(mjlab, "__version__", "unknown"))


class MjlabBackend:
    """MJLab/MuJoCo-Warp adapter with canonical DFS/WXYZ tensor views."""

    capabilities = CapabilitySet.of(
        (
            Capability.BATCHED_SIMULATION,
            Capability.GPU_SIMULATION,
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
            Capability.BODY_MASS_PROPERTIES,
            Capability.EXTERNAL_WRENCH,
            Capability.ROOT_VELOCITY_WRITE,
            Capability.HUMAN_VIEWER,
            Capability.RGB_ARRAY,
        )
    )
    metadata = BackendMetadata(
        name="mjlab",
        version="1",
        engine_version="uninitialized",
        control_semantics="native_implicit_v1",
        contact_force_semantics="net_resultant_world_v1",
        physics={
            "canonical_order": "dfs_v1",
            "quaternion_order": "wxyz",
            "joint_acc_source": "fd_v1",
            "restitution": "solref_dampratio_v1",
        },
    )

    def __init__(self, *, device: str = "cuda:0") -> None:
        self.device = torch.device(device)
        self.num_envs = 0
        self.sim_dt = 0.0
        self.scene: SceneView

        self._scene_spec: SceneSpec | None = None
        self._simulation_spec: SimulationSpec | None = None
        self._mj_scene: Any = None
        self._sim: Any = None
        self._entity: Any = None
        self._entity_name = "robot"
        self._joint_map: CanonicalIndexMap | None = None
        self._body_map: CanonicalIndexMap | None = None
        self._contact_bindings: dict[str, _ContactBinding] = {}
        self._geoms_by_native_body: dict[int, torch.Tensor] = {}
        self._effort_mode_mask: torch.Tensor | None = None
        self._effort_mode_active = False
        self._supports_effort_control = True
        self._effort_actuator_enabled = False
        self._last_joint_acc_native: torch.Tensor | None = None
        self._previous_joint_velocity_native: torch.Tensor | None = None
        self._all_env_ids: torch.Tensor | None = None
        self._all_joint_ids: torch.Tensor | None = None
        self._last_control_mode: ControlMode | None = None
        self._last_control_value: torch.Tensor | None = None
        self._last_control_value_version = -1
        self._last_control_velocity: torch.Tensor | None = None
        self._last_control_velocity_version = -1
        self._offscreen_renderer: Any = None
        self._human_viewer: Any = None

    def initialize(
        self,
        scene_spec: SceneSpec,
        simulation_spec: SimulationSpec,
        requirements: RuntimeRequirements,
    ) -> None:
        scene_spec.validate()
        simulation_spec.validate()
        self.capabilities.require(requirements.capabilities, context="MJLab runtime")
        if scene_spec.terrain.terrain_type != "plane":
            raise NotImplementedError("MJLab adapter currently supports only plane terrain")
        if not math.isclose(scene_spec.terrain.restitution, 0.0, abs_tol=1.0e-8):
            raise NotImplementedError(
                "MJLab adapter does not expose coefficient-of-restitution semantics; "
                "non-zero terrain restitution is unsupported"
            )
        for sensor in scene_spec.contact_sensors:
            if sensor.entity_name != scene_spec.primary_entity:
                raise ValueError(
                    f"MJLab contact sensor {sensor.name!r} targets {sensor.entity_name!r}; "
                    f"the SceneSpec robot is named {scene_spec.primary_entity!r}"
                )

        # Engine imports are intentionally delayed until initialize().
        import mjlab
        import mujoco
        from mjlab.actuator import BuiltinMotorActuatorCfg

        try:
            from mjlab.actuator import BuiltinPdActuatorCfg as NativePdActuatorCfg
        except ImportError:
            # Compatibility with MJLab revisions before BuiltinPdActuatorCfg
            # was renamed from its position-actuator implementation.
            from mjlab.actuator import BuiltinPositionActuatorCfg as NativePdActuatorCfg

            self._supports_effort_control = False
            capabilities = set(self.capabilities.values)
            capabilities.discard(Capability.EFFORT_CONTROL)
            self.capabilities = CapabilitySet.of(capabilities)
        self.capabilities.require(requirements.capabilities, context="MJLab runtime")
        self._effort_actuator_enabled = _enable_effort_actuator(
            self._supports_effort_control,
            requirements,
        )
        from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
        from mjlab.scene import Scene, SceneCfg
        from mjlab.sensor import ContactMatch, ContactSensorCfg
        from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
        from mjlab.terrains import TerrainEntityCfg

        robot = scene_spec.robot
        asset = robot.asset_for("mjlab")
        asset.verify()
        asset_path = Path(asset.path)
        if not asset_path.is_file():
            raise FileNotFoundError(f"MJLab asset does not exist: {asset_path}")

        def load_robot_spec() -> Any:
            return _load_mjcf(asset_path, mujoco, asset.load_mode)

        actuator_cfgs: list[Any] = []
        for properties in robot.joint_properties:
            actuator_cfgs.append(
                NativePdActuatorCfg(
                    target_names_expr=(properties.name,),
                    stiffness=properties.stiffness,
                    damping=properties.damping,
                    effort_limit=properties.effort_limit,
                    armature=properties.armature,
                )
            )
        if self._effort_actuator_enabled:
            # A parallel native motor channel makes direct effort mode available.
            # In effort mode step() continuously nulls the PD error before applying
            # this channel, and MuJoCo's joint actfrcrange clamps their total.
            for properties in robot.joint_properties:
                actuator_cfgs.append(
                    BuiltinMotorActuatorCfg(
                        target_names_expr=(properties.name,),
                        effort_limit=properties.effort_limit,
                    )
                )

        entity_cfg = EntityCfg(
            init_state=EntityCfg.InitialStateCfg(
                pos=robot.default_root_pos,
                rot=robot.default_root_quat_wxyz,
                joint_pos={properties.name: properties.default_pos for properties in robot.joint_properties},
                joint_vel={".*": 0.0},
            ),
            spec_fn=load_robot_spec,
            articulation=EntityArticulationInfoCfg(
                actuators=tuple(actuator_cfgs),
                soft_joint_pos_limit_factor=robot.soft_joint_pos_limit_factor,
            ),
        )

        entity_name = scene_spec.primary_entity
        native_sensor_cfgs = []
        for spec in scene_spec.contact_sensors:
            native_bodies = asset.resolve_contact_body_names(spec.body_names)
            native_sensor_cfgs.append(
                ContactSensorCfg(
                    name=spec.name,
                    primary=ContactMatch(
                        mode="body",
                        pattern=native_bodies,
                        entity=entity_name,
                    ),
                    fields=("found", "force"),
                    reduce="netforce",
                    num_slots=1,
                    track_air_time=spec.track_air_time,
                    history_length=spec.history_length,
                )
            )

        def configure_scene(spec: Any) -> None:
            terrain_body = spec.body("terrain")
            terrain_body.pos[2] = scene_spec.terrain.height
            terrain_geom = spec.geom("terrain")
            terrain_geom.friction[0] = scene_spec.terrain.sliding_friction

        native_scene_cfg = SceneCfg(
            num_envs=scene_spec.num_envs,
            env_spacing=scene_spec.env_spacing,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={entity_name: entity_cfg},
            sensors=tuple(native_sensor_cfgs),
            spec_fn=configure_scene,
        )
        mj_scene = Scene(native_scene_cfg, device=str(self.device))
        native_sim_cfg = self._make_simulation_cfg(simulation_spec, MujocoCfg=MujocoCfg, SimulationCfg=SimulationCfg)
        if hasattr(mj_scene, "collect_variant_info"):
            sim = Simulation(
                num_envs=scene_spec.num_envs,
                cfg=native_sim_cfg,
                spec=mj_scene.spec,
                variant_info=mj_scene.collect_variant_info(),
                device=str(self.device),
            )
            mj_scene.initialize(mj_model=sim.mj_model, model=sim.model, data=sim.data)
        else:
            # Compatibility with MJLab revisions whose Simulation accepted a
            # compiled MjModel instead of an MjSpec plus variant metadata.
            compiled_model = mj_scene.compile()
            sim = Simulation(
                num_envs=scene_spec.num_envs,
                cfg=native_sim_cfg,
                model=compiled_model,
                device=str(self.device),
            )
            mj_scene.initialize(compiled_model, sim.model, sim.data)
        if mj_scene.sensor_context is not None:
            sim.set_sensor_context(mj_scene.sensor_context)
        sim.expand_model_fields(_expanded_randomization_fields(requirements))

        self._scene_spec = scene_spec
        self._simulation_spec = simulation_spec
        self._mj_scene = mj_scene
        self._sim = sim
        self._entity_name = entity_name
        self._entity = mj_scene.entities[entity_name]
        self.num_envs = scene_spec.num_envs
        self.sim_dt = simulation_spec.sim_dt

        self._joint_map = CanonicalIndexMap.build(robot.joint_names, self._entity.joint_names, device=self.device)
        self._all_env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        self._all_joint_ids = torch.arange(len(robot.joint_names), dtype=torch.int64, device=self.device)
        self._body_map = CanonicalIndexMap.build(robot.body_names, self._entity.body_names, device=self.device)
        if self._entity.body_names[0] != robot.root_body:
            raise ValueError(
                f"MJCF root body {self._entity.body_names[0]!r} does not match canonical root {robot.root_body!r}"
            )

        self._build_geom_body_map()
        state = ArticulationState.allocate(
            num_envs=self.num_envs,
            num_joints=len(robot.joint_names),
            num_bodies=len(robot.body_names),
            device=self.device,
        )
        materialized = robot.materialize(device=self.device)
        state.default_joint_pos[:] = materialized["default_pos"]
        state.joint_velocity_limits[:] = materialized["velocity_limit"]
        state.joint_effort_limits[:] = materialized["effort_limit"]
        state.soft_joint_pos_limits.copy_(self._joint_map.to_canonical(self._entity.data.soft_joint_pos_limits, dim=1))

        sensors: dict[str, ContactState] = {}
        self._contact_bindings.clear()
        for spec in scene_spec.contact_sensors:
            native_sensor = mj_scene.sensors[spec.name]
            requested_native_names = asset.resolve_contact_body_names(spec.body_names)
            primary_names = getattr(native_sensor, "primary_names", None)
            if primary_names is None:
                primary_names = tuple(dict.fromkeys(slot.primary_name for slot in native_sensor._slots))
            sensor_map = CanonicalIndexMap.build(
                requested_native_names,
                tuple(primary_names),
                device=self.device,
            )
            sensors[spec.name] = ContactState.allocate(
                num_envs=self.num_envs,
                body_names=spec.body_names,
                history_length=spec.history_length,
                device=self.device,
            )
            self._contact_bindings[spec.name] = _ContactBinding(
                native_sensor=native_sensor,
                index_map=sensor_map,
                force_threshold=spec.force_threshold,
                track_air_time=spec.track_air_time,
            )

        articulation = ArticulationView(
            name=entity_name,
            joint_names=robot.joint_names,
            body_names=robot.body_names,
            data=state,
        )
        self.scene = SceneView(
            env_origins=mj_scene.env_origins,
            articulations={entity_name: articulation},
            sensors=sensors,
        )
        self._effort_mode_mask = torch.zeros(
            (self.num_envs, len(robot.joint_names)), dtype=torch.bool, device=self.device
        )
        self._last_joint_acc_native = torch.zeros_like(self._entity.data.joint_vel)
        self._previous_joint_velocity_native = torch.zeros_like(self._entity.data.joint_vel)

        self._write_default_state(self._all_env_ids)
        self._entity.set_joint_position_target(self._entity.data.default_joint_pos)
        self._entity.set_joint_velocity_target(torch.zeros_like(self._entity.data.default_joint_vel))
        self._entity.set_joint_effort_target(torch.zeros_like(self._entity.data.default_joint_vel))

        mjlab_version = _active_mjlab_version(mjlab)
        self.metadata = BackendMetadata(
            name="mjlab",
            version="1",
            engine_version=f"mjlab={mjlab_version}; mujoco={mujoco.__version__}",
            control_semantics="native_implicit_v1",
            contact_force_semantics="net_resultant_world_v1",
            physics={
                "integrator": native_sim_cfg.mujoco.integrator,
                "solver": native_sim_cfg.mujoco.solver,
                "iterations": native_sim_cfg.mujoco.iterations,
                "canonical_order": robot.schema_version,
                "quaternion_order": "wxyz",
                "joint_acc_source": "fd_v1",
                "velocity_limit_enforcement": "not_native_equivalent",
                "effort_control": (
                    "parallel_native_motor_v1"
                    if self._effort_actuator_enabled
                    else ("disabled_not_requested" if self._supports_effort_control else "unsupported_by_mjlab_version")
                ),
                "mass_properties": "full_inertia_tensor_v1",
                "restitution": "solref_dampratio_v1",
            },
        )
        self.synchronize(SensorReadPhase.POST_RESET)

    @staticmethod
    def _make_simulation_cfg(simulation_spec: SimulationSpec, *, MujocoCfg: Any, SimulationCfg: Any) -> Any:
        mujoco_fields = {item.name for item in fields(MujocoCfg)}
        simulation_fields = {item.name for item in fields(SimulationCfg)}
        mujoco_kwargs: dict[str, Any] = {
            "timestep": simulation_spec.sim_dt,
            "gravity": simulation_spec.gravity,
        }
        simulation_kwargs: dict[str, Any] = {}
        for name, value in simulation_spec.engine_options_for("mjlab").items():
            if name in {"timestep", "gravity"}:
                expected = mujoco_kwargs[name]
                if value != expected:
                    raise ValueError(f"engine option {name!r} conflicts with SimulationSpec: {value!r} != {expected!r}")
            elif name in mujoco_fields:
                mujoco_kwargs[name] = value
            elif name in simulation_fields and name != "mujoco":
                simulation_kwargs[name] = value
            else:
                raise ValueError(f"unsupported MJLab engine option: {name!r}")
        simulation_kwargs["mujoco"] = MujocoCfg(**mujoco_kwargs)
        return SimulationCfg(**simulation_kwargs)

    @property
    def native_sim(self) -> Any:
        """Return the underlying MJLab ``Simulation`` after ``initialize()``."""
        self._require_initialized()
        return self._sim

    def _require_initialized(self) -> None:
        if self._sim is None or self._entity is None:
            raise RuntimeError("MJLab backend is not initialized")

    def _validate_entity(self, entity_name: str) -> None:
        if entity_name != self._entity_name:
            raise KeyError(f"unknown MJLab articulation {entity_name!r}; available: {self._entity_name!r}")

    def _validate_env_ids(self, env_ids: torch.Tensor) -> None:
        if env_ids.dtype != torch.int64 or env_ids.ndim != 1:
            raise ValueError("env_ids must be a one-dimensional int64 tensor")
        if env_ids.device != self.device:
            raise ValueError(f"env_ids must be on {self.device}, received {env_ids.device}")
        if torch.any(env_ids < 0) or torch.any(env_ids >= self.num_envs):
            raise ValueError("env_ids contains an out-of-range environment index")

    def _validate_float_tensor(self, name: str, value: torch.Tensor) -> None:
        if value.dtype != torch.float32:
            raise ValueError(f"{name} must use float32, received {value.dtype}")
        if value.device != self.device:
            raise ValueError(f"{name} must be on {self.device}, received {value.device}")

    def _write_default_state(self, env_ids: torch.Tensor) -> None:
        self._entity.write_root_state_to_sim(self._entity.data.default_root_state[env_ids], env_ids=env_ids)
        self._entity.write_joint_state_to_sim(
            self._entity.data.default_joint_pos[env_ids],
            self._entity.data.default_joint_vel[env_ids],
            env_ids=env_ids,
        )

    def _build_geom_body_map(self) -> None:
        global_body_ids = self._entity.indexing.body_ids.to(torch.int64)
        global_geom_ids = self._entity.indexing.geom_ids.to(torch.int64)
        geom_body_ids = torch.as_tensor(self._sim.mj_model.geom_bodyid, dtype=torch.int64, device=self.device)
        self._geoms_by_native_body.clear()
        for local_body_id, global_body_id in enumerate(global_body_ids):
            mask = geom_body_ids[global_geom_ids] == global_body_id
            self._geoms_by_native_body[local_body_id] = global_geom_ids[mask]

    def reset(self, env_ids: torch.Tensor) -> None:
        self._require_initialized()
        self._validate_env_ids(env_ids)
        self._sim.reset(env_ids)
        self._mj_scene.reset(env_ids)
        self.scene.reset(env_ids)
        self._write_default_state(env_ids)
        native_default = self._entity.data.default_joint_pos[env_ids]
        native_zero = torch.zeros_like(native_default)
        self._entity.data.joint_pos_target[env_ids] = native_default
        self._entity.data.joint_vel_target[env_ids] = native_zero
        self._entity.data.joint_effort_target[env_ids] = native_zero
        self._effort_mode_mask[env_ids] = False
        if env_ids.numel() == self.num_envs:
            self._effort_mode_active = False
        self._last_joint_acc_native[env_ids] = 0.0
        self._invalidate_control_cache()

    def write_root_state(self, entity_name: str, state_wxyz: torch.Tensor, env_ids: torch.Tensor) -> None:
        self._require_initialized()
        self._validate_entity(entity_name)
        self._validate_env_ids(env_ids)
        self._validate_float_tensor("root state", state_wxyz)
        if tuple(state_wxyz.shape) != (env_ids.numel(), 13):
            raise ValueError("root state must have shape [len(env_ids), 13]")
        quat_norm = torch.linalg.vector_norm(state_wxyz[:, 3:7], dim=-1)
        if torch.any(torch.abs(quat_norm - 1.0) > 1.0e-4):
            raise ValueError("root state contains a non-unit WXYZ quaternion")
        self._entity.write_root_state_to_sim(state_wxyz, env_ids=env_ids)

    def write_joint_state(
        self,
        entity_name: str,
        position: torch.Tensor,
        velocity: torch.Tensor,
        env_ids: torch.Tensor,
        joint_ids: torch.Tensor | None = None,
    ) -> None:
        self._require_initialized()
        self._validate_entity(entity_name)
        self._validate_env_ids(env_ids)
        self._validate_float_tensor("joint position", position)
        self._validate_float_tensor("joint velocity", velocity)
        canonical_count = len(self._joint_map.canonical_names)
        if joint_ids is None:
            selected_count = canonical_count
            canonical_ids = torch.arange(canonical_count, device=self.device)
        else:
            if joint_ids.dtype != torch.int64 or joint_ids.ndim != 1:
                raise ValueError("joint_ids must be a one-dimensional int64 tensor")
            if joint_ids.device != self.device:
                raise ValueError("joint_ids must be on the backend device")
            if torch.any(joint_ids < 0) or torch.any(joint_ids >= canonical_count):
                raise ValueError("joint_ids contains an out-of-range canonical index")
            selected_count = joint_ids.numel()
            canonical_ids = joint_ids
        expected = (env_ids.numel(), selected_count)
        if tuple(position.shape) != expected or tuple(velocity.shape) != expected:
            raise ValueError(
                f"joint state tensors must have shape {expected}; received "
                f"{tuple(position.shape)} and {tuple(velocity.shape)}"
            )
        native_ids = self._joint_map.native_ids(canonical_ids)
        self._entity.write_joint_state_to_sim(position, velocity, joint_ids=native_ids, env_ids=env_ids)

    def set_joint_control_target(
        self,
        entity_name: str,
        target: JointControlTarget,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        self._require_initialized()
        self._validate_entity(entity_name)
        target.validate(
            num_envs=self.num_envs,
            num_joints=len(self._joint_map.canonical_names),
        )
        if target.value.device != self.device:
            raise ValueError("control target must be on the backend device")
        if target.velocity is not None and target.velocity.device != self.device:
            raise ValueError("velocity target must be on the backend device")
        if target.joint_ids is not None and target.joint_ids.device != self.device:
            raise ValueError("joint_ids must be on the backend device")
        full_control_write = env_ids is None and target.joint_ids is None
        if full_control_write and self._is_repeated_control_target(target):
            return
        if env_ids is None:
            assert self._all_env_ids is not None
            env_ids = self._all_env_ids
        else:
            self._validate_env_ids(env_ids)

        canonical_ids = target.joint_ids
        if canonical_ids is None:
            assert self._all_joint_ids is not None
            canonical_ids = self._all_joint_ids
        native_ids = self._joint_map.native_ids(canonical_ids)
        source = target.value if full_control_write else target.value[env_ids]
        if full_control_write:
            target_index = (slice(None), native_ids)
        else:
            env_grid, joint_grid = torch.meshgrid(env_ids, native_ids, indexing="ij")
            target_index = (env_grid, joint_grid)

        if target.mode is ControlMode.POSITION:
            velocity = (
                torch.zeros_like(source)
                if target.velocity is None
                else (target.velocity if full_control_write else target.velocity[env_ids])
            )
            self._entity.data.joint_pos_target[target_index] = source
            self._entity.data.joint_vel_target[target_index] = velocity
            self._entity.data.joint_effort_target[target_index] = 0.0
            self._effort_mode_mask[target_index] = False
            if full_control_write:
                self._effort_mode_active = False
        elif target.mode is ControlMode.EFFORT:
            if not self._effort_actuator_enabled:
                raise NotImplementedError(
                    "MJLab effort control was not enabled for this runtime; "
                    "declare Capability.EFFORT_CONTROL in RuntimeRequirements"
                )
            if full_control_write:
                limits = self.scene.articulations[entity_name].data.joint_effort_limits
            else:
                limits = self.scene.articulations[entity_name].data.joint_effort_limits[
                    env_grid, canonical_ids[None, :].expand_as(env_grid)
                ]
            if torch.any(torch.abs(source) > limits + 1.0e-6):
                raise ValueError("effort target exceeds a canonical joint effort limit")
            self._entity.data.joint_effort_target[target_index] = source
            self._effort_mode_mask[target_index] = True
            self._effort_mode_active = True
        else:
            raise NotImplementedError("MJLab adapter does not support velocity control")
        if full_control_write:
            self._cache_control_target(target)
        else:
            self._invalidate_control_cache()

    def _is_repeated_control_target(self, target: JointControlTarget) -> bool:
        velocity = target.velocity
        return (
            target.mode is self._last_control_mode
            and target.value is self._last_control_value
            and target.value._version == self._last_control_value_version
            and velocity is self._last_control_velocity
            and (velocity is None or velocity._version == self._last_control_velocity_version)
        )

    def _cache_control_target(self, target: JointControlTarget) -> None:
        self._last_control_mode = target.mode
        self._last_control_value = target.value
        self._last_control_value_version = target.value._version
        self._last_control_velocity = target.velocity
        self._last_control_velocity_version = -1 if target.velocity is None else target.velocity._version

    def _invalidate_control_cache(self) -> None:
        self._last_control_mode = None
        self._last_control_value = None
        self._last_control_value_version = -1
        self._last_control_velocity = None
        self._last_control_velocity_version = -1

    def set_external_wrench(
        self,
        entity_name: str,
        body_ids: torch.Tensor,
        force_w: torch.Tensor,
        torque_w: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        self._require_initialized()
        self._validate_entity(entity_name)
        self._validate_env_ids(env_ids)
        if body_ids.dtype != torch.int64 or body_ids.ndim != 1:
            raise ValueError("body_ids must be a one-dimensional int64 tensor")
        if body_ids.device != self.device:
            raise ValueError("body_ids must be on the backend device")
        if torch.any(body_ids < 0) or torch.any(body_ids >= len(self._body_map.canonical_names)):
            raise ValueError("body_ids contains an out-of-range canonical index")
        self._validate_float_tensor("external force", force_w)
        self._validate_float_tensor("external torque", torque_w)
        expected = (env_ids.numel(), body_ids.numel(), 3)
        if tuple(force_w.shape) != expected or tuple(torque_w.shape) != expected:
            raise ValueError(f"external wrench tensors must have shape {expected}")
        native_ids = self._body_map.native_ids(body_ids).tolist()
        self._entity.write_external_wrench_to_sim(force_w, torque_w, env_ids=env_ids, body_ids=native_ids)

    def set_body_material(self, values: MaterialProperties) -> None:
        self._require_initialized()
        self._validate_entity(values.entity_name)
        self._validate_env_ids(values.env_ids)
        self._validate_float_tensor("sliding friction", values.sliding_friction)
        expected = (values.env_ids.numel(), values.body_ids.numel())
        if tuple(values.sliding_friction.shape) != expected:
            raise ValueError(f"sliding_friction must have shape {expected}")
        if torch.any(values.sliding_friction < 0.0):
            raise ValueError("sliding friction must be non-negative")
        if values.dynamic_friction is not None:
            raise NotImplementedError(
                "MJLab adapter has a single slide-friction coefficient and cannot store dynamic friction"
            )
        if values.restitution is not None:
            self._validate_float_tensor("restitution", values.restitution)
            if tuple(values.restitution.shape) != expected:
                raise ValueError(f"restitution must have shape {expected}")
            if torch.any(values.restitution < 0.0) or torch.any(values.restitution > 1.0):
                raise ValueError("restitution must be within [0, 1]")
        required_fields = ["geom_friction"]
        if values.restitution is not None:
            required_fields.append("geom_solref")
        self._ensure_model_fields(required_fields)
        self._validate_body_ids(values.body_ids)

        for column in range(values.body_ids.numel()):
            native_local_id = int(self._body_map.native_ids(values.body_ids[column : column + 1])[0])
            geom_ids = self._geoms_by_native_body[native_local_id]
            if geom_ids.numel() == 0:
                canonical_name = self._body_map.canonical_names[int(values.body_ids[column])]
                raise ValueError(
                    f"MJLab body {canonical_name!r} has no collision geoms; "
                    "material writes must target RobotSpec.material_body_names"
                )
            env_grid, geom_grid = torch.meshgrid(values.env_ids, geom_ids, indexing="ij")
            friction = values.sliding_friction[:, column, None].expand(-1, geom_ids.numel())
            self._sim.model.geom_friction[env_grid, geom_grid, 0] = friction
            if values.restitution is not None:
                dampratio = _solref_dampratio_from_restitution(
                    values.restitution[:, column, None].expand(-1, geom_ids.numel())
                )
                self._sim.model.geom_solref[env_grid, geom_grid, 1] = dampratio

    def set_body_mass_properties(self, values: MassProperties) -> None:
        self._require_initialized()
        self._validate_entity(values.entity_name)
        self._validate_env_ids(values.env_ids)
        self._validate_body_ids(values.body_ids)
        for name, tensor in (
            ("mass", values.mass),
            ("inertia", values.inertia),
            ("center_of_mass", values.center_of_mass),
        ):
            self._validate_float_tensor(name, tensor)
        prefix = (values.env_ids.numel(), values.body_ids.numel())
        if tuple(values.mass.shape) != prefix:
            raise ValueError(f"mass must have shape {prefix}")
        if tuple(values.center_of_mass.shape) != prefix + (3,):
            raise ValueError(f"center_of_mass must have shape {prefix + (3,)}")
        if tuple(values.inertia.shape) == prefix + (3,):
            principal = values.inertia
            rotations = torch.eye(3, device=self.device).expand(prefix + (3, 3))
        elif tuple(values.inertia.shape) == prefix + (3, 3):
            symmetric = 0.5 * (values.inertia + values.inertia.transpose(-1, -2))
            if torch.any(torch.abs(values.inertia - symmetric) > 1.0e-5):
                raise ValueError("inertia tensor must be symmetric")
            principal, rotations = torch.linalg.eigh(symmetric)
            determinant = torch.linalg.det(rotations)
            if torch.any(determinant < 0.0):
                rotations = rotations.clone()
                rotations[determinant < 0.0, :, 2] *= -1.0
        else:
            raise ValueError(
                f"inertia must have shape {prefix + (3, 3)} (full tensor) or {prefix + (3,)} (principal moments)"
            )
        if torch.any(values.mass <= 0.0) or torch.any(principal <= 0.0):
            raise ValueError("mass and principal inertia moments must be positive")

        from mjlab.managers.event_manager import RecomputeLevel
        from mjlab.utils.lab_api.math import quat_from_matrix

        self._ensure_model_fields(_MASS_MODEL_FIELDS)
        native_local_ids = self._body_map.native_ids(values.body_ids)
        global_body_ids = self._entity.indexing.body_ids[native_local_ids].to(torch.int64)
        env_grid, body_grid = torch.meshgrid(values.env_ids, global_body_ids, indexing="ij")
        self._sim.model.body_mass[env_grid, body_grid] = values.mass
        self._sim.model.body_ipos[env_grid, body_grid] = values.center_of_mass
        self._sim.model.body_inertia[env_grid, body_grid] = principal
        self._sim.model.body_iquat[env_grid, body_grid] = quat_from_matrix(rotations)
        self._sim.recompute_constants(RecomputeLevel.set_const)

    def get_body_mass_properties(
        self,
        entity_name: str,
        env_ids: torch.Tensor,
        body_ids: torch.Tensor,
    ) -> MassProperties:
        self._require_initialized()
        self._validate_entity(entity_name)
        self._validate_env_ids(env_ids)
        self._validate_body_ids(body_ids)
        native_local_ids = self._body_map.native_ids(body_ids)
        global_body_ids = self._entity.indexing.body_ids[native_local_ids].to(torch.int64)
        env_grid, body_grid = torch.meshgrid(env_ids, global_body_ids, indexing="ij")
        return MassProperties(
            entity_name=entity_name,
            body_ids=body_ids,
            env_ids=env_ids,
            mass=self._sim.model.body_mass[env_grid, body_grid].to(dtype=torch.float32).clone(),
            inertia=self._sim.model.body_inertia[env_grid, body_grid].to(dtype=torch.float32).clone(),
            center_of_mass=self._sim.model.body_ipos[env_grid, body_grid].to(dtype=torch.float32).clone(),
        )

    def _validate_body_ids(self, body_ids: torch.Tensor) -> None:
        if body_ids.dtype != torch.int64 or body_ids.ndim != 1:
            raise ValueError("body_ids must be a one-dimensional int64 tensor")
        if body_ids.device != self.device:
            raise ValueError("body_ids must be on the backend device")
        if torch.any(body_ids < 0) or torch.any(body_ids >= len(self._body_map.canonical_names)):
            raise ValueError("body_ids contains an out-of-range canonical index")

    def _ensure_model_fields(self, required: Iterable[str]) -> None:
        if self.num_envs == 1:
            return
        missing = set(required).difference(self._sim.expanded_fields)
        if missing:
            raise RuntimeError(
                "MJLab per-environment write requires RuntimeRequirements."
                f"randomization_fields to include: {sorted(missing)}"
            )

    def step(self) -> None:
        self._require_initialized()
        if self._effort_mode_active:
            effort_mask = self._effort_mode_mask
            assert effort_mask is not None
            # Null the implicit PD channel at the beginning of every substep.
            current_pos = self._entity.data.joint_pos
            current_vel = self._entity.data.joint_vel
            self._entity.data.joint_pos_target[effort_mask] = current_pos[effort_mask]
            self._entity.data.joint_vel_target[effort_mask] = current_vel[effort_mask]

        previous_velocity = self._previous_joint_velocity_native
        assert previous_velocity is not None
        previous_velocity.copy_(self._entity.data.joint_vel)
        self._mj_scene.write_data_to_sim()
        self._sim.step()
        self._mj_scene.update(dt=self.sim_dt)
        current_velocity = self._entity.data.joint_vel
        self._last_joint_acc_native.copy_((current_velocity - previous_velocity) / self.sim_dt)
        self._advance_force_threshold_air_time()

    def synchronize(self, phase: SensorReadPhase) -> None:
        self._require_initialized()
        if not isinstance(phase, SensorReadPhase):
            raise ValueError(f"unknown sensor read phase: {phase!r}")
        if phase in {SensorReadPhase.POST_RESET, SensorReadPhase.POST_EVENT}:
            self._mj_scene.write_data_to_sim()
            self._sim.forward()
        state = self.scene.articulations[self._entity_name].data
        native = self._entity.data

        root_pose = native.root_link_pose_w
        root_velocity = native.root_link_vel_w
        state.root_pos_w.copy_(root_pose[:, :3])
        state.root_quat_w.copy_(root_pose[:, 3:7])
        state.root_lin_vel_w.copy_(root_velocity[:, :3])
        state.root_ang_vel_w.copy_(root_velocity[:, 3:6])

        body_pose = self._body_map.to_canonical(native.body_link_pose_w, dim=1)
        body_velocity = self._body_map.to_canonical(native.body_link_vel_w, dim=1)
        state.body_pos_w.copy_(body_pose[..., :3])
        state.body_quat_w.copy_(body_pose[..., 3:7])
        state.body_lin_vel_w.copy_(body_velocity[..., :3])
        state.body_ang_vel_w.copy_(body_velocity[..., 3:6])
        state.joint_pos.copy_(self._joint_map.to_canonical(native.joint_pos, dim=1))
        state.joint_vel.copy_(self._joint_map.to_canonical(native.joint_vel, dim=1))
        state.joint_acc.copy_(self._joint_map.to_canonical(self._last_joint_acc_native, dim=1))
        state.applied_joint_effort.copy_(self._joint_map.to_canonical(native.qfrc_actuator, dim=1))
        self._refresh_contacts()

    def _refresh_contacts(self) -> None:
        for name, binding in self._contact_bindings.items():
            canonical = self.scene.sensors[name]
            data = binding.native_sensor.data
            if data.force is None:
                raise RuntimeError(f"MJLab contact sensor {name!r} returned no force data")
            force = binding.index_map.to_canonical(data.force, dim=1)
            canonical.net_forces_w.copy_(force)
            if canonical.history_length:
                if data.force_history is None:
                    raise RuntimeError(f"MJLab contact sensor {name!r} returned no force history")
                history = binding.index_map.to_canonical(data.force_history, dim=1)
                canonical.net_forces_w_history.copy_(history.permute(0, 2, 1, 3))
            canonical.update_active(binding.force_threshold)

    def _advance_force_threshold_air_time(self) -> None:
        """Advance air-time from net force, not MuJoCo ``found > 0`` contacts.

        Native MJLab air-time treats any solver contact as stance. Light
        grazing then counts for ``feet_air_time`` while ``feet_slide`` and
        ``illegal_contact`` still use the 1 N threshold. InstinctMJ uses the
        force threshold for both; keep that here on every physics substep.
        """
        for name, binding in self._contact_bindings.items():
            if not binding.track_air_time:
                continue
            data = binding.native_sensor.data
            if data.force is None:
                raise RuntimeError(f"MJLab contact sensor {name!r} returned no force data")
            force = binding.index_map.to_canonical(data.force, dim=1)
            is_contact = torch.linalg.vector_norm(force, dim=-1) > binding.force_threshold
            self.scene.sensors[name].update_air_time(is_contact, self.sim_dt)

    def render(self, mode: str) -> object | None:
        self._require_initialized()
        if mode == "none":
            return None
        if mode == "rgb_array":
            if self._offscreen_renderer is None:
                from mjlab.viewer import OffscreenRenderer, ViewerConfig

                cfg = ViewerConfig(
                    origin_type=ViewerConfig.OriginType.ASSET_ROOT,
                    entity_name=self._entity_name,
                    width=1280,
                    height=720,
                    distance=2.8,
                    elevation=-18.0,
                    azimuth=140.0,
                    max_extra_envs=0,
                )
                self._offscreen_renderer = OffscreenRenderer(
                    self._sim.mj_model,
                    cfg,
                    self._mj_scene,
                    sim_model=self._sim.model,
                    expanded_fields=self._sim.expanded_fields,
                )
                self._offscreen_renderer.initialize()
            self._offscreen_renderer.update(self._sim.data)
            return self._offscreen_renderer.render()
        if mode == "human":
            import mujoco
            import mujoco.viewer

            if self._human_viewer is None:
                self._human_viewer = mujoco.viewer.launch_passive(self._sim.mj_model, self._sim.mj_data)
            host_data = self._sim.mj_data
            host_data.qpos[:] = self._sim.data.qpos[0].detach().cpu().numpy()
            host_data.qvel[:] = self._sim.data.qvel[0].detach().cpu().numpy()
            if self._sim.mj_model.nmocap:
                host_data.mocap_pos[:] = self._sim.data.mocap_pos[0].detach().cpu().numpy()
                host_data.mocap_quat[:] = self._sim.data.mocap_quat[0].detach().cpu().numpy()
            mujoco.mj_forward(self._sim.mj_model, host_data)
            self._human_viewer.sync()
            return None
        raise ValueError(f"unsupported MJLab render mode: {mode!r}")

    def close(self) -> None:
        if self._offscreen_renderer is not None:
            self._offscreen_renderer.close()
            self._offscreen_renderer = None
        if self._human_viewer is not None:
            self._human_viewer.close()
            self._human_viewer = None
        self._contact_bindings.clear()
        self._mj_scene = None
        self._sim = None
        self._entity = None


__all__ = ["MjlabBackend"]
