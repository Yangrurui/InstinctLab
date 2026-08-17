from __future__ import annotations

import torch
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from instinctlab.assets.unitree_g1 import make_g1_29dof_robot_spec
from instinctlab.backends.mjlab.simulator import (
    MjlabBackend,
    _enable_effort_actuator,
    _expanded_randomization_fields,
    _group_equal_pd_joints,
    _load_mjcf,
    _native_contact_sensor_cfg,
    _solref_dampratio_from_restitution,
    _strip_visual_meshes_xml,
)
from instinctlab.sim.backend import CanonicalIndexMap, RuntimeRequirements, SensorReadPhase
from instinctlab.sim.capabilities import Capability
from instinctlab.sim.control import ControlMode, JointControlTarget
from instinctlab.sim.scene import ContactSensorSpec, SimulationSpec
from instinctlab.sim.state import ArticulationState


@dataclass
class _MujocoCfg:
    timestep: float = 0.002
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    iterations: int = 100
    ls_iterations: int = 50
    ccd_iterations: int = 50


@dataclass
class _SimulationCfg:
    nconmax: int | None = None
    njmax: int | None = None
    mujoco: _MujocoCfg = field(default_factory=_MujocoCfg)


def test_mjlab_backend_preserves_native_simulation_defaults() -> None:
    cfg = MjlabBackend._make_simulation_cfg(
        SimulationSpec(),
        MujocoCfg=_MujocoCfg,
        SimulationCfg=_SimulationCfg,
    )

    assert cfg.nconmax is None
    assert cfg.njmax is None
    assert cfg.mujoco.iterations == 100
    assert cfg.mujoco.ls_iterations == 50
    assert cfg.mujoco.ccd_iterations == 50


def test_mjlab_engine_options_override_training_defaults() -> None:
    cfg = MjlabBackend._make_simulation_cfg(
        SimulationSpec(
            engine_options={
                "mjlab": {"njmax": 512, "iterations": 7, "ccd_iterations": 500},
                "isaacsim": {"not_a_mjlab_option": True},
            }
        ),
        MujocoCfg=_MujocoCfg,
        SimulationCfg=_SimulationCfg,
    )

    assert cfg.njmax == 512
    assert cfg.mujoco.iterations == 7
    assert cfg.mujoco.ccd_iterations == 500


def test_mjlab_native_contact_cfg_uses_force_only() -> None:
    feet = ContactSensorSpec(
        name="feet_contact_forces",
        entity_name="robot",
        body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
        force_threshold=1.0,
        track_air_time=True,
    )
    base = ContactSensorSpec(
        name="base_contact_forces",
        entity_name="robot",
        body_names=("torso_link",),
        force_threshold=1.0,
        track_air_time=False,
    )
    from instinctlab.backends.mjlab.contact_sensor import ForceThresholdContactSensorCfg

    feet_cfg = _native_contact_sensor_cfg(feet, feet.body_names, "robot")
    base_cfg = _native_contact_sensor_cfg(base, base.body_names, "robot")

    assert feet_cfg.fields == ("force",)
    assert "found" not in feet_cfg.fields
    assert feet_cfg.track_air_time is True
    assert isinstance(feet_cfg, ForceThresholdContactSensorCfg)
    assert feet_cfg.force_threshold == 1.0
    assert base_cfg.fields == ("force",)
    assert "found" not in base_cfg.fields
    assert base_cfg.track_air_time is False
    assert not isinstance(base_cfg, ForceThresholdContactSensorCfg)


def test_mjlab_contact_aliases_come_from_robot_asset() -> None:
    asset = make_g1_29dof_robot_spec().asset_for("mjlab")
    assert asset.resolve_contact_body_names(("LL_FOOT", "LR_FOOT")) == (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )
    assert asset.load_mode == "strip_visual_meshes"


def test_mjlab_strip_visual_meshes_is_generic() -> None:
    xml = (
        "<mujoco>"
        "<compiler meshdir='meshes'/>"
        "<asset><mesh name='visual' file='visual.stl'/></asset>"
        "<worldbody>"
        "<geom type='mesh' mesh='visual'/>"
        "<geom type='capsule' size='0.1'/>"
        "</worldbody>"
        "</mujoco>"
    )
    stripped = _strip_visual_meshes_xml(xml)
    assert "mesh" not in stripped
    assert "capsule" in stripped
    assert "meshdir" not in stripped


def test_mjlab_default_load_mode_rejects_unknown_mode() -> None:
    class _FakeMujoco:
        class MjSpec:
            @staticmethod
            def from_file(path: str):
                raise AssertionError(f"should not load {path}")

    with pytest.raises(ValueError, match="unsupported MJLab asset load_mode"):
        _load_mjcf(Path("unused.xml"), _FakeMujoco, "not-a-mode")


def test_mjlab_default_load_mode_does_not_strip(tmp_path: Path) -> None:
    xml_path = tmp_path / "robot.xml"
    xml_path.write_text("<mujoco><geom type='mesh' mesh='visual'/></mujoco>")

    class _FakeMujoco:
        class MjSpec:
            @staticmethod
            def from_file(path: str):
                return f"loaded:{path}"

            @staticmethod
            def from_string(xml: str):
                raise AssertionError("default load_mode must not strip meshes")

    assert _load_mjcf(xml_path, _FakeMujoco, "default") == f"loaded:{xml_path}"


def test_mjlab_strip_visual_meshes_falls_back_only_when_from_file_fails(tmp_path: Path) -> None:
    xml_path = tmp_path / "robot.xml"
    xml_path.write_text(
        "<mujoco><asset><mesh name='visual' file='visual.stl'/></asset>"
        "<worldbody><geom type='mesh' mesh='visual'/></worldbody></mujoco>"
    )

    class _FakeMujoco:
        class MjSpec:
            @staticmethod
            def from_file(path: str):
                raise ValueError(f"missing mesh in {path}")

            @staticmethod
            def from_string(xml: str):
                return xml

    stripped = _load_mjcf(xml_path, _FakeMujoco, "strip_visual_meshes")
    assert "mesh" not in stripped


def test_mjlab_restitution_maps_to_solref_dampratio() -> None:
    restitution = torch.tensor([0.0, 0.4, 1.0])
    torch.testing.assert_close(_solref_dampratio_from_restitution(restitution), torch.tensor([1.0, 0.6, 0.0]))


def test_mjlab_expands_solref_when_restitution_dr_is_declared() -> None:
    fields = _expanded_randomization_fields(
        RuntimeRequirements(
            capabilities=frozenset({Capability.DR_RESTITUTION}),
            randomization_fields=frozenset({"restitution"}),
        )
    )
    assert "geom_solref" in fields
    assert "geom_friction" not in fields


def test_mjlab_effort_actuator_is_enabled_only_when_requested() -> None:
    position_only = RuntimeRequirements(capabilities=frozenset({Capability.IMPLICIT_POSITION_CONTROL}))
    effort_required = RuntimeRequirements(capabilities=frozenset({Capability.EFFORT_CONTROL}))

    assert not _enable_effort_actuator(True, position_only)
    assert _enable_effort_actuator(True, effort_required)
    assert not _enable_effort_actuator(False, effort_required)


def test_g1_pd_actuators_group_by_equal_gains() -> None:
    robot = make_g1_29dof_robot_spec()
    grouped = _group_equal_pd_joints(robot.joint_properties)
    names = [name for group_names, _ in grouped for name in group_names]

    assert len(names) == len(robot.joint_properties)
    assert set(names) == {item.name for item in robot.joint_properties}
    assert len(grouped) == 5
    for group_names, properties in grouped:
        assert len(group_names) >= 1
        for name in group_names:
            item = robot.joint_properties[robot.joint_index(name)]
            assert item.stiffness == properties.stiffness
            assert item.damping == properties.damping
            assert item.effort_limit == properties.effort_limit
            assert item.armature == properties.armature


def test_mjlab_control_cache_tracks_in_place_updates() -> None:
    backend = object.__new__(MjlabBackend)
    backend._last_control_mode = None
    backend._last_control_value = None
    backend._last_control_value_version = -1
    backend._last_control_velocity = None
    backend._last_control_velocity_version = -1
    target = JointControlTarget(
        mode=ControlMode.POSITION,
        value=torch.zeros((2, 2)),
        velocity=torch.zeros((2, 2)),
    )

    backend._cache_control_target(target)
    assert backend._is_repeated_control_target(target)

    target.value.add_(1.0)
    assert not backend._is_repeated_control_target(target)


class _FakeSimulation:
    def __init__(self) -> None:
        self.step_calls = 0
        self.forward_calls = 0

    def step(self) -> None:
        self.step_calls += 1

    def forward(self) -> None:
        self.forward_calls += 1


class _FakeScene:
    def __init__(self) -> None:
        self.write_calls = 0
        self.update_calls = 0

    def write_data_to_sim(self) -> None:
        self.write_calls += 1

    def update(self, dt: float) -> None:
        del dt
        self.update_calls += 1


def _make_runtime_backend() -> tuple[MjlabBackend, _FakeSimulation, _FakeScene]:
    backend = object.__new__(MjlabBackend)
    simulation = _FakeSimulation()
    native_scene = _FakeScene()
    native = SimpleNamespace(
        root_link_pose_w=torch.zeros((1, 7)),
        root_link_vel_w=torch.zeros((1, 6)),
        body_link_pose_w=torch.zeros((1, 1, 7)),
        body_link_vel_w=torch.zeros((1, 1, 6)),
        joint_pos=torch.zeros((1, 1)),
        joint_vel=torch.zeros((1, 1)),
        joint_acc=torch.zeros((1, 1)),
        qfrc_actuator=torch.zeros((1, 1)),
    )
    state = ArticulationState.allocate(
        num_envs=1,
        num_joints=1,
        num_bodies=1,
        device="cpu",
    )
    mapping = CanonicalIndexMap.build(("only",), ("only",), device="cpu")
    backend._sim = simulation
    backend._mj_scene = native_scene
    backend._entity = SimpleNamespace(data=native)
    backend._entity_name = "robot"
    backend._joint_map = mapping
    backend._body_map = mapping
    backend._contact_bindings = {}
    backend._effort_mode_mask = torch.zeros((1, 1), dtype=torch.bool)
    backend._effort_mode_active = False
    backend._sync_fast_path = False
    backend._alias_native_views = False
    backend.sim_dt = 0.005
    backend.scene = SimpleNamespace(
        articulations={"robot": SimpleNamespace(data=state)},
        sensors={},
    )
    return backend, simulation, native_scene


def test_mjlab_post_physics_sync_does_not_forward_again() -> None:
    backend, simulation, native_scene = _make_runtime_backend()

    backend.synchronize(SensorReadPhase.POST_PHYSICS)
    assert simulation.forward_calls == 0
    assert native_scene.write_calls == 0

    backend.synchronize(SensorReadPhase.POST_EVENT)
    assert simulation.forward_calls == 1
    assert native_scene.write_calls == 1

    backend.synchronize(SensorReadPhase.POST_INTERVAL)
    assert simulation.forward_calls == 1
    assert native_scene.write_calls == 1


def test_mjlab_pre_observation_sync_forwards_without_write() -> None:
    backend, simulation, native_scene = _make_runtime_backend()

    backend.synchronize(SensorReadPhase.PRE_OBSERVATION)
    assert simulation.forward_calls == 1
    assert native_scene.write_calls == 0


def test_mjlab_position_step_has_no_effort_mask_host_sync(monkeypatch) -> None:
    backend, simulation, native_scene = _make_runtime_backend()

    def fail_on_any(*args, **kwargs):
        raise AssertionError("position-mode step must not call torch.any")

    monkeypatch.setattr(torch, "any", fail_on_any)
    backend.step()

    assert simulation.step_calls == 1
    assert native_scene.write_calls == 1
    assert native_scene.update_calls == 1
