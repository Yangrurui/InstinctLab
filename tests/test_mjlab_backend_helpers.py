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
    _load_mjcf,
    _strip_visual_meshes_xml,
)
from instinctlab.sim.backend import CanonicalIndexMap, RuntimeRequirements, SensorReadPhase
from instinctlab.sim.capabilities import Capability
from instinctlab.sim.control import ControlMode, JointControlTarget
from instinctlab.sim.scene import SimulationSpec
from instinctlab.sim.state import ArticulationState


@dataclass
class _MujocoCfg:
    timestep: float = 0.002
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    iterations: int = 100
    ls_iterations: int = 50


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


def test_mjlab_engine_options_override_training_defaults() -> None:
    cfg = MjlabBackend._make_simulation_cfg(
        SimulationSpec(
            engine_options={
                "mjlab": {"njmax": 512, "iterations": 7},
                "isaacsim": {"not_a_mjlab_option": True},
            }
        ),
        MujocoCfg=_MujocoCfg,
        SimulationCfg=_SimulationCfg,
    )

    assert cfg.njmax == 512
    assert cfg.mujoco.iterations == 7


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


def test_mjlab_effort_actuator_is_enabled_only_when_requested() -> None:
    position_only = RuntimeRequirements(capabilities=frozenset({Capability.IMPLICIT_POSITION_CONTROL}))
    effort_required = RuntimeRequirements(capabilities=frozenset({Capability.EFFORT_CONTROL}))

    assert not _enable_effort_actuator(True, position_only)
    assert _enable_effort_actuator(True, effort_required)
    assert not _enable_effort_actuator(False, effort_required)


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
    backend._last_joint_acc_native = torch.zeros((1, 1))
    backend._previous_joint_velocity_native = torch.zeros((1, 1))
    backend._effort_mode_mask = torch.zeros((1, 1), dtype=torch.bool)
    backend._effort_mode_active = False
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


def test_mjlab_position_step_has_no_effort_mask_host_sync(monkeypatch) -> None:
    backend, simulation, native_scene = _make_runtime_backend()

    def fail_on_any(*args, **kwargs):
        raise AssertionError("position-mode step must not call torch.any")

    monkeypatch.setattr(torch, "any", fail_on_any)
    backend.step()

    assert simulation.step_calls == 1
    assert native_scene.write_calls == 1
    assert native_scene.update_calls == 1
