from __future__ import annotations

import torch
from types import SimpleNamespace

from instinctlab.assets.unitree_g1 import make_g1_29dof_robot_spec
from instinctlab.backends.isaacsim.backend import IsaacSimBackend, _contact_prim_path
from instinctlab.sim.backend import CanonicalIndexMap
from instinctlab.sim.control import ControlMode, JointControlTarget
from instinctlab.sim.scene import ContactSensorSpec, SceneSpec, SimulationSpec
from instinctlab.tasks.locomotion.mdp.unified import _ids


def test_contact_body_aliases_come_from_robot_asset() -> None:
    asset = make_g1_29dof_robot_spec().asset_for("isaacsim")
    canonical_names = ("torso_link", "LL_FOOT", "LR_FOOT")

    assert asset.resolve_contact_body_names(canonical_names) == (
        "torso_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )
    assert (
        _contact_prim_path(
            canonical_names,
            aliases=asset.contact_body_aliases,
            prim_path=str(asset.import_options["prim_path"]),
        )
        == "{ENV_REGEX_NS}/Robot/(torso_link|left_ankle_roll_link|right_ankle_roll_link)"
    )


def test_mdp_ids_are_cached_per_device() -> None:
    names = ("torso_link", "left_ankle_roll_link", "right_ankle_roll_link")
    selected = ("left_ankle_roll_link", "right_ankle_roll_link")

    first = _ids(names, selected, torch.device("cpu"))
    second = _ids(names, selected, torch.device("cpu"))

    assert first is second
    torch.testing.assert_close(first, torch.tensor([1, 2]))


def test_isaacsim_engine_options_ignore_mjlab_scope() -> None:
    spec = SimulationSpec(
        engine_options={
            "isaacsim": {"device": "cuda:1"},
            "mjlab": {"not_an_isaacsim_option": True},
        }
    )
    native_cfg = SimpleNamespace(device="cuda:0")

    IsaacSimBackend._apply_engine_options(
        native_cfg,
        spec.engine_options_for("isaacsim"),
    )

    assert native_cfg.device == "cuda:1"


class _KwargsCfg:
    class InitialStateCfg:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            for name, value in kwargs.items():
                setattr(self, name, value)

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        for name, value in kwargs.items():
            setattr(self, name, value)


class _FakeSimUtils:
    UrdfFileCfg = _KwargsCfg
    RigidBodyPropertiesCfg = _KwargsCfg
    ArticulationRootPropertiesCfg = _KwargsCfg

    class UrdfConverterCfg:
        class JointDriveCfg(_KwargsCfg):
            class PDGainsCfg(_KwargsCfg):
                pass


def test_isaacsim_robot_cfg_reads_asset_and_scene_options() -> None:
    robot = make_g1_29dof_robot_spec()
    scene_spec = SceneSpec(
        num_envs=2,
        env_spacing=1.0,
        robot=robot,
        backend_options={
            "isaacsim": {
                "robot_spawn": {
                    "self_collision": True,
                    "articulation_props": {
                        "solver_position_iteration_count": 8,
                        "solver_velocity_iteration_count": 4,
                    },
                }
            }
        },
    )
    backend = object.__new__(IsaacSimBackend)
    cfg = backend._make_robot_cfg(
        scene_spec,
        sim_utils=_FakeSimUtils,
        articulation_cfg_type=_KwargsCfg,
        actuator_cfg_type=_KwargsCfg,
    )

    assert cfg.kwargs["prim_path"] == "{ENV_REGEX_NS}/Robot"
    spawn = cfg.kwargs["spawn"].kwargs
    assert spawn["fix_base"] is False
    assert spawn["merge_fixed_joints"] is False
    assert spawn["replace_cylinders_with_capsules"] is True
    assert spawn["self_collision"] is True
    assert spawn["activate_contact_sensors"] is False
    assert spawn["articulation_props"].kwargs["solver_position_iteration_count"] == 8


def test_isaacsim_activates_contact_sensors_when_configured() -> None:
    robot = make_g1_29dof_robot_spec()
    scene_spec = SceneSpec(
        num_envs=2,
        env_spacing=1.0,
        robot=robot,
        contact_sensors=(
            ContactSensorSpec(
                name="contact_forces",
                entity_name="robot",
                body_names=("torso_link",),
            ),
        ),
    )
    backend = object.__new__(IsaacSimBackend)
    cfg = backend._make_robot_cfg(
        scene_spec,
        sim_utils=_FakeSimUtils,
        articulation_cfg_type=_KwargsCfg,
        actuator_cfg_type=_KwargsCfg,
    )

    assert cfg.kwargs["spawn"].kwargs["activate_contact_sensors"] is True


class _FakeRobot:
    def __init__(self) -> None:
        self.data = SimpleNamespace(joint_pos=torch.zeros((2, 2)))
        self.actuators = {
            "canonical": SimpleNamespace(
                stiffness=torch.zeros((2, 2)),
                damping=torch.zeros((2, 2)),
            )
        }
        self.position_writes = 0
        self.velocity_writes = 0
        self.effort_writes = 0
        self.stiffness_writes = 0
        self.damping_writes = 0

    def set_joint_position_target(self, *args, **kwargs) -> None:
        self.position_writes += 1

    def set_joint_velocity_target(self, *args, **kwargs) -> None:
        self.velocity_writes += 1

    def set_joint_effort_target(self, *args, **kwargs) -> None:
        self.effort_writes += 1

    def write_joint_stiffness_to_sim(self, *args, **kwargs) -> None:
        self.stiffness_writes += 1

    def write_joint_damping_to_sim(self, *args, **kwargs) -> None:
        self.damping_writes += 1


def _make_control_backend() -> tuple[IsaacSimBackend, _FakeRobot]:
    backend = object.__new__(IsaacSimBackend)
    robot = _FakeRobot()
    names = ("first", "second")
    backend.device = torch.device("cpu")
    backend.num_envs = 2
    backend._sim = object()
    backend._native_scene = object()
    backend._robot = robot
    backend._entity_name = "robot"
    backend._joint_map = CanonicalIndexMap.build(names, names, device="cpu")
    backend._all_env_ids = torch.arange(2, dtype=torch.int64)
    backend._all_joint_ids = torch.arange(2, dtype=torch.int64)
    backend._joint_properties = {
        "stiffness": torch.ones(2),
        "damping": torch.ones(2),
    }
    backend._global_control_mode = ControlMode.POSITION
    backend._position_velocity_is_zero = True
    backend._last_position_target = None
    backend._last_position_target_version = -1
    backend._last_position_velocity = None
    backend._last_position_velocity_version = -1
    backend.scene = SimpleNamespace(
        articulations={"robot": SimpleNamespace(data=SimpleNamespace(joint_effort_limits=torch.full((2, 2), 10.0)))}
    )
    return backend, robot


def test_position_control_skips_unchanged_targets_and_static_gains() -> None:
    backend, robot = _make_control_backend()
    target = JointControlTarget(
        mode=ControlMode.POSITION,
        value=torch.ones((2, 2)),
        velocity=torch.zeros((2, 2)),
    )

    backend.set_joint_control_target("robot", target)
    backend.set_joint_control_target("robot", target)

    assert robot.position_writes == 1
    assert robot.velocity_writes == 1
    assert robot.effort_writes == 0
    assert robot.stiffness_writes == 0
    assert robot.damping_writes == 0

    target.value.add_(1.0)
    backend.set_joint_control_target("robot", target)
    assert robot.position_writes == 2
    assert robot.velocity_writes == 1


def test_control_mode_switch_writes_gains_once() -> None:
    backend, robot = _make_control_backend()
    target = JointControlTarget(
        mode=ControlMode.VELOCITY,
        value=torch.ones((2, 2)),
    )

    backend.set_joint_control_target("robot", target)
    backend.set_joint_control_target("robot", target)

    assert robot.stiffness_writes == 1
    assert robot.damping_writes == 1
