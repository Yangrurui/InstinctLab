from __future__ import annotations

import math
import torch
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from instinctlab.backends.mock import MockSimulatorBackend
from instinctlab.envs.unified_manager_based_rl_env import UnifiedManagerBasedRLEnv, UnifiedManagerBasedRLEnvCfg
from instinctlab.managers.unified import (
    CommandTerm,
    CommandTermCfg,
    EventTermCfg,
    JointPositionAction,
    JointPositionActionCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardGroupCfg,
    RewardTermCfg,
    TerminationGroupCfg,
    TerminationTermCfg,
    UniformNoiseCfg,
)
from instinctlab.rl.config import OnPolicyRunnerCfg
from instinctlab.rl.vecenv_wrapper import InstinctRlVecEnvWrapper
from instinctlab.sim.backend import SensorReadPhase
from instinctlab.sim.capabilities import Capability
from instinctlab.sim.control import ControlMode
from instinctlab.sim.robot_spec import JointProperties, RobotSpec
from instinctlab.sim.scene import SceneSpec, SimulationSpec
from instinctlab.sim.state import ContactState
from instinctlab.tasks.locomotion.commands import UniformVelocityCommand
from instinctlab.tasks.locomotion.mdp.unified import contact_slide
from instinctlab.tasks.locomotion.unified_flat_env_cfg import (
    LOCOMOTION_MATERIAL_BACKEND_PARAMS,
    locomotion_flat_agent_cfg,
    locomotion_flat_env_cfg,
)


class TrackingBackend(MockSimulatorBackend):
    def __init__(self) -> None:
        super().__init__(device="cpu")
        self.phases: list[SensorReadPhase] = []
        self.targets: list[torch.Tensor] = []
        self.step_calls = 0
        self.reset_calls = 0

    def set_joint_control_target(self, entity_name: str, target: Any, env_ids: torch.Tensor | None = None) -> None:
        super().set_joint_control_target(entity_name, target, env_ids)
        self.targets.append(target.value.clone())

    def reset(self, env_ids: torch.Tensor) -> None:
        self.reset_calls += 1
        super().reset(env_ids)

    def step(self) -> None:
        self.step_calls += 1
        super().step()

    def synchronize(self, phase: SensorReadPhase) -> None:
        self.phases.append(phase)
        super().synchronize(phase)


def make_robot() -> RobotSpec:
    names = ("root_joint", "child_joint")
    return RobotSpec(
        name="tiny",
        schema_version="dfs_v1",
        asset_id="tiny_v1",
        root_body="root",
        joint_names=names,
        body_names=("root", "child"),
        joint_properties=(
            JointProperties(names[0], 1.0, 2.0, 0.0, 0.0, 100.0, 10.0, 0.1),
            JointProperties(names[1], -1.0, 2.0, 0.0, 0.0, 100.0, 10.0, 0.2),
        ),
        assets=(),
        default_root_pos=(0.0, 0.0, 1.0),
        default_root_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        soft_joint_pos_limit_factor=0.9,
    )


def make_cfg(
    *,
    observations: Mapping[str, ObservationGroupCfg] | None = None,
    rewards: Mapping[str, RewardGroupCfg] | None = None,
    terminations: TerminationGroupCfg | None = None,
    events: Mapping[str, EventTermCfg] | None = None,
    episode_length_s: float = 10.0,
    seed: int = 7,
) -> UnifiedManagerBasedRLEnvCfg:
    robot = make_robot()
    if observations is None:
        observations = {
            "policy": ObservationGroupCfg(
                terms={
                    "joint_pos": ObservationTermCfg(
                        func=lambda env: env.scene.articulations["robot"].data.joint_pos,
                        shape=(2,),
                    )
                },
                term_order=("joint_pos",),
            )
        }
    if rewards is None:
        rewards = {
            "default": RewardGroupCfg(
                terms={"alive": RewardTermCfg(func=lambda env: torch.ones(env.num_envs, device=env.device))},
                term_order=("alive",),
            )
        }
    return UnifiedManagerBasedRLEnvCfg(
        scene=SceneSpec(num_envs=2, env_spacing=1.0, robot=robot),
        simulation=SimulationSpec(sim_dt=0.01, decimation=2),
        actions={"joint_pos": JointPositionActionCfg()},
        observations=observations,
        rewards=rewards,
        terminations=terminations or TerminationGroupCfg(terms={}, term_order=()),
        events=events or {},
        episode_length_s=episode_length_s,
        seed=seed,
    )


def test_joint_position_action_and_synchronized_step_lifecycle() -> None:
    def interval_event(env: UnifiedManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
        env.scene.articulations["robot"].data.root_pos_w[env_ids, 0] += 1.0

    cfg = make_cfg(
        events={
            "push": EventTermCfg(
                func=interval_event,
                mode="interval",
                interval_range_s=(0.02, 0.02),
                writes_state=True,
            )
        }
    )
    backend = TrackingBackend()
    env = UnifiedManagerBasedRLEnv(cfg, backend)
    backend.phases.clear()
    action = torch.tensor([[2.0, 3.0], [-1.0, 0.5]])

    env.step(action)

    assert backend.step_calls == cfg.simulation.decimation
    assert len(backend.targets) == cfg.simulation.decimation
    expected = torch.tensor([[1.2, -0.4], [0.9, -0.9]])
    torch.testing.assert_close(backend.targets[-1], expected)
    target = env.action_manager.get_term("joint_pos")
    assert isinstance(target, JointPositionAction)
    assert target.control_target.mode is ControlMode.POSITION
    torch.testing.assert_close(target.control_target.velocity, torch.zeros_like(expected))
    assert backend.phases == [
        SensorReadPhase.POST_PHYSICS,
        SensorReadPhase.PRE_OBSERVATION,
        SensorReadPhase.POST_INTERVAL,
    ]


def test_step_without_reset_or_event_refreshes_before_observation() -> None:
    backend = TrackingBackend()
    env = UnifiedManagerBasedRLEnv(make_cfg(), backend)
    assert backend.reset_calls == 0
    env.reset()
    assert backend.reset_calls == 1
    backend.phases.clear()

    env.step(torch.zeros((2, 2)))

    assert backend.phases == [SensorReadPhase.POST_PHYSICS, SensorReadPhase.PRE_OBSERVATION]


def test_velocity_command_resample_follows_instinctmj_order() -> None:
    cfg = replace(
        make_cfg(),
        commands={
            "base_velocity": CommandTermCfg(
                UniformVelocityCommand,
                params={
                    "resampling_time_range": (10.0, 10.0),
                    "rel_standing_envs": 0.2,
                    "rel_heading_envs": 0.5,
                    "rel_world_envs": 0.0,
                    "rel_forward_envs": 0.0,
                    "init_velocity_prob": 0.0,
                    "heading_command": True,
                    "heading_control_stiffness": 0.5,
                    "ranges": {
                        "lin_vel_x": (-0.5, 1.0),
                        "lin_vel_y": (-0.5, 0.5),
                        "ang_vel_z": (-1.5, 1.5),
                        "heading": (-math.pi, math.pi),
                    },
                },
            )
        },
        command_order=("base_velocity",),
    )
    env = UnifiedManagerBasedRLEnv(cfg, TrackingBackend())
    streams: list[str] = []
    original = env.rng.uniform

    def _trace(stream: str, low: float, high: float, shape: tuple[int, ...], **kwargs):
        streams.append(stream)
        return original(stream, low, high, shape, **kwargs)

    env.rng.uniform = _trace  # type: ignore[method-assign]
    env.reset()
    command_streams = [name for name in streams if name.startswith("command.base_velocity.")]
    assert command_streams == [
        "command.base_velocity.resampling_time",
        "command.base_velocity.lin_x",
        "command.base_velocity.lin_y",
        "command.base_velocity.ang_z",
        "command.base_velocity.heading",
        "command.base_velocity.heading_mask",
        "command.base_velocity.standing_mask",
        "command.base_velocity.world_mask",
        "command.base_velocity.forward_mask",
        "command.base_velocity.init_velocity_mask",
    ]
    term = dict(env.command_manager._terms)["base_velocity"]
    assert not bool(term._world.any())
    assert not bool(term._forward.any())
    env.close()


def test_vecenv_wrapper_resets_once_and_fills_obs_dims() -> None:
    backend = TrackingBackend()
    env = UnifiedManagerBasedRLEnv(make_cfg(), backend)
    assert backend.reset_calls == 0
    wrapped = InstinctRlVecEnvWrapper(env)
    assert backend.reset_calls == 1
    assert wrapped.num_obs == 2
    assert wrapped.num_critic_obs is None
    env.close()


def test_step_refreshes_kinematics_before_command() -> None:
    seen: list[torch.Tensor] = []

    class RecordingCommand(CommandTerm):
        def reset(self, env_ids: torch.Tensor) -> None:
            del env_ids

        def compute(self, dt: float) -> None:
            del dt
            seen.append(self.env.scene.articulations["robot"].data.root_quat_w[0].clone())

        @property
        def command(self) -> torch.Tensor:
            return torch.zeros((self.env.num_envs, 3), device=self.env.device)

    class RefreshOnPreObservation(TrackingBackend):
        def synchronize(self, phase: SensorReadPhase) -> None:
            super().synchronize(phase)
            if phase is SensorReadPhase.PRE_OBSERVATION:
                self.scene.articulations["robot"].data.root_quat_w[:] = torch.tensor([0.0, 1.0, 0.0, 0.0])

    backend = RefreshOnPreObservation()
    env = UnifiedManagerBasedRLEnv(
        replace(
            make_cfg(),
            commands={"base_velocity": CommandTermCfg(RecordingCommand)},
            command_order=("base_velocity",),
        ),
        backend,
    )
    seen.clear()
    env.step(torch.zeros((2, 2)))

    assert seen
    torch.testing.assert_close(seen[-1], torch.tensor([0.0, 1.0, 0.0, 0.0]))


def test_reward_dt_termination_order_and_observation_frozen_schema() -> None:
    calls: list[str] = []

    def termination(env: UnifiedManagerBasedRLEnv) -> torch.Tensor:
        calls.append("termination")
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    def reward_a(env: UnifiedManagerBasedRLEnv) -> torch.Tensor:
        calls.append("reward_a")
        return torch.ones(env.num_envs, device=env.device)

    def reward_b(env: UnifiedManagerBasedRLEnv) -> torch.Tensor:
        calls.append("reward_b")
        return torch.full((env.num_envs,), 3.0, device=env.device)

    observations = {
        "policy": ObservationGroupCfg(
            terms={
                "second": ObservationTermCfg(func=lambda env: torch.full((env.num_envs, 1), 2.0), shape=(1,)),
                "first": ObservationTermCfg(func=lambda env: torch.full((env.num_envs, 2), 1.0), shape=(2,)),
            },
            term_order=("first", "second"),
        )
    }
    rewards = {
        "default": RewardGroupCfg(
            terms={
                "b": RewardTermCfg(func=reward_b, weight=4.0),
                "a": RewardTermCfg(func=reward_a, weight=2.0),
            },
            term_order=("a", "b"),
        )
    }
    terminations = TerminationGroupCfg(
        terms={"fall": TerminationTermCfg(func=termination)},
        term_order=("fall",),
    )
    env = UnifiedManagerBasedRLEnv(
        make_cfg(observations=observations, rewards=rewards, terminations=terminations),
        TrackingBackend(),
    )
    calls.clear()

    obs, reward, _, _, _ = env.step(torch.zeros((2, 2)))

    assert calls == ["termination", "reward_a", "reward_b"]
    torch.testing.assert_close(reward, torch.full((2, 1), (2.0 + 12.0) * env.step_dt))
    torch.testing.assert_close(obs["policy"], torch.tensor([[1.0, 1.0, 2.0], [1.0, 1.0, 2.0]]))
    assert env.observation_manager.group_term_names["policy"] == ("first", "second")
    assert tuple(segment.name for segment in env.observation_manager.group_schemas["policy"].segments) == (
        "first",
        "second",
    )


def test_observation_manager_reuses_shared_term_cfg() -> None:
    calls = {"count": 0}

    def shared_term(env: UnifiedManagerBasedRLEnv) -> torch.Tensor:
        calls["count"] += 1
        return torch.full((env.num_envs, 2), 0.5, device=env.device)

    shared = ObservationTermCfg(func=shared_term, noise=UniformNoiseCfg(-0.1, 0.1), shape=(2,))
    extra = ObservationTermCfg(func=lambda env: torch.zeros((env.num_envs, 1), device=env.device), shape=(1,))
    env = UnifiedManagerBasedRLEnv(
        make_cfg(
            observations={
                "policy": ObservationGroupCfg({"shared": shared}, ("shared",), enable_corruption=True),
                "critic": ObservationGroupCfg(
                    {"extra": extra, "shared": shared},
                    ("extra", "shared"),
                    enable_corruption=False,
                ),
            }
        ),
        TrackingBackend(),
    )
    calls["count"] = 0
    observations = env.observation_manager.compute()

    assert calls["count"] == 1
    assert observations["critic"].shape == (2, 3)
    torch.testing.assert_close(observations["critic"][:, 1:], torch.full((2, 2), 0.5))
    assert torch.all(observations["policy"].abs() <= 0.6)


def test_termination_timeout_and_post_reset_observation() -> None:
    reset_count = 0

    def reset_joints(env: UnifiedManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
        nonlocal reset_count
        reset_count += 1
        value = torch.full((env_ids.numel(), 2), float(reset_count), device=env.device)
        env.backend.write_joint_state("robot", value, torch.zeros_like(value), env_ids)

    def failed(env: UnifiedManagerBasedRLEnv) -> torch.Tensor:
        return torch.tensor([True, False], device=env.device)

    def timed_out(env: UnifiedManagerBasedRLEnv) -> torch.Tensor:
        return torch.tensor([False, True], device=env.device)

    cfg = make_cfg(
        terminations=TerminationGroupCfg(
            terms={
                "time_out": TerminationTermCfg(func=timed_out, time_out=True),
                "failed": TerminationTermCfg(func=failed),
            },
            term_order=("failed", "time_out"),
        ),
        events={"reset_joints": EventTermCfg(func=reset_joints, mode="reset", writes_state=True)},
    )
    backend = TrackingBackend()
    env = UnifiedManagerBasedRLEnv(cfg, backend)
    backend.phases.clear()

    obs, _, terminated, truncated, extras = env.step(torch.zeros((2, 2)))

    assert terminated.tolist() == [True, False]
    assert truncated.tolist() == [False, True]
    assert extras["time_outs"].tolist() == [False, True]
    assert env.episode_length_buf.tolist() == [0, 0]
    torch.testing.assert_close(obs["policy"], torch.full((2, 2), 1.0))
    assert backend.phases == [SensorReadPhase.POST_PHYSICS, SensorReadPhase.POST_RESET]


def test_max_episode_length_truncates_and_uniform_noise_is_seeded() -> None:
    observations = {
        "policy": ObservationGroupCfg(
            terms={
                "value": ObservationTermCfg(
                    func=lambda env: torch.zeros((env.num_envs, 2), device=env.device),
                    noise=UniformNoiseCfg(-0.1, 0.1),
                    shape=(2,),
                )
            },
            term_order=("value",),
        )
    }
    first = UnifiedManagerBasedRLEnv(
        make_cfg(observations=observations, episode_length_s=0.02, seed=11),
        TrackingBackend(),
    )
    first_obs, _, _, first_truncated, _ = first.step(torch.zeros((2, 2)))
    first.close()
    second = UnifiedManagerBasedRLEnv(
        make_cfg(observations=observations, episode_length_s=0.02, seed=11),
        TrackingBackend(),
    )
    second_obs, _, _, second_truncated, _ = second.step(torch.zeros((2, 2)))
    second.close()

    assert first.max_episode_length == 1
    assert first_truncated.all() and second_truncated.all()
    torch.testing.assert_close(first_obs["policy"], second_obs["policy"])
    assert torch.all(first_obs["policy"].abs() <= 0.1)


def test_new_unified_modules_do_not_import_engine_packages() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = (
        root / "source/instinctlab/instinctlab/managers/unified.py",
        root / "source/instinctlab/instinctlab/envs/unified_manager_based_rl_env.py",
        root / "source/instinctlab/instinctlab/sim/robot_spec.py",
        root / "source/instinctlab/instinctlab/sim/scene.py",
        root / "source/instinctlab/instinctlab/assets/unitree_g1.py",
    )
    forbidden_imports = (
        "import isaaclab",
        "from isaaclab",
        "import mjlab",
        "from mjlab",
    )
    for path in paths:
        source = path.read_text()
        for statement in forbidden_imports:
            assert statement not in source, f"{path} contains {statement!r}"


def test_backends_do_not_embed_robot_catalog_knowledge() -> None:
    root = Path(__file__).resolve().parents[1] / "source/instinctlab/instinctlab"
    forbidden = ("LL_FOOT", "LR_FOOT", "Unitree", "G1")
    for path in (
        root / "backends/isaacsim/backend.py",
        root / "backends/mjlab/simulator.py",
        root / "sim/robot_spec.py",
        root / "sim/scene.py",
        root / "managers/unified.py",
        root / "rl/config.py",
    ):
        source = path.read_text()
        for token in forbidden:
            assert token not in source, f"{path} contains {token!r}"


def test_g1_catalog_lives_in_assets_unitree_g1() -> None:
    import instinctlab.assets.unitree_g1 as g1

    spec = g1.make_g1_29dof_robot_spec()
    assert spec.name == "unitree_g1_29dof"
    assert spec.asset_for("isaacsim").import_options["merge_fixed_joints"] is False


def test_sim_package_does_not_export_g1_catalog() -> None:
    import instinctlab.sim as sim

    assert not hasattr(sim, "make_g1_29dof_robot_spec")
    assert not hasattr(sim, "G1_29DOF_DFS_JOINT_NAMES")


def test_shared_layers_do_not_import_robot_catalog() -> None:
    root = Path(__file__).resolve().parents[1] / "source/instinctlab/instinctlab"
    forbidden = "assets.unitree_g1"
    for path in (
        root / "sim/robot_spec.py",
        root / "sim/scene.py",
        root / "sim/backend.py",
        root / "managers/unified.py",
        root / "rl/config.py",
        root / "backends/isaacsim/backend.py",
        root / "backends/mjlab/simulator.py",
    ):
        assert forbidden not in path.read_text(), f"{path} imports the robot catalog"


def test_runner_defaults_are_task_neutral() -> None:
    assert OnPolicyRunnerCfg().experiment_name == "instinctlab"
    assert locomotion_flat_agent_cfg().experiment_name == "g1_locomotion_flat"


def test_unified_locomotion_matches_g1_flat_task() -> None:
    cfg = locomotion_flat_env_cfg(num_envs=2)
    rewards = cfg.rewards["default"].terms
    commands = cfg.commands["base_velocity"].params
    feet_sensor, base_sensor = cfg.scene.contact_sensors

    assert rewards["termination_penalty"].weight == -200.0
    assert rewards["feet_air_time"].weight == 1.0
    assert rewards["feet_air_time"].params["threshold"] == 0.5
    assert rewards["feet_air_time"].params["sensor_name"] == "feet_contact_forces"
    assert rewards["feet_air_time"].params["body_names"] == (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )
    assert rewards["feet_slide"].params["sensor_name"] == "feet_contact_forces"
    assert rewards["feet_slide"].params["threshold"] == 0.1
    assert rewards["feet_slide"].weight == -0.1
    assert rewards["stand_still"].weight == -0.8
    assert rewards["lin_vel_z"].weight == -0.1
    assert rewards["action_rate"].weight == -0.05
    assert rewards["knee_deviation"].weight == -0.05
    assert "ang_vel_xy" not in rewards
    assert commands["rel_standing_envs"] == 0.2
    assert commands["rel_heading_envs"] == 0.5
    assert commands["rel_world_envs"] == 0.0
    assert commands["rel_forward_envs"] == 0.0
    assert commands["init_velocity_prob"] == 0.0
    assert commands["heading_command"] is True
    assert commands["ranges"]["lin_vel_x"] == (-0.5, 1.0)
    assert commands["ranges"]["ang_vel_z"] == (-1.5, 1.5)
    assert commands["ranges"]["heading"] == (-math.pi, math.pi)
    assert feet_sensor.name == "feet_contact_forces"
    assert feet_sensor.body_names == (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )
    assert feet_sensor.track_air_time is True
    assert base_sensor.name == "base_contact_forces"
    assert "torso_link" in base_sensor.body_names
    assert "left_ankle_roll_link" not in base_sensor.body_names
    assert base_sensor.track_air_time is False
    assert cfg.terminations.terms["base_contact"].params["sensor_name"] == "base_contact_forces"
    assert cfg.events["reset_joints"].params["position_range"] == (0.8, 1.2)
    material = cfg.events["randomize_material"].params
    assert material["body_names"] == (".*",)
    assert material["backend_params"] is LOCOMOTION_MATERIAL_BACKEND_PARAMS
    backend_params = LOCOMOTION_MATERIAL_BACKEND_PARAMS
    assert backend_params["mjlab"]["shared_random"] is True
    assert backend_params["mjlab"]["separate_dynamic_friction"] is False
    assert backend_params["mjlab"]["restitution_range"] is None
    assert backend_params["isaacsim"]["shared_random"] is False
    assert backend_params["isaacsim"]["separate_dynamic_friction"] is True
    assert backend_params["isaacsim"]["restitution_range"] == (0.0, 0.8)
    assert backend_params["isaacsim"]["static_friction_range"] == (0.25, 0.8)
    assert backend_params["isaacsim"]["dynamic_friction_range"] == (0.2, 0.6)
    assert backend_params["isaacsim"]["num_buckets"] == 64
    assert backend_params["isaacsim"]["assign_per_shape"] is True
    assert Capability.DR_RESTITUTION in cfg.requirements.optional_capabilities
    assert Capability.DR_RESTITUTION not in cfg.requirements.capabilities
    assert cfg.events["add_base_mass"].params == {
        "body_names": ("torso_link",),
        "mass_range": (-5.0, 5.0),
        "operation": "add",
    }
    assert cfg.event_order[:2] == ("randomize_material", "add_base_mass")


def test_locomotion_flat_configuration_runs_with_mock_backend() -> None:
    cfg = locomotion_flat_env_cfg(num_envs=2)
    assert cfg.simulation.engine_options_for("mjlab") == {
        "njmax": 300,
        "solver": "newton",
        "iterations": 10,
        "ls_iterations": 20,
        "ccd_iterations": 500,
    }
    isaac_options = cfg.scene.backend_options_for("isaacsim")
    assert isaac_options["scene"]["lazy_sensor_update"] is True
    assert isaac_options["robot_spawn"]["self_collision"] is True
    assert isaac_options["robot_spawn"]["articulation_props"]["solver_position_iteration_count"] == 8
    assert cfg.simulation.engine_options_for("isaacsim") == {}
    backend = MockSimulatorBackend(device="cpu")
    env = UnifiedManagerBasedRLEnv(cfg, backend)
    assert backend.mass_properties is not None
    assert tuple(backend.mass_properties.mass.shape) == (2, 1)
    assert torch.all(backend.mass_properties.mass > 0.0)
    assert backend.material_properties is not None
    friction = backend.material_properties.sliding_friction
    dynamic = backend.material_properties.dynamic_friction
    written = tuple(cfg.scene.robot.body_names[int(index)] for index in backend.material_properties.body_ids.tolist())
    assert written == cfg.scene.robot.physical_body_names
    assert "LL_FOOT" not in written
    assert tuple(friction.shape) == (2, len(cfg.scene.robot.physical_body_names))
    assert dynamic is not None
    assert tuple(dynamic.shape) == friction.shape
    assert torch.all(friction >= 0.25)
    assert torch.all(friction <= 0.8)
    assert torch.all(dynamic >= 0.2)
    assert torch.all(dynamic <= 0.6)
    assert backend.material_properties.restitution is not None
    restitution = backend.material_properties.restitution
    assert torch.all(restitution >= 0.0)
    assert torch.all(restitution <= 0.8)

    num_joints = len(cfg.scene.robot.joint_names)
    observations, rewards, terminated, truncated, _ = env.step(torch.zeros((2, num_joints)))

    assert observations["policy"].shape == (2, 3 + 3 + 3 + num_joints * 3)
    assert observations["critic"].shape == (2, 3 + 3 + 3 + 3 + num_joints * 3)
    assert rewards.shape == (2, 1)
    assert terminated.shape == truncated.shape == (2,)
    env.close()


def test_material_dr_follows_backend_specific_policy() -> None:
    cfg = locomotion_flat_env_cfg(num_envs=4)
    isaac = MockSimulatorBackend(device="cpu")
    isaac.metadata = replace(isaac.metadata, name="isaacsim")
    mjlab = MockSimulatorBackend(device="cpu")
    mjlab.metadata = replace(mjlab.metadata, name="mjlab")
    env_isaac = UnifiedManagerBasedRLEnv(replace(cfg, seed=7), isaac)
    env_mjlab = UnifiedManagerBasedRLEnv(replace(cfg, seed=7), mjlab)
    try:
        assert isaac.material_properties is not None
        assert mjlab.material_properties is not None
        assert isaac.material_properties.dynamic_friction is not None
        assert isaac.material_properties.restitution is not None
        assert isaac.material_properties.layout == "shape"
        assert mjlab.material_properties.dynamic_friction is None
        assert mjlab.material_properties.restitution is None
        assert mjlab.material_properties.layout == "body"
        torch.testing.assert_close(
            mjlab.material_properties.sliding_friction,
            mjlab.material_properties.sliding_friction[:, :1].expand_as(mjlab.material_properties.sliding_friction),
        )
        assert not torch.equal(
            isaac.material_properties.sliding_friction,
            mjlab.material_properties.sliding_friction,
        )
        triples = torch.stack(
            (
                isaac.material_properties.sliding_friction,
                isaac.material_properties.dynamic_friction,
                isaac.material_properties.restitution,
            ),
            dim=-1,
        ).reshape(-1, 3)
        assert int(torch.unique(triples, dim=0).shape[0]) <= 64
    finally:
        env_isaac.close()
        env_mjlab.close()


def test_same_seed_mock_mass_and_material_samples_match() -> None:
    cfg = locomotion_flat_env_cfg(num_envs=4)
    first = MockSimulatorBackend(device="cpu")
    second = MockSimulatorBackend(device="cpu")
    env_first = UnifiedManagerBasedRLEnv(replace(cfg, seed=11), first)
    env_second = UnifiedManagerBasedRLEnv(replace(cfg, seed=11), second)
    try:
        assert first.mass_properties is not None
        assert second.mass_properties is not None
        assert first.material_properties is not None
        assert second.material_properties is not None
        torch.testing.assert_close(first.mass_properties.mass, second.mass_properties.mass)
        torch.testing.assert_close(
            first.material_properties.sliding_friction,
            second.material_properties.sliding_friction,
        )
        assert first.material_properties.dynamic_friction is not None
        assert second.material_properties.dynamic_friction is not None
        torch.testing.assert_close(
            first.material_properties.dynamic_friction,
            second.material_properties.dynamic_friction,
        )
        assert first.material_properties.restitution is not None
        assert second.material_properties.restitution is not None
        torch.testing.assert_close(
            first.material_properties.restitution,
            second.material_properties.restitution,
        )
    finally:
        env_first.close()
        env_second.close()


def test_isaacsim_material_dr_samples_per_shape_buckets() -> None:
    class MultiShapeMock(MockSimulatorBackend):
        def material_shape_counts(self, entity_name, body_ids):
            del entity_name
            return torch.full((body_ids.numel(),), 3, device=body_ids.device, dtype=torch.int64)

    cfg = locomotion_flat_env_cfg(num_envs=4)
    backend = MultiShapeMock(device="cpu")
    backend.metadata = replace(backend.metadata, name="isaacsim")
    env = UnifiedManagerBasedRLEnv(replace(cfg, seed=7), backend)
    try:
        assert backend.material_properties is not None
        n_shapes = 3 * len(cfg.scene.robot.physical_body_names)
        assert backend.material_properties.layout == "shape"
        assert tuple(backend.material_properties.sliding_friction.shape) == (4, n_shapes)
        assert backend.material_properties.dynamic_friction is not None
        assert backend.material_properties.restitution is not None
        static = backend.material_properties.sliding_friction
        dynamic = backend.material_properties.dynamic_friction
        assert not torch.equal(static[0], static[0, :1].expand_as(static[0]))
        triples = torch.stack(
            (static, dynamic, backend.material_properties.restitution),
            dim=-1,
        ).reshape(-1, 3)
        assert int(torch.unique(triples, dim=0).shape[0]) <= 64
    finally:
        env.close()


def test_material_dr_rejects_frames() -> None:
    cfg = locomotion_flat_env_cfg(num_envs=2)
    event = cfg.events["randomize_material"]
    cfg = replace(
        cfg,
        events={
            **cfg.events,
            "randomize_material": replace(
                event,
                params={**event.params, "body_names": ("torso_link", "LL_FOOT")},
            ),
        },
    )
    backend = MockSimulatorBackend(device="cpu")
    try:
        UnifiedManagerBasedRLEnv(cfg, backend)
    except ValueError as error:
        assert "frames" in str(error)
    else:
        raise AssertionError("expected material DR to reject frames")
    backend.close()


def test_contact_slide_uses_three_frame_force_history() -> None:
    feet = ("left_ankle_roll_link", "right_ankle_roll_link")
    sensor = ContactState.allocate(num_envs=2, body_names=feet, history_length=3, device="cpu")
    # Newest sample is history index 0. Env 0 only has 0.5 N on an older frame.
    sensor.net_forces_w_history[0, 2, 0] = torch.tensor([0.5, 0.0, 0.0])
    # Env 1 stays below the 0.1 N gate on every frame.
    sensor.net_forces_w_history[1, 0, 0] = torch.tensor([0.05, 0.0, 0.0])
    robot = SimpleNamespace(
        body_names=("torso_link", *feet),
        data=SimpleNamespace(
            body_lin_vel_w=torch.tensor(
                [
                    [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 0.0]],
                ]
            )
        ),
    )
    env = SimpleNamespace(
        device=torch.device("cpu"),
        cfg=SimpleNamespace(scene=SimpleNamespace(primary_entity="robot")),
        scene=SimpleNamespace(articulations={"robot": robot}, sensors={"feet_contact_forces": sensor}),
    )

    reward = contact_slide(env, "feet_contact_forces", feet, threshold=0.1)

    torch.testing.assert_close(reward, torch.tensor([5.0, 0.0]))
