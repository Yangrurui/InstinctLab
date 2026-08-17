"""Backend-independent manager-based reinforcement-learning environment."""

from __future__ import annotations

import math
import torch
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from instinctlab.managers.unified import (
    ActionManager,
    ActionTermCfg,
    CommandManager,
    CommandTerm,
    CommandTermCfg,
    CurriculumManager,
    EventManager,
    EventTermCfg,
    JointPositionActionCfg,
    MonitorManager,
    ObservationGroupCfg,
    ObservationManager,
    RewardGroupCfg,
    RewardManager,
    TerminationGroupCfg,
    TerminationManager,
)
from instinctlab.sim.backend import RuntimeRequirements, SensorReadPhase, SimulatorBackend
from instinctlab.sim.rng import RngManager
from instinctlab.sim.scene import SceneSpec, SimulationSpec


@dataclass(frozen=True)
class UnifiedManagerBasedRLEnvCfg:
    """Complete public configuration for the minimal unified environment."""

    scene: SceneSpec
    actions: Mapping[str, ActionTermCfg | JointPositionActionCfg]
    observations: Mapping[str, ObservationGroupCfg]
    rewards: Mapping[str, RewardGroupCfg]
    terminations: TerminationGroupCfg = field(default_factory=lambda: TerminationGroupCfg(terms={}, term_order=()))
    events: Mapping[str, EventTermCfg] = field(default_factory=dict)
    commands: Mapping[str, CommandTermCfg | CommandTerm] = field(default_factory=dict)
    simulation: SimulationSpec = field(default_factory=SimulationSpec)
    requirements: RuntimeRequirements = field(default_factory=lambda: RuntimeRequirements(capabilities=frozenset()))
    episode_length_s: float = 20.0
    is_finite_horizon: bool = False
    seed: int = 0
    action_order: tuple[str, ...] | None = None
    observation_group_order: tuple[str, ...] | None = None
    reward_group_order: tuple[str, ...] | None = None
    event_order: tuple[str, ...] | None = None
    command_order: tuple[str, ...] | None = None
    curriculum: object | None = None
    monitors: object | None = None

    def __post_init__(self) -> None:
        if self.episode_length_s <= 0.0:
            raise ValueError("episode_length_s must be positive")
        self.scene.validate()
        self.simulation.validate()


class UnifiedManagerBasedRLEnv:
    """A small Gym-style vector environment over ``SimulatorBackend``."""

    def __init__(self, cfg: UnifiedManagerBasedRLEnvCfg, backend: SimulatorBackend) -> None:
        self.cfg = cfg
        self.backend = backend
        self.backend.initialize(cfg.scene, cfg.simulation, cfg.requirements)
        self.cfg.requirements.validate_backend_metadata(self.backend.metadata)

        self.num_envs = backend.num_envs
        self.device = backend.device
        self.step_dt = cfg.simulation.policy_dt
        self.max_episode_length = math.ceil(cfg.episode_length_s / self.step_dt)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        self.common_step_counter = 0
        self.extras: dict[str, Any] = {
            "log": {},
            "step": {},
            "episode": {},
            "observations": {},
            "time_outs": torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
        }
        self.rng_manager = RngManager(cfg.seed, self.device)

        self.action_manager = ActionManager(
            cfg.actions,
            self,
            term_order=cfg.action_order,
        )
        self.observation_manager = ObservationManager(
            cfg.observations,
            self,
            self.rng_manager,
            group_order=cfg.observation_group_order,
        )
        self.reward_manager = RewardManager(
            cfg.rewards,
            self,
            group_order=cfg.reward_group_order,
        )
        self.termination_manager = TerminationManager(cfg.terminations, self)
        self.event_manager = EventManager(
            cfg.events,
            self,
            self.rng_manager,
            term_order=cfg.event_order,
        )
        self.command_manager = CommandManager(
            cfg.commands,
            self,
            term_order=cfg.command_order,
        )
        self.curriculum_manager = CurriculumManager(cfg.curriculum, self)
        self.monitor_manager = MonitorManager(cfg.monitors, self)

        if self.event_manager.apply("startup"):
            self.backend.synchronize(SensorReadPhase.POST_EVENT)
        self.reset()

    @property
    def scene(self) -> Any:
        return self.backend.scene

    @property
    def num_rewards(self) -> int:
        return self.reward_manager.num_rewards

    @property
    def rng(self) -> RngManager:
        return self.rng_manager

    def seed(self, seed: int) -> int:
        self.rng_manager = RngManager(seed, self.device)
        self.observation_manager.rng = self.rng_manager
        self.event_manager.rng = self.rng_manager
        self.event_manager.reset_timers(None)
        return seed

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, torch.Tensor | dict[str, torch.Tensor]], dict[str, Any]]:
        del options
        if seed is not None:
            self.seed(seed)
        self._reset_idx(torch.arange(self.num_envs, device=self.device, dtype=torch.int64))
        observations = self.observation_manager.compute()
        self.extras["observations"] = observations
        return observations, self.extras

    def _reset_idx(self, env_ids: torch.Tensor | Sequence[int]) -> None:
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.int64)
        if ids.ndim != 1:
            raise ValueError("env_ids must be one-dimensional")
        if not ids.numel():
            return

        self.backend.reset(ids)
        self.event_manager.apply("reset", env_ids=ids)
        self.backend.synchronize(SensorReadPhase.POST_RESET)

        self.extras["episode"].update(self.reward_manager.reset(ids))
        self.curriculum_manager.reset(ids)
        self.action_manager.reset(ids)
        self.observation_manager.reset(ids)
        self.termination_manager.reset(ids)
        self.event_manager.reset(ids)
        self.command_manager.reset(ids)
        self.monitor_manager.reset(ids, is_episode=True)
        self.command_manager.compute(0.0)
        self.episode_length_buf[ids] = 0

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        dict[str, torch.Tensor | dict[str, torch.Tensor]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        self.action_manager.process_action(action)
        for _ in range(self.cfg.simulation.decimation):
            self.action_manager.apply_action()
            self.backend.step()

        self.backend.synchronize(SensorReadPhase.POST_PHYSICS)
        self.episode_length_buf += 1
        self.common_step_counter += 1

        terminated_view, time_out_view = self.termination_manager.compute()
        terminated = terminated_view.clone()
        time_outs = time_out_view.clone()
        rewards = self.reward_manager.compute(self.step_dt)

        done = terminated | time_outs
        done_ids = done.nonzero(as_tuple=False).flatten()
        if done_ids.numel():
            self.curriculum_manager.compute(done_ids)
            self._reset_idx(done_ids)

        self.command_manager.compute(self.step_dt)
        events_wrote_state = self.event_manager.apply("interval", dt=self.step_dt)
        if events_wrote_state:
            self.backend.synchronize(SensorReadPhase.POST_EVENT)

        self.extras["step"].update(self.monitor_manager.update(self.step_dt))
        self.extras["time_outs"] = time_outs
        observations = self.observation_manager.compute()
        self.extras["observations"] = observations
        return observations, rewards, terminated, time_outs, self.extras

    def render(self, mode: str = "human") -> object | None:
        return self.backend.render(mode)

    def close(self) -> None:
        self.backend.close()


UnifiedManagerBasedRlEnv = UnifiedManagerBasedRLEnv
UnifiedManagerBasedRlEnvCfg = UnifiedManagerBasedRLEnvCfg
InstinctManagerBasedRLEnv = UnifiedManagerBasedRLEnv
InstinctManagerBasedRlEnv = UnifiedManagerBasedRLEnv


__all__ = [
    "InstinctManagerBasedRLEnv",
    "InstinctManagerBasedRlEnv",
    "UnifiedManagerBasedRLEnv",
    "UnifiedManagerBasedRLEnvCfg",
    "UnifiedManagerBasedRlEnv",
    "UnifiedManagerBasedRlEnvCfg",
]
