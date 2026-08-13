"""Small backend-independent managers used by the unified RL environment."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from instinctlab.sim.control import ControlMode, JointControlTarget
from instinctlab.sim.rng import RngManager
from instinctlab.sim.schema import ObservationGroupSchema, TensorSegment

TermFunction = Callable[..., torch.Tensor]
EventFunction = Callable[..., object]


def _frozen_items(
    values: Mapping[str, Any],
    order: Sequence[str],
    *,
    context: str,
) -> tuple[tuple[str, Any], ...]:
    frozen_order = tuple(order)
    if len(set(frozen_order)) != len(frozen_order):
        raise ValueError(f"{context} order contains duplicate names")
    missing = set(values).difference(frozen_order)
    unknown = set(frozen_order).difference(values)
    if missing or unknown:
        raise ValueError(f"{context} order mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}")
    return tuple((name, values[name]) for name in frozen_order)


def _env_ids(env: Any, env_ids: torch.Tensor | Sequence[int] | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.int64)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.int64)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.int64)


def _vector(value: torch.Tensor, env: Any, *, context: str) -> torch.Tensor:
    value = torch.as_tensor(value, device=env.device)
    if value.shape == (env.num_envs, 1):
        value = value[:, 0]
    if value.shape != (env.num_envs,):
        raise ValueError(f"{context} must return shape ({env.num_envs},) or ({env.num_envs}, 1), got {tuple(value.shape)}")
    return value


@dataclass(frozen=True)
class UniformNoiseCfg:
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError("uniform noise bounds are reversed")


@dataclass(frozen=True)
class ActionTermCfg:
    class_type: type["ActionTerm"]


@dataclass(frozen=True)
class JointPositionActionCfg:
    entity_name: str = "robot"
    clip: tuple[float, float] | None = None
    class_type: type["ActionTerm"] = field(default_factory=lambda: JointPositionAction, init=False)


class ActionTerm(ABC):
    def __init__(self, cfg: ActionTermCfg | JointPositionActionCfg, env: Any) -> None:
        self.cfg = cfg
        self.env = env

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Number of policy action columns consumed by this term."""

    @abstractmethod
    def process_actions(self, actions: torch.Tensor) -> None:
        """Process and hold one policy action."""

    @abstractmethod
    def apply_actions(self) -> None:
        """Apply the held action for one physics step."""

    def reset(self, env_ids: torch.Tensor) -> None:
        del env_ids


class JointPositionAction(ActionTerm):
    """Canonical DFS joint-position action using values from ``RobotSpec``."""

    cfg: JointPositionActionCfg

    def __init__(self, cfg: JointPositionActionCfg, env: Any) -> None:
        super().__init__(cfg, env)
        robot = env.cfg.scene.robot
        values = robot.materialize(device=env.device)
        self._default_pos = values["default_pos"].unsqueeze(0)
        self._action_scale = values["action_scale"].unsqueeze(0)
        self.raw_actions = torch.zeros((env.num_envs, len(robot.joint_names)), device=env.device)
        self.processed_actions = torch.zeros_like(self.raw_actions)
        self.control_target = JointControlTarget(
            mode=ControlMode.POSITION,
            value=self.processed_actions,
            velocity=torch.zeros_like(self.processed_actions),
        )

    @property
    def action_dim(self) -> int:
        return self.raw_actions.shape[1]

    def process_actions(self, actions: torch.Tensor) -> None:
        actions = actions.to(device=self.env.device, dtype=torch.float32)
        if self.cfg.clip is not None:
            actions = actions.clamp(*self.cfg.clip)
        self.raw_actions.copy_(actions)
        self.processed_actions.copy_(self._default_pos + self._action_scale * actions)

    def apply_actions(self) -> None:
        self.env.backend.set_joint_control_target(self.cfg.entity_name, self.control_target)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.raw_actions[env_ids] = 0.0
        self.processed_actions[env_ids] = self._default_pos


class ActionManager:
    def __init__(
        self,
        cfg: Mapping[str, ActionTermCfg | JointPositionActionCfg],
        env: Any,
        *,
        term_order: Sequence[str] | None = None,
    ) -> None:
        self.env = env
        order = tuple(cfg) if term_order is None else tuple(term_order)
        items = _frozen_items(cfg, order, context="action")
        self._terms = tuple((name, term_cfg.class_type(term_cfg, env)) for name, term_cfg in items)
        self.term_names = tuple(name for name, _ in self._terms)
        self.action = torch.zeros((env.num_envs, self.total_action_dim), device=env.device)
        self.prev_action = torch.zeros_like(self.action)

    @property
    def total_action_dim(self) -> int:
        return sum(term.action_dim for _, term in self._terms)

    def process_action(self, action: torch.Tensor) -> None:
        action = action.to(device=self.env.device, dtype=torch.float32)
        expected = (self.env.num_envs, self.total_action_dim)
        if action.shape != expected:
            raise ValueError(f"action has shape {tuple(action.shape)}, expected {expected}")
        self.prev_action.copy_(self.action)
        self.action.copy_(action)
        offset = 0
        for _, term in self._terms:
            term.process_actions(action[:, offset : offset + term.action_dim])
            offset += term.action_dim

    def apply_action(self) -> None:
        for _, term in self._terms:
            term.apply_actions()

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> None:
        ids = _env_ids(self.env, env_ids)
        self.action[ids] = 0.0
        self.prev_action[ids] = 0.0
        for _, term in self._terms:
            term.reset(ids)

    def get_term(self, name: str) -> ActionTerm:
        return dict(self._terms)[name]


@dataclass(frozen=True)
class ObservationTermCfg:
    func: TermFunction
    params: Mapping[str, Any] = field(default_factory=dict)
    noise: UniformNoiseCfg | None = None
    scale: float | torch.Tensor = 1.0
    clip: tuple[float, float] | None = None
    shape: tuple[int, ...] | None = None
    semantic: str = ""


@dataclass(frozen=True)
class ObservationGroupCfg:
    terms: Mapping[str, ObservationTermCfg]
    term_order: tuple[str, ...]
    concatenate_terms: bool = True
    enable_corruption: bool = True

    def __post_init__(self) -> None:
        _frozen_items(self.terms, self.term_order, context="observation group")


class ObservationManager:
    def __init__(
        self,
        cfg: Mapping[str, ObservationGroupCfg],
        env: Any,
        rng: RngManager,
        *,
        group_order: Sequence[str] | None = None,
    ) -> None:
        self.env = env
        self.rng = rng
        order = tuple(cfg) if group_order is None else tuple(group_order)
        self._groups = _frozen_items(cfg, order, context="observation group")
        self.group_names = tuple(name for name, _ in self._groups)
        self.group_term_names = {name: group.term_order for name, group in self._groups}
        self.group_obs_dim: dict[str, int] = {}
        self.group_schemas: dict[str, ObservationGroupSchema] = {}

    @property
    def active_terms(self) -> dict[str, tuple[str, ...]]:
        return self.group_term_names

    @property
    def group_obs_term_dim(self) -> dict[str, list[tuple[int, ...]]]:
        return {
            group_name: [segment.shape for segment in self.group_schemas[group_name].segments]
            for group_name in self.group_names
            if group_name in self.group_schemas
        }

    def compute(self) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        observations: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {}
        for group_name, group_cfg in self._groups:
            term_values: dict[str, torch.Tensor] = {}
            segments: list[TensorSegment] = []
            for term_name, term_cfg in _frozen_items(
                group_cfg.terms, group_cfg.term_order, context=f"observation group {group_name!r}"
            ):
                value = torch.as_tensor(term_cfg.func(self.env, **term_cfg.params), device=self.env.device)
                if value.ndim < 2 or value.shape[0] != self.env.num_envs:
                    raise ValueError(
                        f"observation {group_name}.{term_name} must have leading shape ({self.env.num_envs}, ...)"
                    )
                value = value * torch.as_tensor(term_cfg.scale, device=self.env.device)
                if group_cfg.enable_corruption and term_cfg.noise is not None:
                    noise = self.rng.uniform(
                        f"observation_noise.{group_name}.{term_name}",
                        term_cfg.noise.low,
                        term_cfg.noise.high,
                        tuple(value.shape),
                        dtype=value.dtype,
                    )
                    value = value + noise
                if term_cfg.clip is not None:
                    value = value.clamp(*term_cfg.clip)
                actual_shape = tuple(value.shape[1:])
                if term_cfg.shape is not None and actual_shape != term_cfg.shape:
                    raise ValueError(
                        f"observation {group_name}.{term_name} has shape {actual_shape}, expected {term_cfg.shape}"
                    )
                term_values[term_name] = value
                segments.append(TensorSegment(term_name, actual_shape, term_cfg.semantic))
            schema = ObservationGroupSchema(group_name, tuple(segments))
            self.group_schemas[group_name] = schema
            self.group_obs_dim[group_name] = schema.flat_dim
            if group_cfg.concatenate_terms:
                observations[group_name] = torch.cat(
                    [term_values[name].reshape(self.env.num_envs, -1) for name in group_cfg.term_order], dim=-1
                )
            else:
                observations[group_name] = term_values
        return observations

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> None:
        del env_ids


@dataclass(frozen=True)
class RewardTermCfg:
    func: TermFunction
    weight: float = 1.0
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RewardGroupCfg:
    terms: Mapping[str, RewardTermCfg]
    term_order: tuple[str, ...]

    def __post_init__(self) -> None:
        _frozen_items(self.terms, self.term_order, context="reward group")


class RewardManager:
    def __init__(
        self,
        cfg: Mapping[str, RewardGroupCfg],
        env: Any,
        *,
        group_order: Sequence[str] | None = None,
    ) -> None:
        self.env = env
        order = tuple(cfg) if group_order is None else tuple(group_order)
        self._groups = _frozen_items(cfg, order, context="reward group")
        self.group_names = tuple(name for name, _ in self._groups)
        self.group_term_names = {name: group.term_order for name, group in self._groups}
        self._episode_sums = {
            (group_name, term_name): torch.zeros(env.num_envs, device=env.device)
            for group_name, group in self._groups
            for term_name in group.term_order
        }

    @property
    def num_rewards(self) -> int:
        return len(self._groups)

    def compute(self, dt: float) -> torch.Tensor:
        groups: list[torch.Tensor] = []
        for group_name, group_cfg in self._groups:
            group_value = torch.zeros(self.env.num_envs, device=self.env.device)
            for term_name, term_cfg in _frozen_items(
                group_cfg.terms, group_cfg.term_order, context=f"reward group {group_name!r}"
            ):
                value = _vector(term_cfg.func(self.env, **term_cfg.params), self.env, context=f"reward {term_name}")
                weighted = value * term_cfg.weight * dt
                group_value += weighted
                self._episode_sums[(group_name, term_name)] += weighted
            groups.append(group_value)
        if not groups:
            return torch.zeros((self.env.num_envs, 0), device=self.env.device)
        return torch.stack(groups, dim=-1)

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        ids = _env_ids(self.env, env_ids)
        result = {
            f"{group}/{term}": values[ids].mean().clone()
            for (group, term), values in self._episode_sums.items()
            if ids.numel()
        }
        for values in self._episode_sums.values():
            values[ids] = 0.0
        return result


@dataclass(frozen=True)
class TerminationTermCfg:
    func: TermFunction
    params: Mapping[str, Any] = field(default_factory=dict)
    time_out: bool = False


@dataclass(frozen=True)
class TerminationGroupCfg:
    terms: Mapping[str, TerminationTermCfg]
    term_order: tuple[str, ...]

    def __post_init__(self) -> None:
        _frozen_items(self.terms, self.term_order, context="termination")


class TerminationManager:
    def __init__(self, cfg: TerminationGroupCfg, env: Any) -> None:
        self.env = env
        self._terms = _frozen_items(cfg.terms, cfg.term_order, context="termination")
        self.term_names = cfg.term_order
        self.terminated = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.time_outs = torch.zeros_like(self.terminated)

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.terminated.zero_()
        self.time_outs.zero_()
        for name, cfg in self._terms:
            value = _vector(cfg.func(self.env, **cfg.params), self.env, context=f"termination {name}").bool()
            if cfg.time_out:
                self.time_outs |= value
            else:
                self.terminated |= value
        self.time_outs |= self.env.episode_length_buf >= self.env.max_episode_length
        return self.terminated, self.time_outs

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> None:
        ids = _env_ids(self.env, env_ids)
        self.terminated[ids] = False
        self.time_outs[ids] = False


@dataclass(frozen=True)
class EventTermCfg:
    func: EventFunction
    mode: str
    params: Mapping[str, Any] = field(default_factory=dict)
    interval_range_s: tuple[float, float] | None = None
    writes_state: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"startup", "reset", "interval"}:
            raise ValueError(f"unsupported event mode {self.mode!r}")
        if self.mode == "interval":
            if self.interval_range_s is None:
                raise ValueError("interval events require interval_range_s")
            if self.interval_range_s[0] <= 0.0 or self.interval_range_s[1] < self.interval_range_s[0]:
                raise ValueError("invalid interval_range_s")
        elif self.interval_range_s is not None:
            raise ValueError("interval_range_s is only valid for interval events")


class EventManager:
    def __init__(
        self,
        cfg: Mapping[str, EventTermCfg],
        env: Any,
        rng: RngManager,
        *,
        term_order: Sequence[str] | None = None,
    ) -> None:
        self.env = env
        self.rng = rng
        order = tuple(cfg) if term_order is None else tuple(term_order)
        self._terms = _frozen_items(cfg, order, context="event")
        self.term_names = tuple(name for name, _ in self._terms)
        self._time_left = {
            name: torch.zeros(env.num_envs, device=env.device)
            for name, term_cfg in self._terms
            if term_cfg.mode == "interval"
        }
        self.reset_timers(None)

    def _sample_interval(self, name: str, cfg: EventTermCfg, count: int) -> torch.Tensor:
        assert cfg.interval_range_s is not None
        return self.rng.uniform(
            f"event_interval.{name}",
            cfg.interval_range_s[0],
            cfg.interval_range_s[1],
            (count,),
        )

    def reset_timers(self, env_ids: torch.Tensor | Sequence[int] | None) -> None:
        ids = _env_ids(self.env, env_ids)
        for name, cfg in self._terms:
            if cfg.mode == "interval":
                self._time_left[name][ids] = self._sample_interval(name, cfg, ids.numel())

    def apply(
        self,
        mode: str,
        *,
        env_ids: torch.Tensor | Sequence[int] | None = None,
        dt: float | None = None,
    ) -> bool:
        ids = _env_ids(self.env, env_ids)
        wrote_state = False
        for name, cfg in self._terms:
            if cfg.mode != mode:
                continue
            call_ids = ids
            if mode == "interval":
                if dt is None:
                    raise ValueError("interval event application requires dt")
                self._time_left[name][ids] -= dt
                due = self._time_left[name][ids] <= 0.0
                call_ids = ids[due]
                if not call_ids.numel():
                    continue
            result = cfg.func(self.env, env_ids=call_ids, **cfg.params)
            wrote_state = wrote_state or cfg.writes_state or (isinstance(result, bool) and result)
            if mode == "interval":
                self._time_left[name][call_ids] = self._sample_interval(name, cfg, call_ids.numel())
        return wrote_state

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> None:
        self.reset_timers(env_ids)


@dataclass(frozen=True)
class CommandTermCfg:
    class_type: type["CommandTerm"]
    params: Mapping[str, Any] = field(default_factory=dict)


class CommandTerm(ABC):
    """Interface for one batched command generator."""

    def __init__(self, cfg: CommandTermCfg, env: Any) -> None:
        self.cfg = cfg
        self.env = env

    @abstractmethod
    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset command state for selected environments."""

    @abstractmethod
    def compute(self, dt: float) -> None:
        """Advance command state by one policy step."""

    @property
    @abstractmethod
    def command(self) -> torch.Tensor:
        """Current batched command tensor."""


class CommandManager:
    def __init__(
        self,
        cfg: Mapping[str, CommandTermCfg | CommandTerm],
        env: Any,
        *,
        term_order: Sequence[str] | None = None,
    ) -> None:
        self.env = env
        order = tuple(cfg) if term_order is None else tuple(term_order)
        items = _frozen_items(cfg, order, context="command")
        terms: list[tuple[str, CommandTerm]] = []
        for name, value in items:
            term = value if isinstance(value, CommandTerm) else value.class_type(value, env)
            terms.append((name, term))
        self._terms = tuple(terms)
        self.term_names = tuple(name for name, _ in self._terms)

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> None:
        ids = _env_ids(self.env, env_ids)
        for _, term in self._terms:
            term.reset(ids)

    def compute(self, dt: float) -> None:
        for _, term in self._terms:
            term.compute(dt)

    def get_command(self, name: str) -> torch.Tensor:
        return dict(self._terms)[name].command


class CurriculumManager:
    """Usable no-op curriculum manager."""

    def __init__(self, cfg: object | None, env: Any) -> None:
        self.cfg = cfg
        self.env = env

    def compute(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        del env_ids
        return {}

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        del env_ids
        return {}


class MonitorManager:
    """Usable no-op monitor manager."""

    def __init__(self, cfg: object | None, env: Any) -> None:
        self.cfg = cfg
        self.env = env

    def update(self, dt: float) -> dict[str, torch.Tensor]:
        del dt
        return {}

    def reset(
        self,
        env_ids: torch.Tensor | Sequence[int] | None = None,
        *,
        is_episode: bool = False,
    ) -> dict[str, torch.Tensor]:
        del env_ids, is_episode
        return {}


__all__ = [
    "ActionManager",
    "ActionTerm",
    "ActionTermCfg",
    "CommandManager",
    "CommandTerm",
    "CommandTermCfg",
    "CurriculumManager",
    "EventManager",
    "EventTermCfg",
    "JointPositionAction",
    "JointPositionActionCfg",
    "MonitorManager",
    "ObservationGroupCfg",
    "ObservationManager",
    "ObservationTermCfg",
    "RewardGroupCfg",
    "RewardManager",
    "RewardTermCfg",
    "TerminationGroupCfg",
    "TerminationManager",
    "TerminationTermCfg",
    "UniformNoiseCfg",
]
