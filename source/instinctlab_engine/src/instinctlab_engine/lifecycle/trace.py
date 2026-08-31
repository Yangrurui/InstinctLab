"""Episode trace capture and deterministic same-engine replay."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from .snapshot import (
    EnvironmentSnapshot,
    load_archive,
    save_archive,
)

if TYPE_CHECKING:
    from .runtime import LifecycleRuntime


class TraceError(RuntimeError):
    """A trace operation violates episode or artifact semantics."""


class ReplayMismatch(TraceError):
    """Strict replay observed a value different from the recorded transition."""


@dataclass(frozen=True)
class TraceStep:
    """One normalized policy-boundary transition."""

    action: torch.Tensor
    observation: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    timeout: torch.Tensor
    active: torch.Tensor

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {
            "action": self.action,
            "observation": self.observation,
            "reward": self.reward,
            "done": self.done,
            "timeout": self.timeout,
            "active": self.active,
        }


@dataclass(frozen=True)
class EpisodeTrace:
    """One episode per selected vector-environment index from a common snapshot."""

    schema_version: int
    engine: str
    task_id: str
    num_envs: int
    env_ids: torch.Tensor
    episode_ids: torch.Tensor
    initial_snapshot: EnvironmentSnapshot
    steps: tuple[TraceStep, ...]
    complete: bool

    def save(self, path: str | Path) -> Path:
        return save_archive(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> EpisodeTrace:
        value = load_archive(path)
        if not isinstance(value, dict):
            raise TraceError("Trace archive root must be a mapping.")
        try:
            snapshot = EnvironmentSnapshot(**value.pop("initial_snapshot"))
            steps = tuple(TraceStep(**step) for step in value.pop("steps"))
            trace = cls(initial_snapshot=snapshot, steps=steps, **value)
        except (KeyError, TypeError, ValueError) as exc:
            raise TraceError(f"Invalid episode trace document: {exc}") from exc
        trace.validate()
        return trace

    def validate(self) -> None:
        if self.schema_version != 1:
            raise TraceError(f"Unsupported trace schema {self.schema_version}.")
        if self.env_ids.dtype != torch.long or self.env_ids.ndim != 1:
            raise TraceError("Trace env_ids must be a one-dimensional int64 tensor.")
        if self.episode_ids.shape != self.env_ids.shape:
            raise TraceError("Trace episode_ids shape does not match env_ids.")
        if self.env_ids.numel() == 0:
            raise TraceError("Trace must select at least one environment.")
        if bool((self.env_ids < 0).any()) or bool((self.env_ids >= self.num_envs).any()):
            raise TraceError("Trace environment index is outside the vectorized environment.")
        if self.env_ids.unique().numel() != self.env_ids.numel():
            raise TraceError("Trace environment indices must be unique.")
        snapshot_identity = (
            self.initial_snapshot.engine,
            self.initial_snapshot.task_id,
            self.initial_snapshot.num_envs,
        )
        if snapshot_identity != (self.engine, self.task_id, self.num_envs):
            raise TraceError("Trace identity does not match its initial snapshot.")
        active = torch.ones(self.env_ids.numel(), dtype=torch.bool)
        for index, step in enumerate(self.steps):
            selected = self.env_ids.numel()
            if not all(
                isinstance(getattr(step, name), torch.Tensor)
                for name in (
                    "action",
                    "observation",
                    "reward",
                    "done",
                    "timeout",
                    "active",
                )
            ):
                raise TraceError(f"Trace step {index} fields must be tensors.")
            if step.action.shape[0] != self.num_envs:
                raise TraceError(f"Trace step {index} does not contain full-vector actions.")
            for name in ("observation", "reward", "done", "timeout", "active"):
                value = getattr(step, name)
                if value.shape[0] != selected:
                    raise TraceError(
                        f"Trace step {index} field {name} has the wrong environment dimension."
                    )
            if step.done.shape != (selected,) or step.timeout.shape != (selected,):
                raise TraceError(f"Trace step {index} done/timeout shapes are invalid.")
            if step.done.dtype != torch.bool or step.timeout.dtype != torch.bool:
                raise TraceError(f"Trace step {index} done/timeout fields must be boolean.")
            if step.active.dtype != torch.bool or step.active.shape != (selected,):
                raise TraceError(f"Trace step {index} active mask must be one-dimensional bool.")
            if not torch.equal(step.active, active):
                raise TraceError(f"Trace step {index} active mask is not sequential.")
            active &= ~step.done
        if self.complete != (not bool(active.any())):
            raise TraceError("Trace completeness does not match its transition sequence.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "engine": self.engine,
            "task_id": self.task_id,
            "num_envs": self.num_envs,
            "env_ids": self.env_ids,
            "episode_ids": self.episode_ids,
            "initial_snapshot": self.initial_snapshot.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
            "complete": self.complete,
        }


class EpisodeTraceRecorder:
    """Mutable recorder owned by one :class:`LifecycleRuntime`."""

    def __init__(
        self,
        runtime: LifecycleRuntime,
        env_ids: torch.Tensor,
        initial_snapshot: EnvironmentSnapshot,
    ) -> None:
        self.runtime = runtime
        self.env_ids = env_ids.detach().cpu().to(dtype=torch.long)
        self.episode_ids = runtime.episode_id[env_ids].detach().cpu().clone()
        self.initial_snapshot = initial_snapshot
        self.steps: list[TraceStep] = []
        self.active = torch.ones(self.env_ids.numel(), dtype=torch.bool)
        self.invalid_reason: str | None = None

    @property
    def complete(self) -> bool:
        return not bool(self.active.any())

    def invalidate(self, reason: str) -> None:
        self.invalid_reason = reason

    def record(
        self,
        *,
        actions: torch.Tensor,
        observations: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        timeouts: torch.Tensor | None,
    ) -> None:
        if self.invalid_reason is not None:
            raise TraceError(f"Active episode trace is invalid: {self.invalid_reason}.")
        if self.complete:
            return
        if actions.shape[0] != self.runtime.num_envs:
            raise TraceError("Trace actions do not cover the complete vectorized environment.")
        ids = self.env_ids.to(observations.device)
        selected_dones = dones.to(dtype=torch.bool)[ids]
        selected_timeouts = (
            torch.zeros_like(selected_dones)
            if timeouts is None
            else timeouts.to(dtype=torch.bool)[ids]
        )
        self.steps.append(
            TraceStep(
                action=actions.detach().cpu().clone(),
                observation=observations[ids].detach().cpu().clone(),
                reward=rewards[ids].detach().cpu().clone(),
                done=selected_dones.detach().cpu().clone(),
                timeout=selected_timeouts.detach().cpu().clone(),
                active=self.active.clone(),
            )
        )
        self.active &= ~selected_dones.detach().cpu()

    def finish(self, *, require_complete: bool) -> EpisodeTrace:
        if self.invalid_reason is not None:
            raise TraceError(f"Episode trace is invalid: {self.invalid_reason}.")
        if require_complete and not self.complete:
            pending = self.env_ids[self.active].tolist()
            raise TraceError(f"Selected episodes are not complete for env_ids={pending}.")
        trace = EpisodeTrace(
            schema_version=self.runtime.trace_schema_version,
            engine=self.runtime.engine,
            task_id=self.runtime.task_id,
            num_envs=self.runtime.num_envs,
            env_ids=self.env_ids.clone(),
            episode_ids=self.episode_ids.clone(),
            initial_snapshot=self.initial_snapshot,
            steps=tuple(self.steps),
            complete=self.complete,
        )
        trace.validate()
        return trace


@dataclass(frozen=True)
class ReplayDifference:
    step: int
    field: str
    max_absolute_error: float
    max_index: tuple[int, ...] | None = None
    actual_at_max: float | bool | None = None
    expected_at_max: float | bool | None = None


@dataclass(frozen=True)
class ReplayReport:
    matched: bool
    compared_steps: int
    differences: tuple[ReplayDifference, ...]


def replay_trace(
    env: Any,
    trace: EpisodeTrace,
    *,
    strict: bool = True,
    atol: float = 1.0e-5,
    rtol: float = 1.0e-5,
    field_tolerances: Mapping[str, tuple[float, float]] | None = None,
) -> ReplayReport:
    """Restore and replay a trace through the normalized RL environment boundary."""
    trace.validate()
    tolerances = _resolve_tolerances(atol, rtol, field_tolerances)
    lifecycle = env.lifecycle
    if lifecycle.trace_active:
        raise TraceError("Cannot replay while another episode trace is active.")
    if trace.schema_version != lifecycle.trace_schema_version:
        raise TraceError("Trace schema does not match the environment lifecycle.")
    lifecycle.restore(trace.initial_snapshot)
    ids = trace.env_ids.to(env.device)
    differences: list[ReplayDifference] = []
    for step_index, expected in enumerate(trace.steps):
        observation, reward, done, extras = env.step(expected.action.to(env.device))
        timeout = extras.get("time_outs")
        if timeout is None:
            timeout = torch.zeros_like(done, dtype=torch.bool)
        actual = {
            "observation": observation[ids],
            "reward": reward[ids],
            "done": done[ids].to(dtype=torch.bool),
            "timeout": timeout[ids].to(dtype=torch.bool),
        }
        active = expected.active
        for field, actual_value in actual.items():
            expected_value = getattr(expected, field)
            selected_actual = actual_value.detach().cpu()[active]
            selected_expected = expected_value[active]
            if field in {"done", "timeout"}:
                equal = torch.equal(selected_actual, selected_expected)
                error = 0.0 if equal else 1.0
            else:
                field_atol, field_rtol = tolerances[field]
                equal = torch.allclose(
                    selected_actual,
                    selected_expected,
                    atol=field_atol,
                    rtol=field_rtol,
                )
                error = _max_absolute_error(selected_actual, selected_expected)
            if not equal:
                max_index, actual_at_max, expected_at_max = _max_error_details(
                    selected_actual,
                    selected_expected,
                )
                difference = ReplayDifference(
                    step_index,
                    field,
                    error,
                    max_index,
                    actual_at_max,
                    expected_at_max,
                )
                differences.append(difference)
                if strict:
                    raise ReplayMismatch(
                        f"Replay differs at step {step_index} field {field}; "
                        f"max_abs_error={error:.6g}."
                    )
    return ReplayReport(
        matched=not differences,
        compared_steps=len(trace.steps),
        differences=tuple(differences),
    )


def _max_absolute_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    if actual.numel() == 0:
        return 0.0
    return float((actual - expected).abs().max().item())


def _max_error_details(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[tuple[int, ...] | None, float | bool | None, float | bool | None]:
    if actual.numel() == 0:
        return None, None, None
    if actual.dtype == torch.bool:
        flat_index = int((actual != expected).flatten().nonzero()[0].item())
    else:
        flat_index = int((actual - expected).abs().flatten().argmax().item())
    coordinates: list[int] = []
    remaining = flat_index
    for size in reversed(actual.shape):
        coordinates.append(remaining % size)
        remaining //= size
    index = tuple(reversed(coordinates))
    actual_value = actual.flatten()[flat_index].item()
    expected_value = expected.flatten()[flat_index].item()
    return index, actual_value, expected_value


def _resolve_tolerances(
    atol: float,
    rtol: float,
    field_tolerances: Mapping[str, tuple[float, float]] | None,
) -> dict[str, tuple[float, float]]:
    tolerances = {
        "observation": (float(atol), float(rtol)),
        "reward": (float(atol), float(rtol)),
    }
    if field_tolerances is not None:
        unknown = set(field_tolerances) - set(tolerances)
        if unknown:
            raise TraceError(f"Unknown replay tolerance fields: {sorted(unknown)}.")
        tolerances.update(field_tolerances)
    for field, values in tolerances.items():
        if len(values) != 2:
            raise TraceError(f"Replay tolerance for {field} must be (atol, rtol).")
        absolute, relative = (float(value) for value in values)
        if not all(math.isfinite(value) and value >= 0.0 for value in (absolute, relative)):
            raise TraceError(f"Replay tolerances for {field} must be finite and non-negative.")
        tolerances[field] = (absolute, relative)
    return tolerances


__all__ = [
    "EpisodeTrace",
    "EpisodeTraceRecorder",
    "ReplayDifference",
    "ReplayMismatch",
    "ReplayReport",
    "TraceError",
    "TraceStep",
    "replay_trace",
]
