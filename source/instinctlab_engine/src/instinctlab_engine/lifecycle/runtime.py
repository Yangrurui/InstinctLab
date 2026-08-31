"""Integer-clock lifecycle runtime attached to a native environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from .component import validate_stateful_component
from .snapshot import EnvironmentSnapshot, SnapshotError, SnapshotProvider

if TYPE_CHECKING:
    from instinctlab_engine.spec.lifecycle import (
        ResolvedClockDomain,
    )
    from instinctlab_engine.spec.task import TaskSpec


@dataclass(frozen=True, slots=True)
class ClockReading:
    """One named clock at a step boundary.

    ``tick`` and ``time_s`` are scalars for global clocks and tensors for clocks
    reset independently per environment.
    """

    name: str
    tick: Any
    time_s: Any


@dataclass(slots=True)
class _RegisteredComponent:
    value: object
    managed_reset: bool


def _clock_dict(clock: ResolvedClockDomain, physics_dt: float) -> dict[str, Any]:
    period = clock.period_physics_steps
    phase = clock.phase_physics_steps
    return {
        "period_physics_ticks": {
            "numerator": period.numerator,
            "denominator": period.denominator,
        },
        "phase_physics_ticks": {
            "numerator": phase.numerator,
            "denominator": phase.denominator,
        },
        "period_s": float(period) * physics_dt,
        "reset": clock.reset,
    }


def lifecycle_manifest(task: TaskSpec) -> dict[str, Any]:
    """Return a readable, JSON-safe resolved lifecycle contract."""
    clocks, components = task.lifecycle_contract()
    return {
        "trace_schema_version": task.lifecycle.trace_schema_version,
        "snapshot_schema_version": task.lifecycle.snapshot_schema_version,
        "clocks": {
            name: _clock_dict(clock, task.sim.physics_dt)
            for name, clock in sorted(clocks.items())
        },
        "components": {
            name: asdict(component)
            for name, component in sorted(components.items())
        },
    }


class LifecycleRuntime:
    """Clock and reset state for one vectorized native environment."""

    def __init__(self, env: Any, task: TaskSpec, *, engine: str):
        import torch

        self.env = env
        self.task_id = task.task_id
        self.engine = engine
        self.physics_dt = task.sim.physics_dt
        self.decimation = task.sim.decimation
        self.clocks, self.component_contracts = task.lifecycle_contract()
        self.trace_schema_version = task.lifecycle.trace_schema_version
        self.snapshot_schema_version = task.lifecycle.snapshot_schema_version
        self.num_envs = int(env.num_envs)
        self.device = getattr(env, "device", "cpu")

        self.physics_tick = 0
        self.policy_tick = 0
        self.episode_id = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.episode_physics_tick = torch.zeros_like(self.episode_id)
        self.reset_count = torch.zeros_like(self.episode_id)
        self._step_open = False
        self._reset_during_step = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._registered: dict[str, _RegisteredComponent] = {}
        self._snapshot_provider: SnapshotProvider | None = None

    @property
    def manifest(self) -> dict[str, Any]:
        return {
            "trace_schema_version": self.trace_schema_version,
            "snapshot_schema_version": self.snapshot_schema_version,
            "clocks": {
                name: _clock_dict(clock, self.physics_dt)
                for name, clock in sorted(self.clocks.items())
            },
            "components": {
                name: asdict(component)
                for name, component in sorted(self.component_contracts.items())
            },
        }

    def register_component(
        self,
        name: str,
        component: object,
        *,
        managed_reset: bool = False,
    ) -> None:
        """Bind the object implementing one declared component contract.

        Native managers keep ownership of their reset calls and register with
        ``managed_reset=False``. Application-owned components may opt into the
        lifecycle runtime invoking reset exactly once.
        """
        try:
            contract = self.component_contracts[name]
        except KeyError:
            raise KeyError(
                f"Cannot register undeclared lifecycle component {name!r}."
            ) from None
        existing = self._registered.get(name)
        if existing is not None and existing.value is not component:
            raise RuntimeError(
                f"Lifecycle component {name!r} is already bound to "
                f"{type(existing.value).__name__}."
            )
        if contract.state == "snapshot":
            validate_stateful_component(name, component)
        elif managed_reset and not callable(getattr(component, "reset", None)):
            raise RuntimeError(
                f"Managed lifecycle component {name!r} has no reset(env_ids) hook."
            )
        self._registered[name] = _RegisteredComponent(component, managed_reset)

    def set_snapshot_provider(self, provider: SnapshotProvider) -> None:
        """Bind the engine implementation that owns native state restoration."""
        if self._snapshot_provider is not None and self._snapshot_provider is not provider:
            raise RuntimeError("A lifecycle snapshot provider is already attached.")
        self._snapshot_provider = provider

    def snapshot(self, *, metadata: dict[str, Any] | None = None) -> EnvironmentSnapshot:
        """Capture a complete same-engine snapshot at a step boundary."""
        if self._step_open:
            raise SnapshotError("Cannot capture a snapshot while a policy step is open.")
        provider = self._require_snapshot_provider()
        component_states: dict[str, Any] = {}
        for name, registered in sorted(self._registered.items()):
            contract = self.component_contracts[name]
            if contract.state != "snapshot":
                continue
            component_states[name] = dict(registered.value.snapshot_state())  # type: ignore[attr-defined]
        return EnvironmentSnapshot(
            schema_version=self.snapshot_schema_version,
            engine=self.engine,
            task_id=self.task_id,
            num_envs=self.num_envs,
            provider_id=provider.provider_id,
            provider_version=provider.provider_version,
            native_state=dict(provider.capture()),
            lifecycle_state=self.state_dict(),
            component_states=component_states,
            metadata={} if metadata is None else dict(metadata),
        )

    def restore(self, snapshot: EnvironmentSnapshot) -> None:
        """Restore a compatible snapshot without invoking reset randomization."""
        if self._step_open:
            raise SnapshotError("Cannot restore a snapshot while a policy step is open.")
        provider = self._require_snapshot_provider()
        identity = (snapshot.engine, snapshot.task_id, snapshot.num_envs)
        expected_identity = (self.engine, self.task_id, self.num_envs)
        if identity != expected_identity:
            raise SnapshotError(
                "Snapshot environment identity does not match: "
                f"got {identity}, expected {expected_identity}."
            )
        if snapshot.schema_version != self.snapshot_schema_version:
            raise SnapshotError(
                f"Snapshot schema {snapshot.schema_version} is not supported; "
                f"expected {self.snapshot_schema_version}."
            )
        provider_identity = (snapshot.provider_id, snapshot.provider_version)
        expected_provider = (provider.provider_id, provider.provider_version)
        if provider_identity != expected_provider:
            raise SnapshotError(
                "Snapshot provider does not match: "
                f"got {provider_identity}, expected {expected_provider}."
            )
        expected_components = {
            name
            for name, registered in self._registered.items()
            if self.component_contracts[name].state == "snapshot"
        }
        if set(snapshot.component_states) != expected_components:
            raise SnapshotError(
                "Snapshot registered-component schema does not match: "
                f"got {sorted(snapshot.component_states)}, "
                f"expected {sorted(expected_components)}."
            )
        provider.restore(snapshot.native_state)
        self.load_state_dict(dict(snapshot.lifecycle_state))
        for name in sorted(expected_components):
            self._registered[name].value.restore_state(  # type: ignore[attr-defined]
                snapshot.component_states[name]
            )
        self._reset_during_step.zero_()

    def before_step(self) -> None:
        """Open one policy transition before native managers consume actions."""
        if self._step_open:
            raise RuntimeError("Lifecycle before_step() called twice without after_step().")
        self._step_open = True
        self._reset_during_step.zero_()

    def on_reset(self, env_ids: Any = None) -> None:
        """Record a native full/partial reset and reset managed components."""
        ids = self._resolve_env_ids(env_ids)
        self.episode_id[ids] += 1
        self.episode_physics_tick[ids] = 0
        self.reset_count[ids] += 1
        if self._step_open:
            self._reset_during_step[ids] = True

        for name, registered in self._registered.items():
            if not registered.managed_reset:
                continue
            contract = self.component_contracts[name]
            if contract.reset == "stateless":
                continue
            target_ids = None if contract.reset == "full" else ids
            registered.value.reset(target_ids)  # type: ignore[attr-defined]

    def after_step(self, dones: Any = None) -> None:
        """Close a transition, advancing all clocks exactly once."""
        import torch

        if not self._step_open:
            raise RuntimeError("Lifecycle after_step() called without before_step().")
        self.physics_tick += self.decimation
        self.policy_tick += 1

        reset_mask = self._reset_during_step.clone()
        if dones is not None:
            done_mask = torch.as_tensor(dones, device=self.device, dtype=torch.bool)
            if done_mask.ndim > 1:
                done_mask = done_mask.reshape(self.num_envs, -1).any(dim=1)
            missing_native_reset = done_mask & ~reset_mask
            if bool(missing_native_reset.any()):
                self.on_reset(missing_native_reset.nonzero(as_tuple=False).flatten())
                reset_mask |= missing_native_reset
        self.episode_physics_tick[~reset_mask] += self.decimation
        self._step_open = False

    def cancel_step(self) -> None:
        """Close a failed native step without advancing time."""
        self._step_open = False
        self._reset_during_step.zero_()

    def reading(self, name: str) -> ClockReading:
        """Read one named clock using exact rational integer arithmetic."""
        try:
            clock = self.clocks[name]
        except KeyError:
            raise KeyError(
                f"Unknown lifecycle clock {name!r}; declared: {sorted(self.clocks)}."
            ) from None
        elapsed = (
            self.episode_physics_tick
            if clock.reset == "episode"
            else self.physics_tick
        )
        tick = self._ticks(elapsed, clock)
        time_s = tick * (float(clock.period_physics_steps) * self.physics_dt)
        if hasattr(tick, "clone"):
            tick = tick.clone()
            time_s = time_s.clone()
        return ClockReading(name=name, tick=tick, time_s=time_s)

    def readings(self) -> dict[str, ClockReading]:
        return {name: self.reading(name) for name in sorted(self.clocks)}

    def state_dict(self) -> dict[str, Any]:
        """Copy clock state for inclusion in an environment snapshot."""
        return {
            "physics_tick": self.physics_tick,
            "policy_tick": self.policy_tick,
            "episode_id": self.episode_id.clone(),
            "episode_physics_tick": self.episode_physics_tick.clone(),
            "reset_count": self.reset_count.clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore clock state after validating its complete schema and shapes."""
        expected = {
            "physics_tick",
            "policy_tick",
            "episode_id",
            "episode_physics_tick",
            "reset_count",
        }
        if set(state) != expected:
            raise ValueError(
                f"Lifecycle clock snapshot fields are {sorted(state)}, expected {sorted(expected)}."
            )
        tensors = {
            name: state[name]
            for name in ("episode_id", "episode_physics_tick", "reset_count")
        }
        wrong = {
            name: tuple(value.shape)
            for name, value in tensors.items()
            if tuple(value.shape) != (self.num_envs,)
        }
        if wrong:
            raise ValueError(
                f"Lifecycle clock snapshot has incompatible environment shapes: {wrong}."
            )
        self.physics_tick = int(state["physics_tick"])
        self.policy_tick = int(state["policy_tick"])
        self.episode_id.copy_(state["episode_id"].to(self.device))
        self.episode_physics_tick.copy_(
            state["episode_physics_tick"].to(self.device)
        )
        self.reset_count.copy_(state["reset_count"].to(self.device))

    def _resolve_env_ids(self, env_ids: Any) -> Any:
        import torch

        if env_ids is None:
            return slice(None)
        if isinstance(env_ids, slice):
            return env_ids
        value = torch.as_tensor(env_ids, device=self.device)
        if value.dtype == torch.bool:
            if tuple(value.shape) != (self.num_envs,):
                raise ValueError(
                    f"Boolean reset mask has shape {tuple(value.shape)}, expected {(self.num_envs,)}."
                )
            return value
        return value.to(dtype=torch.long).flatten()

    def _require_snapshot_provider(self) -> SnapshotProvider:
        if self._snapshot_provider is None:
            raise SnapshotError(
                f"Engine {self.engine!r} did not attach a snapshot provider."
            )
        return self._snapshot_provider

    @staticmethod
    def _ticks(elapsed: Any, clock: ResolvedClockDomain) -> Any:
        period = clock.period_physics_steps
        phase = clock.phase_physics_steps
        common_denominator = math_lcm(period.denominator, phase.denominator)
        period_units = period.numerator * (
            common_denominator // period.denominator
        )
        phase_units = phase.numerator * (common_denominator // phase.denominator)
        elapsed_units = elapsed * common_denominator - phase_units
        if hasattr(elapsed_units, "clamp_min"):
            return elapsed_units.clamp_min(0) // period_units
        return max(0, elapsed_units) // period_units


def math_lcm(first: int, second: int) -> int:
    """Small local LCM to keep the runtime import-safe on supported Python."""
    import math

    return math.lcm(first, second)


def attach_lifecycle(env: Any, task: TaskSpec, *, engine: str) -> LifecycleRuntime:
    """Attach exactly one lifecycle runtime to a constructed native environment."""
    existing = getattr(env, "lifecycle", None)
    if existing is not None:
        if existing.task_id != task.task_id or existing.engine != engine:
            raise RuntimeError(
                "A native environment cannot be rebound to a different lifecycle contract."
            )
        return existing
    runtime = LifecycleRuntime(env, task, engine=engine)
    env.lifecycle = runtime
    return runtime


__all__ = [
    "ClockReading",
    "LifecycleRuntime",
    "attach_lifecycle",
    "lifecycle_manifest",
]
