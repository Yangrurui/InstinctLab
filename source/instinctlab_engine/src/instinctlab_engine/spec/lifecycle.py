"""Engine-neutral timing, reset, and recoverable-component contracts.

Lifecycle declarations use integer clock ratios.  Floating point time is only a
presentation value; scheduling is derived from physics ticks so a long run does
not drift differently on two engines.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .task import SimSpec, TaskSpec

ClockReset = Literal["never", "episode"]
ComponentPhase = Literal[
    "startup",
    "pre_step",
    "pre_physics",
    "post_physics",
    "post_step",
    "on_reset",
    "on_demand",
]
ComponentReset = Literal["stateless", "full", "partial"]
ComponentState = Literal["stateless", "snapshot"]

_BUILTIN_CLOCKS = frozenset({"physics", "policy", "episode"})


def _is_dotted_name(value: str) -> bool:
    return bool(value) and all(part.isidentifier() for part in value.split("."))


@dataclass(frozen=True, slots=True)
class ClockDomainSpec:
    """A named integer subdivision of another clock.

    ``tick_divider=4`` means one tick here for every four ticks of ``parent``.
    ``phase`` is expressed in parent ticks and must be smaller than the divider.
    The built-in ``physics``, ``policy``, and ``episode`` clocks are derived from
    :class:`SimSpec`; declarations add clocks below those roots.
    """

    name: str
    parent: str = "physics"
    tick_divider: int = 1
    phase: int = 0
    reset: ClockReset = "never"

    def __post_init__(self) -> None:
        if not _is_dotted_name(self.name):
            raise ValueError(
                f"Clock domain names must be dotted identifiers, got {self.name!r}."
            )
        if self.name in _BUILTIN_CLOCKS:
            raise ValueError(
                f"Clock domain {self.name!r} is built in and cannot be redeclared."
            )
        if not _is_dotted_name(self.parent):
            raise ValueError(
                f"Clock parent names must be dotted identifiers, got {self.parent!r}."
            )
        if isinstance(self.tick_divider, bool) or self.tick_divider < 1:
            raise ValueError("Clock tick_divider must be a positive integer.")
        if isinstance(self.phase, bool) or not 0 <= self.phase < self.tick_divider:
            raise ValueError(
                f"Clock phase must satisfy 0 <= phase < tick_divider, got "
                f"phase={self.phase}, divider={self.tick_divider}."
            )
        if self.reset not in {"never", "episode"}:
            raise ValueError(f"Unknown clock reset semantics {self.reset!r}.")


@dataclass(frozen=True, slots=True)
class ResolvedClockDomain:
    """A clock reduced to exact physics-step units."""

    name: str
    period_numerator: int
    period_denominator: int
    phase_numerator: int
    phase_denominator: int
    reset: ClockReset

    def __post_init__(self) -> None:
        period = self.period_physics_steps
        phase = self.phase_physics_steps
        if period <= 0:
            raise ValueError("Resolved clock periods must be positive.")
        if not 0 <= phase < period:
            raise ValueError("Resolved clock phase is outside its period.")

    @classmethod
    def from_fractions(
        cls,
        name: str,
        period: Fraction,
        phase: Fraction,
        reset: ClockReset,
    ) -> ResolvedClockDomain:
        return cls(
            name=name,
            period_numerator=period.numerator,
            period_denominator=period.denominator,
            phase_numerator=phase.numerator,
            phase_denominator=phase.denominator,
            reset=reset,
        )

    @property
    def period_physics_steps(self) -> Fraction:
        return Fraction(self.period_numerator, self.period_denominator)

    @property
    def phase_physics_steps(self) -> Fraction:
        return Fraction(self.phase_numerator, self.phase_denominator)


@dataclass(frozen=True, slots=True)
class ComponentLifecycleSpec:
    """When a component runs, how it resets, and whether state is recoverable."""

    clock: str
    phase: ComponentPhase
    reset: ComponentReset
    state: ComponentState
    latency_ticks: int = 0

    def __post_init__(self) -> None:
        if not _is_dotted_name(self.clock):
            raise ValueError(
                f"Component clock must be a dotted identifier, got {self.clock!r}."
            )
        if self.phase not in {
            "startup",
            "pre_step",
            "pre_physics",
            "post_physics",
            "post_step",
            "on_reset",
            "on_demand",
        }:
            raise ValueError(f"Unknown component phase {self.phase!r}.")
        if self.reset not in {"stateless", "full", "partial"}:
            raise ValueError(f"Unknown component reset semantics {self.reset!r}.")
        if self.state not in {"stateless", "snapshot"}:
            raise ValueError(f"Unknown component state semantics {self.state!r}.")
        if isinstance(self.latency_ticks, bool) or self.latency_ticks < 0:
            raise ValueError("Component latency_ticks must be a non-negative integer.")
        if self.state == "snapshot" and self.reset == "stateless":
            raise ValueError(
                "A recoverable stateful component must declare full or partial reset semantics."
            )


@dataclass(frozen=True)
class LifecycleSpec:
    """The stable lifecycle extension point carried by every :class:`TaskSpec`.

    Built-in component contracts are derived from scene and MDP declarations.
    ``components`` is a complete, explicit replacement for a derived contract
    when a component has stronger semantics than its family default.
    """

    clocks: tuple[ClockDomainSpec, ...] = ()
    components: Mapping[str, ComponentLifecycleSpec] = field(default_factory=dict)
    trace_schema_version: int = 1
    snapshot_schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "clocks", tuple(self.clocks))
        object.__setattr__(self, "components", dict(self.components))
        names = tuple(clock.name for clock in self.clocks)
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Lifecycle clock names must be unique; repeated: {duplicates}.")
        empty_components = sorted(name for name in self.components if not name)
        if empty_components:
            raise ValueError("Lifecycle component keys must be non-empty.")
        if self.trace_schema_version != 1 or self.snapshot_schema_version != 1:
            raise ValueError(
                "This release supports lifecycle trace and snapshot schema version 1 only."
            )

    def resolved_clocks(self, sim: SimSpec) -> dict[str, ResolvedClockDomain]:
        """Resolve every clock to integer physics steps and reject cycles."""
        resolved = {
            "physics": ResolvedClockDomain.from_fractions(
                "physics", Fraction(1), Fraction(0), "never"
            ),
            "policy": ResolvedClockDomain.from_fractions(
                "policy", Fraction(sim.decimation), Fraction(0), "never"
            ),
            "episode": ResolvedClockDomain.from_fractions(
                "episode", Fraction(sim.decimation), Fraction(0), "episode"
            ),
        }
        pending = {clock.name: clock for clock in self.clocks}
        while pending:
            progressed = False
            for name, clock in tuple(pending.items()):
                parent = resolved.get(clock.parent)
                if parent is None:
                    continue
                period = parent.period_physics_steps * clock.tick_divider
                phase = (
                    parent.phase_physics_steps
                    + parent.period_physics_steps * clock.phase
                ) % period
                reset: ClockReset = (
                    "episode"
                    if parent.reset == "episode" or clock.reset == "episode"
                    else "never"
                )
                resolved[name] = ResolvedClockDomain.from_fractions(
                    name, period, phase, reset
                )
                pending.pop(name)
                progressed = True
            if progressed:
                continue
            unresolved = {
                name: clock.parent for name, clock in sorted(pending.items())
            }
            raise ValueError(
                "Lifecycle clocks contain an unknown parent or cycle: "
                f"{unresolved}."
            )
        return resolved


def _period_clock(
    *,
    component_key: str,
    update_period: float | None,
    sim: SimSpec,
    clocks: dict[str, ResolvedClockDomain],
) -> str:
    if update_period is None:
        return "physics"
    raw_ratio = update_period / sim.physics_dt
    physics_steps = Fraction(str(raw_ratio)).limit_denominator(1_000_000)
    if physics_steps <= 0 or not math.isclose(
        raw_ratio, float(physics_steps), rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError(
            f"Lifecycle component {component_key!r} update period {update_period} s "
            f"cannot be represented as a stable rational multiple of "
            f"physics_dt={sim.physics_dt} s."
        )
    matching = [
        name
        for name, clock in clocks.items()
        if clock.period_physics_steps == physics_steps
        and clock.phase_physics_steps == 0
        and clock.reset == "never"
    ]
    if matching:
        return min(matching, key=lambda name: (name not in _BUILTIN_CLOCKS, name))
    generated_name = component_key.replace("/", ".")
    clocks[generated_name] = ResolvedClockDomain.from_fractions(
        generated_name, physics_steps, Fraction(0), "never"
    )
    return generated_name


def _derived_components(
    task: TaskSpec,
    clocks: dict[str, ResolvedClockDomain],
) -> dict[str, ComponentLifecycleSpec]:
    components: dict[str, ComponentLifecycleSpec] = {}

    def sensor_contract(
        name: str,
        update_period: float | None,
        *,
        stateful: bool,
        latency_s: float = 0.0,
    ) -> None:
        key = f"sensor/{name}"
        clock_name = _period_clock(
            component_key=key,
            update_period=update_period,
            sim=task.sim,
            clocks=clocks,
        )
        clock = clocks[clock_name]
        latency_ratio = latency_s / (
            clock.period_physics_steps * task.sim.physics_dt
        )
        latency_ticks = round(latency_ratio)
        if not math.isclose(
            latency_ratio, latency_ticks, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise ValueError(
                f"Lifecycle component {key!r} latency {latency_s} s is not an "
                f"integer number of {clock_name!r} ticks."
            )
        components[key] = ComponentLifecycleSpec(
            clock=clock_name,
            phase="post_physics",
            reset="partial",
            state="snapshot" if stateful else "stateless",
            latency_ticks=latency_ticks,
        )

    for sensor in task.scene.contact_sensors:
        sensor_contract(
            sensor.name,
            task.sim.physics_dt,
            stateful=sensor.track_air_time or sensor.history_length > 0,
        )
    for sensor in task.scene.ray_casters:
        sensor_contract(
            sensor.name,
            sensor.update_period,
            stateful=True,
        )
    for sensor in task.scene.motion_references:
        sensor_contract(sensor.name, sensor.update_period, stateful=True)
    for sensor in task.scene.volume_points:
        sensor_contract(sensor.name, sensor.update_period, stateful=True)
    for sensor in task.scene.native_sensors:
        sensor_contract(
            sensor.name,
            sensor.update_period,
            stateful=True,
            latency_s=sensor.latency,
        )

    family_defaults = {
        "action": ComponentLifecycleSpec(
            "policy", "pre_step", "partial", "snapshot"
        ),
        "command": ComponentLifecycleSpec(
            "policy", "pre_step", "partial", "snapshot"
        ),
        "observation": ComponentLifecycleSpec(
            "policy", "post_physics", "stateless", "stateless"
        ),
        "reward": ComponentLifecycleSpec(
            "policy", "post_physics", "stateless", "stateless"
        ),
        "termination": ComponentLifecycleSpec(
            "policy", "post_physics", "stateless", "stateless"
        ),
        "curriculum": ComponentLifecycleSpec(
            "episode", "on_reset", "partial", "snapshot"
        ),
    }
    for key, term in task.mdp.terms().items():
        family = key.partition("/")[0]
        if family == "event":
            if term.mode == "startup":  # type: ignore[attr-defined]
                contract = ComponentLifecycleSpec(
                    "physics", "startup", "full", "snapshot"
                )
            elif term.mode == "reset":  # type: ignore[attr-defined]
                contract = ComponentLifecycleSpec(
                    "episode", "on_reset", "partial", "snapshot"
                )
            else:
                contract = ComponentLifecycleSpec(
                    "physics", "post_physics", "partial", "snapshot"
                )
        else:
            contract = family_defaults[family]
        if family == "observation" and getattr(term, "history_length", 0) > 0:
            contract = ComponentLifecycleSpec(
                contract.clock,
                contract.phase,
                "partial",
                "snapshot",
            )
        components[key] = contract
    return components


def resolve_lifecycle_contract(
    task: TaskSpec,
) -> tuple[dict[str, ResolvedClockDomain], dict[str, ComponentLifecycleSpec]]:
    """Return the complete clock/component contract for a materialized task."""
    clocks = task.lifecycle.resolved_clocks(task.sim)
    components = _derived_components(task, clocks)
    unknown = sorted(set(task.lifecycle.components) - set(components))
    if unknown:
        raise ValueError(
            f"Lifecycle overrides name undeclared components: {unknown}. "
            f"Declared components: {sorted(components)}."
        )
    components.update(task.lifecycle.components)
    unknown_clocks = sorted(
        {component.clock for component in components.values()} - set(clocks)
    )
    if unknown_clocks:
        raise ValueError(
            f"Lifecycle components refer to unknown clocks: {unknown_clocks}."
        )
    return clocks, components


__all__ = [
    "ClockDomainSpec",
    "ComponentLifecycleSpec",
    "LifecycleSpec",
    "ResolvedClockDomain",
    "resolve_lifecycle_contract",
]
