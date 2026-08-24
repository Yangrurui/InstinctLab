"""The MDP of a task, declared once for every engine.

A term is stated in one of two ways, and which one applies is a property of the family rather than
a choice the task author makes case by case:

**Portable families carry a function.** Observations, rewards, terminations and commands are
written as ``ObsTermSpec(mdp.base_ang_vel, ...)`` -- a direct reference to a function in
:mod:`instinctlab.mdp` that runs unmodified under either engine's native manager. This is not a
convenience. It is the reason migration is cheap: an Isaac Lab task's term definitions transfer
almost verbatim, with the import redirected and the class renamed, because the function itself did
not have to change.

**Per-engine families carry a semantic name.** Actions, events and domain randomisation are stated
as ``EventTermSpec(kind="randomize_friction")`` and resolved through each engine's term registry.
These families are where the engines genuinely differ -- Isaac Lab randomises friction per shape
across 64 buckets, mjlab shares one friction per environment -- and pretending otherwise would mean
picking one engine's idiom and making the other emulate it badly. Naming the intent instead lets
each engine keep the implementation its own users would recognise, which is what "each engine keeps
its own characteristics" has to mean in practice.

The split is visible in :class:`TermSpec`: exactly one of ``func`` and ``kind`` is set, and which
one is set determines whether the compiler needs a per-engine mapping at all.

Engine-specific values reach a term through ``engine_params``, keyed by engine name. That is data,
not a branch -- ``spec/`` may not contain ``if engine == ...`` anywhere, since a conditional here
would mean the declaration is no longer engine-agnostic and every future engine would have to edit
task files to be supported.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from .capability import Requirement
from .entity import EntityRef

__all__ = [
    "ActionTermSpec",
    "CommandTermSpec",
    "CurriculumTermSpec",
    "DoneTermSpec",
    "EventTermSpec",
    "MdpSpec",
    "NoiseSpec",
    "ObsGroupSpec",
    "ObsTermSpec",
    "RewardTermSpec",
    "TermSpec",
]


def walk_parameter_values(values: Iterable[Any]) -> Iterable[Any]:
    """Yield parameter values recursively through mappings and sequences."""
    for value in values:
        yield value
        if isinstance(value, Mapping):
            yield from walk_parameter_values(value.values())
        elif isinstance(value, (tuple, list, set, frozenset)):
            yield from walk_parameter_values(value)


@dataclass(frozen=True)
class NoiseSpec:
    """Observation noise, in the two shapes both engines' noise models already share.

    Args:
        kind: ``"uniform"`` over ``[lo, hi]``, or ``"gaussian"`` with mean ``lo`` and standard
            deviation ``hi``. The reuse of the field names for the gaussian case is deliberate --
            they are the two numbers each distribution takes, and naming them per-distribution
            would need a class per distribution for no gain.
        lo: Lower bound, or the mean.
        hi: Upper bound, or the standard deviation.
        operation: Whether the sample is added to the observation or scales it.
    """

    kind: Literal["uniform", "gaussian"]
    lo: float
    hi: float
    operation: Literal["add", "scale", "abs"] = "add"

    def __post_init__(self) -> None:
        if self.kind == "uniform" and self.lo > self.hi:
            raise ValueError(f"Uniform noise has lo={self.lo} above hi={self.hi}.")
        if self.kind == "gaussian" and self.hi < 0.0:
            raise ValueError(f"Gaussian noise has a negative standard deviation: {self.hi}.")


@dataclass(frozen=True)
class TermSpec:
    """Base of every MDP term.

    Args:
        func: The portable implementation, for observation / reward / termination / command terms.
            Written exactly as an Isaac Lab task would write it.
        kind: The semantic name, for action / event / curriculum terms, looked up in the running
            engine's registry. Exactly one of ``func`` and ``kind`` is set.
        params: Arguments passed to the term, identical on every engine.
        target: The entity subset the term acts on, if any.
        level: What the compiler does when this engine cannot provide the term.
        engine_params: Per-engine parameter overrides, merged over ``params`` at compile time by
            :meth:`resolved_params`. Data rather than a branch, so that adding an engine never
            edits a task file.
    """

    func: Callable[..., Any] | None = None
    kind: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    target: EntityRef | None = None
    level: Requirement = Requirement.OPTIONAL
    engine_params: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.func is None) == (self.kind is None):
            stated = "both" if self.func is not None else "neither"
            raise ValueError(
                f"{type(self).__name__} must set exactly one of func / kind; {stated} was given. "
                "Portable terms carry the function itself; per-engine terms carry a semantic name."
            )
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(self, "engine_params", {k: dict(v) for k, v in self.engine_params.items()})

    @property
    def is_portable(self) -> bool:
        """Whether this term runs the same implementation on every engine."""
        return self.func is not None

    def resolved_params(self, engine: str) -> dict[str, Any]:
        """``params`` with this engine's overrides applied. The only place the two are combined.

        Nested mappings merge rather than replace, so a shared ``velocity_ranges`` dict can keep
        the nine common sub-terrains and put only the tenth name in ``engine_params``. A shallow
        update would drop the shared keys on that engine and the command would raise at init.
        """
        merged = dict(self.params)
        for key, value in self.engine_params.get(engine, {}).items():
            existing = merged.get(key)
            if isinstance(existing, Mapping) and isinstance(value, Mapping):
                nested = dict(existing)
                nested.update(value)
                merged[key] = nested
            else:
                merged[key] = value
        return merged

    def engines_named(self) -> frozenset[str]:
        """Engines this term mentions by name, so the compiler can reject typos in engine keys."""
        return frozenset(self.engine_params)


@dataclass(frozen=True)
class ObsTermSpec(TermSpec):
    """One observation. REQUIRED by default: dropping it changes the policy's input width."""

    noise: NoiseSpec | None = None
    scale: float | None = None
    clip: tuple[float, float] | None = None
    history_length: int = 0
    level: Requirement = Requirement.REQUIRED


@dataclass(frozen=True)
class RewardTermSpec(TermSpec):
    """One reward. OPTIONAL by default, and therefore the term family most worth watching.

    A skipped reward leaves a run that trains, converges and looks healthy while optimising a
    different objective than the one written down. That is why every skip is recorded and printed
    rather than warned about, and why a task that depends on a particular reward should raise its
    level rather than trusting the report to be read.
    """

    weight: float = 0.0


@dataclass(frozen=True)
class DoneTermSpec(TermSpec):
    """One termination. REQUIRED by default: dropping it changes the episode structure.

    Args:
        time_out: Whether this termination is a time limit rather than a failure. Both engines use
            the distinction for bootstrapping the value function at the episode boundary, and
            getting it wrong biases the value estimate rather than raising anything.
    """

    time_out: bool = False
    level: Requirement = Requirement.REQUIRED


@dataclass(frozen=True)
class EventTermSpec(TermSpec):
    """One event or domain randomisation.

    Args:
        mode: When it fires. ``"startup"`` once at construction, ``"reset"`` per episode,
            ``"interval"`` on a timer.
        interval_range_s: Seconds between firings, for ``"interval"`` mode.
    """

    mode: Literal["startup", "reset", "interval"] = "reset"
    interval_range_s: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.mode not in {"startup", "reset", "interval"}:
            raise ValueError(f"Unknown event mode {self.mode!r}; expected startup, reset, or interval.")
        if (self.mode == "interval") != (self.interval_range_s is not None):
            raise ValueError(
                f"Event in {self.mode!r} mode with interval_range_s={self.interval_range_s}: an "
                "interval range is required in interval mode and meaningless otherwise."
            )
        if self.interval_range_s is not None:
            lo, hi = self.interval_range_s
            if not math.isfinite(lo) or not math.isfinite(hi) or lo <= 0.0 or hi < lo:
                raise ValueError(
                    f"Event interval_range_s must be finite with 0 < min <= max, got {self.interval_range_s}."
                )


@dataclass(frozen=True)
class ActionTermSpec(TermSpec):
    """One action term. REQUIRED by default: without it the policy cannot act.

    Always per-engine. The two engines drive joints through different objects -- Isaac Lab's
    articulation actuators against mjlab's MuJoCo actuators -- so this family is stated with
    ``kind`` and resolved per engine even though the resulting behaviour is meant to match.
    """

    level: Requirement = Requirement.REQUIRED


@dataclass(frozen=True)
class CommandTermSpec(TermSpec):
    """One command generator. REQUIRED by default: an observation term reads its output."""

    level: Requirement = Requirement.REQUIRED


@dataclass(frozen=True)
class CurriculumTermSpec(TermSpec):
    """One curriculum term."""


@dataclass(frozen=True)
class ObsGroupSpec:
    """One observation group, such as the policy's input or the critic's.

    Args:
        terms: The group's terms. **Insertion order is the concatenation order**, and therefore
            part of the task's contract with any checkpoint trained from it -- reordering this
            mapping silently permutes the policy's input vector.
        enable_corruption: Whether the terms' noise is applied. Off for critic groups by
            convention on both engines.
        concatenate_terms: Whether the group is flattened into one tensor.
        history_length: Group-level override. ``None`` (the default) is unset — each
            term keeps its own history. An integer, including ``0``, overrides every
            term. Both engines' managers test ``is not None``, so ``0`` cannot mean
            "unspecified": that value is a real instruction to drop history.
    """

    terms: Mapping[str, ObsTermSpec]
    enable_corruption: bool = True
    concatenate_terms: bool = True
    history_length: int | None = None

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError("An observation group with no terms produces an empty input tensor.")
        if self.history_length is not None and self.history_length < 0:
            raise ValueError(f"Observation group history_length={self.history_length} is negative.")
        object.__setattr__(self, "terms", dict(self.terms))


@dataclass(frozen=True)
class MdpSpec:
    """Every term of a task's MDP, grouped by family.

    Rewards are grouped rather than flat because both this project's ``MultiRewardManager`` and the
    agent configs that consume it treat reward groups as separate signals. A task with one group
    writes ``rewards={"rewards": {...}}``.
    """

    observations: Mapping[str, ObsGroupSpec] = field(default_factory=dict)
    actions: Mapping[str, ActionTermSpec] = field(default_factory=dict)
    rewards: Mapping[str, Mapping[str, RewardTermSpec]] = field(default_factory=dict)
    terminations: Mapping[str, DoneTermSpec] = field(default_factory=dict)
    events: Mapping[str, EventTermSpec] = field(default_factory=dict)
    commands: Mapping[str, CommandTermSpec] = field(default_factory=dict)
    curriculum: Mapping[str, CurriculumTermSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("observations", "actions", "terminations", "events", "commands", "curriculum"):
            object.__setattr__(self, name, dict(getattr(self, name)))
        object.__setattr__(self, "rewards", {group: dict(terms) for group, terms in self.rewards.items()})

    def terms(self) -> dict[str, TermSpec]:
        """Every term keyed by ``family/name``, the key used throughout a compilation's report.

        Reward keys carry their group -- ``reward/rewards/track_lin_vel_xy_exp`` -- so that two
        groups may use the same term name without colliding in the report.
        """
        out: dict[str, TermSpec] = {}
        for group, group_spec in self.observations.items():
            for name, term in group_spec.terms.items():
                out[f"observation/{group}/{name}"] = term
        for group, group_terms in self.rewards.items():
            for name, term in group_terms.items():
                out[f"reward/{group}/{name}"] = term
        for family, mapping in (
            ("action", self.actions),
            ("termination", self.terminations),
            ("event", self.events),
            ("command", self.commands),
            ("curriculum", self.curriculum),
        ):
            for name, term in mapping.items():
                out[f"{family}/{name}"] = term
        return out

    def entity_refs(self) -> tuple[EntityRef, ...]:
        """Every entity reference the MDP uses, including those passed through ``params``.

        The backend needs all of them to check its selector support up front, and they hide in two
        places: the ``target`` field, and parameters like an ``asset_cfg`` that a portable term
        takes as an argument.
        """
        found: list[EntityRef] = []
        for term in self.terms().values():
            if term.target is not None:
                found.append(term.target)
            found.extend(v for v in walk_parameter_values(term.params.values()) if isinstance(v, EntityRef))
            for overrides in term.engine_params.values():
                found.extend(v for v in walk_parameter_values(overrides.values()) if isinstance(v, EntityRef))
        return tuple(found)
