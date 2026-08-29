"""Turning a declared task into native terms, and accounting for what did not survive.

This is the whole of "skip what the engine cannot do". It is a small amount of code and most of it
is bookkeeping, which is the correct proportion: deciding to skip is easy, and the part that makes
skipping safe is that nothing gets skipped quietly.

:class:`CompileCtx` is the other half. It is the single place where an engine-agnostic reference
becomes an engine-native config, so :meth:`CompileCtx.entity` is the only code in the project that
has to know about decision D1 -- that joint order is the depth-first walk of the kinematic tree,
and that a task pinning an explicit joint order needs ``preserve_order`` set. Every other module
just passes ``EntityRef`` around.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from instinctlab.compat import entity as compat_entity
from instinctlab.engines.capabilities import CapabilitySet
from instinctlab.spec.capability import Requirement
from instinctlab.spec.entity import EntityRef, resolve_entity_names
from instinctlab.spec.mdp import MdpSpec, NoiseSpec, TermSpec
from instinctlab.spec.task import TaskSpec

from .base import Resolution, UnsupportedTerm
from .registry import TermRegistry

__all__ = [
    "CompileCtx",
    "compile_family",
    "compile_mdp",
    "contract_report",
    "flatten_reward_groups",
    "joint_position_target",
    "observation_group_settings",
    "qualname_of",
    "record_reward_omissions",
]


def qualname_of(obj: Any) -> str:
    """A stable name for whatever a builder returned, for the resolution report.

    Reports the wrapped function where there is one, since a native term config is mostly a
    container and its own type name says nothing about what will run.
    """
    inner = getattr(obj, "func", None)
    target = inner if callable(inner) else obj
    module = getattr(target, "__module__", "")
    name = getattr(target, "__qualname__", None) or getattr(target, "__name__", None)
    if name is None:
        name = type(target).__qualname__
        module = type(target).__module__
    return f"{module}.{name}" if module else str(name)


@dataclass
class CompileCtx:
    """Everything a term builder needs, and the place canonical names become native ones.

    Backends subclass this to supply :meth:`noise`, which needs the engine's own noise config
    classes. Everything else is engine-independent and lives here.

    Args:
        engine: Engine being compiled for.
        spec: The task.
        resolution: The report being filled in as compilation proceeds.
        profile: Solver settings for this engine, adapter defaults already merged under the task's
            overrides.
        num_envs: Parallel environments, a launch argument rather than part of the task.
        device: Torch device.
        strict: Promote every OPTIONAL term to REQUIRED. Used in CI and for runs meant to be
            compared against each other, where a quietly missing reward term would make the
            comparison meaningless.
    """

    engine: str
    spec: TaskSpec
    resolution: Resolution
    profile: Mapping[str, Any] = field(default_factory=dict)
    num_envs: int = 1
    device: str = "cpu"
    strict: bool = False

    def entity(self, ref: EntityRef | None) -> Any:
        """Lower an :class:`EntityRef` onto this engine's ``SceneEntityCfg``.

        The single point where a canonical selector becomes a native one, and therefore the single
        place D1's joint-order decision is applied. Raises
        :class:`~instinctlab.compat.entity.UnsupportedSelector` when the engine has no selector for
        a kind the reference names, rather than dropping that selector -- a reference that asked
        for geoms and got bodies selects the wrong things and reports nothing.
        """
        if ref is None:
            return None
        if ref.entity == "robot" and ref.joints is not None and ref.preserve_order:
            # A lone ``.*`` does not make either native resolver use the canonical order: it
            # preserves the one regex, then still enumerates matches in the articulation's own
            # order (BFS on Isaac). Expand every order-sensitive robot selector against the
            # canonical RobotSpec before lowering it so native resolution receives exact DFS
            # names, one per policy column.
            canonical_names = resolve_entity_names(
                ref.joints,
                self.spec.robot.joint_names,
                preserve_order=False,
            )
            ref = replace(ref, joints=canonical_names)
        return compat_entity.lower(ref, self.engine)

    def noise(self, noise: NoiseSpec | None) -> Any:
        """Build this engine's noise config. Supplied by the backend."""
        raise NotImplementedError(f"{type(self).__name__} must implement noise() for engine {self.engine!r}.")

    def params(self, spec: TermSpec) -> dict[str, Any]:
        """A term's parameters for this engine, with any ``EntityRef`` among them lowered.

        Portable terms take their entity as a parameter the way Isaac Lab tasks always have, so the
        lowering has to reach into ``params`` and not only into ``target``.
        """
        return {name: self._lower_parameter(value) for name, value in spec.resolved_params(self.engine).items()}

    def _lower_parameter(self, value: Any) -> Any:
        """Lower entity references without disturbing a term's nested parameter shape."""
        if isinstance(value, EntityRef):
            return self.entity(value)
        if isinstance(value, Mapping):
            return {key: self._lower_parameter(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(self._lower_parameter(item) for item in value)
        if isinstance(value, list):
            return [self._lower_parameter(item) for item in value]
        if isinstance(value, set):
            return {self._lower_parameter(item) for item in value}
        if isinstance(value, frozenset):
            return frozenset(self._lower_parameter(item) for item in value)
        return value


def _effective_level(spec: TermSpec, strict: bool) -> Requirement:
    """``--strict-capabilities`` promotes OPTIONAL to REQUIRED, and leaves the others alone."""
    if strict and spec.level is Requirement.OPTIONAL:
        return Requirement.REQUIRED
    return spec.level


def compile_family(
    family: str,
    specs: Mapping[str, TermSpec],
    ctx: CompileCtx,
    registry: TermRegistry,
    *,
    prefix: str = "",
) -> dict[str, Any]:
    """Build one family's terms, recording every outcome on ``ctx.resolution``.

    Args:
        family: Term family, one of :data:`~instinctlab.engines.registry.FAMILIES`.
        specs: The family's terms, keyed by name. Iteration order is preserved, which matters for
            observation groups, where it is the concatenation order.
        ctx: Compilation context.
        registry: The engine's term registry.
        prefix: Group name, for the families that have groups.

    Returns:
        Native term configs for the terms that survived, in declaration order.

    Raises:
        UnsupportedTerm: A REQUIRED term has no implementation here.
    """
    out: dict[str, Any] = {}
    for name, spec in specs.items():
        key = f"{family}/{prefix}{name}" if prefix else f"{family}/{name}"
        built, outcome, reason = _build(key, family, spec, ctx, registry)
        if outcome == "skipped":
            ctx.resolution.skipped[key] = reason
            continue
        out[name] = built
        if outcome == "emulated":
            ctx.resolution.emulated[key] = qualname_of(built)
        ctx.resolution.resolved[key] = qualname_of(built)
    return out


def _build(key: str, family: str, spec: TermSpec, ctx: CompileCtx, registry: TermRegistry) -> tuple[Any, str, str]:
    """One term: build it, emulate it, or explain why it is being skipped."""
    level = _effective_level(spec, ctx.strict)

    if spec.is_portable:
        builder = registry.lookup_portable(family)
        if builder is None:
            # Not a task problem: the engine's registry is incomplete, and no task can work around it.
            raise UnsupportedTerm(
                key,
                ctx.engine,
                None,
                detail=f"Its registry has no portable builder for the {family!r} family.",
            )
        return builder(spec, ctx), "resolved", ""

    builder = registry.lookup(family, spec.kind)  # type: ignore[arg-type]
    if builder is not None:
        return builder(spec, ctx), "resolved", ""

    if spec.level is Requirement.EMULATE:
        stand_in = registry.lookup_emulation(family, spec.kind)  # type: ignore[arg-type]
        if stand_in is not None:
            return stand_in(spec, ctx), "emulated", ""
        # No stand-in: EMULATE degrades to OPTIONAL, and strict mode still overrides that.
        level = Requirement.REQUIRED if ctx.strict else Requirement.OPTIONAL

    if level is Requirement.REQUIRED:
        known = sorted(registry.kinds(family))
        raise UnsupportedTerm(
            key,
            ctx.engine,
            spec.kind,
            detail=(
                f"It implements {known} in this family."
                + (
                    " The term is OPTIONAL but strict capabilities are on."
                    if ctx.strict and spec.level is not Requirement.REQUIRED
                    else ""
                )
            ),
        )
    return None, "skipped", f"engine {ctx.engine!r} has no {family} kind {spec.kind!r}"


_OBSERVATION_GROUP_FIELDS = ("enable_corruption", "concatenate_terms", "history_length")


def observation_group_settings(source: Any) -> dict[str, Any]:
    """Group fields both engines accept with the same sentinels.

    ``enable_corruption`` and ``concatenate_terms`` are real booleans and are
    always emitted. ``history_length`` is ``None`` when the group does not
    override terms and an ``int`` (including ``0``) when it does. Both engines'
    managers do ``if group_cfg.history_length is not None``, so the IR uses that
    sentinel rather than leaving each adapter to guess whether ``0`` means
    "unset". Accepts an :class:`~instinctlab.spec.mdp.ObsGroupSpec` or the
    compiled mapping ``compile_mdp`` emits from one.
    """
    get = source.__getitem__ if isinstance(source, Mapping) else lambda key: getattr(source, key)
    return {key: get(key) for key in _OBSERVATION_GROUP_FIELDS}


def flatten_reward_groups(groups: Mapping[str, Mapping[str, Any]], *, omit: tuple[str, ...] = ()) -> dict[str, Any]:
    """Flatten reward groups without silently overwriting repeated term names."""
    omitted = set(omit)
    counts: dict[str, int] = {}
    for terms in groups.values():
        for name in terms:
            if name not in omitted:
                counts[name] = counts.get(name, 0) + 1

    flattened: dict[str, Any] = {}
    for group, terms in groups.items():
        for name, term in terms.items():
            if name in omitted:
                continue
            native_name = name if counts[name] == 1 else f"{group}__{name}"
            if native_name in flattened:
                raise ValueError(
                    f"Reward groups produce the same native term name {native_name!r}; "
                    "rename the group or term so every compiled reward remains visible."
                )
            flattened[native_name] = term
    return flattened


def joint_position_target(spec: TermSpec, ctx: CompileCtx) -> EntityRef:
    """Return an unambiguous joint selector for a position action.

    With no selector, both engines control the whole robot in canonical order. A portable task may
    not request native order: PhysX and MuJoCo enumerate the same joints differently, so the same
    policy tensor would command different motors.
    """
    target = spec.target
    if target is None:
        target = EntityRef(entity="robot", joints=ctx.spec.robot.joint_names, preserve_order=True)
    if target.entity != "robot":
        raise ValueError(
            f"joint_position actions may target only the TaskSpec robot, got entity {target.entity!r}; "
            "no canonical joint schema is declared for another articulation."
        )
    if target.bodies is not None or target.other:
        raise ValueError("joint_position actions may select joints only")
    if target.joints is None:
        target = EntityRef(entity=target.entity, joints=ctx.spec.robot.joint_names, preserve_order=True)
    if not target.preserve_order:
        raise ValueError(
            "A joint_position action must set preserve_order=True; the policy joint axis is the "
            "RobotSpec canonical order, not an engine's native articulation order."
        )
    # Return exact names, not patterns.  Isaac lowers the target through ``ctx.entity`` while
    # MJLab's preserving action consumes this reference directly; normalising here keeps both
    # builders on the same engine-neutral policy axis.
    return replace(
        target,
        joints=resolve_entity_names(
            target.joints,
            ctx.spec.robot.joint_names,
            preserve_order=False,
        ),
    )


def record_reward_omissions(
    resolution: Resolution,
    groups: Mapping[str, Mapping[str, Any]],
    omit: tuple[str, ...],
) -> None:
    """Move profile-omitted rewards out of ``resolved`` into an explicit manifest section."""
    omitted = set(omit)
    for group, terms in groups.items():
        for name in terms:
            if name not in omitted:
                continue
            key = f"reward/{group}/{name}"
            resolution.resolved.pop(key, None)
            resolution.omitted[key] = "omitted by this engine profile to match its reference task"


def contract_report(
    spec: TaskSpec,
    *,
    engine: str,
    registry: TermRegistry,
    capabilities: CapabilitySet,
    omitted_rewards: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Describe registry resolution accurately without importing an engine SDK."""
    spec.validate_for_engine(engine)
    missing: dict[str, str] = {}
    for key, term in spec.mdp.terms().items():
        family = key.split("/", 1)[0]
        if term.is_portable:
            if registry.lookup_portable(family) is None:
                missing[key] = f"no portable builder for family {family!r}"
            continue
        if registry.lookup(family, term.kind) is not None:
            continue
        if term.level is Requirement.EMULATE and registry.lookup_emulation(family, term.kind) is not None:
            missing[key] = "emulated"
        else:
            missing[key] = f"unsupported kind {term.kind!r}"

    omitted = set(omitted_rewards)
    omitted_keys = [
        f"reward/{group}/{name}" for group, terms in spec.mdp.rewards.items() for name in terms if name in omitted
    ]
    return {
        "engine": engine,
        "task_id": spec.task_id,
        "capabilities": sorted(capabilities.values),
        "missing": missing,
        "omitted": sorted(omitted_keys),
        "engine_extras_used": sorted(spec.engine_extras.get(engine, {})),
    }


def compile_mdp(mdp: MdpSpec, ctx: CompileCtx, registry: TermRegistry) -> dict[str, Any]:
    """Compile every family of an MDP, preserving group structure.

    Returns:
        A mapping with the same shape as :class:`~instinctlab.spec.mdp.MdpSpec` -- grouped
        observations and rewards, flat everything else -- holding native term configs. The backend
        assembles these into its own env config; what that assembly looks like is the one thing
        that genuinely differs per engine.
    """
    observations = {
        group: {
            "terms": compile_family("observation", group_spec.terms, ctx, registry, prefix=f"{group}/"),
            **observation_group_settings(group_spec),
        }
        for group, group_spec in mdp.observations.items()
    }
    rewards = {
        group: compile_family("reward", terms, ctx, registry, prefix=f"{group}/")
        for group, terms in mdp.rewards.items()
    }
    return {
        "observations": observations,
        "actions": compile_family("action", mdp.actions, ctx, registry),
        "rewards": rewards,
        "terminations": compile_family("termination", mdp.terminations, ctx, registry),
        "events": compile_family("event", mdp.events, ctx, registry),
        "commands": compile_family("command", mdp.commands, ctx, registry),
        "curriculum": compile_family("curriculum", mdp.curriculum, ctx, registry),
    }
