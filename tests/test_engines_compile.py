"""Guard: a task compiles, and everything that did not compile is accounted for.

The compiler's job is easy to get half right. Building the terms an engine supports is the half
that fails loudly if it is wrong. The other half -- deciding what to do about the terms it does not
support, and making sure a person finds out -- fails silently by construction, which is why most of
what follows is about skips rather than about builds.

A mock engine stands in for a real backend here. That is the whole acceptance criterion for this
layer: if a registry and a context are enough to compile a task, then adding an engine really is
bounded by those two pieces, and the ``N + M`` claim holds. The one place a real engine appears is
``ctx.entity``, which is exercised against mjlab, since lowering a reference onto a native selector
config is not something a mock can honestly stand in for.
"""

from __future__ import annotations

import pytest

from instinctlab.engines import CompileCtx, Resolution, TermRegistry, UnsupportedTerm, compile_family, compile_mdp
from instinctlab.engines.compile import qualname_of
from instinctlab.sim.capabilities import Capability
from instinctlab.spec import (
    ActionTermSpec,
    CommandTermSpec,
    DoneTermSpec,
    EntityRef,
    EventTermSpec,
    MdpSpec,
    NoiseSpec,
    ObsGroupSpec,
    ObsTermSpec,
    Requirement,
    RewardTermSpec,
)


class _NativeTerm:
    """Stand-in for an engine's native term config, which is mostly a container for a callable."""

    def __init__(self, func=None, **kwargs):
        self.func = func
        self.kwargs = kwargs


def _mock_registry() -> TermRegistry:
    registry = TermRegistry("mock")

    for family in ("observation", "reward", "termination", "command"):
        registry.portable(family)(lambda spec, ctx: _NativeTerm(spec.func, **ctx.params(spec)))

    @registry.action("joint_position")
    def _joint_position(spec, ctx):
        return _NativeTerm(_joint_position, target=ctx.entity(spec.target))

    @registry.event("randomize_friction", provides=(Capability.DR_SLIDING_FRICTION, Capability.DR_RESTITUTION))
    def _friction(spec, ctx):
        return _NativeTerm(_friction, **ctx.profile.get("friction", {}))

    @registry.event("push_robot", provides=(Capability.EXTERNAL_WRENCH,))
    def _push(spec, ctx):
        return _NativeTerm(_push)

    @registry.emulation("event", "randomize_gravity")
    def _emulated_gravity(spec, ctx):
        return _NativeTerm(_emulated_gravity)

    return registry


class _MockCtx(CompileCtx):
    """A backend supplies exactly one thing beyond the base context: its noise config."""

    def noise(self, noise: NoiseSpec | None):
        return None if noise is None else (noise.kind, noise.lo, noise.hi)


def _ctx(*, engine: str = "mock", strict: bool = False, profile: dict | None = None) -> _MockCtx:
    return _MockCtx(
        engine=engine,
        spec=None,  # type: ignore[arg-type]
        resolution=Resolution(engine=engine, task_id="Test-Task-v0", strict=strict),
        profile=profile or {},
        strict=strict,
    )


def _term(env, asset_cfg=None, std=None):  # noqa: ANN001 - a stand-in for a portable term
    return env


"""
The registry is the capability matrix.
"""


def test_capabilities_are_derived_from_the_builders_rather_than_declared():
    """A hand-written capability list drifts; this one cannot, because there is only one copy."""
    capabilities = _mock_registry().capabilities()
    assert capabilities.supports(Capability.DR_SLIDING_FRICTION)
    assert capabilities.supports(Capability.DR_RESTITUTION)
    assert capabilities.supports(Capability.EXTERNAL_WRENCH)
    assert not capabilities.supports(Capability.CONTACT_FORCE_VECTOR)


def test_an_unregistered_kind_is_simply_absent():
    registry = _mock_registry()
    assert registry.lookup("event", "randomize_friction") is not None
    assert registry.lookup("event", "randomize_gravity") is None  # only an emulation exists
    assert registry.kinds("event") == {"randomize_friction", "push_robot"}


def test_registering_the_same_kind_twice_is_an_error():
    registry = _mock_registry()
    with pytest.raises(ValueError, match="already registered"):
        registry.event("push_robot")(lambda spec, ctx: None)
    with pytest.raises(ValueError, match="already registered"):
        registry.portable("reward")(lambda spec, ctx: None)


def test_an_unknown_family_is_rejected_at_registration_and_at_lookup():
    registry = TermRegistry("mock")
    with pytest.raises(KeyError, match="Unknown term family"):
        registry.register("rewards", "x", lambda spec, ctx: None)
    with pytest.raises(KeyError, match="Unknown term family"):
        registry.lookup("rewards", "x")


"""
Building terms, and skipping them.
"""


def test_a_portable_term_is_built_from_the_function_it_carries():
    ctx, registry = _ctx(), _mock_registry()
    built = compile_family("reward", {"alive": RewardTermSpec(_term, weight=1.0)}, ctx, registry)
    assert built["alive"].func is _term
    assert ctx.resolution.resolved["reward/alive"].endswith("._term")
    assert ctx.resolution.is_clean


def test_a_kind_term_is_built_through_the_registry():
    ctx, registry = _ctx(), _mock_registry()
    built = compile_family(
        "event", {"friction": EventTermSpec(kind="randomize_friction", mode="startup")}, ctx, registry
    )
    assert "friction" in built
    assert ctx.resolution.skipped == {}


def test_an_optional_term_the_engine_lacks_is_skipped_with_a_reason():
    ctx, registry = _ctx(), _mock_registry()
    built = compile_family("event", {"gravity": EventTermSpec(kind="randomize_gravity", mode="startup")}, ctx, registry)
    assert built == {}
    assert "no event kind 'randomize_gravity'" in ctx.resolution.skipped["event/gravity"]
    assert not ctx.resolution.is_clean


def test_a_required_term_the_engine_lacks_stops_the_run():
    ctx, registry = _ctx(), _mock_registry()
    with pytest.raises(UnsupportedTerm, match="cannot provide 'action/joint_effort'") as excinfo:
        compile_family("action", {"joint_effort": ActionTermSpec(kind="joint_effort")}, ctx, registry)
    assert excinfo.value.kind == "joint_effort"
    assert "joint_position" in str(excinfo.value)  # says what it does have


def test_an_emulated_term_runs_a_stand_in_and_is_reported_as_emulated():
    """A run with emulated terms is a different experiment, and the report has to say so."""
    ctx, registry = _ctx(), _mock_registry()
    spec = EventTermSpec(kind="randomize_gravity", mode="startup", level=Requirement.EMULATE)
    built = compile_family("event", {"gravity": spec}, ctx, registry)
    assert "gravity" in built
    assert "event/gravity" in ctx.resolution.emulated
    assert "event/gravity" in ctx.resolution.resolved
    assert not ctx.resolution.is_clean


def test_emulate_falls_back_to_optional_when_the_engine_has_no_stand_in():
    ctx, registry = _ctx(), _mock_registry()
    spec = EventTermSpec(kind="randomize_wind", mode="startup", level=Requirement.EMULATE)
    assert compile_family("event", {"wind": spec}, ctx, registry) == {}
    assert "event/wind" in ctx.resolution.skipped
    assert ctx.resolution.emulated == {}


def test_a_portable_family_the_engine_never_registered_is_an_adapter_bug():
    """Distinguished from a task problem, because no task can work around it."""
    ctx = _ctx()
    with pytest.raises(UnsupportedTerm, match="no portable builder"):
        compile_family("reward", {"alive": RewardTermSpec(_term)}, ctx, TermRegistry("empty"))


"""
Strict capabilities.
"""


def test_strict_promotes_an_optional_skip_into_a_failure():
    ctx, registry = _ctx(strict=True), _mock_registry()
    with pytest.raises(UnsupportedTerm, match="strict capabilities are on"):
        compile_family("event", {"gravity": EventTermSpec(kind="randomize_gravity", mode="startup")}, ctx, registry)


def test_strict_also_catches_an_emulate_term_with_no_stand_in():
    ctx, registry = _ctx(strict=True), _mock_registry()
    spec = EventTermSpec(kind="randomize_wind", mode="startup", level=Requirement.EMULATE)
    with pytest.raises(UnsupportedTerm):
        compile_family("event", {"wind": spec}, ctx, registry)


def test_strict_leaves_a_term_the_engine_supports_alone():
    ctx, registry = _ctx(strict=True), _mock_registry()
    assert compile_family("event", {"push": EventTermSpec(kind="push_robot", mode="reset")}, ctx, registry)


"""
Context: parameters, profiles, entity lowering.
"""


def test_engine_parameters_and_profiles_reach_the_builder():
    ctx = _ctx(profile={"friction": {"num_buckets": 64}})
    registry = _mock_registry()
    built = compile_family(
        "event", {"friction": EventTermSpec(kind="randomize_friction", mode="startup")}, ctx, registry
    )
    assert built["friction"].kwargs == {"num_buckets": 64}

    term = RewardTermSpec(_term, params={"std": 0.5}, engine_params={"mock": {"std": 0.6}})
    built = compile_family("reward", {"track": term}, ctx, registry)
    assert built["track"].kwargs == {"std": 0.6}


def test_an_entity_ref_in_params_is_lowered_the_same_as_one_in_target():
    """Portable terms pass their entity as ``asset_cfg``; that path has to be lowered too."""
    pytest.importorskip("mjlab")
    ctx = _ctx(engine="mjlab")
    registry = _mock_registry()
    ref = EntityRef(entity="robot", joints=(".*_hip_.*",))
    built = compile_family("reward", {"joint": RewardTermSpec(_term, params={"asset_cfg": ref})}, ctx, registry)
    lowered = built["joint"].kwargs["asset_cfg"]
    assert lowered.name == "robot"
    assert tuple(lowered.joint_names) == (".*_hip_.*",)


def test_a_selector_the_engine_cannot_express_is_refused_rather_than_dropped():
    pytest.importorskip("mjlab")
    from instinctlab.compat.entity import UnsupportedSelector

    ctx = _ctx(engine="mjlab")
    with pytest.raises(UnsupportedSelector):
        ctx.entity(EntityRef(other={"fixed_tendon": ("a",)}))


def test_the_base_context_refuses_to_invent_a_noise_config():
    with pytest.raises(NotImplementedError, match="must implement noise"):
        CompileCtx(engine="mock", spec=None, resolution=Resolution("mock", "t")).noise(NoiseSpec("uniform", -1, 1))


"""
The whole MDP, and the report.
"""


def _mdp() -> MdpSpec:
    return MdpSpec(
        observations={
            "policy": ObsGroupSpec(
                enable_corruption=True,
                terms={
                    "base_ang_vel": ObsTermSpec(_term, noise=NoiseSpec("uniform", -0.2, 0.2)),
                    "joint_pos": ObsTermSpec(_term),
                    "actions": ObsTermSpec(_term),
                },
            ),
            "critic": ObsGroupSpec(enable_corruption=False, terms={"base_lin_vel": ObsTermSpec(_term)}),
        },
        actions={"joint_pos": ActionTermSpec(kind="joint_position")},
        rewards={"rewards": {"alive": RewardTermSpec(_term, weight=1.0), "effort": RewardTermSpec(_term, weight=-0.1)}},
        terminations={"time_out": DoneTermSpec(_term, time_out=True)},
        events={
            "friction": EventTermSpec(kind="randomize_friction", mode="startup"),
            "gravity": EventTermSpec(kind="randomize_gravity", mode="startup"),
        },
        commands={"base_velocity": CommandTermSpec(_term)},
    )


def test_a_whole_mdp_compiles_and_keeps_its_group_structure():
    ctx, registry = _ctx(), _mock_registry()
    compiled = compile_mdp(_mdp(), ctx, registry)
    assert set(compiled["observations"]) == {"policy", "critic"}
    assert compiled["observations"]["policy"]["enable_corruption"] is True
    assert compiled["observations"]["critic"]["enable_corruption"] is False
    assert set(compiled["rewards"]["rewards"]) == {"alive", "effort"}
    assert set(compiled["events"]) == {"friction"}  # gravity was skipped


def test_observation_order_survives_compilation():
    """It is the concatenation order, and therefore part of the contract with any checkpoint."""
    ctx, registry = _ctx(), _mock_registry()
    compiled = compile_mdp(_mdp(), ctx, registry)
    assert list(compiled["observations"]["policy"]["terms"]) == ["base_ang_vel", "joint_pos", "actions"]


def test_the_report_names_every_term_and_every_skip():
    ctx, registry = _ctx(), _mock_registry()
    compile_mdp(_mdp(), ctx, registry)
    resolution = ctx.resolution
    assert "observation/policy/base_ang_vel" in resolution.resolved
    assert "reward/rewards/alive" in resolution.resolved
    assert set(resolution.skipped) == {"event/gravity"}

    table = resolution.summary_table()
    assert "10 terms resolved" in table
    assert "skipped:" in table and "event/gravity" in table

    manifest = resolution.manifest()
    assert manifest["portable"] is False
    assert manifest["skipped"] == resolution.skipped


def test_a_clean_compilation_says_so_in_one_line():
    """Absence of a report must never be how a clean compilation is recognised."""
    resolution = Resolution("mock", "Test-Task-v0", resolved={"reward/alive": "x"})
    assert resolution.is_clean
    assert resolution.summary_table() == "[mock] Test-Task-v0: 1 terms resolved, none skipped or emulated."
    assert resolution.manifest()["portable"] is True


def test_the_report_flags_a_task_that_reached_for_an_escape_hatch():
    resolution = Resolution("isaacsim", "T", resolved={"a": "b"}, engine_extras_used=("tiled_camera",))
    assert "not portable" in resolution.summary_table()
    assert resolution.manifest()["portable"] is False


def test_strict_mode_is_recorded_in_the_report():
    assert "strict" in Resolution("mock", "T", strict=True).summary_table()


def test_qualname_reports_the_function_rather_than_the_container():
    assert qualname_of(_NativeTerm(_term)).endswith("._term")
    assert qualname_of(_NativeTerm()).endswith("._NativeTerm")
