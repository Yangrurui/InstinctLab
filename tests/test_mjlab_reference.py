"""The declared task, checked against InstinctMJ's locomotion config.

The counterpart of ``test_parity_static.py``, and the pair is the actual claim. That file checks one
``TaskSpec`` against main's Isaac Lab config; this one checks the same ``TaskSpec`` against
InstinctMJ's mjlab config. Neither reference knows about the other, and neither was adjusted to
meet the declaration -- so agreeing with both is evidence that the declaration carries the task
rather than a translation of one engine's spelling of it.

Reading rather than importing: InstinctMJ is not installed, per decision D3. The facts come off its
syntax tree, so these tests state what its source says.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import reference_mjlab as reference

from instinctlab.spec.capability import Requirement
from instinctlab.tasks.locomotion.flat_g1 import flat_g1

pytestmark = pytest.mark.skipif(not reference.available(), reason="InstinctMJ is not checked out")


@pytest.fixture(scope="module")
def spec():
    return flat_g1()


@pytest.fixture(scope="module")
def rewards(spec):
    return {name: term for group in spec.mdp.rewards.values() for name, term in group.items()}


# --------------------------------------------------------------------------------------------
# Differences that are meant to be there.
#
# Kept as data so a new one has to be written down. Each entry says what differs and why the two
# still describe the same task; anything not listed is a failure.
# --------------------------------------------------------------------------------------------

EXPECTED_DIFFERENCES = {
    "observations.critic": (
        "The reference declares a critic group with an extra base_lin_vel term, and so does the "
        "declaration; only the policy group is compared term by term because it is the one whose "
        "layout a trained policy depends on."
    ),
    "scene.sensors": (
        "The reference declares two narrow contact sensors, feet and base; the declaration names "
        "one covering every body and terms select within it. Same elements measured. Isaac Lab's "
        "config is arranged the second way, and mjlab accepts either. This exemption is the one "
        "blind spot in this file, and it is structural rather than lazy: two arrangements of the "
        "same measurement cannot be compared field by field. It already cost one silent failure "
        "-- see tests/test_mjlab_contact_wiring.py -- so what the sensor does is checked against a "
        "running engine there instead of against the reference here."
    ),
    "scene.sensors.force_threshold": (
        "The reference's foot sensor is a ForceThresholdContactSensorCfg with a 1 N threshold, "
        "which is how it reproduces Isaac Lab's force-thresholded air time. The portable terms "
        "take contact from the sensor's own contact duration instead, so no threshold is imposed "
        "at the sensor. Air time therefore starts marginally earlier here."
    ),
    "rewards.groups": (
        "The reference groups rewards for MultiRewardManager; both backends flatten the groups the "
        "declaration states, since each engine's stock manager takes a flat namespace."
    ),
    "events.physics_material": (
        "The reference collapses main's static and dynamic friction intervals into MuJoCo's single "
        "sliding coefficient and drops restitution. The declaration states neither: the interval "
        "is in the engine profile, and it is the same collapse."
    ),
}


def test_the_policy_observation_vector_is_the_reference_vector(spec):
    """Same terms in the same order. A policy's input layout is this list."""
    assert [name for name, _ in reference.observation_terms("policy")] == list(spec.mdp.observations["policy"].terms)


def test_every_policy_observation_carries_the_reference_noise(spec):
    declared = {
        name: (term.noise.lo, term.noise.hi)
        for name, term in spec.mdp.observations["policy"].terms.items()
        if term.noise is not None
    }
    assert declared == reference.observation_noise("policy")


def test_the_reward_set_matches_the_reference_exactly(rewards):
    """Including the three terms that are engine-specific rather than portable."""
    assert list(rewards) == list(reference.rewards())


def test_every_reward_weight_matches_the_reference(rewards):
    assert {name: term.weight for name, term in rewards.items()} == reference.rewards()


def test_the_terms_the_reference_implements_itself_are_the_ones_declared_by_kind(rewards):
    """The three terms whose two implementations differ in what they measure.

    The reference reaches outside mjlab for exactly these, which is the same judgement the
    declaration makes when it names them by kind instead of by function.
    """
    by_kind = {name for name, term in rewards.items() if term.kind is not None}
    assert by_kind == {"feet_slide", "dof_acc_l2", "dof_torques_l2"}
    for name in by_kind:
        assert rewards[name].level is Requirement.REQUIRED, (
            f"{name} is stated by kind, so an engine without it would silently change training; "
            "it must be REQUIRED rather than skipped."
        )


def test_the_terminations_match(spec):
    assert list(spec.mdp.terminations) == list(reference.terminations())


def test_the_events_match_by_name_mode_and_period(spec):
    declared = {name: (term.mode, term.interval_range_s) for name, term in spec.mdp.events.items()}
    expected = {name: (event["mode"], event["interval_range_s"]) for name, event in reference.events().items()}
    assert declared == expected


def test_the_timing_matches(spec):
    timing = reference.timing()
    assert spec.sim.physics_dt == timing["timestep"]
    assert spec.sim.decimation == timing["decimation"]
    assert spec.sim.episode_length_s == timing["episode_length_s"]


def test_the_solver_settings_live_in_the_profile_not_the_task(spec):
    """A task states none of these, and the mjlab profile states all of them at the reference's values."""
    from instinctlab.engines.mjlab.scene import PROFILE_DEFAULTS

    timing = reference.timing()
    for field in ("solver", "iterations", "ls_iterations", "ccd_iterations"):
        assert PROFILE_DEFAULTS[field] == timing[field]
    assert not any(field in spec.sim.profiles.get("mjlab", {}) for field in ("solver", "iterations"))


def test_the_command_ranges_match(spec):
    command = reference.commands()
    declared = spec.mdp.commands[command["name"]].params
    for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        assert declared[key] == command["ranges"][key], key
    assert declared["resampling_time_range"] == command["resampling_time_range"]
    assert declared["rel_standing_envs"] == command["rel_standing_envs"]
    assert declared["heading_control_stiffness"] == command["heading_control_stiffness"]


def test_the_friction_range_is_the_collapse_the_reference_performs():
    """MuJoCo has one sliding coefficient where PhysX has two, and this is where they meet.

    The reference merges main's static ``(0.25, 0.8)`` and dynamic ``(0.2, 0.6)`` intervals by
    taking the lower low and the higher high. The profile has to hold that same interval, or the
    two mjlab runs randomise differently.
    """
    from instinctlab.engines.mjlab.scene import PROFILE_DEFAULTS

    source = reference.REFERENCE.read_text()
    module = ast.parse(source)
    material = reference.events()["physics_material"]["params"]
    expected = (
        min(material["static_friction_range"][0], material["dynamic_friction_range"][0]),
        max(material["static_friction_range"][1], material["dynamic_friction_range"][1]),
    )
    assert PROFILE_DEFAULTS["friction_dr"]["ranges"] == expected
    assert PROFILE_DEFAULTS["friction_dr"]["operation"] == "abs"
    assert PROFILE_DEFAULTS["friction_dr"]["shared_random"] is True
    assert isinstance(module, ast.Module)


def test_the_action_is_the_reference_action(spec):
    action = spec.mdp.actions["joint_pos"]
    source = reference.REFERENCE.read_text()
    assert action.kind == "joint_position"
    assert "use_default_offset=True" in source
    assert action.params["use_default_offset"] is True


def test_every_expected_difference_says_why():
    for path, reason in EXPECTED_DIFFERENCES.items():
        assert len(reason) > 60, f"{path} needs a reason, not a label."


def test_the_mjlab_backend_can_be_imported_without_mjlab(monkeypatch):
    """Registry keys must exist without mjlab, so a task can be checked against this engine anywhere."""
    import importlib

    for name in [name for name in sys.modules if name.startswith(("mjlab", "instinctlab.engines.mjlab"))]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def blocked(name, *args, **kwargs):
        if name == "mjlab" or name.startswith("mjlab."):
            raise ImportError(f"{name} is unavailable in this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked)
    module = importlib.import_module("instinctlab.engines.mjlab")
    assert module.MjlabAdapter().capabilities().values


def test_the_backend_reports_that_it_can_run_this_task(spec):
    """No term unsupported, no term emulated: the report is empty or it is not the same task."""
    from instinctlab.engines.mjlab import MjlabAdapter

    report = MjlabAdapter().contract_report(spec)
    assert report["missing"] == {}
    assert report["engine_extras_used"] == []
