"""The declared task, checked against InstinctMJ's locomotion config.

This is now the only structural comparison the declaration has against an outside implementation.
It had a counterpart on the Isaac side, checking the same ``TaskSpec`` against main's env config,
and agreeing with two references that knew nothing about each other was the evidence that the
declaration carried the task rather than one engine's spelling of it. That half went with D3, so
what remains is one reference -- and it is the one this repository cannot edit, which is most of
what made it worth having.

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
from instinctlab.tasks.locomotion.config.g1 import FEET_CONTACT, UPPER_BODY_CONTACT, flat_g1

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
        "config is arranged the second way, and mjlab accepts either. Structural rather than lazy: "
        "two arrangements of the same measurement cannot be compared field by field. What is "
        "comparable is compared -- test_the_contact_groups_are_the_reference_groups resolves both "
        "arrangements against the robot's bodies and requires the same two groups, and "
        "test_the_declared_sensor_keeps_the_references_timing carries over the history depth and "
        "air-time tracking. For a while this entry exempted the sensors and nothing else looked at "
        "them at all, which is how a missing field left every contact timer at zero for a whole "
        "training run; see tests/test_mjlab_contact_wiring.py for the live check."
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


# The reference and the declaration spell one parameter differently for the same quantity. Stated as
# data so a second one has to be added here rather than absorbed into an assertion.
EVENT_PARAM_ALIASES = {"add_range": "mass_distribution_params"}


def test_every_event_randomises_over_the_reference_range(spec):
    """The ranges, not just the names and periods.

    Weights and term names are what a config diff shows; the intervals are what the randomisation
    actually is. A mass range of +/-5 kg against +/-2 kg is the same term, the same mode and the
    same period, and a different task.
    """
    for name, event in reference.events().items():
        if name == "physics_material":
            continue  # See EXPECTED_DIFFERENCES: the interval is the engine profile's, by design.
        declared = dict(spec.mdp.events[name].params)
        for ours, theirs in EVENT_PARAM_ALIASES.items():
            if ours in declared:
                declared[theirs] = declared.pop(ours)
        expected = {
            key: value
            for key, value in event["params"].items()
            # Entity configs read off a syntax tree are markers rather than values; what they select
            # is compared where it can be resolved, in the reward tests below.
            if not (isinstance(value, str) and value.startswith("<"))
        }
        assert {key: declared.get(key) for key in expected} == expected, name


def test_every_reward_is_computed_by_the_reference_implementation(rewards):
    """Name and weight agreeing is not the same as measuring the same thing.

    Two of these differ from what the name suggests -- ``track_lin_vel_xy_exp`` is computed by
    ``track_lin_vel_xy_yaw_frame_exp`` and ``track_ang_vel_z_exp`` by ``track_ang_vel_z_world_exp``
    -- so the term names alone would not have caught a substitution.
    """
    for name, function in reference.reward_functions().items():
        term = rewards[name]
        ours = term.func.__name__ if term.func is not None else term.kind
        assert ours == function, f"{name} is computed by {ours} where the reference uses {function}"


def test_every_reward_charges_for_the_reference_elements(spec, rewards):
    """Which joints and bodies a term selects, resolved rather than compared as text.

    The reference names both feet outright where the declaration matches them with a pattern. That
    is the same selection only if it resolves to the same names in the same order, so this resolves
    both against the robot's own lists instead of accepting the difference on sight.
    """
    import re

    catalogue = {"joint": list(spec.robot.joint_names), "body": list(spec.robot.body_names)}

    def resolve(kind: str, patterns) -> list[str]:
        return [name for name in catalogue[kind] if any(re.fullmatch(p, name) for p in patterns)]

    checked = 0
    for name, params in reference.entity_selectors().items():
        for key, selection in params.items():
            declared = rewards[name].params.get(key)
            selectors = declared.selectors() if hasattr(declared, "selectors") else {}
            for kind in ("joint", "body"):
                theirs = selection.get(f"{kind}_names")
                if theirs is None:
                    continue
                assert resolve(kind, selectors[kind]) == resolve(kind, theirs), f"{name}.{key} {kind}s"
                checked += 1
    assert checked >= 8, f"only {checked} selections compared; the extractor stopped seeing them"


def test_every_reward_takes_the_reference_parameters(rewards):
    """The scalars: tracking widths, contact thresholds, which command a term reads."""
    for name, params in reference.reward_params().items():
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("<"):
                continue  # An entity config; compared by what it selects, above.
            if key == "sensor_name":
                continue  # The reference's two narrow sensors against one wide one, per EXPECTED_DIFFERENCES.
            assert rewards[name].params.get(key) == value, f"{name}.{key}"


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


def test_the_action_is_the_reference_action(spec):
    action = spec.mdp.actions["joint_pos"]
    assert action.kind == "joint_position"
    assert action.params["use_default_offset"] is True


def _matching(patterns, names):
    """Bodies the patterns select, as a set. Both sides spell the same selection differently."""
    import re

    chosen = {name for name in names for pattern in patterns if re.fullmatch(pattern, name)}
    unused = [pattern for pattern in patterns if not any(re.fullmatch(pattern, name) for name in names)]
    assert not unused, f"{unused} match no body; a pattern that selects nothing hides what it meant to select"
    return chosen


def test_the_contact_groups_are_the_reference_groups(spec):
    """What the two arrangements of contact sensors actually measure, which is comparable.

    The reference splits feet from upper body at the sensor and this task splits them at the term,
    so the declarations cannot be diffed field by field -- which is why ``scene.sensors`` is an
    expected difference. That exemption was doing more work than it should: the extractor that reads
    these sensors out of the reference had no caller at all, so nothing compared the two
    arrangements in any form, and contact is where this project's one silent training failure came
    from.

    Resolved against the robot's bodies, the arrangements do become comparable. Each side ends up
    measuring two groups; the groups have to hold the same bodies, however each side spells them.
    """
    sensors = reference.scene_sensors()
    assert set(sensors) == {"feet_contact_forces", "base_contact_forces"}, (
        f"the reference now declares {sorted(sensors)}; the mapping onto this task's single sensor "
        "plus per-term selections has to be redrawn"
    )

    bodies = spec.robot.body_names
    groups = {
        "feet_contact_forces": FEET_CONTACT,
        "base_contact_forces": UPPER_BODY_CONTACT,
    }
    for name, ref in groups.items():
        elements = (ref.elements,) if isinstance(ref.elements, str) else ref.elements
        assert _matching(elements, bodies) == _matching(
            sensors[name]["primary"]["pattern"], bodies
        ), f"the {name} group covers different bodies than the term that replaces it"
        assert sensors[name]["primary"]["mode"] == "body"
        assert sensors[name]["primary"]["entity"] == ref.entity


def test_the_declared_sensor_keeps_the_references_timing(spec):
    """History depth and air-time tracking, which survive the regrouping and have to carry over.

    ``track_air_time`` is what makes mjlab accumulate contact and air duration at all, and the
    portable terms take contact from that duration rather than from a force threshold. Getting it
    from the reference rather than from a constant here means an upstream that stops asking for it
    shows up as a failure instead of as a foot that is never in contact.
    """
    sensors = reference.scene_sensors()
    declared = {sensor.name: sensor for sensor in spec.scene.contact_sensors}
    assert set(declared) == {
        "contact_forces"
    }, f"one sensor was the arrangement compared above; found {sorted(declared)}"
    sensor = declared["contact_forces"]

    assert sensor.track_air_time is sensors["feet_contact_forces"]["track_air_time"] is True
    assert sensor.history_length == sensors["feet_contact_forces"]["history_length"]

    # The one property that deliberately does not carry over, asserted so that it stays deliberate.
    # The reference's subclass rederives contact from a newton threshold; a plain ContactSensorCfg
    # has no such threshold and must request the "found" field instead, which is exactly the field
    # whose absence once left every contact timer at zero for a whole training run.
    assert sensors["feet_contact_forces"]["cfg_class"] == "ForceThresholdContactSensorCfg"
    assert "found" not in sensors["feet_contact_forces"]["fields"]


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
