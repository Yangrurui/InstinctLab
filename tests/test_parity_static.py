"""Parity checks that need the golden file but not the engine.

``scripts/check_parity.py`` compares a compiled config against main's field by field, and it needs
Isaac Sim to run. The invariants below are the subset that can be checked from the declaration and
the golden dump alone, so they run in the ordinary test suite on any machine.

They are not a weaker version of the same check. Two of them -- observation order and reward
weights -- are the things that decide whether a policy trained against main's config can be loaded
against a compiled one, and a field-by-field diff does not test them: the diff compares paths, and
paths are keyed by name, so a reordered observation vector produces no difference at all.
"""

from __future__ import annotations

import ast
import json
import math
import re
import sys
from pathlib import Path

import pytest

from instinctlab.tasks.locomotion.flat_g1 import flat_g1

REPO = Path(__file__).resolve().parent.parent
GOLDEN_FILE = REPO / "tests/parity/isaacsim.locomotion_flat.golden.json"
WHITELIST_FILE = REPO / "tests/parity/isaacsim.locomotion_flat.whitelist.json"
ENGINE_DIR = REPO / "source/instinctlab/instinctlab/engines/isaacsim"

NOT_PORTABLE = {"dof_acc_l2", "dof_torques_l2", "feet_slide"}
"""Main's rewards that the portable task deliberately omits, per the design's P3 findings."""


@pytest.fixture(scope="module")
def golden() -> dict:
    if not GOLDEN_FILE.exists():
        pytest.skip(f"No golden dump at {GOLDEN_FILE}; run scripts/dump_golden.py under Isaac Sim.")
    return json.loads(GOLDEN_FILE.read_text())["config"]


@pytest.fixture(scope="module")
def task():
    return flat_g1()


def test_the_observation_vector_has_the_same_terms_in_the_same_order(golden, task):
    """Order is the whole content of this test.

    An observation group is concatenated in declaration order, so a policy's input layout is
    positional. Two configs listing the same terms in different orders compare equal field by field
    and produce checkpoints that cannot be loaded against each other.
    """
    settings = {"concatenate_terms", "concatenate_dim", "enable_corruption", "history_length", "flatten_history_dim"}
    for group in ("policy", "critic"):
        expected = [name for name in golden["observations"][group] if name not in settings]
        assert expected, f"The golden group {group!r} has no terms; the dump lost them."
        assert list(task.mdp.observations[group].terms) == expected, group


def test_the_reward_weights_match_main_except_where_a_term_was_dropped(golden, task):
    declared = {name: term.weight for name, term in task.mdp.rewards["rewards"].items()}
    expected = {name: cfg["weight"] for name, cfg in golden["rewards"].items() if name not in NOT_PORTABLE}
    assert declared == expected


def test_the_dropped_rewards_are_the_ones_the_design_names(golden, task):
    """Guards against a term going missing by accident and being read as a decision."""
    missing = set(golden["rewards"]) - set(task.mdp.rewards["rewards"])
    assert missing == NOT_PORTABLE


def test_every_joint_gets_the_action_scale_main_gives_it(golden, task):
    """The task keys action scale per joint; main keys it by actuator-group pattern.

    Equivalence has to be checked by expanding the patterns, and it is worth checking rather than
    asserting: the two are computed from different sources, so a change to the robot catalog could
    move one without the other.
    """
    patterns = golden["actions"]["joint_pos"]["scale"]
    declared = task.mdp.actions["joint_pos"].params["scale"]
    assert set(declared) == set(task.robot.joint_names)
    for joint, scale in declared.items():
        matched = [value for pattern, value in patterns.items() if re.fullmatch(pattern, joint)]
        assert len(matched) == 1, f"{joint} matched {len(matched)} of main's patterns"
        assert math.isclose(matched[0], scale, rel_tol=1e-12), joint


def test_the_terminations_match(golden, task):
    assert set(task.mdp.terminations) == set(golden["terminations"])
    for name, term in task.mdp.terminations.items():
        assert term.time_out == golden["terminations"][name]["time_out"], name


def test_the_events_match_by_name_and_mode(golden, task):
    assert set(task.mdp.events) == set(golden["events"])
    for name, term in task.mdp.events.items():
        assert term.mode == golden["events"][name]["mode"], name


def test_the_interval_event_keeps_its_period(golden, task):
    """The one event field that a builder can drop while everything still constructs."""
    assert task.mdp.events["push_robot"].interval_range_s == tuple(golden["events"]["push_robot"]["interval_range_s"])


def test_the_timing_matches(golden, task):
    assert task.sim.decimation == golden["decimation"]
    assert task.sim.episode_length_s == golden["episode_length_s"]
    assert task.sim.physics_dt == golden["sim"]["dt"]


def test_the_task_declares_no_engine_specific_escape_hatch(task):
    """A task needing ``engine_extras`` is not portable, and this one claims to be."""
    assert not task.engine_extras
    assert set(task.engines) == {"isaacsim", "mjlab"}


def test_the_declaration_imports_no_engine():
    """Static check, because the runtime one would pass on a machine with neither engine."""
    source = (REPO / "source/instinctlab/instinctlab/tasks/locomotion/flat_g1.py").read_text()
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = [name for name in imported if name.split(".")[0] in {"isaaclab", "mjlab", "omni", "mujoco", "isaacsim"}]
    assert not forbidden, f"The task declaration imports {forbidden}."


def test_the_isaac_backend_can_be_imported_without_isaac(monkeypatch):
    """The property that lets a task be checked against this engine on a machine without it.

    The registry's keys have to exist at import time, so ``contract_report`` can answer; the
    builders' bodies must not, so importing does not need the app running. Blocking the modules
    outright is the only honest way to test that, since the machine running this may have them.
    """
    for name in list(sys.modules):
        if name.startswith(("instinctlab.engines.isaacsim", "isaaclab")):
            monkeypatch.delitem(sys.modules, name, raising=False)

    class Blocker:
        def find_module(self, name, path=None):
            if name.split(".")[0] in {"isaaclab", "omni", "pxr", "carb", "isaacsim"}:
                raise AssertionError(f"Importing the Isaac Sim backend pulled in {name!r}.")

    monkeypatch.setattr(sys, "meta_path", [Blocker(), *sys.meta_path])
    import instinctlab.engines.isaacsim as backend

    assert backend.TERMS.capabilities(), "The registry advertises no capabilities at all."
    assert "joint_position" in backend.TERMS.kinds("action")


def test_the_backend_reports_that_it_can_run_this_task(task):
    from instinctlab.engines.isaacsim import IsaacSimAdapter

    report = IsaacSimAdapter().contract_report(task)
    assert report["missing"] == {}, report["missing"]
    assert report["engine_extras_used"] == []


def test_every_whitelisted_difference_carries_a_reason():
    """A whitelist entry is a decision someone has to be able to read back."""
    if not WHITELIST_FILE.exists():
        pytest.skip("No whitelist yet.")
    allow = json.loads(WHITELIST_FILE.read_text())
    assert allow
    for path, reason in allow.items():
        assert path == path.strip(".") and path, f"{path!r} is not a usable path prefix."
        assert len(reason.split()) >= 8, f"{path} is waved through with {reason!r}."


def test_no_whitelist_entry_would_swallow_a_whole_family():
    """A broad enough prefix silently accepts every future difference beneath it.

    ``rewards`` would explain away a changed weight; ``rewards.feet_slide`` explains one decision.
    The families named here are the ones where a blanket entry would make the parity check
    meaningless rather than merely lenient.
    """
    if not WHITELIST_FILE.exists():
        pytest.skip("No whitelist yet.")
    allow = json.loads(WHITELIST_FILE.read_text())
    too_broad = {"observations", "rewards", "terminations", "events", "actions", "commands", "scene", "sim"}
    assert not (too_broad & set(allow))
