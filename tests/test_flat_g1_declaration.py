"""What holds for the flat G1 declaration now that it is the only description of the task.

This file used to be the engine-free half of a parity check: ``scripts/check_parity.py`` compiled
the declaration and diffed it against a dump of main's ``G1FlatEnvCfg``, and the assertions here
covered the parts a field-by-field diff cannot see. That reference is gone. D3 was retired, main's
Isaac-only config was deleted, and ``config/flat_g1.py`` is now where the task is defined rather
than where it is restated.

So the questions changed. Nothing here asks whether the declaration agrees with something else --
there is no longer a something else. What is left are the properties the declaration has to have on
its own: that it names an engine nowhere, that the Isaac backend can answer for it without Isaac Sim
being installed, that the joint axis is pinned, and that the reward set is what it was. The last one
is a snapshot rather than a comparison, and is labelled as such where it sits: it catches a term
being lost by accident, and it says nothing at all about main.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from instinctlab.tasks.locomotion.config.flat_g1 import flat_g1

REPO = Path(__file__).resolve().parent.parent
DECLARATION = REPO / "source/instinctlab/instinctlab/tasks/locomotion/config/flat_g1.py"

NOT_PORTABLE = {"dof_acc_l2", "dof_torques_l2", "feet_slide"}
"""Rewards each backend implements itself, because the two engines do not measure the quantity the
same way. Present in the task, named by kind rather than by function."""

REWARDS = {
    "termination_penalty": -200.0,
    "track_lin_vel_xy_exp": 1.0,
    "track_ang_vel_z_exp": 1.0,
    "feet_air_time": 1.0,
    "feet_slide": -0.1,
    "flat_orientation_l2": -1.0,
    "stand_still": -0.8,
    "dof_pos_limits": -1.0,
    "joint_deviation_hip": -0.1,
    "joint_deviation_arms": -0.1,
    "joint_deviation_torso": -0.1,
    "joint_deviation_knee": -0.05,
    "lin_vel_z_l2": -0.1,
    "action_rate_l2": -0.05,
    "dof_acc_l2": -2e-07,
    "dof_torques_l2": -4e-06,
}
"""A snapshot of the objective, not a claim about any other implementation.

Sixteen weights decide what the policy optimises, and losing one is the kind of edit that keeps
training and changes the result. Written out so that changing the objective takes two edits and a
sentence in the commit message rather than one edit nobody sees.
"""


@pytest.fixture(scope="module")
def task():
    return flat_g1()


def test_the_objective_is_the_one_recorded_here(task) -> None:
    declared = {name: term.weight for name, term in task.mdp.rewards["rewards"].items()}
    assert declared == REWARDS, (
        "the reward set or a weight changed. If that was deliberate, update REWARDS in the same "
        "commit and say why; existing checkpoints were trained against the old objective."
    )


def test_the_terms_named_by_kind_are_the_ones_the_design_names(task) -> None:
    """Guards the reverse mistake: a portable term being replaced by an engine-specific one."""
    by_kind = {name for name, term in task.mdp.rewards["rewards"].items() if term.kind is not None}
    assert by_kind == NOT_PORTABLE


def test_the_task_declares_no_engine_specific_escape_hatch(task) -> None:
    """A task needing ``engine_extras`` is not portable, and this one claims to be."""
    assert not task.engine_extras
    assert set(task.engines) == {"isaacsim", "mjlab"}


def test_the_declaration_imports_no_engine() -> None:
    """Static check, because the runtime one would pass on a machine with neither engine."""
    imported = set()
    for node in ast.walk(ast.parse(DECLARATION.read_text())):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = [name for name in imported if name.split(".")[0] in {"isaaclab", "mjlab", "omni", "mujoco", "isaacsim"}]
    assert not forbidden, f"The task declaration imports {forbidden}."


def test_the_isaac_backend_can_be_imported_without_isaac(monkeypatch) -> None:
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


def test_the_backend_reports_that_it_can_run_this_task(task) -> None:
    from instinctlab.engines.isaacsim import IsaacSimAdapter

    report = IsaacSimAdapter().contract_report(task)
    assert report["missing"] == {}, report["missing"]
    assert report["engine_extras_used"] == []


def test_the_joint_axis_is_pinned_to_the_canonical_order() -> None:
    """Decision D1, asserted where it is actually decided rather than where it is documented.

    Selecting with a lone ``".*"`` picks the same twenty-nine joints, and ``preserve_order=True``
    next to it looks like it settles their order. It does not: ``resolve_matching_names`` orders a
    selection by the *patterns* it was given, so a single pattern falls back to the entity's own
    order and the flag becomes a no-op. The task therefore has to name every joint, and the
    difference used to be invisible in every check that did not compare sequences: the asset test
    compared sets, and the term-value comparison reindexed both engines to this order before diffing,
    so it agreed either way. Both have since been fixed -- the asset test pins the sequence and the
    comparison no longer reindexes -- and this is the assertion on the declaration itself.

    Both the action term and the two joint observations are checked, because pinning one without the
    other leaves a policy whose inputs and outputs are indexed differently per engine.
    """
    spec = flat_g1()
    canonical = tuple(spec.robot.joint_names)

    selectors = {"actions.joint_pos": spec.mdp.actions["joint_pos"].target}
    for group, group_spec in spec.mdp.observations.items():
        for name in ("joint_pos", "joint_vel"):
            selectors[f"observations.{group}.{name}"] = group_spec.terms[name].params["asset_cfg"]

    for path, ref in selectors.items():
        assert ref is not None, f"{path} selects no entity"
        assert tuple(ref.joints) == canonical, (
            f"{path} selects {list(ref.joints)!r}; a lone pattern leaves the engine's own order in "
            "place, so the joints have to be named in the canonical depth-first order"
        )
        assert ref.preserve_order is True, f"{path} names the joints but does not ask for their order"
