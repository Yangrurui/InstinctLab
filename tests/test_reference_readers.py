"""Guard: the reference readers must answer, not shrug.

``tests/reference_main_parkour.py`` and ``tests/reference_mjlab_parkour.py`` read facts off the
references' syntax trees. Every audit assertion is only as true as they are, and their failure mode
is not an exception -- it is an empty dict, or a dict of ``None``, which reads exactly like "the
reference does not set this".

That failure has already shipped once. ``sim_params`` probed for substrings in ``ast.unparse``
output, which writes ``2 ** 29`` where the file writes ``2**29``, so every power of two came back
``None``. The ``gpu_collision_stack_size`` drift row then reported main's value as the string
``"None"`` and passed for weeks, because a drift row only has to differ from ours.

So: no reader may return nothing, and no reader may return all-``None`` unless it is listed below as
legitimately absent from the reference. A new reader that cannot parse its target fails here rather
than quietly widening the set of things we believe the references do not do.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from tests import reference_main_parkour as main_ref
from tests import reference_mjlab_parkour as mj_ref

OBS_GROUPS = ("policy", "critic", "amp_policy", "amp_reference")

# (module, accessor, group or None) whose values are all None *in the reference itself*.
LEGITIMATELY_ABSENT = {
    ("reference_main_parkour", "observation_clip", "policy"),
    ("reference_main_parkour", "observation_clip", "critic"),
    ("reference_main_parkour", "observation_clip", "amp_policy"),
    ("reference_main_parkour", "observation_clip", "amp_reference"),
    ("reference_main_parkour", "observation_noise", "critic"),
    ("reference_main_parkour", "observation_noise", "amp_policy"),
    ("reference_main_parkour", "observation_noise", "amp_reference"),
}

# Readers that legitimately come back empty, with the reason they do.
# mjlab's observation_noise collects only the terms that carry a noise call, unlike main's, which
# returns every term with None where there is none. InstinctMJ's critic group has no noisy term, so
# empty is that reader's way of saying so.
LEGITIMATELY_EMPTY = {
    ("reference_mjlab_parkour", "observation_noise", "critic"),
}

# Readers that answer a yes/no question, where False is an answer rather than a parse failure.
BOOLEAN_READERS = frozenset(
    {
        "available",
        "isaac_observation_term_flatten_history_default",
        "main_wrapper_sets_missing_step_dict",
        "train_script_calls_configure_torch_backends",
        "uses_instinct_rl_env",
        "uses_multi_reward_cfg",
        "wrapper_sets_missing_step_dict",
    }
)


def _nullary_readers(module: Any) -> list[tuple[str, Any]]:
    return [
        (name, fn)
        for name, fn in sorted(vars(module).items())
        if not name.startswith("_")
        and inspect.isfunction(fn)
        and fn.__module__ == module.__name__
        and not inspect.signature(fn).parameters
    ]


def _check(module_name: str, accessor: str, group: str | None, value: Any) -> None:
    where = f"{module_name}.{accessor}" + (f"({group!r})" if group else "()")
    assert value is not None, f"{where} returned None; a reader that cannot parse must raise"
    if isinstance(value, bool):
        return
    if not len(value):
        assert (module_name, accessor, group) in LEGITIMATELY_EMPTY, (
            f"{where} returned nothing. Either the reference really has none of these -- add it to "
            "LEGITIMATELY_EMPTY with that fact -- or the parser stopped finding them."
        )
        return
    if isinstance(value, dict) and all(item is None for item in value.values()):
        assert (module_name, accessor, group) in LEGITIMATELY_ABSENT, (
            f"{where} parsed {len(value)} entries but every value is None. Either the reference "
            "really sets none of them -- add it to LEGITIMATELY_ABSENT with that fact -- or the "
            "parser is missing them and every assertion built on it is vacuous."
        )


@pytest.mark.parametrize("module", (main_ref, mj_ref), ids=("main", "mjlab"))
def test_every_nullary_reader_answers(module) -> None:
    if module is mj_ref and not mj_ref.available():
        pytest.skip("InstinctMJ is not checked out")
    readers = _nullary_readers(module)
    assert len(readers) >= 15, f"only found {len(readers)} readers in {module.__name__}"
    for name, fn in readers:
        value = fn()
        assert not (name in BOOLEAN_READERS) or isinstance(value, bool), f"{name} is not boolean"
        _check(module.__name__.rsplit(".", 1)[-1], name, None, value)


@pytest.mark.parametrize("module", (main_ref, mj_ref), ids=("main", "mjlab"))
def test_every_per_group_reader_answers(module) -> None:
    if module is mj_ref and not mj_ref.available():
        pytest.skip("InstinctMJ is not checked out")
    short = module.__name__.rsplit(".", 1)[-1]
    for name, fn in sorted(vars(module).items()):
        if name.startswith("_") or not inspect.isfunction(fn) or fn.__module__ != module.__name__:
            continue
        if list(inspect.signature(fn).parameters) != ["group"]:
            continue
        for group in OBS_GROUPS:
            try:
                value = fn(group)
            except KeyError:
                continue  # The group is absent from this reference, and saying so loudly is fine.
            _check(short, name, group, value)


def test_sim_params_would_catch_the_unparse_spacing_bug() -> None:
    """The specific regression, kept as its own row: powers of two must survive the reader."""
    params = main_ref.sim_params()
    assert params["gpu_collision_stack_size"] == 2**29
    assert params["gpu_max_rigid_patch_count"] == 10 * 2**15
