"""The PPO configuration, pinned against both reference implementations.

Written because ``flat_g1_ppo.py`` claimed in its own docstring that this file pinned every field,
and this file did not exist. The configuration had been moved out of ``config/g1/agents/`` so it
could be read without Isaac Sim, and a move is exactly when values go missing quietly: nothing
imports a hyperparameter by name, so a dropped one shows up as a slightly different learning curve
weeks later.

Two references, checked differently. Main's are read out of ``main`` itself and compared whole:
its copy of this file is still there, and building it here needs nothing but a substitute for the
one Isaac Lab import it makes. InstinctMJ's are read from the ``agent.yaml`` its own training run
wrote, which is better evidence than its source: it is what the reference run actually trained with.

Main's values used to be transcribed into a table here, and the transcription was the flaw. A table
can only be consulted in one direction -- every entry was looked up in the configuration and found
to agree -- so a *new* field, one main never had, was invisible. Training would have quietly used a
hyperparameter the reference does not, which is the failure this file exists to prevent.
"""

from __future__ import annotations

import glob
import subprocess
import sys
import types
import yaml
from pathlib import Path

import pytest

from instinctlab.tasks.locomotion.flat_g1_ppo import G1FlatPPORunnerCfg
from instinctlab.utils.configclass import class_to_dict

MAIN_CFG = "source/instinctlab/instinctlab/tasks/locomotion/config/g1/agents/instinct_rl_ppo_cfg.py"
REPO = Path(__file__).resolve().parents[1]

REFERENCE_RUN = "/root/InstinctMJ/logs/instinct_rl/g1_locomotion_flat/*/params/agent.yaml"


def _flatten(tree: dict, prefix: str = "") -> dict:
    flat: dict = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


@pytest.fixture(scope="module")
def declared() -> dict:
    return _flatten(class_to_dict(G1FlatPPORunnerCfg()))


@pytest.fixture(scope="module")
def mains() -> dict:
    """Main's own runner configuration, built from the source on that branch.

    The single Isaac Lab import is swapped for the vendored ``configclass``, which
    ``tests/test_configclass_vendor.py`` pins against upstream function by function. Nothing else in
    main's file touches a simulator, so this runs with no engine installed.
    """
    shown = subprocess.run(("git", "show", f"main:{MAIN_CFG}"), cwd=REPO, capture_output=True, text=True)
    assert shown.returncode == 0, f"cannot read {MAIN_CFG} on main: {shown.stderr.strip()}"
    source = shown.stdout.replace(
        "from isaaclab.utils import configclass", "from instinctlab.utils.configclass import configclass"
    )
    assert "isaaclab" not in source, "main's configuration grew an Isaac Lab import that this cannot substitute"

    # Registered in sys.modules because dataclasses resolves a field's annotations by looking its
    # class's module back up there; an unregistered module makes that lookup return None.
    name = "_mains_ppo_cfg"
    module = types.ModuleType(name)
    module.__dict__["__name__"] = name
    sys.modules[name] = module
    try:
        exec(compile(source, f"main:{MAIN_CFG}", "exec"), module.__dict__)  # noqa: S102 - the reference, read-only
        return _flatten(class_to_dict(module.G1FlatPPORunnerCfg()))
    finally:
        del sys.modules[name]


def test_the_configuration_is_mains_field_for_field(declared, mains) -> None:
    """Both directions at once: a changed value, a dropped field and an added one all fail here."""
    assert declared == mains, {
        key: (mains.get(key, "<absent on main>"), declared.get(key, "<absent here>"))
        for key in sorted(set(mains) | set(declared))
        if mains.get(key, "<absent on main>") != declared.get(key, "<absent here>")
    }


def test_the_comparison_covers_the_whole_configuration(mains) -> None:
    """A reference that came back empty would make the test above pass without comparing anything."""
    assert len(mains) >= 30, f"main's configuration flattened to only {len(mains)} fields"


def test_the_run_this_trains_matches_the_reference_run(declared) -> None:
    """Against the yaml InstinctMJ's own 5000-iteration run wrote, not against its source.

    The reference's runner declares many more fields than this one -- MoE, VAE, distillation and an
    AMP discriminator -- and every one of them is inert in that run. Only the fields both declare
    are compared; a field only it has is checked to be switched off, since a reference that was
    quietly doing something extra would make the two runs incomparable however well the shared
    fields agreed.
    """
    matches = sorted(glob.glob(REFERENCE_RUN))
    if not matches:
        pytest.skip("no InstinctMJ training run to compare against")
    reference = _flatten(yaml.safe_load(Path(matches[-1]).read_text()))

    shared = {key: value for key, value in reference.items() if key in declared}
    assert len(shared) >= 30, f"only {len(shared)} fields in common; the reference's format changed"
    differing = {key: (value, declared[key]) for key, value in shared.items() if declared[key] != value}
    assert not differing, f"the two runs differ: {differing}"

    extra_features = ("kl_loss", "teacher", "distill", "discriminator", "moe", "vae", "encoder")
    inert = (None, 0, 0.0, False, "", {}, [])
    switched_on = {
        key: value
        for key, value in reference.items()
        if key not in declared and any(part in key for part in extra_features) and value not in inert
    }
    assert not switched_on, f"the reference run enabled {switched_on}, so it is not the same algorithm"
