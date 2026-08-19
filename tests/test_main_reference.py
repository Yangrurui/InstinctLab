"""The files that are supposed to *be* main's are checked against main.

Decision D3 makes ``G1FlatEnvCfg`` the single golden: the Isaac path is correct when it reproduces
main, and ``tests/parity`` measures everything against a dump of it. That argument only holds while
the file is actually main's, and nothing checked. It had carried twenty-two lines of local edits --
spawn self-collision, rigid-body and solver settings, plus three scene flags -- since the backend
cleanup in 4806241, so the golden was a dump of a modified main and every "matches main" result was
measured against the wrong reference. Four separate audits read this file and none noticed, because
reading it tells you nothing: it looks like a plausible env config either way.

So the reference is consulted rather than remembered. Files listed as untouched must be byte-equal
to main; files listed as edited must differ, so an entry whose edit was reverted upstream shows up
instead of sitting here claiming a difference that no longer exists.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

UNTOUCHED = (
    "source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py",
    "source/instinctlab/instinctlab/tasks/locomotion/config/g1/__init__.py",
    "source/instinctlab/instinctlab/tasks/locomotion/config/g1/agents/__init__.py",
    "source/instinctlab/instinctlab/tasks/locomotion/config/__init__.py",
    "source/instinctlab/instinctlab/tasks/locomotion/__init__.py",
    "source/instinctlab/instinctlab/tasks/locomotion/mdp/rewards.py",
    "source/instinctlab/instinctlab/tasks/locomotion/mdp/curriculums.py",
)
"""Main's, verbatim. The golden is dumped from the first of these."""

EDITED = {
    "source/instinctlab/instinctlab/tasks/locomotion/mdp/__init__.py": (
        "star imports became a lazy __getattr__ with the same precedence, after the retired unified "
        "stack shadowed feet_air_time_positive_biped with an incompatible signature"
    ),
    "source/instinctlab/instinctlab/tasks/locomotion/config/g1/agents/instinct_rl_ppo_cfg.py": (
        "re-exports flat_g1_ppo so reading a hyperparameter does not require Isaac Sim"
    ),
    "source/instinctlab/instinctlab/assets/unitree_g1.py": (
        "actuator constants and spawn paths now come from unitree_g1_spec instead of being repeated"
    ),
    "source/instinctlab/instinctlab/assets/__init__.py": "names the two G1 modules",
    # The training path, which the parity argument covers just as much as the env config does.
    "source/instinctlab/instinctlab/utils/wrappers/instinct_rl/vecenv_wrapper.py": (
        "num_rewards falls back to 1, because a task whose rewards are one flat container runs on a "
        "plain ManagerBasedRLEnv, which does not declare the attribute InstinctRlEnv does"
    ),
    "source/instinctlab/instinctlab/utils/wrappers/instinct_rl/module_cfg.py": (
        "imports the vendored configclass, so reading a config does not start Isaac Sim"
    ),
    "source/instinctlab/instinctlab/utils/wrappers/instinct_rl/rl_cfg.py": "same vendored configclass swap",
    "source/instinctlab/instinctlab/utils/wrappers/instinct_rl/__init__.py": "resolves wrappers on access",
    "source/instinctlab/instinctlab/utils/wrappers/__init__.py": (
        "resolves wrappers on access, so this package does not require every engine to be installed"
    ),
    "source/instinctlab/instinctlab/utils/math.py": (
        "imports compat.math rather than isaaclab.utils.math; test_compat_math pins the two bitwise"
    ),
    # Package fronts that used to import Isaac Sim as a side effect of importing InstinctLab.
    "source/instinctlab/instinctlab/__init__.py": (
        "stops registering Gym ids on import, so the package is engine-neutral"
    ),
    "source/instinctlab/instinctlab/tasks/__init__.py": (
        "the same registration moved into register_legacy_isaac_tasks()"
    ),
    "source/instinctlab/instinctlab/envs/__init__.py": "exports lazily",
    "source/instinctlab/instinctlab/managers/__init__.py": "exports lazily",
    "source/instinctlab/instinctlab/sim/__init__.py": (
        "now fronts the engine-neutral contract; legacy spawners load lazily"
    ),
    "source/instinctlab/instinctlab/tasks/parkour/scripts/play.py": (
        "calls register_legacy_isaac_tasks() before using a Gym id"
    ),
    "source/instinctlab/instinctlab/tasks/shadowing/play.py": (
        "calls register_legacy_isaac_tasks() before using a Gym id"
    ),
    "source/instinctlab/setup.py": "declares the new packages",
}
"""Main's, with a deliberate edit. The text says which one."""


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(("git", *args), cwd=REPO, capture_output=True, text=True)


@pytest.fixture(scope="module")
def main_ref() -> str:
    """Whichever ref names the release branch here.

    Resolution failure is an error rather than a skip: a skip would read as a pass and leave the
    comparison silently not happening, which is the failure mode this module exists to close.
    """
    for candidate in ("main", "origin/main"):
        if _git("rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}").returncode == 0:
            return candidate
    raise AssertionError("neither 'main' nor 'origin/main' resolves, so nothing can be compared to the release")


def _on_main(ref: str, path: str) -> str:
    shown = _git("show", f"{ref}:{path}")
    assert shown.returncode == 0, f"{path} does not exist on {ref}: {shown.stderr.strip()}"
    return shown.stdout


@pytest.mark.parametrize("path", UNTOUCHED)
def test_the_files_that_are_mains_still_are(main_ref: str, path: str) -> None:
    ours = (REPO / path).read_text()
    theirs = _on_main(main_ref, path)
    assert ours == theirs, (
        f"{path} differs from {main_ref}, but the parity argument treats it as main's own. "
        "Either revert the edit or move the file into EDITED with the reason, and re-dump the golden."
    )


@pytest.mark.parametrize("path", sorted(EDITED))
def test_the_recorded_edits_are_still_edits(main_ref: str, path: str) -> None:
    ours = (REPO / path).read_text()
    theirs = _on_main(main_ref, path)
    assert ours != theirs, f"{path} now matches {main_ref}; drop it from EDITED so the list keeps meaning something"


def test_every_file_here_is_one_main_has(main_ref: str) -> None:
    """Neither list may name a file the release does not have, which would make its entry vacuous."""
    for path in (*UNTOUCHED, *sorted(EDITED)):
        assert _git("cat-file", "-e", f"{main_ref}:{path}").returncode == 0, f"{path} is not on {main_ref}"


def test_no_edit_of_mains_goes_unrecorded(main_ref: str) -> None:
    """The tables above are only as good as their coverage, and coverage was how the last one hid.

    ``flat_env_cfg.py`` was not exempted from a check; it was never listed, and a list nobody is
    obliged to complete says nothing about what is missing from it. So the question is asked the
    other way round here: git names every file of main's we touched, and each one has to appear
    above with a reason. Adding a file to the list is cheap; forgetting to is what costs.
    """
    changed = _git("diff", main_ref, "--name-only", "--diff-filter=M", "--", "source/")
    assert changed.returncode == 0, changed.stderr
    unrecorded = {path for path in changed.stdout.split() if path} - set(EDITED)
    assert not unrecorded, (
        f"these files of {main_ref}'s were modified without saying why:\n  "
        + "\n  ".join(sorted(unrecorded))
        + "\n"
        "Put each in EDITED with the reason, or revert it. Anything reachable from G1FlatEnvCfg or "
        "the training loop also needs the golden re-dumped."
    )


def test_the_isaac_profile_randomises_friction_the_way_main_does() -> None:
    """``PROFILE_DEFAULTS['friction_dr']`` restates main's event, so it can fall behind it.

    The ranges cannot come from the TaskSpec: the two engines randomise friction differently enough
    that the spec states no range at all, and this profile is where Isaac's shape of it lives. That
    makes it a second copy of four numbers main also writes, which is the arrangement that produced
    the self-collision drift. ``check_parity`` would catch it, but only where Isaac Sim is installed.
    """
    import ast

    from instinctlab.engines.isaacsim.scene import PROFILE_DEFAULTS

    config = REPO / "source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py"
    for node in ast.walk(ast.parse(config.read_text())):
        if not (isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "physics_material"):
            continue
        params = next(k.value for k in node.value.keywords if k.arg == "params")  # type: ignore[union-attr]
        declared = {
            key.value: ast.literal_eval(value)
            for key, value in zip(params.keys, params.values)  # type: ignore[union-attr]
            if isinstance(key, ast.Constant) and key.value != "asset_cfg"
        }
        assert declared == dict(PROFILE_DEFAULTS["friction_dr"]), (
            "the Isaac profile's friction randomisation no longer matches the one G1FlatEnvCfg "
            f"declares: {PROFILE_DEFAULTS['friction_dr']} vs {declared}"
        )
        return
    raise AssertionError("G1FlatEventsCfg no longer assigns physics_material; this check needs rewriting")
