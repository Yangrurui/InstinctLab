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
