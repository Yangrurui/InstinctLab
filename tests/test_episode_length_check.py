"""Guard: the episode-length check can tell a wired-up run from one where nothing terminates.

This check exists because of a failure that every other layer missed -- contact timers left at zero,
so the only way an episode could end was by running out of time, while training converged and the
reward curve looked ordinary. The detector is therefore the last line rather than a nicety, and a
detector nobody checks is worth what the missing sensor field was worth.

Synthetic curves rather than recorded logs: what is being asserted is the discrimination, and the
interesting cases are ones no log in the repository happens to contain -- a trained policy that
legitimately survives nearly every episode is the case most likely to be confused with a broken one.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "check_episode_length.py"


@pytest.fixture(scope="module")
def check():
    spec = importlib.util.spec_from_file_location("_check_episode_length", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_horizon_comes_from_the_task(check) -> None:
    """Not a constant in the script: an episode length change would silently retune the detector."""
    assert check.horizon() == 1000


def test_it_reads_the_curve_the_runner_logs(check, tmp_path) -> None:
    log = tmp_path / "run.log"
    log.write_text("noise\nMean episode length: 14.58\nmore noise\nMean episode length: 1000.00\n")
    assert check.curve(log) == [14.58, 1000.00]


def _run(check, monkeypatch, tmp_path, values: list[float]) -> int:
    log = tmp_path / "run.log"
    log.write_text("".join(f"Mean episode length: {v:.2f}\n" for v in values))
    monkeypatch.setattr("sys.argv", ["check_episode_length.py", str(log)])
    return check.main()


def test_a_run_pinned_at_the_horizon_fails(check, monkeypatch, tmp_path) -> None:
    """The signature of the real failure: 277 iterations, every one of them exactly 1000."""
    assert _run(check, monkeypatch, tmp_path, [1000.0] * 30) == 1


def test_a_policy_that_learns_to_survive_passes(check, monkeypatch, tmp_path) -> None:
    """The case a cruder rule would get wrong.

    A good locomotion policy ends up finishing almost every episode, so "mostly at the limit" is not
    evidence of anything. What distinguishes the broken run is that it starts there -- an untrained
    policy that never falls over has nothing to fall over from.
    """
    learning = [14.6, 37.1, 27.7, *(min(1000.0, 40.0 + 40.0 * i) for i in range(26))]
    assert _run(check, monkeypatch, tmp_path, learning) == 0


def test_a_short_run_is_not_judged(check, monkeypatch, tmp_path) -> None:
    """Early episodes are legitimately long before terminations have anything to fire on."""
    assert _run(check, monkeypatch, tmp_path, [1000.0] * 3) == 0


def test_a_log_without_a_curve_fails(check, monkeypatch, tmp_path) -> None:
    """Silence has to be a failure; the check is worthless if pointing it at the wrong file passes."""
    assert _run(check, monkeypatch, tmp_path, []) == 1
