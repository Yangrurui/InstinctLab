"""The files that are supposed to *be* main's are checked against main.

This repo is a branch of main and keeps most of its files verbatim -- the Isaac-only tasks, the
wrappers, the managers. Editing one of them is allowed and sometimes necessary, but doing it
silently is not: a reader who assumes a file is upstream's will reason about it wrongly, and the one
time that happened it cost a whole verification apparatus. ``G1FlatEnvCfg`` was the golden every
Isaac result was measured against, and it had carried twenty-two lines of local edits since 4806241,
so "matches main" was measured against a modified main. Four separate audits read the file and none
noticed, because reading it tells you nothing: it looks like a plausible env config either way.

So the reference is consulted rather than remembered. Files listed as untouched must be byte-equal
to main; files listed as edited must differ, so an entry whose edit was reverted upstream shows up
instead of sitting here claiming a difference that no longer exists. And because the last failure
was a file nobody had listed at all, ``test_no_edit_of_mains_goes_unrecorded`` asks git for the full
set rather than trusting the tables to be complete.

The locomotion MDP package is gone from both tables along with the files: that task's Isaac-only
rewards lived there and were deleted when D3 was retired. The robot package is back at
``config/g1/``, but ``flat_env_cfg.py`` is now a ``TaskSpec`` and the Gym ids are gone.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

UNTOUCHED = (
    "source/instinctlab/instinctlab/tasks/locomotion/config/__init__.py",
    "source/instinctlab/instinctlab/tasks/locomotion/config/g1/agents/__init__.py",
    "source/instinctlab/instinctlab/tasks/locomotion/__init__.py",
    "source/instinctlab/instinctlab/envs/manager_based_rl_env.py",
    "source/instinctlab/instinctlab/managers/reward_manager.py",
)
"""Main's, verbatim.

The parkour rewards and env entries are here because those are the largest of main's files this
repo still runs unchanged, and the next adaptation will touch them. Listing them now means the
first edit has to say why.
"""

EDITED = {
    "source/instinctlab/instinctlab/envs/mdp/events/randomization.py": (
        "updates randomized action offsets by joint name because shared actions are canonical DFS "
        "while Isaac articulations are native BFS; main uses one native order for both"
    ),
    "source/instinctlab/instinctlab/tasks/shadowing/README.md": "documents the unified train/play surface",
    "source/instinctlab/instinctlab/tasks/shadowing/beyondmimic/README.md": (
        "documents the shared BeyondMimic declaration"
    ),
    "source/instinctlab/instinctlab/tasks/shadowing/beyondmimic/config/g1/agents/beyondmimic_ppo_cfg.py": (
        "uses the engine-neutral configclass so MJLab can resolve the shared agent without Kit"
    ),
    "source/instinctlab/instinctlab/tasks/shadowing/beyondmimic/beyondmimic_env_cfg.py": (
        "native EnvCfg translated to the engine-neutral BeyondMimic TaskSpec declaration"
    ),
    "source/instinctlab/instinctlab/tasks/shadowing/beyondmimic/config/g1/beyondmimic_plane_cfg.py": (
        "keeps the G1 BeyondMimic data and factories engine-neutral"
    ),
    "source/instinctlab/instinctlab/tasks/shadowing/perceptive/perceptive_env_cfg.py": (
        "native EnvCfg translated to the engine-neutral Perceptive and VAE TaskSpec declarations"
    ),
    "source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/perceptive_shadowing_cfg.py": (
        "keeps the G1 Perceptive data and factories engine-neutral"
    ),
    "source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/perceptive_vae_cfg.py": (
        "keeps the G1 Perceptive VAE data and factories engine-neutral"
    ),
    "source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/agents/instinct_rl_ppo_cfg.py": (
        "uses the engine-neutral configclass and removes an unused Isaac-only observation import"
    ),
    "source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/agents/instinct_rl_vae_cfg.py": (
        "uses the engine-neutral configclass and removes an unused Isaac-only observation import"
    ),
    "source/instinctlab/instinctlab/tasks/shadowing/perceptive_hoi/config/g1/agents/instinct_rl_ppo_cfg.py": (
        "uses the engine-neutral configclass and removes an unused Isaac-only observation import"
    ),
    "source/instinctlab/instinctlab/tasks/shadowing/perceptive_hoi/perceptive_env_cfg.py": (
        "native EnvCfg translated to the engine-neutral Perceptive HOI TaskSpec declaration"
    ),
    "source/instinctlab/instinctlab/tasks/shadowing/perceptive_hoi/config/g1/perceptive_shadowing_cfg.py": (
        "keeps the G1 Perceptive HOI data and factories engine-neutral"
    ),
    "source/instinctlab/instinctlab/tasks/shadowing/whole_body/config/g1/agents/instinct_rl_ppo_cfg.py": (
        "uses the engine-neutral configclass so MJLab can resolve the shared agent without Kit"
    ),
    "source/instinctlab/instinctlab/tasks/shadowing/whole_body/shadowing_env_cfg.py": (
        "native EnvCfg translated to the engine-neutral Whole Body TaskSpec declaration"
    ),
    "source/instinctlab/instinctlab/tasks/shadowing/whole_body/config/g1/plane_shadowing_cfg.py": (
        "keeps the G1 Whole Body data and factories engine-neutral"
    ),
    "source/instinctlab/instinctlab/tasks/parkour/README.md": (
        "documents the engine-neutral task id and unified train/play entry points"
    ),
    "source/instinctlab/instinctlab/tasks/parkour/config/g1/g1_parkour_target_amp_cfg.py": (
        "Isaac-only EnvCfg replaced by the engine-neutral TaskSpec shared with MJLab"
    ),
    "source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py": (
        "Isaac-only runner config replaced by the engine-neutral runner shared by both engines"
    ),
    "source/instinctlab/instinctlab/assets/__init__.py": "each robot is an engine-neutral catalog package",
    # The training path, which the parity argument covers just as much as the env config does.
    "source/instinctlab/instinctlab/utils/wrappers/instinct_rl/vecenv_wrapper.py": (
        "num_rewards falls back to 1, because a task whose rewards are one flat container runs on a "
        "plain ManagerBasedRLEnv, which does not declare the attribute InstinctRlEnv does; "
        "step/episode extras are created so WasabiPPO can write discriminator_reward"
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
    "source/instinctlab/instinctlab/tasks/__init__.py": "exports only the engine-neutral registry",
    "source/instinctlab/instinctlab/envs/__init__.py": "exports lazily",
    "source/instinctlab/instinctlab/managers/__init__.py": "exports lazily",
    "source/instinctlab/instinctlab/sim/__init__.py": (
        "now fronts the engine-neutral contract; legacy spawners load lazily"
    ),
    "source/instinctlab/setup.py": "declares the new packages",
    "source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py": (
        "Isaac EnvCfg replaced by the engine-free TaskSpec; the Gym ids are gone"
    ),
    "source/instinctlab/instinctlab/tasks/locomotion/config/g1/__init__.py": (
        "no longer registers Gym ids; re-exports the TaskSpec factory so the package stays engine-free"
    ),
    "source/instinctlab/instinctlab/tasks/parkour/config/g1/__init__.py": (
        "no longer registers Isaac-only Gym ids; re-exports the shared TaskSpec factory"
    ),
    "source/instinctlab/instinctlab/tasks/locomotion/config/g1/agents/instinct_rl_ppo_cfg.py": (
        "same hyperparameters; configclass is vendored so reading them does not start Isaac Sim"
    ),
    "source/instinctlab/instinctlab/tasks/shadowing/__init__.py": "exports engine-neutral task factories",
    "source/instinctlab/instinctlab/tasks/shadowing/whole_body/config/g1/__init__.py": (
        "Gym registration replaced by TaskSpec exports"
    ),
    "source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/__init__.py": (
        "Gym registrations replaced by TaskSpec exports"
    ),
    "source/instinctlab/instinctlab/tasks/shadowing/perceptive_hoi/config/g1/__init__.py": (
        "Gym registration replaced by TaskSpec exports"
    ),
    "source/instinctlab/instinctlab/tasks/shadowing/beyondmimic/config/g1/__init__.py": (
        "Gym registration replaced by TaskSpec exports"
    ),
    "source/instinctlab/instinctlab/sensors/volume_points/volume_points_cfg.py": (
        "optional body_order so discovery must match the declared attach list; default None keeps legacy; "
        "velocity defaults to com so Isaac-only parkour Gym ids keep PhysX COM speeds"
    ),
    "source/instinctlab/instinctlab/sensors/volume_points/volume_points.py": (
        "empty or 0-cylinder registration raises; registered_cylinder_count is observable; "
        "body_order mismatch raises so the two engines cannot sum different clouds; "
        "velocity='attach_link' converts PhysX COM linear to link origin"
    ),
    "source/instinctlab/instinctlab/utils/warp/kernels.py": (
        "on-axis cylinder hit used to divide by zero and write NaN into the reward; "
        "depth is now the radius along a stable perpendicular"
    ),
}
"""Main's, with a deliberate edit. The text says which one."""

REMOVED = {
    "scripts/amass_visualize.py": "stale Isaac-only tool imported task symbols removed by the unified config",
    "scripts/instinct_rl/cli_args.py": "the unified train/play entry points own argument parsing",
    "scripts/instinct_rl/play.py": "replaced by scripts/play.py for both engines",
    "scripts/instinct_rl/plotter.py": "only the removed legacy player imported it",
    "scripts/instinct_rl/train.py": "replaced by scripts/train.py for both engines",
    "source/instinctlab/instinctlab/tasks/shadowing/play.py": "generic scripts/play.py serves both engines",
    "source/instinctlab/instinctlab/tasks/shadowing/cli_args.py": (
        "generic train/play argument parsing owns the unified tasks"
    ),
    "source/instinctlab/instinctlab/tasks/shadowing/grid_search.sh": "depended on the removed Isaac-only player",
    "source/instinctlab/instinctlab/tasks/shadowing/mdp/__init__.py": (
        "shadowing terms move to shared mdp or engine registries"
    ),
    "source/instinctlab/instinctlab/tasks/shadowing/mdp/curriculums.py": (
        "shadowing curriculum is declared semantically once"
    ),
    "source/instinctlab/instinctlab/tasks/shadowing/mdp/events.py": "shadowing events are declared semantically once",
    "source/instinctlab/instinctlab/tasks/parkour/mdp/__init__.py": (
        "the shared Parkour TaskSpec uses the portable instinctlab.mdp package"
    ),
    "source/instinctlab/instinctlab/tasks/parkour/mdp/commands/__init__.py": (
        "the Isaac-only Parkour command package has no production consumers"
    ),
    "source/instinctlab/instinctlab/tasks/parkour/mdp/commands/commands_cfg.py": (
        "the shared TaskSpec declares its command without an Isaac config class"
    ),
    "source/instinctlab/instinctlab/tasks/parkour/mdp/commands/pose_velocity_command.py": (
        "the shared TaskSpec uses the portable command implementation"
    ),
    "source/instinctlab/instinctlab/tasks/parkour/mdp/curriculums.py": (
        "portable curriculum terms live in instinctlab.mdp"
    ),
    "source/instinctlab/instinctlab/tasks/parkour/mdp/events.py": "portable event terms live in instinctlab.mdp",
    "source/instinctlab/instinctlab/tasks/parkour/mdp/rewards.py": (
        "portable rewards live in instinctlab.mdp or their engine adapter"
    ),
    "source/instinctlab/instinctlab/tasks/parkour/mdp/terminations.py": (
        "portable termination terms live in instinctlab.mdp"
    ),
    "source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py": (
        "Isaac-only Parkour EnvCfg was replaced by the shared G1 TaskSpec"
    ),
    "source/instinctlab/instinctlab/tasks/parkour/scripts/play.py": (
        "the unified scripts/play.py handles Parkour on either engine"
    ),
    "source/instinctlab/instinctlab/tasks/parkour/scripts/onnxer.py": (
        "its only consumer was the removed Isaac-only Parkour player"
    ),
    "source/instinctlab/instinctlab/assets/unitree_g1.py": (
        "became the unitree_g1 package: numbers and RobotSpec in isaacsim.py"
    ),
    "source/instinctlab/instinctlab/tasks/locomotion/mdp/__init__.py": "only flat_env_cfg imported this package",
    "source/instinctlab/instinctlab/tasks/locomotion/mdp/rewards.py": (
        "the four terms it held are in instinctlab/mdp/, written to read quantities both engines have"
    ),
    "source/instinctlab/instinctlab/tasks/locomotion/mdp/curriculums.py": (
        "terrain_levels_vel, which the flat task had commented out and nothing else called"
    ),
}
"""Main's, and gone. Deleting upstream's work needs a louder reason than editing it."""


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
        + "\nPut each in EDITED with the reason, or revert it."
    )


def test_no_deletion_of_mains_goes_unrecorded(main_ref: str) -> None:
    """Removing upstream's file is the larger act, and it was the one the tables did not cover.

    ``--no-renames`` matters here. Git once saw the agent config leave this path as a rename at 61%
    similarity, so it reported neither a deletion nor a modification: the file left its original
    path without appearing in any of these lists.
    """
    removed = _git("diff", main_ref, "--name-only", "--diff-filter=D", "--no-renames", "--", "source/", "scripts/")
    assert removed.returncode == 0, removed.stderr
    unrecorded = {path for path in removed.stdout.split() if path} - set(REMOVED)
    assert not unrecorded, (
        f"these files of {main_ref}'s were deleted without saying why:\n  "
        + "\n  ".join(sorted(unrecorded))
        + "\nPut each in REMOVED with the reason, or restore it."
    )


def test_the_recorded_removals_are_still_removed(main_ref: str) -> None:
    """An entry describing nothing is worse than no entry, because it reads as coverage."""
    for path in sorted(REMOVED):
        assert _git("cat-file", "-e", f"{main_ref}:{path}").returncode == 0, f"{path} was never on {main_ref}"
        assert not (REPO / path).exists(), f"{path} is back; drop it from REMOVED"
