"""Source-level guard for the two external shadowing references audited for the rewrite.

The parser follows registration calls and their literal IDs.  It intentionally does not import an
engine SDK, so this audit stays runnable while the legacy task modules are being removed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

MAIN = Path("/root/InstinctLab-main/source/instinctlab/instinctlab/tasks/shadowing")
MJ = Path("/root/InstinctMJ/src/instinct_mj/tasks/shadowing")

COMMON_IDS = {
    "Instinct-Shadowing-WholeBody-Plane-G1-v0",
    "Instinct-Shadowing-WholeBody-Plane-G1-Play-v0",
    "Instinct-Perceptive-Shadowing-G1-v0",
    "Instinct-Perceptive-Shadowing-G1-Play-v0",
    "Instinct-Perceptive-Vae-G1-v0",
    "Instinct-Perceptive-Vae-G1-Play-v0",
    "Instinct-Perceptive-HOI-Shadowing-G1-v0",
    "Instinct-Perceptive-HOI-Shadowing-G1-Play-v0",
    "Instinct-BeyondMimic-Plane-G1-v0",
    "Instinct-BeyondMimic-Plane-G1-Play-v0",
}
MJ_ONLY_IDS = {
    "Instinct-Perceptive-Shadowing-G1-OneMotion-v0",
    "Instinct-Perceptive-Shadowing-G1-OneMotion-Play-v0",
}


def _registered_ids(root: Path, call_name: str, keyword: str) -> set[str]:
    found: set[str] = set()
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
            if name != call_name:
                continue
            for item in node.keywords:
                if item.arg == keyword and isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
                    found.add(item.value.value)
    return found


@pytest.mark.skipif(not MAIN.is_dir(), reason="main reference checkout is unavailable")
def test_main_effective_registration_inventory() -> None:
    assert _registered_ids(MAIN, "register", "id") == COMMON_IDS


@pytest.mark.skipif(not MJ.is_dir(), reason="InstinctMJ reference checkout is unavailable")
def test_instinctmj_effective_registration_inventory() -> None:
    assert _registered_ids(MJ, "register_instinct_task", "task_id") == COMMON_IDS | MJ_ONLY_IDS


@pytest.mark.skipif(not MJ.is_dir(), reason="InstinctMJ reference checkout is unavailable")
def test_instinctmj_train_registrations_reach_factories_not_play_classes() -> None:
    whole_body = (MJ / "whole_body/config/g1/__init__.py").read_text()
    beyondmimic = (MJ / "beyondmimic/config/g1/__init__.py").read_text()
    assert "env_cfg_factory=lambda: _plane_shadowing_env_cfg(play=False)" in whole_body
    assert "env_cfg_factory=lambda: _beyondmimic_plane_env_cfg(play=False)" in beyondmimic


def test_legacy_local_shadowing_surface_is_fully_inventoried() -> None:
    root = Path("source/instinctlab/instinctlab/tasks/shadowing")
    families = {path.name for path in root.iterdir() if path.is_dir() and not path.name.startswith("__")}
    assert {"whole_body", "perceptive", "perceptive_hoi", "beyondmimic", "mdp"} <= families
    assert (root / "play.py").is_file()
    assert (root / "cli_args.py").is_file()
    assert (root / "grid_search.sh").is_file()
