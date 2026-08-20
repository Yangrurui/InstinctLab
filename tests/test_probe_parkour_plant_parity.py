"""Offline tests for scripts/probe_parkour_plant_parity.py (no GPU, no engine bootstrap)."""

from __future__ import annotations

import ast
import importlib.util
import json
import numpy as np
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/probe_parkour_plant_parity.py"
INSTINCTMJ_REG = Path("/root/InstinctMJ/src/instinct_mj/tasks/parkour/config/g1/__init__.py")


def _load_probe_module():
    spec = importlib.util.spec_from_file_location("probe_parkour_plant_parity", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe():
    return _load_probe_module()


def test_module_top_level_has_no_engine_imports(probe):
    hits = probe.module_top_level_engine_imports(SCRIPT)
    assert hits == [], f"module must lazy-import engines; found {hits}"


@pytest.mark.skipif(not INSTINCTMJ_REG.is_file(), reason="InstinctMJ not checked out")
def test_instinctmj_registration_uses_train_path(probe):
    info = probe.read_instinctmj_train_registration(INSTINCTMJ_REG)
    assert info["task_id"] == probe.INSTINCTMJ_TASK_ID
    assert info["play_false_in_env_cfg_factory"] is True


def test_root_height_margin_formula(probe):
    assert probe.root_height_margin(1.2, 0.4) == pytest.approx(1.2 - 0.4 - 0.5)
    assert probe.root_height_margin(0.9, -1.0) == pytest.approx(0.9 - 0.0 - 0.5)


def test_name_alignment_failure(probe):
    with pytest.raises(ValueError, match="joint names differ"):
        probe.align_names_or_fail(["a"], ["b"], label="joint")


def test_consecutive_two_step_rule(probe):
    assert probe.first_consecutive_two_step_exceedance([0.0, 0.2, 0.2, 0.1], 0.15) == 1
    assert probe.first_consecutive_two_step_exceedance([0.2, 0.1, 0.2], 0.15) is None


def _synthetic_payload(joint_names: list[str], qpos_steps: list[np.ndarray], *, side: str) -> dict:
    companion = Path(f"/tmp/probe_test_{side}.npz")
    arrays = {}
    steps = []
    for idx, qpos in enumerate(qpos_steps):
        key = f"step_{idx}_qpos"
        arrays[key] = qpos
        steps.append({"phase": "post_control_step", "step_index": idx})
    np.savez(companion, **arrays)
    return {
        "metadata": {"side": side},
        "static": {
            "joint_names": joint_names,
            "action_target_names": joint_names,
        },
        "companion_npz": str(companion),
        "steps": steps,
    }


def test_compare_reports_first_consecutive_exceedance(probe, tmp_path):
    names = ["j0", "j1"]
    left = _synthetic_payload(names, [np.array([[0.0, 0.0]]), np.array([[0.0, 0.0]])], side="ours")
    right = _synthetic_payload(names, [np.array([[0.0, 0.0]]), np.array([[0.0, 0.001]])], side="mj")
    report = probe.compare_rollout_payloads(
        left, right, thresholds={"qpos": 1e-4, "root_z": 1e-3, "qfrc": 0.05, "action": 1e-3}
    )
    assert report["passed"] is True

    right2 = _synthetic_payload(
        names,
        [np.array([[0.0, 0.0]]), np.array([[0.001, 0.0]]), np.array([[0.001, 0.0]])],
        side="mj2",
    )
    left2 = _synthetic_payload(
        names,
        [np.array([[0.0, 0.0]]), np.array([[0.0, 0.0]]), np.array([[0.0, 0.0]])],
        side="ours2",
    )
    report2 = probe.compare_rollout_payloads(
        left2, right2, thresholds={"qpos": 1e-4, "root_z": 1e-3, "qfrc": 0.05, "action": 1e-3}
    )
    assert report2["passed"] is False
    fail = report2["first_consecutive_two_step_exceedance"]
    assert fail is not None
    assert fail["field"] == "qpos"
    assert fail["first_step_index"] == 1


def test_output_schema_keys(probe):
    assert probe.output_schema_keys() == frozenset({"metadata", "static", "steps"})


def test_cli_help_without_engine():
    result = __import__("subprocess").run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "compare" in result.stdout
