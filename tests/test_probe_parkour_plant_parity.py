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


def _synthetic_payload(
    joint_names: list[str],
    qpos_steps: list[np.ndarray],
    *,
    side: str,
    action_names: list[str] | None = None,
    with_required: bool = True,
) -> dict:
    companion = Path(f"/tmp/probe_test_{side}.npz")
    arrays = {}
    steps = []
    action_names = list(action_names or joint_names)
    for idx, qpos in enumerate(qpos_steps):
        arrays[f"step_{idx}_qpos"] = qpos
        if with_required:
            n_act = len(action_names)
            arrays[f"step_{idx}_qvel"] = np.zeros_like(qpos)
            arrays[f"step_{idx}_qfrc_actuator"] = np.zeros_like(qpos)
            arrays[f"step_{idx}_raw_action"] = np.zeros((qpos.shape[0], n_act), dtype=np.float32)
            arrays[f"step_{idx}_processed_action"] = np.zeros((qpos.shape[0], n_act), dtype=np.float32)
            arrays[f"step_{idx}_root_link_pos_w"] = np.zeros((qpos.shape[0], 3), dtype=np.float32)
            arrays[f"step_{idx}_depth_raw"] = np.zeros((qpos.shape[0], 36, 64, 1), dtype=np.float32)
            arrays[f"step_{idx}_depth_processed"] = np.zeros((qpos.shape[0], 8, 18, 32), dtype=np.float32)
        steps.append({"phase": "post_control_step", "step_index": idx})
    np.savez(companion, **arrays)
    return {
        "metadata": {"side": side},
        "static": {
            "joint_names": joint_names,
            "action_target_names": action_names,
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


def test_permute_by_names_reorders_last_axis(probe):
    src = ["waist", "hip", "knee"]
    dst = ["knee", "waist", "hip"]
    values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = probe.permute_by_names(values, src, dst)
    np.testing.assert_array_equal(out, np.array([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]]))


def test_compare_remaps_qpos_by_joint_name(probe, tmp_path):
    left_names = ["a", "b"]
    right_names = ["b", "a"]
    left_q = [np.array([[0.0, 1.0]]), np.array([[0.0, 1.0]]), np.array([[0.0, 1.0]])]
    right_q = [np.array([[1.0, 0.0]]), np.array([[1.0, 0.0]]), np.array([[1.0, 0.0]])]
    left = _synthetic_payload(left_names, left_q, side="ours", action_names=left_names)
    right = _synthetic_payload(right_names, right_q, side="mj", action_names=right_names)
    report = probe.compare_rollout_payloads(left, right)
    assert report["passed"] is True


def test_compare_missing_required_action_is_failure(probe, tmp_path):
    names = ["j0"]
    left = _synthetic_payload(names, [np.array([[0.0]]), np.array([[0.0]])], side="ours", with_required=False)
    right = _synthetic_payload(names, [np.array([[0.0]]), np.array([[0.0]])], side="mj", with_required=False)
    report = probe.compare_rollout_payloads(left, right)
    assert report["passed"] is False
    fail = report["first_consecutive_two_step_exceedance"]
    assert fail["field"] in {"raw_action", "processed_action", "qfrc_actuator", "root_link_pos_w"}
    assert "missing" in fail.get("reason", "")


def test_compare_depth_before_qpos_in_causal_order(probe, tmp_path):
    names = ["j0"]
    zeros = [np.array([[0.0]]), np.array([[0.0]]), np.array([[0.0]])]
    left = _synthetic_payload(names, zeros, side="ours", with_required=True)
    right = _synthetic_payload(names, zeros, side="mj", with_required=True)
    companion_l = Path(left["companion_npz"])
    companion_r = Path(right["companion_npz"])
    with np.load(companion_l) as archive:
        arrays_l = {key: archive[key] for key in archive.files}
    with np.load(companion_r) as archive:
        arrays_r = {key: archive[key] for key in archive.files}
    for idx in range(3):
        arrays_l[f"step_{idx}_depth_processed"] = np.zeros((1, 8, 18, 32), dtype=np.float32)
        arrays_r[f"step_{idx}_depth_processed"] = np.ones((1, 8, 18, 32), dtype=np.float32) * 0.5
        arrays_l[f"step_{idx}_qpos"] = np.array([[0.0]])
        arrays_r[f"step_{idx}_qpos"] = np.array([[0.2]])
    np.savez(companion_l, **arrays_l)
    np.savez(companion_r, **arrays_r)
    report = probe.compare_rollout_payloads(left, right)
    assert report["passed"] is False
    assert report["first_consecutive_two_step_exceedance"]["field"] == "depth_processed"


def test_inference_action_detaches_grad(probe):
    import torch

    class _Pol(torch.nn.Module):
        def forward(self, obs):
            return obs * 2.0

    obs = torch.ones(2, 3, requires_grad=True)
    out = probe._inference_action(_Pol(), obs)
    assert isinstance(out, torch.Tensor)
    assert out.requires_grad is False
    assert torch.allclose(out, torch.full((2, 3), 2.0))


def test_action_buffers_read_private_processed(probe):
    class _Term:
        raw_action = "raw"
        _processed_actions = "processed"

    raw, processed = probe._action_buffers(_Term())
    assert raw == "raw"
    assert processed == "processed"


def test_instinctmj_python_hint_names_training_venv(probe):
    text = probe.instinctmj_python_hint("/usr/bin/python")
    assert str(probe.DEFAULT_INSTINCTMJ_PYTHON) in text
    assert "/usr/bin/python" in text
    assert "instinct-train" in text
    assert "CUDA_VISIBLE_DEVICES" in text
    assert "--device cuda:0" in text


def test_dump_state_output_path_does_not_clobber_input(probe, tmp_path):
    out = tmp_path / "mj_dump.json"
    incoming = tmp_path / "ours.state.npz"
    incoming.write_bytes(b"x")
    written = probe.dump_state_output_path(out=out, state_npz=incoming, incoming_exists=True)
    assert written == out.with_suffix(".state.npz")
    assert written != incoming
    fresh = tmp_path / "new.state.npz"
    assert probe.dump_state_output_path(out=out, state_npz=fresh, incoming_exists=False) == fresh
    assert probe.dump_state_output_path(out=out, state_npz=None, incoming_exists=False) == out.with_suffix(".state.npz")


class _WpArray:
    def __init__(self, values):
        self._v = np.asarray(values)

    def numpy(self):
        return self._v


def test_first_int_reads_warp_like_array(probe):
    assert probe._first_int(_WpArray([12])) == 12
    assert probe._max_int(_WpArray([1, 8, 3])) == 8
    with pytest.raises(TypeError):
        int(_WpArray([12]))


def test_contact_snapshot_does_not_call_int_on_warp_array(probe):
    class _Cfg:
        nconmax = 256
        njmax = 768

    class _Wp:
        nacon = _WpArray([12])
        nefc = _WpArray([1, 8, 3])

    class _Sim:
        wp_data = _Wp()
        cfg = _Cfg()

    class _Env:
        sim = _Sim()

    snap = probe._mjlab_contact_snapshot(_Env())
    assert snap == {"available": True, "nacon": 12, "nefc_max": 8, "nconmax": 256, "njmax": 768}


def test_depth_raw_washes_nonfinite_to_inf(probe):
    import torch

    class _Cfg:
        image_plane_max = 10.0

    class _Data:
        output = {"distance_to_image_plane": torch.tensor([[[[float("nan")], [12.0], [3.0]]]])}

    class _Sensor:
        data = _Data()
        cfg = _Cfg()

    out = probe._depth_raw_from_sensor(_Sensor())
    assert out.shape == (1, 1, 3, 1)
    assert torch.isinf(out[0, 0, 0, 0])
    assert torch.isinf(out[0, 0, 1, 0])
    assert out[0, 0, 2, 0].item() == pytest.approx(3.0)


def test_hash_stable_across_copies(probe):
    arr = np.array([[1.0, 2.0], [3.0, np.inf]], dtype=np.float64)
    assert probe.tensor_summary(arr)["sha256"] == probe.tensor_summary(arr.copy())["sha256"]


def test_compare_relative_root_z_ignores_origin_offset(probe):
    names = ["j0"]
    zeros = [np.array([[0.0]]), np.array([[0.0]]), np.array([[0.0]])]
    left = _synthetic_payload(names, zeros, side="ours_z")
    right = _synthetic_payload(names, zeros, side="mj_z")
    companion_l = Path(left["companion_npz"])
    companion_r = Path(right["companion_npz"])
    with np.load(companion_l) as archive:
        arrays_l = {key: archive[key] for key in archive.files}
    with np.load(companion_r) as archive:
        arrays_r = {key: archive[key] for key in archive.files}
    for idx in range(3):
        arrays_l[f"step_{idx}_root_link_pos_w"] = np.array([[0.0, 0.0, 0.90]], dtype=np.float32)
        arrays_r[f"step_{idx}_root_link_pos_w"] = np.array([[0.0, 0.0, 1.49]], dtype=np.float32)
        arrays_l[f"step_{idx}_env_origins_z"] = np.array([0.00], dtype=np.float32)
        arrays_r[f"step_{idx}_env_origins_z"] = np.array([0.59], dtype=np.float32)
        left["steps"][idx]["env_origins_z"] = [0.00]
        right["steps"][idx]["env_origins_z"] = [0.59]
    np.savez(companion_l, **arrays_l)
    np.savez(companion_r, **arrays_r)
    report = probe.compare_rollout_payloads(left, right)
    assert report["passed"] is True
    for row in report["per_step"]:
        assert row["fields"]["root_link_pos_w"]["max_abs_diff"] == pytest.approx(0.0, abs=1e-6)


def test_compare_single_step_dump_is_not_vacuous_pass(probe):
    names = ["j0"]
    left = _synthetic_payload(names, [np.array([[0.0]])], side="ours_dump")
    right = _synthetic_payload(names, [np.array([[0.0]])], side="mj_dump")
    companion_l = Path(left["companion_npz"])
    companion_r = Path(right["companion_npz"])
    with np.load(companion_l) as archive:
        arrays_l = {key: archive[key] for key in archive.files}
    with np.load(companion_r) as archive:
        arrays_r = {key: archive[key] for key in archive.files}
    arrays_l["step_0_depth_processed"] = np.zeros((1, 8, 18, 32), dtype=np.float32)
    arrays_r["step_0_depth_processed"] = np.ones((1, 8, 18, 32), dtype=np.float32)
    np.savez(companion_l, **arrays_l)
    np.savez(companion_r, **arrays_r)
    report = probe.compare_rollout_payloads(left, right)
    assert report["passed"] is False
    fail = report["first_consecutive_two_step_exceedance"]
    assert fail["field"] == "depth_processed"
    assert "single-step" in fail.get("reason", "")
