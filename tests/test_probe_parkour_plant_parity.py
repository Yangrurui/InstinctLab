"""Offline tests for scripts/probe_parkour_plant_parity.py (no GPU, no engine bootstrap)."""

from __future__ import annotations

import ast
import importlib.util
import json
import numpy as np
import subprocess
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
    assert probe.policy_eval_schema_keys() == frozenset({"metadata", "static", "eval"})
    assert "terrain_mapping" in probe.policy_eval_eval_keys()
    assert "per_terrain" in probe.policy_eval_summary_keys()
    assert "episode_length_stat_scope" in probe.policy_eval_summary_keys()
    assert "completed_episode_mean_length" in probe.policy_eval_summary_keys()
    assert "termination_rate_per_1000_env_steps" in probe.policy_eval_summary_keys()
    assert "termination_rates_per_1000_env_steps" in probe.policy_eval_summary_keys()


def test_instinctmj_reference_camera_uses_groups_012(probe):
    assert probe.instinctmj_reference_camera_geom_groups() == (0, 1, 2)


def test_reference_camera_reader_works_without_repo_on_pythonpath():
    code = (
        "import importlib.util; "
        f"p={str(SCRIPT)!r}; "
        "s=importlib.util.spec_from_file_location('probe', p); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "assert m.instinctmj_reference_camera_geom_groups() == (0, 1, 2)"
    )
    subprocess.run([sys.executable, "-c", code], cwd="/tmp", check=True)


def test_native_camera_metadata_reports_min_distance_hop(probe):
    class _Cfg:
        name = "camera"
        include_geom_groups = (0, 1, 2)
        min_distance = 0.1

    class _Sensor:
        cfg = _Cfg()
        _hop_max = 6

        def _apply_min_distance_hop(self):
            return None

    meta = probe.camera_semantics_metadata(_Sensor(), probe.CAMERA_SEMANTICS_NATIVE)
    assert meta["camera_filter"] == "geom_groups_min_distance_hop"
    assert meta["hop_max"] == 6
    assert meta["hop_epsilon_m"] == pytest.approx(1e-4)
    assert meta["min_distance_m"] == pytest.approx(0.1)
    assert meta["native_already_aligned"] is True


def test_instinctmj_alias_detects_native_already_aligned(probe):
    class _Cfg:
        name = "camera"
        include_geom_groups = (0, 1, 2)
        min_distance = 0.1

    class _Sensor:
        cfg = _Cfg()
        _hop_max = 6

        def _apply_min_distance_hop(self):
            return None

    sensor = _Sensor()
    assert probe.native_camera_already_instinctmj_aligned(sensor) is True
    meta = probe.camera_semantics_metadata(sensor, probe.CAMERA_SEMANTICS_INSTINCTMJ)
    assert meta["alias_note"].startswith("instinctmj_geom_groups is a no-op alias")


def test_legacy_body_mask_metadata_still_detectable(probe):
    class _Sensor:
        cfg = type("Cfg", (), {"name": "camera", "include_geom_groups": None})()
        _allowed_geom_mask = __import__("torch").ones(10, dtype=bool)

        def _filter_and_continue(self):
            return None

    meta = probe.camera_semantics_metadata(_Sensor(), probe.CAMERA_SEMANTICS_NATIVE)
    assert meta["camera_filter"] == "body_mesh_mask_with_hop"
    assert meta["hop_max"] == 6
    assert meta["native_already_aligned"] is False


def test_summarize_policy_eval_counts_root_height(probe):
    episodes = [
        {
            "control_step": 10,
            "episode_length": 50,
            "termination_reasons": {"root_height": True},
            "primary_reason": "root_height",
            "root_height_margin": -0.1,
            "terrain_name": "pyramid_stairs",
        },
        {
            "control_step": 20,
            "episode_length": 80,
            "termination_reasons": {"time_out": True},
            "primary_reason": "time_out",
            "root_height_margin": 0.4,
            "terrain_name": "perlin_rough",
        },
        {
            "control_step": 5,
            "episode_length": 30,
            "termination_reasons": {"root_height": True},
            "primary_reason": "root_height",
            "root_height_margin": -0.2,
            "terrain_name": "pyramid_stairs",
        },
    ]
    summary = probe.summarize_policy_eval(episodes, control_steps=100, num_envs=4, warmup_steps=10)
    assert summary["completed_episodes"] == 2
    assert summary["root_height_count"] == 1
    assert summary["root_height_rate_per_1000_env_steps"] == pytest.approx(1 * 1000.0 / (100 * 4))
    assert summary["termination_rate_per_1000_env_steps"] == pytest.approx(2 * 1000.0 / (100 * 4))
    assert summary["termination_rates_per_1000_env_steps"]["root_height"] == pytest.approx(2.5)
    assert summary["mean_episode_length"] == pytest.approx(65.0)
    assert summary["completed_episode_mean_length"] == pytest.approx(65.0)
    assert summary["completed_episode_median_length"] == pytest.approx(65.0)
    assert "right-censored" in summary["episode_length_stat_scope"]
    assert summary["per_terrain"]["pyramid_stairs"]["completed_episodes"] == 1
    assert summary["per_terrain"]["perlin_rough"]["mean_episode_length"] == pytest.approx(80.0)
    assert "0" not in summary["per_terrain"]
    assert summary["root_height_margin"]["mean_at_root_height_term"] == pytest.approx(-0.1)


def test_snapshot_pre_reset_reads_term_dones(probe):
    import torch

    class _TermMgr:
        active_terms = ["root_height", "time_out"]

        @staticmethod
        def get_term(name):
            if name == "root_height":
                return torch.tensor([True, False])
            return torch.tensor([False, True])

        @property
        def terminated(self):
            return torch.tensor([True, False])

        @property
        def time_outs(self):
            return torch.tensor([False, True])

    class _Data:
        root_link_pos_w = torch.tensor([[0.0, 0.0, 0.6], [0.0, 0.0, 1.0]])

    class _Robot:
        data = _Data()

    class _Scene:
        env_origins = torch.tensor([[0.0, 0.0, 0.1], [0.0, 0.0, 0.0]])
        robot = _Robot()

    class _Env:
        device = "cpu"
        termination_manager = _TermMgr()
        reset_terminated = torch.tensor([True, False])
        reset_time_outs = torch.tensor([False, True])
        episode_length_buf = torch.tensor([12, 34])
        scene = _Scene()

    events = probe.snapshot_pre_reset_episodes(_Env(), torch.tensor([0, 1]), control_step=7)
    assert events[0]["episode_length"] == 12
    assert events[0]["termination_reasons"]["root_height"] is True
    assert events[0]["primary_reason"] == "root_height"
    assert events[1]["primary_reason"] == "time_out"
    assert events[0]["control_step"] == 7
    assert events[0]["terrain_name"] is None
    assert events[0]["terrain_type_id"] is None


def test_classify_episode_termination_prefers_failure_over_timeout(probe):
    reasons = {"root_height": True, "time_out": True}
    assert probe.classify_episode_termination(termination_reasons=reasons, truncated=True) == "root_height"


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
        nconmax = 128
        njmax = 700

    class _Wp:
        nacon = _WpArray([12])
        nefc = _WpArray([1, 8, 3])

    class _Sim:
        wp_data = _Wp()
        cfg = _Cfg()

    class _Env:
        sim = _Sim()

    snap = probe._mjlab_contact_snapshot(_Env())
    assert snap == {"available": True, "nacon": 12, "nefc_max": 8, "nconmax": 128, "njmax": 700}


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


@pytest.mark.skipif(not INSTINCTMJ_REG.is_file(), reason="InstinctMJ not checked out")
def test_instinctmj_policy_eval_uses_play_false(probe):
    import inspect

    source = inspect.getsource(probe._build_instinctmj_eval)
    assert "play=False" in source.replace(" ", "")
    assert "play=True" not in source.replace(" ", "")
    info = probe.read_instinctmj_train_registration(INSTINCTMJ_REG)
    assert info["play_false_in_env_cfg_factory"] is True


def test_resolve_terrain_mapping_one_column_per_type(probe):
    class _Tile:
        proportion = 0.5

    class _Gen:
        curriculum = True
        num_cols = 20
        sub_terrains = {"stairs": _Tile(), "perlin": _Tile()}

    class _Cfg:
        terrain_generator = _Gen()

    class _Terrain:
        cfg = _Cfg()
        terrain_origins = __import__("numpy").zeros((10, 2, 3))

    mapping = probe.resolve_terrain_name_mapping(_Terrain())
    assert mapping["available"] is True
    assert mapping["allocation"] == "one_column_per_type"
    assert mapping["column_to_name"] == ["stairs", "perlin"]
    assert mapping["declared_num_cols"] == 20
    assert mapping["num_cols"] == 2


def test_resolve_terrain_mapping_isaac_proportion(probe):
    class _Tile:
        def __init__(self, proportion):
            self.proportion = proportion

    class _Gen:
        curriculum = True
        num_cols = 4
        sub_terrains = {"a": _Tile(0.25), "b": _Tile(0.75)}

    class _Cfg:
        terrain_generator = _Gen()

    class _Terrain:
        cfg = _Cfg()
        terrain_origins = __import__("numpy").zeros((3, 4, 3))

    mapping = probe.resolve_terrain_name_mapping(_Terrain())
    assert mapping["available"] is True
    assert mapping["allocation"] == "isaac_cumulative_proportion"
    assert mapping["num_cols"] == 4
    assert set(mapping["column_to_name"]) == {"a", "b"}
    assert mapping["column_to_name"].count("b") > mapping["column_to_name"].count("a")


def test_resolve_terrain_mapping_refuses_non_curriculum(probe):
    class _Gen:
        curriculum = False
        num_cols = 4
        sub_terrains = {"a": type("T", (), {"proportion": 1.0})()}

    class _Terrain:
        cfg = type("C", (), {"terrain_generator": _Gen()})()
        terrain_origins = __import__("numpy").zeros((2, 4, 3))

    mapping = probe.resolve_terrain_name_mapping(_Terrain())
    assert mapping["available"] is False
    assert "unresolvable" in mapping["reason"]


def test_snapshot_maps_terrain_id_to_name(probe):
    import torch

    class _TermMgr:
        active_terms = ["root_height"]

        @staticmethod
        def get_term(name):
            return torch.tensor([True])

        @property
        def terminated(self):
            return torch.tensor([True])

        @property
        def time_outs(self):
            return torch.tensor([False])

    class _Tile:
        proportion = 1.0

    class _Gen:
        curriculum = True
        num_cols = 2
        sub_terrains = {"perlin_rough": _Tile(), "pyramid_stairs": _Tile()}

    class _Terrain:
        cfg = type("C", (), {"terrain_generator": _Gen()})()
        terrain_origins = np.zeros((2, 2, 3))
        terrain_types = torch.tensor([1])
        terrain_levels = torch.tensor([3])

    class _Env:
        device = "cpu"
        termination_manager = _TermMgr()
        reset_terminated = torch.tensor([True])
        reset_time_outs = torch.tensor([False])
        episode_length_buf = torch.tensor([40])
        scene = type(
            "S",
            (),
            {
                "robot": type(
                    "R", (), {"data": type("D", (), {"root_link_pos_w": torch.tensor([[0.0, 0.0, 0.6]])})()}
                )(),
                "env_origins": torch.tensor([[0.0, 0.0, 0.0]]),
                "terrain": _Terrain(),
            },
        )()

    mapping = probe.resolve_terrain_name_mapping(_Terrain())
    events = probe.snapshot_pre_reset_episodes(_Env(), torch.tensor([0]), control_step=12, terrain_mapping=mapping)
    assert events[0]["terrain_type_id"] == 1
    assert events[0]["terrain_name"] == "pyramid_stairs"
    assert events[0]["terrain_level"] == 3
    assert events[0]["primary_reason"] == "root_height"


def test_compare_marks_mdp_parity_false_without_command(probe, tmp_path):
    names = ["j0"]
    left = _synthetic_payload(names, [np.array([[0.0]])], side="ours_cmd")
    right = _synthetic_payload(names, [np.array([[0.0]])], side="mj_cmd")
    left["metadata"]["command_state"] = "absent"
    right["metadata"]["command_state"] = "loaded_absent"
    report = probe.compare_rollout_payloads(left, right)
    assert report["command_state"]["command_dependent_mdp_parity"] is False
    assert "note" in report["command_state"]


def test_compare_marks_mdp_parity_true_with_command(probe):
    names = ["j0"]
    left = _synthetic_payload(names, [np.array([[0.0]])], side="ours_cmd2")
    right = _synthetic_payload(names, [np.array([[0.0]])], side="mj_cmd2")
    left["metadata"]["command_state"] = "present"
    right["metadata"]["command_state"] = "loaded_present"
    report = probe.compare_rollout_payloads(left, right)
    assert report["command_state"]["command_dependent_mdp_parity"] is True


def _fake_command_term(num_envs: int = 2):
    import torch

    class _Term:
        def __init__(self):
            self.vel_command_b = torch.zeros(num_envs, 3)
            self.pos_command_w = torch.zeros(num_envs, 3)
            self.heading_command_w = torch.zeros(num_envs)
            self.pos_command_b = torch.zeros(num_envs, 3)
            self.max_command_b = torch.zeros(num_envs, 3)
            self.is_standing_env = torch.zeros(num_envs, dtype=torch.bool)
            self.lin_vel_x_range = torch.tensor([[-1.0, 1.0]])
            self.lin_vel_y_range = torch.tensor([[-1.0, 1.0]])
            self.ang_vel_z_range = torch.tensor([[-1.0, 1.0]])
            self.random_lin_vel_x_range = torch.tensor([[-0.5, 0.5]])
            self.random_lin_vel_y_range = torch.tensor([[-0.5, 0.5]])
            self.random_ang_vel_z_range = torch.tensor([[-0.5, 0.5]])
            self.random_velocity_indices = torch.zeros(num_envs, dtype=torch.long)
            self.random_lin_vel_x = torch.zeros(num_envs)
            self.random_lin_vel_y = torch.zeros(num_envs)
            self.random_ang_vel_z = torch.zeros(num_envs)
            self.time_left = torch.ones(num_envs)
            self.command_counter = torch.zeros(num_envs, dtype=torch.long)

    return _Term()


def _synthetic_plant_state(num_envs: int = 2) -> dict:
    return {
        "joint_names": ["j0", "j1"],
        "action_target_names": ["j0", "j1"],
        "root_pos": np.zeros((num_envs, 3), dtype=np.float32),
        "root_quat": np.array([[1.0, 0.0, 0.0, 0.0]] * num_envs, dtype=np.float32),
        "root_lin_vel": np.zeros((num_envs, 3), dtype=np.float32),
        "root_ang_vel": np.zeros((num_envs, 3), dtype=np.float32),
        "joint_pos": np.zeros((num_envs, 2), dtype=np.float32),
        "joint_vel": np.zeros((num_envs, 2), dtype=np.float32),
    }


def test_command_state_roundtrip_npz(probe, tmp_path):
    term = _fake_command_term()
    term.vel_command_b[0, 2] = 0.75
    term.heading_command_w[1] = 1.25
    term.is_standing_env[0] = True
    captured = probe.capture_command_state(term)
    state = _synthetic_plant_state()
    state["command_state"] = captured
    path = tmp_path / "roundtrip.state.npz"
    probe._write_state_npz(path, state)
    loaded = probe._load_state_npz(path)
    assert loaded["command_state"]["schema"] == probe.COMMAND_STATE_SCHEMA
    assert set(loaded["command_state"]["fields"]) == set(probe.COMMAND_FIELD_NAMES)
    np.testing.assert_allclose(loaded["command_state"]["fields"]["vel_command_b"], captured["fields"]["vel_command_b"])
    np.testing.assert_array_equal(
        loaded["command_state"]["fields"]["is_standing_env"], captured["fields"]["is_standing_env"]
    )

    target = _fake_command_term()
    report = probe.apply_command_state(target, loaded["command_state"])
    assert report["missing_on_term"] == []
    assert report["missing_in_snapshot"] == []
    assert set(report["applied"]) == set(probe.COMMAND_FIELD_NAMES)
    assert target.vel_command_b[0, 2].item() == pytest.approx(0.75)
    assert target.heading_command_w[1].item() == pytest.approx(1.25)
    assert target.is_standing_env[0].item() is True


def test_apply_command_state_shape_mismatch_raises(probe):
    term = _fake_command_term()
    snapshot = probe.capture_command_state(term)
    snapshot["fields"]["vel_command_b"] = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="vel_command_b: shape"):
        probe.apply_command_state(term, snapshot)


def test_apply_command_state_missing_fields_reported(probe):
    full = _fake_command_term()
    snapshot = probe.capture_command_state(full)
    del snapshot["fields"]["heading_command_w"]
    term = _fake_command_term()
    term.vel_command_b = None  # type: ignore[assignment]
    del term.vel_command_b
    report = probe.apply_command_state(term, snapshot)
    assert "vel_command_b" in report["missing_on_term"]
    assert "heading_command_w" in report["missing_in_snapshot"]
    assert "vel_command_b" not in report["applied"]


def test_load_legacy_state_npz_without_command(probe, tmp_path):
    state = _synthetic_plant_state()
    path = tmp_path / "legacy.state.npz"
    probe._write_state_npz(path, state)
    loaded = probe._load_state_npz(path)
    assert loaded["command_state"] is None
    assert probe.command_state_status(captured=False, loaded=True, loaded_present=False) == "loaded_absent"


def test_command_state_status_helpers(probe):
    assert probe.command_state_status(captured=True, loaded=False, loaded_present=False) == "present"
    assert probe.command_state_status(captured=False, loaded=True, loaded_present=True) == "loaded_present"
    assert probe.command_npz_key("vel_command_b") == "cmd__vel_command_b"


def test_reweight_and_2x2_effects(probe):
    ours = {
        "metadata": {"side": "ours", "checkpoint": "/ckpt/ours.pt", "seed": 42},
        "eval": {
            "summary": {
                "mean_episode_length": 100.0,
                "root_height_rate_per_1000_env_steps": 8.0,
                "completed_episodes": 10,
                "per_terrain": {
                    "stairs": {"completed_episodes": 8, "mean_episode_length": 120.0},
                    "perlin": {"completed_episodes": 2, "mean_episode_length": 20.0},
                },
            },
            "terrain_mapping": {"allocation": "isaac_cumulative_proportion"},
        },
    }
    ref = {
        "metadata": {"side": "instinctmj", "checkpoint": "/ckpt/ours.pt", "seed": 42},
        "eval": {
            "summary": {
                "mean_episode_length": 80.0,
                "root_height_rate_per_1000_env_steps": 5.0,
                "completed_episodes": 10,
                "per_terrain": {
                    "stairs": {"completed_episodes": 5, "mean_episode_length": 90.0},
                    "perlin": {"completed_episodes": 5, "mean_episode_length": 70.0},
                },
            },
            "terrain_mapping": {"allocation": "one_column_per_type"},
        },
    }
    reweighted = probe.reweight_mean_length_to_mix(
        ours["eval"]["summary"]["per_terrain"], {"stairs": 0.5, "perlin": 0.5}
    )
    assert reweighted["available"] is True
    assert reweighted["value"] == pytest.approx(70.0)
    report = probe.analyze_policy_eval_2x2([ours, ref])
    assert len(report["factory_effect_same_ckpt"]) == 1
    pair = report["factory_effect_same_ckpt"][0]
    assert pair["mean_len_delta"] == pytest.approx(20.0)
    assert pair["reweighted_left_len_onto_right_mix"]["value"] == pytest.approx(70.0)
