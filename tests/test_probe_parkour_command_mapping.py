"""Offline tests for scripts/probe_parkour_command_mapping.py (no GPU, no engine bootstrap)."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/probe_parkour_command_mapping.py"
INSTINCTMJ_REG = Path("/root/InstinctMJ/src/instinct_mj/tasks/parkour/config/g1/__init__.py")

PARKOUR_NAMES = [
    "perlin_rough",
    "perlin_rough_stand",
    "square_gaps",
    "pyramid_stairs",
    "pyramid_stairs_high",
    "pyramid_stairs_inv",
    "pyramid_stairs_inv_high",
    "boxes",
    "dense_boxes",
    "hf_pyramid_slope_inv",
]
PARKOUR_PROPORTIONS = [0.05, 0.05, 0.10, 0.15, 0.10, 0.15, 0.10, 0.10, 0.10, 0.10]


def _load_probe():
    spec = importlib.util.spec_from_file_location("probe_parkour_command_mapping", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe():
    return _load_probe()


def _synthetic_row(
    probe,
    *,
    name: str,
    type_ids: list[int],
    effective,
    declared,
    source: str | None,
    bind_class: str,
):
    return {
        "name": name,
        "env_count": 16,
        "env_fraction": 16 / 256,
        "type_ids": type_ids,
        "effective_range": list(effective) if effective is not None else None,
        "live_range_unique": [tuple(effective)] if effective is not None else [],
        "declared_box": list(declared) if declared is not None else None,
        "predicted_declared_col_source": source,
        "predicted_declared_col_sources": [source] if source else [],
        "predicted_name_loop_source": name,
        "bind_class": bind_class,
        "name_loop_class": "aligned",
        "sample": probe.summarize_samples([0.0] * 64, [False] * 64, [False] * 64),
    }


def _official_payload(probe, *, side: str, by_name: dict, play: bool = False):
    mapping = probe.name_mapping_from_lists(
        names=PARKOUR_NAMES,
        proportions=PARKOUR_PROPORTIONS,
        built_num_cols=10 if side == "instinctmj" else 20,
        declared_num_cols=20,
    )
    return {
        "metadata": {
            "side": side,
            "seed": 42,
            "num_envs": 256,
            "task_id": probe.INSTINCTMJ_TASK_ID if side == "instinctmj" else probe.OURS_TASK_ID,
            "play": play,
            "declared_num_cols": 20,
            "built_num_cols": mapping["built_num_cols"],
            "command_name": probe.COMMAND_NAME,
        },
        "name_mapping_source": mapping,
        "by_terrain_name": by_name,
        "misbinds": probe.collect_misbinds(by_name),
    }


def test_module_top_level_has_no_engine_imports(probe):
    hits = probe.module_top_level_engine_imports(SCRIPT)
    assert hits == [], f"module must lazy-import engines; found {hits}"


def test_cli_help_without_engine():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--side" in result.stdout
    assert "instinctmj" in result.stdout


@pytest.mark.skipif(not INSTINCTMJ_REG.is_file(), reason="InstinctMJ not checked out")
def test_instinctmj_registration_and_builder_use_play_false(probe):
    info = probe.read_instinctmj_train_registration(INSTINCTMJ_REG)
    assert info["task_id"] == probe.INSTINCTMJ_TASK_ID
    assert info["play_false_in_env_cfg_factory"] is True
    source = inspect.getsource(probe._build_instinctmj)
    compact = source.replace(" ", "")
    assert "play=False" in compact
    assert "play=True" not in compact


def test_official_256_env_schema(probe):
    by_name = {
        "pyramid_stairs": _synthetic_row(
            probe,
            name="pyramid_stairs",
            type_ids=[3],
            effective=(0.0, 0.0),
            declared=(0.45, 0.8),
            source=None,
            bind_class="unbound_zero",
        )
    }
    payload = _official_payload(probe, side="instinctmj", by_name=by_name)
    probe.validate_payload_schema(payload, official=True)
    small = json.loads(json.dumps(payload))
    small["metadata"]["num_envs"] = 2
    with pytest.raises(ValueError, match="num_envs=256"):
        probe.validate_payload_schema(small, official=True)


def test_schema_refuses_digit_type_id_keys(probe):
    by_name = {
        "3": _synthetic_row(
            probe,
            name="3",
            type_ids=[3],
            effective=(0.0, 0.0),
            declared=(0.45, 0.8),
            source=None,
            bind_class="unbound_zero",
        )
    }
    payload = _official_payload(probe, side="ours", by_name=by_name)
    with pytest.raises(ValueError, match="bare type-id"):
        probe.validate_payload_schema(payload)


def test_name_mapping_uses_built_grid_not_declared_cols(probe):
    ten = probe.name_mapping_from_lists(
        names=PARKOUR_NAMES,
        proportions=PARKOUR_PROPORTIONS,
        built_num_cols=10,
        declared_num_cols=20,
    )
    assert ten["allocation"] == "one_column_per_type"
    assert ten["column_to_name"][3] == "pyramid_stairs"
    assert ten["built_num_cols"] == 10
    assert ten["declared_num_cols"] == 20
    twenty = probe.name_mapping_from_lists(
        names=PARKOUR_NAMES,
        proportions=PARKOUR_PROPORTIONS,
        built_num_cols=20,
        declared_num_cols=20,
    )
    assert twenty["allocation"] == "isaac_cumulative_proportion"
    assert twenty["column_to_name"].count("pyramid_stairs") == 3
    assert 3 not in [i for i, n in enumerate(twenty["column_to_name"]) if n == "pyramid_stairs"]


def test_synthetic_10_col_plus_20_col_recipe_catches_misbind(probe):
    unique = probe.unique_box_recipe(PARKOUR_NAMES)
    declared = probe.declared_column_box_for_type_ids(
        names=PARKOUR_NAMES,
        proportions=PARKOUR_PROPORTIONS,
        declared_num_cols=20,
        velocity_ranges=unique,
    )
    name_loop = probe.name_loop_box_for_type_ids(
        names=PARKOUR_NAMES,
        proportions=PARKOUR_PROPORTIONS,
        built_num_cols=10,
        velocity_ranges=unique,
    )
    stairs_id = PARKOUR_NAMES.index("pyramid_stairs")
    gaps_id = PARKOUR_NAMES.index("square_gaps")
    assert name_loop[stairs_id]["source_name"] == "pyramid_stairs"
    assert declared[stairs_id]["source_name"] == "square_gaps"
    assert declared[stairs_id]["lin_vel_x"] == unique["square_gaps"]
    assert declared[stairs_id]["lin_vel_x"] != unique["pyramid_stairs"]
    assert declared[gaps_id]["source_name"] == "square_gaps"

    terrain_types = [i % 10 for i in range(256)]
    live = [declared[tid]["lin_vel_x"] for tid in terrain_types]
    mapping = probe.name_mapping_from_lists(
        names=PARKOUR_NAMES,
        proportions=PARKOUR_PROPORTIONS,
        built_num_cols=10,
        declared_num_cols=20,
    )
    issued = [0.0] * (256 * 64)
    standing = [False] * (256 * 64)
    near = [False] * (256 * 64)
    by_name = probe.aggregate_by_terrain_name(
        type_ids=terrain_types,
        mapping=mapping,
        live_ranges=live,
        declared_boxes=unique,
        predicted_declared=declared,
        predicted_name_loop=name_loop,
        issued=issued,
        standing=standing,
        target_near=near,
        resample_rounds=64,
    )
    stairs = by_name["pyramid_stairs"]
    assert stairs["type_ids"] == [3]
    assert stairs["effective_range"] == list(unique["square_gaps"])
    assert stairs["predicted_declared_col_source"] == "square_gaps"
    assert stairs["bind_class"] == "wrong_box"
    wrong = {row["name"]: row for row in probe.collect_misbinds(by_name)}
    assert "pyramid_stairs" in wrong


def test_parkour_value_collision_is_source_mismatch_not_zero(probe):
    """Real recipe: stairs and gaps share (0.45, 0.8), so the 10/20 loop does not zero stairs."""
    real = {name: (0.45, 0.8) for name in PARKOUR_NAMES}
    real["perlin_rough"] = (0.45, 1.0)
    real["perlin_rough_stand"] = (0.0, 0.0)
    declared = probe.declared_column_box_for_type_ids(
        names=PARKOUR_NAMES,
        proportions=PARKOUR_PROPORTIONS,
        declared_num_cols=20,
        velocity_ranges=real,
    )
    stairs_id = PARKOUR_NAMES.index("pyramid_stairs")
    assert declared[stairs_id]["source_name"] == "square_gaps"
    assert declared[stairs_id]["lin_vel_x"] == (0.45, 0.8)
    klass = probe.classify_bind(
        name="pyramid_stairs",
        effective=(0.45, 0.8),
        declared=(0.45, 0.8),
        predicted_source="square_gaps",
    )
    assert klass == "value_ok_source_mismatch"
    assert (
        probe.classify_bind(
            name="pyramid_stairs",
            effective=(0.0, 0.0),
            declared=(0.45, 0.8),
            predicted_source=None,
        )
        == "unbound_zero"
    )


def test_compare_is_by_name_not_type_id(probe):
    ours = _official_payload(
        probe,
        side="ours",
        by_name={
            "pyramid_stairs": _synthetic_row(
                probe,
                name="pyramid_stairs",
                type_ids=[4, 5, 6],
                effective=(0.45, 0.8),
                declared=(0.45, 0.8),
                source="pyramid_stairs",
                bind_class="aligned",
            )
        },
    )
    ref = _official_payload(
        probe,
        side="instinctmj",
        by_name={
            "pyramid_stairs": _synthetic_row(
                probe,
                name="pyramid_stairs",
                type_ids=[3],
                effective=(0.45, 0.8),
                declared=(0.45, 0.8),
                source="square_gaps",
                bind_class="value_ok_source_mismatch",
            )
        },
    )
    report = probe.compare_payloads(ours, ref)
    assert "pyramid_stairs" in report["per_name"]
    assert report["per_name"]["pyramid_stairs"]["same_effective_range"] is True
    assert (
        report["per_name"]["pyramid_stairs"]["left_type_ids"] != report["per_name"]["pyramid_stairs"]["right_type_ids"]
    )
    assert "pyramid_stairs" in report["source_or_class_diffs"]


def test_instinctmj_python_hint_names_training_venv(probe):
    hint = probe.instinctmj_python_hint("/usr/bin/python")
    assert str(probe.DEFAULT_INSTINCTMJ_PYTHON) in hint


@pytest.mark.skipif(not INSTINCTMJ_REG.is_file(), reason="InstinctMJ not checked out")
def test_synthetic_recipe_names_match_instinctmj_ast():
    from tests.reference_mjlab_parkour import terrain_recipe

    recipe = terrain_recipe()
    assert list(recipe["sub_terrains"]) == PARKOUR_NAMES
    props = [float(recipe["sub_terrains"][name]["proportion"]) for name in PARKOUR_NAMES]
    assert props == pytest.approx(PARKOUR_PROPORTIONS)


def test_round_major_samples_stay_with_the_named_env(probe):
    mapping = probe.name_mapping_from_lists(
        names=["a", "b"],
        proportions=[0.5, 0.5],
        built_num_cols=2,
        declared_num_cols=2,
    )
    type_ids = [0, 1, 0, 1]
    n_env, rounds = 4, 3
    issued = []
    standing = []
    near = []
    for _round in range(rounds):
        issued.extend([10.0, 20.0, 10.0, 20.0])
        standing.extend([False, True, False, True])
        near.extend([False] * n_env)
    by_name = probe.aggregate_by_terrain_name(
        type_ids=type_ids,
        mapping=mapping,
        live_ranges=[(0.1, 0.2), (0.3, 0.4), (0.1, 0.2), (0.3, 0.4)],
        declared_boxes={"a": (0.1, 0.2), "b": (0.3, 0.4)},
        predicted_declared={
            0: {"source_name": "a", "lin_vel_x": (0.1, 0.2)},
            1: {"source_name": "b", "lin_vel_x": (0.3, 0.4)},
        },
        predicted_name_loop={
            0: {"source_name": "a", "lin_vel_x": (0.1, 0.2)},
            1: {"source_name": "b", "lin_vel_x": (0.3, 0.4)},
        },
        issued=issued,
        standing=standing,
        target_near=near,
        resample_rounds=rounds,
    )
    assert by_name["a"]["sample"]["min"] == 10.0
    assert by_name["a"]["sample"]["max"] == 10.0
    assert by_name["b"]["sample"]["min"] == 20.0
    assert by_name["b"]["sample"]["standing_fraction"] == 1.0
    assert by_name["a"]["sample"]["standing_fraction"] == 0.0


def test_builder_ast_does_not_call_prepare_probe_hooks(probe):
    """This probe must not reuse plant-probe mutations (auto_reset=False / term nulling)."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.add(node.func.id)
    assert "_prepare_probe_env_cfg" not in names
    assert "_disable_terminations" not in names
