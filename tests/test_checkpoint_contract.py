from __future__ import annotations

import json
from dataclasses import replace

import pytest

from instinctlab.checkpoint import (
    add_task_contract,
    latest_checkpoint,
    latest_run_checkpoint,
    task_contract,
    validate_checkpoint_contract,
)
from instinctlab.tasks.parkour.config.g1.g1_parkour_target_amp_cfg import parkour_target_g1


def test_task_contract_is_stable_and_backend_independent() -> None:
    spec = parkour_target_g1()
    first = task_contract(spec)
    second = task_contract(parkour_target_g1())
    assert first == second
    assert first["task_id"] == spec.task_id
    assert first["joint_names"] == list(spec.robot.joint_names)


def test_contract_changes_when_tensor_order_changes() -> None:
    spec = parkour_target_g1()
    reversed_robot = replace(spec.robot, joint_names=tuple(reversed(spec.robot.joint_names)))
    changed = replace(spec, robot=reversed_robot)
    assert task_contract(changed)["hash"] != task_contract(spec)["hash"]


def test_contract_does_not_depend_on_absolute_backend_asset_paths() -> None:
    spec = parkour_target_g1()
    moved_assets = tuple(replace(asset, path=f"/different/clone/{asset.backend}.asset") for asset in spec.robot.assets)
    moved = replace(spec, robot=replace(spec.robot, assets=moved_assets))
    assert task_contract(moved)["hash"] == task_contract(spec)["hash"]


def test_checkpoint_contract_accepts_the_same_task_across_engines(tmp_path) -> None:
    spec = parkour_target_g1()
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    manifest = add_task_contract({"engine": "isaacsim"}, spec)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    validate_checkpoint_contract(checkpoint, spec)


def test_checkpoint_contract_rejects_a_different_task(tmp_path) -> None:
    spec = parkour_target_g1()
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    (tmp_path / "manifest.json").write_text(json.dumps(add_task_contract({}, spec)))
    changed = replace(spec, task_id="different-task")
    with pytest.raises(ValueError, match="Checkpoint task contract mismatch"):
        validate_checkpoint_contract(checkpoint, changed)


def test_checkpoint_contract_does_not_reject_hash_drift(tmp_path) -> None:
    spec = parkour_target_g1()
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    manifest = add_task_contract({}, spec)
    manifest["task_contract"]["hash"] = "deadbeef" * 8
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    validate_checkpoint_contract(checkpoint, spec)


def test_checkpoint_contract_rejects_joint_order_drift(tmp_path) -> None:
    spec = parkour_target_g1()
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    (tmp_path / "manifest.json").write_text(json.dumps(add_task_contract({}, spec)))
    changed = replace(spec, robot=replace(spec.robot, joint_names=tuple(reversed(spec.robot.joint_names))))

    with pytest.raises(ValueError, match="canonical joint order mismatch"):
        validate_checkpoint_contract(checkpoint, changed)


def test_checkpoint_contract_rejects_robot_joint_schema_drift(tmp_path) -> None:
    spec = parkour_target_g1()
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    (tmp_path / "manifest.json").write_text(json.dumps(add_task_contract({}, spec)))
    changed = replace(spec, robot=replace(spec.robot, schema_version="different_joint_schema"))

    with pytest.raises(ValueError, match="robot joint schema mismatch"):
        validate_checkpoint_contract(checkpoint, changed)


def test_legacy_checkpoint_without_manifest_remains_loadable_with_warning(tmp_path) -> None:
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    with pytest.warns(RuntimeWarning, match="compatibility cannot be verified"):
        validate_checkpoint_contract(checkpoint, parkour_target_g1())


def test_latest_checkpoint_sorts_iterations_numerically(tmp_path) -> None:
    for name in ("model_9.pt", "model_1000.pt", "model_100.pt"):
        (tmp_path / name).touch()
    assert latest_checkpoint(tmp_path).name == "model_1000.pt"


def test_playback_checkpoint_discovery_can_skip_an_empty_latest_run(tmp_path) -> None:
    complete = tmp_path / "20260101_000000"
    complete.mkdir()
    (complete / "model_100.pt").touch()
    (tmp_path / "20260102_000000").mkdir()

    checkpoint = latest_run_checkpoint(tmp_path, skip_empty_runs=True)

    assert checkpoint == complete / "model_100.pt"
