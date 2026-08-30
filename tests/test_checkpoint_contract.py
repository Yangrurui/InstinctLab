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
from instinctlab.tasks import registry
from tests.task_specs import task_spec

PARKOUR_ID = "Instinct-Parkour-Target-G1"


def test_task_contract_is_stable_and_backend_independent() -> None:
    spec = task_spec(PARKOUR_ID)
    first = task_contract(spec)
    second = task_contract(task_spec(PARKOUR_ID, "isaacsim"))
    assert first["policy_io"] == second["policy_io"]
    assert first["experiment_semantics"] == second["experiment_semantics"]
    assert first["provenance"] != second["provenance"]
    assert first["task_id"] == spec.task_id
    assert first["joint_names"] == list(spec.robot.joint_names)


def test_contract_changes_when_tensor_order_changes() -> None:
    spec = task_spec(PARKOUR_ID)
    reversed_robot = replace(spec.robot, joint_names=tuple(reversed(spec.robot.joint_names)))
    changed = replace(spec, robot=reversed_robot)
    assert task_contract(changed)["policy_io"] != task_contract(spec)["policy_io"]


def test_contract_does_not_depend_on_absolute_backend_asset_paths() -> None:
    spec = task_spec(PARKOUR_ID)
    moved_assets = tuple(replace(asset, path=f"/different/clone/{asset.backend}.asset") for asset in spec.robot.assets)
    moved = replace(spec, robot=replace(spec.robot, assets=moved_assets))
    current = task_contract(spec)
    changed = task_contract(moved)
    assert changed["policy_io"] == current["policy_io"]
    assert changed["experiment_semantics"] == current["experiment_semantics"]
    assert changed["provenance"] != current["provenance"]


def test_checkpoint_contract_accepts_the_same_task_across_engines(tmp_path) -> None:
    spec = task_spec(PARKOUR_ID)
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    manifest = add_task_contract({"engine": "isaacsim"}, spec)
    assert manifest["portability"]["contract_portability"]["multi_engine"]
    assert manifest["portability"]["clean_resolution"]["known"] is False
    assert manifest["task_contract"]["portability"]["native_extras"]["count"] == 0
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    validate_checkpoint_contract(checkpoint, spec)


def test_checkpoint_contract_rejects_a_different_task(tmp_path) -> None:
    spec = task_spec(PARKOUR_ID)
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    (tmp_path / "manifest.json").write_text(json.dumps(add_task_contract({}, spec)))
    changed = replace(spec, task_id="different-task")
    with pytest.raises(ValueError, match="Checkpoint task contract mismatch"):
        validate_checkpoint_contract(checkpoint, changed)


def test_checkpoint_contract_accepts_an_explicit_play_training_pair(tmp_path) -> None:
    train_id = "Instinct-Perceptive-Shadowing-G1-v0"
    play_id = "Instinct-Perceptive-Shadowing-G1-Play-v0"
    train_spec = task_spec(train_id)
    play_spec = task_spec(play_id)
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    (tmp_path / "manifest.json").write_text(json.dumps(add_task_contract({}, train_spec)))

    validate_checkpoint_contract(
        checkpoint,
        play_spec,
        checkpoint_task_id=train_id,
        experiment_policy="ignore",
    )


def test_checkpoint_contract_keeps_unpaired_tasks_strict(tmp_path) -> None:
    train_spec = task_spec("Instinct-Perceptive-Shadowing-G1-v0")
    wrong_play = task_spec("Instinct-BeyondMimic-Plane-G1-Play-v0")
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    (tmp_path / "manifest.json").write_text(json.dumps(add_task_contract({}, train_spec)))

    with pytest.raises(ValueError, match="Checkpoint task contract mismatch"):
        validate_checkpoint_contract(
            checkpoint,
            wrong_play,
            checkpoint_task_id=registry.checkpoint_task_id(wrong_play.task_id),
        )


def test_checkpoint_contract_does_not_reject_provenance_drift(tmp_path) -> None:
    spec = task_spec(PARKOUR_ID)
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    manifest = add_task_contract({}, spec)
    manifest["task_contract"]["provenance"]["hash"] = "deadbeef" * 8
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    validate_checkpoint_contract(checkpoint, spec)


def test_checkpoint_contract_rejects_observation_scale_drift(tmp_path) -> None:
    spec = task_spec(PARKOUR_ID)
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    (tmp_path / "manifest.json").write_text(json.dumps(add_task_contract({}, spec)))
    group_name, group = next(iter(spec.mdp.observations.items()))
    term_name, term = next(iter(group.terms.items()))
    changed_term = replace(term, scale=2.0 if term.scale != 2.0 else 3.0)
    changed_group = replace(group, terms={**group.terms, term_name: changed_term})
    changed = replace(
        spec,
        mdp=replace(
            spec.mdp,
            observations={**spec.mdp.observations, group_name: changed_group},
        ),
    )

    with pytest.raises(ValueError, match="policy I/O contract mismatch"):
        validate_checkpoint_contract(checkpoint, changed)


def test_checkpoint_contract_rejects_action_layout_drift(tmp_path) -> None:
    spec = task_spec(PARKOUR_ID)
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    (tmp_path / "manifest.json").write_text(json.dumps(add_task_contract({}, spec)))
    action_name, action = next(iter(spec.mdp.actions.items()))
    changed_action = replace(action, params={**action.params, "scale": 0.12345})
    changed = replace(
        spec,
        mdp=replace(spec.mdp, actions={**spec.mdp.actions, action_name: changed_action}),
    )

    with pytest.raises(ValueError, match="policy I/O contract mismatch"):
        validate_checkpoint_contract(checkpoint, changed)


def test_resume_requires_an_explicit_experiment_drift_override(tmp_path) -> None:
    spec = task_spec(PARKOUR_ID)
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    (tmp_path / "manifest.json").write_text(json.dumps(add_task_contract({}, spec)))
    reward_group_name, rewards = next(iter(spec.mdp.rewards.items()))
    reward_name, reward = next(iter(rewards.items()))
    changed_reward = replace(reward, weight=reward.weight + 1.0)
    changed = replace(
        spec,
        mdp=replace(
            spec.mdp,
            rewards={
                **spec.mdp.rewards,
                reward_group_name: {**rewards, reward_name: changed_reward},
            },
        ),
    )

    with pytest.raises(ValueError, match="experiment semantics mismatch"):
        validate_checkpoint_contract(checkpoint, changed)
    with pytest.warns(RuntimeWarning, match="experiment semantics mismatch"):
        validate_checkpoint_contract(checkpoint, changed, experiment_policy="warn")
    validate_checkpoint_contract(checkpoint, changed, experiment_policy="ignore")


def test_legacy_v1_contract_remains_loadable_with_warning(tmp_path) -> None:
    spec = task_spec(PARKOUR_ID)
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    legacy = {
        "version": "task_spec_v1",
        "task_id": spec.task_id,
        "robot_schema_version": spec.robot.schema_version,
        "asset_id": spec.robot.asset_id,
        "joint_names": list(spec.robot.joint_names),
        "body_names": list(spec.robot.body_names),
        "hash": "deadbeef" * 8,
    }
    (tmp_path / "manifest.json").write_text(json.dumps({"task_contract": legacy}))

    with pytest.warns(RuntimeWarning, match="policy I/O compatibility cannot be verified"):
        validate_checkpoint_contract(checkpoint, spec)


def test_checkpoint_contract_rejects_joint_order_drift(tmp_path) -> None:
    spec = task_spec(PARKOUR_ID)
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    (tmp_path / "manifest.json").write_text(json.dumps(add_task_contract({}, spec)))
    changed = replace(spec, robot=replace(spec.robot, joint_names=tuple(reversed(spec.robot.joint_names))))

    with pytest.raises(ValueError, match="canonical joint order mismatch"):
        validate_checkpoint_contract(checkpoint, changed)


def test_checkpoint_contract_rejects_robot_joint_schema_drift(tmp_path) -> None:
    spec = task_spec(PARKOUR_ID)
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
        validate_checkpoint_contract(checkpoint, task_spec(PARKOUR_ID))


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
