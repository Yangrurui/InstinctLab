from __future__ import annotations

import json
from dataclasses import replace

import instinctlab.checkpoint as checkpoint_module
import pytest
from instinctlab.checkpoint import (
    add_task_contract,
    checkpoint_load_semantics,
    latest_checkpoint,
    latest_run_checkpoint,
    runtime_provenance,
    task_contract,
    validate_checkpoint_contract,
)

from tests.task_specs import task_spec

PARKOUR_ID = "Instinct-Parkour-Target-G1"


def _agent_config(spec) -> dict:
    return spec.agent.resolve()(**dict(spec.agent.overrides)).to_dict()


def _write_manifest(tmp_path, spec, *, agent_config=None) -> None:
    manifest = add_task_contract({}, spec, agent_config=agent_config)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))


def test_task_manifest_is_readable_data_without_compatibility_hashes() -> None:
    spec = task_spec(PARKOUR_ID)
    contract = task_contract(spec)

    assert contract["version"] == "task_manifest_v4"
    assert contract["checkpoint_format_version"] == "instinct_rl_on_policy_runner_v1"
    assert contract["task_id"] == spec.task_id
    assert contract["joint_names"] == list(spec.robot.joint_names)
    assert contract["effective_agent_config"]["algorithm"]["class_name"] == "WasabiPPO"
    assert contract["task_declaration"]["type"].endswith("TaskSpec")
    assert "policy_io" not in contract
    assert "experiment_semantics" not in contract
    assert "provenance" not in contract
    assert "hash" not in json.dumps(contract)
    assert contract["strict_resume"]["task_id"] == spec.task_id
    assert set(contract["strict_resume"]) == {
        "task_id",
        "robot",
        "policy_io",
        "effective_agent_config",
        "training_semantics",
    }


def test_effective_engine_and_cli_agent_configuration_is_recorded_directly() -> None:
    spec = task_spec(PARKOUR_ID)
    overridden = replace(
        spec,
        agent=replace(
            spec.agent,
            engine_overrides={"mjlab": {"num_steps_per_env": 48}},
        ),
    )
    mjlab = task_contract(overridden, engine="mjlab")
    assert mjlab["effective_agent_config"]["num_steps_per_env"] == 48

    final_config = _agent_config(spec)
    final_config["seed"] = 123
    final_config["max_iterations"] = 456
    final = task_contract(spec, agent_config=final_config)
    assert final["effective_agent_config"]["seed"] == 123
    assert final["effective_agent_config"]["max_iterations"] == 456


def test_agent_snapshot_is_detached_from_later_mutation() -> None:
    spec = task_spec(PARKOUR_ID)
    agent_config = _agent_config(spec)
    contract = task_contract(spec, agent_config=agent_config)

    agent_config["algorithm"]["discriminator_kwargs"]["hidden_sizes"][0] = 2048

    assert contract["effective_agent_config"]["algorithm"]["discriminator_kwargs"][
        "hidden_sizes"
    ] == [1024, 512]


def test_runtime_provenance_records_dataset_checksum_and_run_identity(
    tmp_path, monkeypatch
) -> None:
    clip = tmp_path / "motion.npz"
    clip.write_bytes(b"fixed motion bytes")
    spec = task_spec(PARKOUR_ID)
    reference = replace(
        spec.scene.motion_references[0],
        clip=str(clip),
        engine_clips={},
    )
    spec = replace(
        spec,
        scene=replace(spec.scene, motion_references=(reference,)),
    )
    monkeypatch.setattr(
        checkpoint_module,
        "_accelerator_provenance",
        lambda device: {"requested_device": device},
    )
    monkeypatch.setattr(
        checkpoint_module,
        "_installed_versions",
        lambda engine: {"selected-backend": engine},
    )
    monkeypatch.setattr(
        checkpoint_module,
        "_git_provenance",
        lambda repository: {"available": True, "root": str(repository)},
    )
    monkeypatch.setattr(
        checkpoint_module,
        "_critical_editable_repositories",
        lambda engine: [
            {
                "root": "/workspace/dependency",
                "commit": "abc123",
                "dirty": False,
                "distributions": [{"name": engine, "version": "1.0"}],
            }
        ],
    )

    provenance = runtime_provenance(
        spec,
        engine="mjlab",
        device="cuda:1",
        num_envs=128,
        argv=["train.py", "--seed", "7"],
        repository=tmp_path,
    )

    assert provenance["command"] == {
        "argv": ["train.py", "--seed", "7"],
        "engine": "mjlab",
        "device": "cuda:1",
        "num_envs": 128,
    }
    assert provenance["version"] == "runtime_provenance_v2"
    assert provenance["repositories"][0]["commit"] == "abc123"
    assert provenance["repositories"][0]["dirty"] is False
    assert provenance["datasets"][0]["sha256"] == (
        "68ac8a95dbff11408b29efb6e2e76c6a095e13ff15b9a4a2fa7caebd4ee24b59"
    )
    assert provenance["datasets"][0]["resolved"] == str(clip.resolve())


def test_editable_repository_provenance_groups_distributions_by_git_root(
    monkeypatch,
) -> None:
    class _Distribution:
        version = "1.2.3"

        def __init__(self, source: str) -> None:
            self.source = source

        def read_text(self, name: str) -> str:
            assert name == "direct_url.json"
            return json.dumps(
                {"dir_info": {"editable": True}, "url": f"file://{self.source}"}
            )

    sources = {
        "instinctlab": "/workspace/project/source/instinctlab",
        "instinctlab-engine-core": "/workspace/project/source/core",
        "instinctlab-engine-mjlab": "/workspace/project/source/mjlab-engine",
        "instinct-rl": "/workspace/instinct_rl",
        "mjlab": "/workspace/mjlab",
    }

    def distribution(name: str):
        if name not in sources:
            raise checkpoint_module.importlib.metadata.PackageNotFoundError(name)
        return _Distribution(sources[name])

    def git(source):
        source = str(source)
        if source.startswith("/workspace/project"):
            root, commit, dirty = "/workspace/project", "project-sha", True
        else:
            root, commit, dirty = source, f"{source}-sha", False
        return {"available": True, "root": root, "commit": commit, "dirty": dirty}

    monkeypatch.setattr(checkpoint_module.importlib.metadata, "distribution", distribution)
    monkeypatch.setattr(checkpoint_module, "_git_provenance", git)

    repositories = checkpoint_module._critical_editable_repositories("mjlab")

    project = next(item for item in repositories if item["root"] == "/workspace/project")
    assert project["commit"] == "project-sha"
    assert project["dirty"] is True
    assert [item["name"] for item in project["distributions"]] == [
        "instinctlab",
        "instinctlab-engine-core",
        "instinctlab-engine-mjlab",
    ]


def test_manifest_carries_portability_and_task_metadata() -> None:
    spec = task_spec(PARKOUR_ID)
    manifest = add_task_contract({"engine": "isaacsim"}, spec)

    assert manifest["portability"]["contract_portability"]["multi_engine"]
    assert manifest["portability"]["clean_resolution"]["known"] is False
    assert manifest["task_contract"]["portability"]["native_extras"]["count"] == 0


def test_checkpoint_validation_accepts_supported_readable_format(tmp_path) -> None:
    spec = task_spec(PARKOUR_ID)
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    _write_manifest(tmp_path, spec)

    validate_checkpoint_contract(checkpoint, spec)


def test_strict_resume_accepts_an_unchanged_contract(tmp_path) -> None:
    spec = task_spec(PARKOUR_ID)
    agent_config = _agent_config(spec)
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    _write_manifest(tmp_path, spec, agent_config=agent_config)

    validate_checkpoint_contract(
        checkpoint,
        spec,
        mode="resume",
        agent_config=agent_config,
    )


def test_strict_resume_rejects_robot_schema_drift(tmp_path) -> None:
    spec = task_spec(PARKOUR_ID)
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    _write_manifest(tmp_path, spec)
    changed = replace(
        spec,
        robot=replace(spec.robot, schema_version="changed_dfs"),
    )

    with pytest.raises(ValueError, match="robot schema drift"):
        validate_checkpoint_contract(checkpoint, changed, mode="resume")


def test_strict_resume_rejects_observation_drift(tmp_path) -> None:
    spec = task_spec(PARKOUR_ID)
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    _write_manifest(tmp_path, spec)
    group_name, group = next(iter(spec.mdp.observations.items()))
    term_name, term = next(iter(group.terms.items()))
    changed_terms = dict(group.terms)
    changed_terms[term_name] = replace(term, scale=1.25)
    changed_observations = dict(spec.mdp.observations)
    changed_observations[group_name] = replace(group, terms=changed_terms)
    changed = replace(
        spec,
        mdp=replace(spec.mdp, observations=changed_observations),
    )

    with pytest.raises(ValueError, match="policy I/O drift"):
        validate_checkpoint_contract(checkpoint, changed, mode="resume")


def test_strict_resume_rejects_reward_drift(tmp_path) -> None:
    spec = task_spec(PARKOUR_ID)
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    _write_manifest(tmp_path, spec)
    group_name, terms = next(iter(spec.mdp.rewards.items()))
    term_name, term = next(iter(terms.items()))
    changed_terms = dict(terms)
    changed_terms[term_name] = replace(term, weight=term.weight + 0.5)
    changed_rewards = dict(spec.mdp.rewards)
    changed_rewards[group_name] = changed_terms
    changed = replace(spec, mdp=replace(spec.mdp, rewards=changed_rewards))

    with pytest.raises(ValueError, match="training semantics drift"):
        validate_checkpoint_contract(checkpoint, changed, mode="resume")


def test_strict_resume_rejects_effective_agent_drift(tmp_path) -> None:
    spec = task_spec(PARKOUR_ID)
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    stored_agent = _agent_config(spec)
    _write_manifest(tmp_path, spec, agent_config=stored_agent)
    current_agent = _agent_config(spec)
    current_agent["num_steps_per_env"] += 1

    with pytest.raises(ValueError, match="effective agent drift"):
        validate_checkpoint_contract(
            checkpoint,
            spec,
            mode="resume",
            agent_config=current_agent,
        )


def test_explicit_transfer_accepts_declared_training_drift(tmp_path) -> None:
    spec = task_spec(PARKOUR_ID)
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    _write_manifest(tmp_path, spec)
    group_name, terms = next(iter(spec.mdp.rewards.items()))
    term_name, term = next(iter(terms.items()))
    changed_terms = dict(terms)
    changed_terms[term_name] = replace(term, weight=term.weight + 0.5)
    changed_rewards = dict(spec.mdp.rewards)
    changed_rewards[group_name] = changed_terms
    changed = replace(spec, mdp=replace(spec.mdp, rewards=changed_rewards))

    validate_checkpoint_contract(checkpoint, changed, mode="transfer")


def test_checkpoint_load_semantics_explicitly_use_fresh_environment_state() -> None:
    resume = checkpoint_load_semantics("resume")
    transfer = checkpoint_load_semantics("transfer")

    assert resume["lifecycle_snapshot"] == "not restored"
    assert resume["environment_state"] == "fresh environment construction and reset"
    assert resume["common_rng_state"] == (
        "not restored; initialized from the current run seed"
    )
    assert "learning iteration restored" in resume["runner_state"]
    assert "learning iteration reset to zero" in transfer["runner_state"]


def test_checkpoint_validation_rejects_unknown_manifest_schema(tmp_path) -> None:
    spec = task_spec(PARKOUR_ID)
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    manifest = add_task_contract({}, spec)
    manifest["task_contract"]["version"] = "task_manifest_v999"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="manifest schema version mismatch"):
        validate_checkpoint_contract(checkpoint, spec)


def test_checkpoint_validation_rejects_unknown_checkpoint_format(tmp_path) -> None:
    spec = task_spec(PARKOUR_ID)
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    manifest = add_task_contract({}, spec)
    manifest["task_contract"]["checkpoint_format_version"] = "future_runner_v9"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="Checkpoint format version mismatch"):
        validate_checkpoint_contract(checkpoint, spec)


def test_task_and_declaration_drift_are_metadata_not_load_gates(tmp_path) -> None:
    spec = task_spec(PARKOUR_ID)
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    _write_manifest(tmp_path, spec)

    changed_robot = replace(
        spec.robot,
        schema_version="different",
        joint_names=tuple(reversed(spec.robot.joint_names)),
    )
    changed = replace(spec, robot=changed_robot)
    validate_checkpoint_contract(checkpoint, changed)

    different_task = replace(changed, task_id="different-task")
    with pytest.warns(RuntimeWarning, match="metadata task"):
        validate_checkpoint_contract(checkpoint, different_task)


def test_play_training_pair_uses_registered_metadata_identity(tmp_path) -> None:
    train_id = "Instinct-Perceptive-Shadowing-G1-v0"
    play_id = "Instinct-Perceptive-Shadowing-G1-Play-v0"
    train_spec = task_spec(train_id)
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    _write_manifest(tmp_path, train_spec)

    validate_checkpoint_contract(
        checkpoint,
        task_spec(play_id),
        checkpoint_task_id=train_id,
    )


@pytest.mark.parametrize("legacy_version", ("task_spec_v1", "task_contract_v2"))
def test_legacy_manifest_remains_loadable_with_warning(tmp_path, legacy_version) -> None:
    spec = task_spec(PARKOUR_ID)
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    legacy = {"version": legacy_version, "task_id": spec.task_id}
    (tmp_path / "manifest.json").write_text(json.dumps({"task_contract": legacy}))

    with pytest.warns(RuntimeWarning, match="legacy metadata"):
        validate_checkpoint_contract(checkpoint, spec)


def test_legacy_checkpoint_without_manifest_remains_loadable_with_warning(tmp_path) -> None:
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.touch()
    with pytest.warns(RuntimeWarning, match="format version"):
        validate_checkpoint_contract(checkpoint, task_spec(PARKOUR_ID))


def test_checkpoint_path_must_exist_before_environment_construction(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        validate_checkpoint_contract(tmp_path / "missing.pt", task_spec(PARKOUR_ID))


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


def test_latest_run_checkpoint_can_require_the_latest_matching_run(tmp_path) -> None:
    complete = tmp_path / "20260101_000000"
    complete.mkdir()
    (complete / "model_100.pt").touch()
    latest = tmp_path / "20260102_000000"
    latest.mkdir()

    with pytest.raises(FileNotFoundError, match=str(latest)):
        latest_run_checkpoint(tmp_path)
