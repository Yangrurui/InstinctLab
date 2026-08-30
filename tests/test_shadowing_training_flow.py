"""End-to-end contract guards around shadowing train, resume, play and export."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from instinctlab.checkpoint import task_contract
from instinctlab_engine_isaacsim.adapter import IsaacSimAdapter
from instinctlab_engine_mjlab.adapter import MjlabAdapter
from instinctlab.shadowing_probe import shadowing_task_with_motion
from instinctlab.tasks import registry
from instinctlab.training import (
    DistributedRun,
    destroy_process_group,
    distributed_run,
    load_runner_checkpoint,
    rank_device,
)
from tests.task_specs import task_spec

SHADOW_IDS = tuple(
    task_id for task_id in registry.ids() if any(token in task_id for token in ("Shadowing", "Mimic", "Vae"))
)


def test_vae_uses_one_frozen_teacher_bundle_on_both_engines() -> None:
    spec = task_spec("Instinct-Perceptive-Vae-G1-v0")
    runner_cfg = spec.agent.resolve()()
    teacher_dirs = {
        adapter.name: spec.agent.resolve()(
            **spec.agent.resolved_overrides(adapter.name)
        ).algorithm.teacher_logdir
        for adapter in (IsaacSimAdapter(), MjlabAdapter())
    }

    assert len(set(teacher_dirs.values())) == 1
    assert os.path.basename(next(iter(teacher_dirs.values()))) == "mjlab_gpu7_iter22000"
    assert tuple(spec.mdp.observations["critic"].terms) == tuple(
        runner_cfg.algorithm.teacher_policy["obs_format"]["policy"]
    )


def test_motion_probe_override_is_explicit_and_keeps_the_registered_identity(tmp_path) -> None:
    clip = tmp_path / "motion.npz"
    clip.touch()
    task_id = "Instinct-Shadowing-WholeBody-Plane-G1-v0"
    original = task_spec(task_id)
    overridden = shadowing_task_with_motion(task_id, "mjlab", clip)

    assert overridden.task_id == task_id
    assert overridden.scene.motion_references[0].clip == str(clip.resolve())
    assert task_contract(overridden) != task_contract(original)


def test_motion_probe_override_accepts_a_dataset_directory(tmp_path) -> None:
    dataset = tmp_path / "motions"
    dataset.mkdir()
    task_id = "Instinct-Perceptive-Vae-G1-v0"

    overridden = shadowing_task_with_motion(task_id, "mjlab", dataset)

    assert overridden.scene.motion_references[0].clip == str(dataset.resolve())
    assert overridden.scene.motion_references[0].first_motion_only is True
    assert overridden.scene.terrain.params["engine_paths"] == {
        "isaacsim": str(dataset.resolve()),
        "mjlab": str(dataset.resolve()),
    }


def test_motion_probe_override_refuses_a_missing_clip(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="motion source not found"):
        shadowing_task_with_motion(
            "Instinct-Shadowing-WholeBody-Plane-G1-v0",
            "mjlab",
            tmp_path / "missing.npz",
        )


def test_train_and_play_compile_the_same_shadowing_contract() -> None:
    for task_id in SHADOW_IDS:
        for adapter in (IsaacSimAdapter(), MjlabAdapter()):
            spec = task_spec(task_id, adapter.name)
            fingerprint = task_contract(spec)
            assert adapter.contract_report(spec)["missing"] == {}
            assert task_contract(spec) == fingerprint
            expected_agent = spec.agent.resolve()(**spec.agent.resolved_overrides(adapter.name)).to_dict()
            if adapter.name == "mjlab":
                compiled = adapter.compile(spec, num_envs=1, device="cpu", strict=True)
                assert compiled.resolution.task_id == task_id
                assert compiled.agent_cfg.to_dict()["policy"] == expected_agent["policy"]


def test_registered_play_checkpoint_pairs_keep_the_runner_architecture() -> None:
    for play_id, train_id in registry.PLAY_CHECKPOINT_TASKS.items():
        for adapter in (IsaacSimAdapter(), MjlabAdapter()):
            play = task_spec(play_id, adapter.name)
            train = task_spec(train_id, adapter.name)
            play_agent = play.agent.resolve()(**play.agent.resolved_overrides(adapter.name)).to_dict()
            train_agent = train.agent.resolve()(**train.agent.resolved_overrides(adapter.name)).to_dict()
            assert play_agent == train_agent, (adapter.name, play_id, train_id)


def test_distributed_rank_device_and_seed_streams_are_stable(monkeypatch) -> None:
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "4")
    run = distributed_run()
    assert run == DistributedRun(enabled=True, rank=2, local_rank=1, world_size=4)
    assert rank_device("cuda:7", run) == "cuda:1"
    assert rank_device("cpu", run) == "cpu"
    assert [DistributedRun(True, rank, rank, 4).seed(42) for rank in range(4)] == [
        42,
        43,
        44,
        45,
    ]


@pytest.mark.parametrize(
    ("name", "value", "message"),
    (("WORLD_SIZE", "0", "WORLD_SIZE"), ("RANK", "1", "RANK"), ("LOCAL_RANK", "-1", "LOCAL_RANK")),
)
def test_distributed_coordinates_are_validated(monkeypatch, name, value, message) -> None:
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=message):
        distributed_run(requested=True)


def test_distributed_shutdown_does_not_wait_for_failed_peers(monkeypatch) -> None:
    import torch.distributed as dist

    destroyed = []
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "destroy_process_group", lambda: destroyed.append(True))
    monkeypatch.setattr(
        dist,
        "barrier",
        lambda: pytest.fail("shutdown must not barrier after another rank can have failed"),
    )

    destroy_process_group(DistributedRun(True, 0, 0, 2))
    assert destroyed == [True]


class _Stateful:
    def __init__(self) -> None:
        self.loaded = None

    def load_state_dict(self, state) -> None:
        self.loaded = state


def test_distributed_resume_restores_optimizer_noise_normalizer_and_iteration(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "model_37.pt"
    state = {
        "model_state_dict": {"noise_std": torch.tensor([0.31])},
        "optimizer_state_dict": {"state": {7: {"step": torch.tensor(19)}}},
        "policy_normalizer_state_dict": {"mean": torch.tensor([1.5])},
        "iter": 37,
        "infos": {"source": "resume"},
    }
    torch.save(state, checkpoint)
    algorithm = _Stateful()
    normalizer = _Stateful()
    runner = SimpleNamespace(
        cfg={},
        device="cpu",
        alg=algorithm,
        normalizers={"policy": normalizer},
        current_learning_iteration=0,
    )
    infos = load_runner_checkpoint(runner, checkpoint, DistributedRun(True, 1, 1, 2))

    assert algorithm.loaded == state
    assert normalizer.loaded == state["policy_normalizer_state_dict"]
    assert runner.current_learning_iteration == 37
    assert infos == {"source": "resume"}


def test_shadowing_play_and_export_validate_checkpoint_contract_before_loading() -> None:
    text = (Path(__file__).resolve().parents[1] / "scripts" / "play.py").read_text()
    validation = text.index("validate_checkpoint_contract(")
    assert validation < text.index("runner.load(str(checkpoint))")
    assert 'experiment_policy="ignore"' in text[validation : validation + 300]
    assert text.index("runner.load(str(checkpoint))") < text.index("runner.export_as_onnx")
    assert '"task_contract": task_contract(' in text
    assert "spec, agent_config=agent_config" in text
