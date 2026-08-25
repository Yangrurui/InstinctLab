"""End-to-end contract guards around shadowing train, resume, play and export."""

from __future__ import annotations

import torch
from pathlib import Path
from types import SimpleNamespace

from instinctlab.checkpoint import task_contract
from instinctlab.engines.isaacsim.adapter import IsaacSimAdapter
from instinctlab.engines.mjlab.adapter import MjlabAdapter
from instinctlab.tasks import registry
from instinctlab.training import DistributedRun, distributed_run, load_runner_checkpoint, rank_device

SHADOW_IDS = tuple(
    task_id for task_id in registry.ids() if any(token in task_id for token in ("Shadowing", "Mimic", "Vae"))
)


def test_train_and_play_compile_the_same_shadowing_contract() -> None:
    for task_id in SHADOW_IDS:
        spec = registry.spec(task_id)
        fingerprint = task_contract(spec)
        for adapter in (IsaacSimAdapter(), MjlabAdapter()):
            assert adapter.contract_report(spec)["missing"] == {}
            assert task_contract(spec) == fingerprint
            expected_agent = spec.agent.resolve()(**spec.agent.resolved_overrides(adapter.name)).to_dict()
            if adapter.name == "mjlab":
                compiled = adapter.compile(spec, num_envs=1, device="cpu", strict=True)
                assert compiled.resolution.task_id == task_id
                assert compiled.agent_cfg.to_dict()["policy"] == expected_agent["policy"]


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
    assert text.index("validate_checkpoint_contract(checkpoint, spec)") < text.index("runner.load(str(checkpoint))")
    assert text.index("runner.load(str(checkpoint))") < text.index("runner.export_as_onnx")
    assert '"task_contract": task_contract(spec)' in text
