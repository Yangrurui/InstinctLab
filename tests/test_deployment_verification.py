"""A deployed policy is one self-contained, executable, fail-closed ONNX file."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from instinctlab.deployment import (
    POLICY_FILE,
    POLICY_SCHEMA,
    DeploymentVerificationError,
    export_deployment_policy,
    verify_deployment_policy,
)
from instinctlab.deployment_cli import main as verify_main


class _Normalizer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor([1.0, -1.0, 0.5]))
        self.register_buffer("std", torch.tensor([2.0, 4.0, 0.5]))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return (observations - self.mean) / self.std


class _Policy(torch.nn.Module):
    is_recurrent = False

    def __init__(self) -> None:
        super().__init__()
        self.actor = torch.nn.Linear(3, 2)
        with torch.no_grad():
            self.actor.weight.copy_(torch.tensor([[0.25, -0.5, 1.0], [-0.75, 0.5, 0.125]]))
            self.actor.bias.copy_(torch.tensor([0.1, -0.2]))

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.actor(observations))


def _policy(path: Path) -> Path:
    runner = SimpleNamespace(
        alg=SimpleNamespace(actor_critic=_Policy()),
        normalizers={"policy": _Normalizer()},
    )
    return export_deployment_policy(
        runner,
        torch.tensor([[1.5, 2.0, -0.5]], dtype=torch.float32),
        path,
        {
            "checkpoint": "/not/copied/model_100.pt",
            "checkpoint_task_id": "Instinct-Test-G1",
            "task_contract": {"schema_version": "test_contract_v1"},
        },
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_export_is_one_self_contained_policy_and_passes_reference_runtime(tmp_path: Path) -> None:
    policy = _policy(tmp_path / "exported")

    report = verify_deployment_policy(policy, runtime="reference", warmup=1, runs=3)

    assert policy.name == POLICY_FILE
    assert list(policy.parent.iterdir()) == [policy]
    assert report["status"] == "passed"
    assert report["provenance"]["checkpoint_task_id"] == "Instinct-Test-G1"
    assert report["contract"]["input"] == {
        "name": "policy_observation",
        "dtype": "float32",
        "shape": [1, 3],
    }
    assert report["contract"]["output"]["shape"] == [1, 2]
    assert report["runtime"]["name"] == "onnx-reference"
    assert report["parity"]["max_absolute_error"] < 1.0e-6
    assert report["latency_ms"]["measured_runs"] == 3


def test_policy_passes_onnx_runtime(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    policy = _policy(tmp_path / "exported")

    report = verify_deployment_policy(policy, runtime="onnxruntime", warmup=1, runs=3)

    assert report["runtime"]["name"] == "onnxruntime"
    assert "CPUExecutionProvider" in report["runtime"]["providers"]


def test_embedded_content_checksum_rejects_model_tampering(tmp_path: Path) -> None:
    import onnx

    policy = _policy(tmp_path / "exported")
    model = onnx.load(str(policy))
    model.graph.name = "tampered"
    onnx.save(model, str(policy))

    with pytest.raises(DeploymentVerificationError, match="content checksum mismatch"):
        verify_deployment_policy(policy, runtime="reference", warmup=0, runs=1)


def test_external_release_checksum_is_enforced_before_loading(tmp_path: Path) -> None:
    policy = _policy(tmp_path / "exported")
    actual_hash = _sha256(policy)

    report = verify_deployment_policy(
        policy,
        expected_sha256=actual_hash,
        runtime="reference",
        warmup=0,
        runs=1,
    )
    assert report["policy_sha256"] == actual_hash
    with pytest.raises(DeploymentVerificationError, match="policy.onnx checksum mismatch"):
        verify_deployment_policy(
            policy,
            expected_sha256="0" * 64,
            runtime="reference",
            warmup=0,
            runs=1,
        )


def test_cli_writes_no_sidecar_unless_report_is_requested(tmp_path: Path, capsys) -> None:
    policy = _policy(tmp_path / "exported")

    assert verify_main([str(policy), "--runtime", "reference", "--warmup", "0", "--runs", "1"]) == 0
    assert list(policy.parent.iterdir()) == [policy]
    assert '"status": "passed"' in capsys.readouterr().out

    report_path = tmp_path / "evidence" / "verification.json"
    assert verify_main(
        [
            str(policy),
            "--runtime",
            "reference",
            "--warmup",
            "0",
            "--runs",
            "1",
            "--report",
            str(report_path),
        ]
    ) == 0
    assert report_path.is_file()


def test_export_refuses_a_mixed_output_directory(tmp_path: Path) -> None:
    export_dir = tmp_path / "exported"
    export_dir.mkdir()
    (export_dir / "old-policy.onnx").write_bytes(b"old")
    runner = SimpleNamespace(alg=SimpleNamespace(actor_critic=_Policy()), normalizers={})

    with pytest.raises(FileExistsError, match="absent or empty"):
        export_deployment_policy(runner, torch.zeros((1, 3)), export_dir, {})


def test_latency_gate_requires_onnx_runtime(tmp_path: Path) -> None:
    policy = _policy(tmp_path / "exported")

    with pytest.raises(DeploymentVerificationError, match="Latency release gates require ONNX Runtime"):
        verify_deployment_policy(
            policy,
            runtime="reference",
            warmup=0,
            runs=1,
            max_p95_latency_ms=1.0,
        )


def test_recurrent_policy_requires_an_explicit_state_contract(tmp_path: Path) -> None:
    policy = _Policy()
    policy.is_recurrent = True
    runner = SimpleNamespace(alg=SimpleNamespace(actor_critic=policy), normalizers={})

    with pytest.raises(ValueError, match="hidden-state input/output contract"):
        export_deployment_policy(runner, torch.zeros((1, 3), dtype=torch.float32), tmp_path, {})


def test_schema_name_is_versioned() -> None:
    assert POLICY_SCHEMA == "instinctlab_policy_onnx_v1"
