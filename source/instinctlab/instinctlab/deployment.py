"""Export and verify a single-file, self-contained ONNX policy artifact."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

POLICY_SCHEMA = "instinctlab_policy_onnx_v1"
DEPLOYMENT_REPORT_SCHEMA = "instinctlab_deployment_verification_v1"
POLICY_FILE = "policy.onnx"
POLICY_METADATA_KEY = "instinctlab.deployment"
CONTENT_SHA256_KEY = "instinctlab.content_sha256"
SHA256 = re.compile(r"[0-9a-f]{64}")


class DeploymentVerificationError(RuntimeError):
    """The deployment policy does not satisfy its embedded contract."""


class _DeploymentPolicy(torch.nn.Module):
    def __init__(self, actor_critic: torch.nn.Module, normalizer: torch.nn.Module | None):
        super().__init__()
        self.actor_critic = actor_critic
        self.normalizer = normalizer

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if self.normalizer is not None:
            observations = self.normalizer(observations)
        return self.actor_critic.act_inference(observations)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _tensor_array(value: torch.Tensor, *, name: str) -> np.ndarray:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    if value.ndim != 2 or value.shape[0] != 1:
        raise ValueError(f"{name} must have shape [1, features], got {tuple(value.shape)}.")
    array = value.detach().cpu().numpy()
    if array.dtype != np.float32:
        raise ValueError(f"{name} must use float32, got {array.dtype}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity.")
    return np.ascontiguousarray(array)


def _array_payload(array: np.ndarray) -> dict[str, Any]:
    little_endian = np.asarray(array, dtype="<f4", order="C")
    return {
        "dtype": "float32",
        "shape": list(little_endian.shape),
        "base64": base64.b64encode(little_endian.tobytes()).decode("ascii"),
    }


def _payload_array(payload: Any, *, name: str) -> np.ndarray:
    if not isinstance(payload, dict) or payload.get("dtype") != "float32":
        raise DeploymentVerificationError(f"Embedded {name} declaration is invalid.")
    shape = payload.get("shape")
    encoded = payload.get("base64")
    if not isinstance(shape, list) or not shape or not all(isinstance(value, int) and value > 0 for value in shape):
        raise DeploymentVerificationError(f"Embedded {name} shape is invalid: {shape!r}.")
    if not isinstance(encoded, str):
        raise DeploymentVerificationError(f"Embedded {name} payload is missing.")
    try:
        raw = base64.b64decode(encoded, validate=True)
        array = np.frombuffer(raw, dtype="<f4").copy().reshape(shape)
    except (ValueError, TypeError) as error:
        raise DeploymentVerificationError(f"Embedded {name} payload cannot be decoded.") from error
    if not np.isfinite(array).all():
        raise DeploymentVerificationError(f"Embedded {name} contains NaN or infinity.")
    return np.ascontiguousarray(array.astype(np.float32, copy=False))


def _checkpoint_provenance(metadata: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(metadata)
    checkpoint = result.get("checkpoint")
    if isinstance(checkpoint, str):
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.is_file():
            result["checkpoint_sha256"] = _sha256(checkpoint_path)
            result["checkpoint_size_bytes"] = checkpoint_path.stat().st_size
    return result


def _provenance_summary(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    summary = {
        key: metadata[key]
        for key in (
            "checkpoint",
            "checkpoint_sha256",
            "checkpoint_size_bytes",
            "checkpoint_task_id",
            "allow_nonclean_resolution",
        )
        if key in metadata
    }
    task_contract = metadata.get("task_contract")
    if isinstance(task_contract, dict):
        serialized = json.dumps(task_contract, sort_keys=True, separators=(",", ":"), default=str).encode()
        summary["task_contract_sha256"] = hashlib.sha256(serialized).hexdigest()
        summary["task_contract_version"] = task_contract.get("version", task_contract.get("schema_version"))
        summary["task_id"] = task_contract.get("task_id")
    return summary


def _content_sha256(model: Any) -> str:
    clone = type(model)()
    clone.CopyFrom(model)
    for index in range(len(clone.metadata_props) - 1, -1, -1):
        if clone.metadata_props[index].key == CONTENT_SHA256_KEY:
            del clone.metadata_props[index]
    serialized = clone.SerializeToString(deterministic=True)
    return hashlib.sha256(serialized).hexdigest()


def _model_properties(model: Any) -> dict[str, str]:
    properties: dict[str, str] = {}
    for item in model.metadata_props:
        if item.key in properties:
            raise DeploymentVerificationError(f"ONNX model repeats metadata key {item.key!r}.")
        properties[item.key] = item.value
    return properties


def _onnx_shape(value_info: Any) -> list[int | str]:
    result: list[int | str] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            result.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            result.append(str(dimension.dim_param))
        else:
            result.append("?")
    return result


def _contract(metadata: Mapping[str, Any], key: str) -> dict[str, Any]:
    policy = metadata.get("policy")
    if not isinstance(policy, dict) or not isinstance(policy.get(key), dict):
        raise DeploymentVerificationError(f"Deployment metadata has no policy {key} contract.")
    contract = policy[key]
    if not isinstance(contract.get("name"), str) or contract.get("dtype") != "float32":
        raise DeploymentVerificationError(f"Policy {key} name or dtype is invalid.")
    shape = contract.get("shape")
    if not isinstance(shape, list) or not shape or not all(isinstance(value, int) and value > 0 for value in shape):
        raise DeploymentVerificationError(f"Policy {key} shape is invalid: {shape!r}.")
    return contract


def _validate_onnx_contract(model: Any, input_contract: Mapping[str, Any], output_contract: Mapping[str, Any]) -> None:
    import onnx

    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise DeploymentVerificationError(
            f"Policy must have one input and one output, got {len(model.graph.input)} and {len(model.graph.output)}."
        )
    for label, value_info, declaration in (
        ("input", model.graph.input[0], input_contract),
        ("output", model.graph.output[0], output_contract),
    ):
        if value_info.name != declaration["name"]:
            raise DeploymentVerificationError(
                f"ONNX {label} name {value_info.name!r} does not match {declaration['name']!r}."
            )
        if value_info.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise DeploymentVerificationError(f"ONNX {label} must use float32.")
        actual_shape = _onnx_shape(value_info)
        if actual_shape != declaration["shape"]:
            raise DeploymentVerificationError(
                f"ONNX {label} shape {actual_shape} does not match {declaration['shape']}."
            )


def _load_policy(path: Path) -> tuple[Any, dict[str, Any]]:
    try:
        import onnx
    except ImportError as error:
        raise DeploymentVerificationError(
            "ONNX verification requires: python -m pip install 'instinctlab[deployment]'."
        ) from error
    try:
        model = onnx.load(str(path), load_external_data=False)
        onnx.checker.check_model(model)
    except Exception as error:
        raise DeploymentVerificationError(f"ONNX structural validation failed: {error}") from error
    if any(initializer.data_location == onnx.TensorProto.EXTERNAL for initializer in model.graph.initializer):
        raise DeploymentVerificationError("policy.onnx must not depend on external tensor data.")
    properties = _model_properties(model)
    declared_content_hash = properties.get(CONTENT_SHA256_KEY)
    if not isinstance(declared_content_hash, str) or not SHA256.fullmatch(declared_content_hash):
        raise DeploymentVerificationError("ONNX policy has no valid embedded content checksum.")
    actual_content_hash = _content_sha256(model)
    if actual_content_hash != declared_content_hash:
        raise DeploymentVerificationError(
            f"ONNX content checksum mismatch: got {actual_content_hash}, expected {declared_content_hash}."
        )
    try:
        metadata = json.loads(properties[POLICY_METADATA_KEY])
    except (KeyError, json.JSONDecodeError) as error:
        raise DeploymentVerificationError("ONNX policy has no valid InstinctLab metadata.") from error
    if not isinstance(metadata, dict) or metadata.get("schema_version") != POLICY_SCHEMA:
        version = metadata.get("schema_version") if isinstance(metadata, dict) else None
        raise DeploymentVerificationError(f"Unsupported ONNX policy schema: {version!r}.")
    policy = metadata.get("policy")
    declared_opset = policy.get("opset_version") if isinstance(policy, dict) else None
    actual_opset = next((item.version for item in model.opset_import if item.domain in {"", "ai.onnx"}), None)
    if not isinstance(declared_opset, int) or declared_opset != actual_opset:
        raise DeploymentVerificationError(
            f"ONNX opset {actual_opset!r} does not match embedded contract {declared_opset!r}."
        )
    return model, metadata


def export_deployment_policy(
    runner: Any,
    observations: torch.Tensor,
    export_dir: str | Path,
    metadata: Mapping[str, Any],
    *,
    atol: float = 1.0e-4,
    rtol: float = 1.0e-4,
    opset_version: int = 17,
) -> Path:
    """Export one policy.onnx containing normalization, provenance, and a self-test oracle."""
    if atol < 0.0 or rtol < 0.0:
        raise ValueError("Deployment tolerances must be non-negative.")
    export_path = Path(export_dir).expanduser().resolve()
    if export_path.exists() and (not export_path.is_dir() or any(export_path.iterdir())):
        raise FileExistsError(f"Deployment export directory must be absent or empty: {export_path}.")
    export_path.mkdir(parents=True, exist_ok=True)

    actor_critic = runner.alg.actor_critic
    if bool(getattr(actor_critic, "is_recurrent", False)):
        raise ValueError(
            "Recurrent policy deployment requires an explicit hidden-state input/output contract; "
            "the v1 policy artifact only accepts feed-forward policies."
        )
    normalizer = runner.normalizers.get("policy")
    if normalizer is not None and not isinstance(normalizer, torch.nn.Module):
        raise TypeError("The policy normalizer must be a torch.nn.Module.")

    observation_array = _tensor_array(observations, name="policy observations")
    policy = _DeploymentPolicy(actor_critic, normalizer)
    policy.eval()
    with torch.no_grad():
        expected_actions = policy(observations)
    action_array = _tensor_array(expected_actions, name="policy actions")

    try:
        import onnx
    except ImportError as error:
        raise RuntimeError(
            "ONNX export requires the 'deployment' extra: "
            "python -m pip install 'instinctlab[deployment]'."
        ) from error

    deployment_metadata = {
        "schema_version": POLICY_SCHEMA,
        "provenance": _checkpoint_provenance(metadata),
        "policy": {
            "opset_version": opset_version,
            "normalizer_embedded": normalizer is not None,
            "input": {"name": "policy_observation", "dtype": "float32", "shape": list(observation_array.shape)},
            "output": {"name": "action", "dtype": "float32", "shape": list(action_array.shape)},
        },
        "self_test": {
            "input": _array_payload(observation_array),
            "expected_output": _array_payload(action_array),
            "atol": atol,
            "rtol": rtol,
        },
    }
    policy_path = export_path / POLICY_FILE
    with tempfile.TemporaryDirectory(prefix=".instinctlab-onnx-", dir=export_path) as temporary_dir:
        raw_path = Path(temporary_dir) / "raw.onnx"
        final_path = Path(temporary_dir) / POLICY_FILE
        with torch.no_grad():
            torch.onnx.export(
                policy,
                observations,
                str(raw_path),
                input_names=["policy_observation"],
                output_names=["action"],
                opset_version=opset_version,
                dynamo=False,
            )
        model = onnx.load(str(raw_path), load_external_data=True)
        input_contract = _contract(deployment_metadata, "input")
        output_contract = _contract(deployment_metadata, "output")
        _validate_onnx_contract(model, input_contract, output_contract)
        onnx.helper.set_model_props(
            model,
            {POLICY_METADATA_KEY: json.dumps(deployment_metadata, sort_keys=True, separators=(",", ":"), default=str)},
        )
        content_hash = _content_sha256(model)
        checksum_property = model.metadata_props.add()
        checksum_property.key = CONTENT_SHA256_KEY
        checksum_property.value = content_hash
        onnx.checker.check_model(model)
        onnx.save_model(model, str(final_path), save_as_external_data=False)
        os.replace(final_path, policy_path)

    entries = list(export_path.iterdir())
    if entries != [policy_path]:
        raise RuntimeError(f"Single-file policy export produced unexpected artifacts: {entries}.")
    _load_policy(policy_path)
    return policy_path


def _runtime(model: Any, policy_path: Path, runtime: str) -> tuple[str, str, list[str], Any]:
    if runtime not in {"auto", "onnxruntime", "reference"}:
        raise ValueError(f"Unknown ONNX runtime {runtime!r}.")
    if runtime in {"auto", "onnxruntime"}:
        try:
            import onnxruntime
        except ImportError as error:
            if runtime == "onnxruntime":
                raise DeploymentVerificationError(
                    "ONNX Runtime verification requires: "
                    "python -m pip install 'instinctlab[deployment]'."
                ) from error
        else:
            available = onnxruntime.get_available_providers()
            providers = ["CPUExecutionProvider"] if "CPUExecutionProvider" in available else available
            if not providers:
                raise DeploymentVerificationError("ONNX Runtime reports no available execution providers.")
            try:
                session = onnxruntime.InferenceSession(str(policy_path), providers=providers)
            except Exception as error:
                raise DeploymentVerificationError(f"ONNX Runtime rejected the policy: {error}") from error
            return "onnxruntime", onnxruntime.__version__, session.get_providers(), session.run

    import onnx
    from onnx.reference import ReferenceEvaluator

    try:
        evaluator = ReferenceEvaluator(model)
    except Exception as error:
        raise DeploymentVerificationError(f"ONNX reference runtime rejected the policy: {error}") from error
    return "onnx-reference", onnx.__version__, [], evaluator.run


def verify_deployment_policy(
    policy_file: str | Path,
    *,
    runtime: str = "onnxruntime",
    expected_sha256: str | None = None,
    atol: float | None = None,
    rtol: float | None = None,
    warmup: int = 10,
    runs: int = 100,
    max_p95_latency_ms: float | None = None,
) -> dict[str, Any]:
    """Fail closed unless a policy is intact and reproduces its embedded PyTorch oracle."""
    if warmup < 0 or runs < 1:
        raise ValueError("warmup must be non-negative and runs must be positive.")
    if max_p95_latency_ms is not None and max_p95_latency_ms <= 0.0:
        raise ValueError("max_p95_latency_ms must be positive.")
    policy_path = Path(policy_file).expanduser().resolve()
    if not policy_path.is_file() or policy_path.name != POLICY_FILE:
        raise DeploymentVerificationError(f"Expected a policy.onnx file, got: {policy_path}")
    file_hash = _sha256(policy_path)
    if expected_sha256 is not None:
        if not SHA256.fullmatch(expected_sha256):
            raise ValueError("expected_sha256 must contain 64 lowercase hexadecimal characters.")
        if file_hash != expected_sha256:
            raise DeploymentVerificationError(
                f"policy.onnx checksum mismatch: got {file_hash}, expected {expected_sha256}."
            )

    model, metadata = _load_policy(policy_path)
    input_contract = _contract(metadata, "input")
    output_contract = _contract(metadata, "output")
    _validate_onnx_contract(model, input_contract, output_contract)
    self_test = metadata.get("self_test")
    if not isinstance(self_test, dict):
        raise DeploymentVerificationError("ONNX policy has no embedded self-test.")
    observations = _payload_array(self_test.get("input"), name="self-test input")
    expected_actions = _payload_array(self_test.get("expected_output"), name="self-test output")
    if list(observations.shape) != input_contract["shape"]:
        raise DeploymentVerificationError("Embedded self-test input does not match the ONNX contract.")
    if list(expected_actions.shape) != output_contract["shape"]:
        raise DeploymentVerificationError("Embedded self-test output does not match the ONNX contract.")

    selected_runtime, runtime_version, providers, run = _runtime(model, policy_path, runtime)
    if max_p95_latency_ms is not None and selected_runtime != "onnxruntime":
        raise DeploymentVerificationError("Latency release gates require ONNX Runtime, not the reference evaluator.")
    feeds = {input_contract["name"]: observations}
    try:
        actual_actions = np.asarray(run([output_contract["name"]], feeds)[0])
    except Exception as error:
        raise DeploymentVerificationError(f"ONNX inference failed: {error}") from error
    if actual_actions.dtype != np.float32 or actual_actions.shape != expected_actions.shape:
        raise DeploymentVerificationError(
            f"ONNX action contract mismatch: dtype={actual_actions.dtype}, shape={actual_actions.shape}."
        )
    if not np.isfinite(actual_actions).all():
        raise DeploymentVerificationError("ONNX action contains NaN or infinity.")

    declared_atol = self_test.get("atol")
    declared_rtol = self_test.get("rtol")
    if not isinstance(declared_atol, (int, float)) or not isinstance(declared_rtol, (int, float)):
        raise DeploymentVerificationError("Embedded self-test tolerances are invalid.")
    selected_atol = float(declared_atol if atol is None else atol)
    selected_rtol = float(declared_rtol if rtol is None else rtol)
    if selected_atol < 0.0 or selected_rtol < 0.0:
        raise DeploymentVerificationError("Deployment tolerances must be non-negative.")
    absolute_error = np.abs(actual_actions - expected_actions)
    denominator = np.maximum(np.abs(expected_actions), np.finfo(np.float32).tiny)
    relative_error = absolute_error / denominator
    max_absolute_error = float(absolute_error.max(initial=0.0))
    max_relative_error = float(relative_error.max(initial=0.0))
    if not np.allclose(actual_actions, expected_actions, atol=selected_atol, rtol=selected_rtol):
        raise DeploymentVerificationError(
            "ONNX/PyTorch parity failed: "
            f"max_abs={max_absolute_error:.8g}, max_rel={max_relative_error:.8g}, "
            f"atol={selected_atol:.8g}, rtol={selected_rtol:.8g}."
        )

    timings_ms = []
    try:
        for _ in range(warmup):
            run([output_contract["name"]], feeds)
        for _ in range(runs):
            started = time.perf_counter_ns()
            run([output_contract["name"]], feeds)
            timings_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    except Exception as error:
        raise DeploymentVerificationError(f"ONNX latency probe failed: {error}") from error
    p95_latency_ms = float(np.percentile(timings_ms, 95))
    if max_p95_latency_ms is not None and p95_latency_ms > max_p95_latency_ms:
        raise DeploymentVerificationError(
            f"ONNX p95 latency {p95_latency_ms:.4f} ms exceeds {max_p95_latency_ms:.4f} ms."
        )

    return {
        "schema_version": DEPLOYMENT_REPORT_SCHEMA,
        "status": "passed",
        "policy": str(policy_path),
        "policy_sha256": file_hash,
        "content_sha256": _model_properties(model)[CONTENT_SHA256_KEY],
        "provenance": _provenance_summary(metadata.get("provenance")),
        "runtime": {"name": selected_runtime, "version": runtime_version, "providers": providers},
        "contract": {"input": input_contract, "output": output_contract},
        "parity": {
            "atol": selected_atol,
            "rtol": selected_rtol,
            "max_absolute_error": max_absolute_error,
            "max_relative_error": max_relative_error,
        },
        "latency_ms": {
            "warmup_runs": warmup,
            "measured_runs": runs,
            "p50": float(np.percentile(timings_ms, 50)),
            "p95": p95_latency_ms,
            "max": float(max(timings_ms)),
            "threshold_p95": max_p95_latency_ms,
        },
    }


def write_verification_report(path: str | Path, report: Mapping[str, Any]) -> None:
    """Write optional CI/release evidence outside the deployment artifact directory."""
    _atomic_json(Path(path).expanduser().resolve(), report)
