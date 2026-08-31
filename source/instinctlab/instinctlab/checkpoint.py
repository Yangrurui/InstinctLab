"""Checkpoint compatibility metadata for the unified training entry points."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import re
import subprocess
import sys
import warnings
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from instinctlab_engine.spec import TaskSpec, portability_report
from instinctlab_engine.spec.robot import BackendAsset

_CONTRACT_VERSION = "task_manifest_v4"
_CHECKPOINT_FORMAT_VERSION = "instinct_rl_on_policy_runner_v1"
_LEGACY_CONTRACT_VERSIONS = frozenset(
    {"task_spec_v1", "task_contract_v2", "task_manifest_v3"}
)
_RUNTIME_PROVENANCE_VERSION = "runtime_provenance_v1"
_DEFAULT_CHECKPOINT_PATTERN = r"model_.*\.pt"
_NON_TRAINING_AGENT_FIELDS = frozenset(
    {
        "device",
        "experiment_name",
        "load_checkpoint",
        "load_run",
        "log_interval",
        "max_iterations",
        "resume",
        "run_name",
        "save_interval",
    }
)
_PUBLIC_TYPE_MODULES = {
    "instinctlab_engine.spec.motion_reference": "instinctlab_engine.spec.sensor",
    "instinctlab_engine.spec.volume": "instinctlab_engine.spec.sensor",
}


def _type_name(value: Any) -> str:
    value_type = type(value)
    # Contract v1 records the public declaration path, not its physical source file. Keeping this
    # mapping stable lets the implementation be split into cohesive modules without invalidating
    # checkpoints whose tensor contract did not change.
    module = _PUBLIC_TYPE_MODULES.get(value_type.__module__, value_type.__module__)
    return f"{module}.{value_type.__qualname__}"


def _canonical(
    value: Any,
    *,
    include_asset_paths: bool = False,
    include_omitted_fields: bool = False,
) -> Any:
    """Convert a declaration to deterministic JSON without importing an engine."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Checkpoint contracts cannot contain non-finite value {value!r}.")
        return value
    if isinstance(value, Enum):
        return {
            "type": _type_name(value),
            "value": _canonical(
                value.value,
                include_asset_paths=include_asset_paths,
                include_omitted_fields=include_omitted_fields,
            ),
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BackendAsset):
        # ``asset_id`` identifies the robot. Paths are included only in full
        # provenance, where clone/machine differences are useful operator data.
        return {
            "type": _type_name(value),
            "fields": [
                [
                    field.name,
                    _canonical(
                        getattr(value, field.name),
                        include_asset_paths=include_asset_paths,
                        include_omitted_fields=include_omitted_fields,
                    ),
                ]
                for field in fields(value)
                if include_asset_paths or field.name != "path"
            ],
        }
    if is_dataclass(value) and not isinstance(value, type):

        def include(field) -> bool:
            if field.metadata.get("contract_omit", False) and not include_omitted_fields:
                return False
            if not field.metadata.get("contract_omit_if_default", False):
                return True
            default = field.default
            if default is MISSING:
                default = field.default_factory()
            return getattr(value, field.name) != default

        return {
            "type": _type_name(value),
            "fields": [
                [
                    field.name,
                    _canonical(
                        getattr(value, field.name),
                        include_asset_paths=include_asset_paths,
                        include_omitted_fields=include_omitted_fields,
                    ),
                ]
                for field in fields(value)
                if include(field)
            ],
        }
    if isinstance(value, Mapping):
        # Preserve declaration order for observations, actions, and rewards.
        return {
            "mapping": [
                [
                    _canonical(
                        key,
                        include_asset_paths=include_asset_paths,
                        include_omitted_fields=include_omitted_fields,
                    ),
                    _canonical(
                        item,
                        include_asset_paths=include_asset_paths,
                        include_omitted_fields=include_omitted_fields,
                    ),
                ]
                for key, item in value.items()
            ]
        }
    if isinstance(value, (tuple, list)):
        return [
            _canonical(
                item,
                include_asset_paths=include_asset_paths,
                include_omitted_fields=include_omitted_fields,
            )
            for item in value
        ]
    if isinstance(value, (set, frozenset)):
        items = [
            _canonical(
                item,
                include_asset_paths=include_asset_paths,
                include_omitted_fields=include_omitted_fields,
            )
            for item in value
        ]
        return {"set": sorted(items, key=lambda item: json.dumps(item, sort_keys=True))}
    if callable(value):
        module = getattr(value, "__module__", None)
        qualname = getattr(value, "__qualname__", None)
        if module and qualname and "<locals>" not in qualname:
            return {"callable": f"{module}.{qualname}"}
    raise TypeError(f"Task contract cannot serialize {_type_name(value)}: {value!r}.")


def _as_agent_config(config: Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return deepcopy(dict(config))
    to_dict = getattr(config, "to_dict", None)
    if not callable(to_dict):
        raise TypeError(
            f"Agent config {type(config).__module__}.{type(config).__qualname__} "
            "must be a mapping or expose to_dict() for checkpoint contracts."
        )
    return to_dict()


def _agent_config(spec: TaskSpec, engine: str | None = None) -> Mapping[str, Any]:
    overrides = (
        dict(spec.agent.overrides)
        if engine is None
        else spec.agent.resolved_overrides(engine)
    )
    config = spec.agent.resolve()(**overrides)
    return _as_agent_config(config)


def _strict_agent_config(config: Mapping[str, Any] | object) -> dict[str, Any]:
    """Return learning semantics without checkpoint, logging, or run-length controls."""
    return {
        key: value
        for key, value in _as_agent_config(config).items()
        if key not in _NON_TRAINING_AGENT_FIELDS
    }


def _strict_resume_contract(
    spec: TaskSpec,
    agent_config: Mapping[str, Any] | object,
) -> dict[str, Any]:
    """Readable fields which must match before restoring training state."""
    return {
        "task_id": spec.task_id,
        "robot": _canonical(
            spec.robot,
            include_omitted_fields=True,
        ),
        "policy_io": {
            "observations": _canonical(
                spec.mdp.observations,
                include_omitted_fields=True,
            ),
            "actions": _canonical(
                spec.mdp.actions,
                include_omitted_fields=True,
            ),
        },
        "effective_agent_config": _strict_agent_config(agent_config),
        "training_semantics": {
            "scene": _canonical(spec.scene, include_omitted_fields=True),
            "simulation": _canonical(spec.sim, include_omitted_fields=True),
            "rewards": _canonical(spec.mdp.rewards, include_omitted_fields=True),
            "terminations": _canonical(
                spec.mdp.terminations,
                include_omitted_fields=True,
            ),
            "events": _canonical(spec.mdp.events, include_omitted_fields=True),
            "commands": _canonical(
                spec.mdp.commands,
                include_omitted_fields=True,
            ),
            "curriculum": _canonical(
                spec.mdp.curriculum,
                include_omitted_fields=True,
            ),
            "engines": list(spec.engines),
            "engine_extras": _canonical(
                spec.engine_extras,
                include_omitted_fields=True,
            ),
            "lifecycle": _canonical(
                spec.lifecycle,
                include_omitted_fields=True,
            ),
        },
    }


def task_contract(
    spec: TaskSpec,
    *,
    engine: str | None = None,
    agent_config: Mapping[str, Any] | object | None = None,
) -> dict[str, Any]:
    """Readable task/run metadata stored beside the runner checkpoint.

    ``engine`` resolves declaration-time engine overrides. Production callers
    should pass the final ``agent_config`` used to construct the runner after
    CLI and distributed overrides. This metadata is deliberately not a tensor
    compatibility hash: the runner's strict state-dict loader owns key and
    shape validation.
    """
    resolved_agent_config = (
        _agent_config(spec, engine)
        if agent_config is None
        else _as_agent_config(agent_config)
    )
    return {
        "version": _CONTRACT_VERSION,
        "checkpoint_format_version": _CHECKPOINT_FORMAT_VERSION,
        "task_id": spec.task_id,
        "robot_schema_version": spec.robot.schema_version,
        "asset_id": spec.robot.asset_id,
        "joint_names": list(spec.robot.joint_names),
        "body_names": list(spec.robot.body_names),
        "portability": portability_report(spec),
        "effective_agent_config": resolved_agent_config,
        "strict_resume": _strict_resume_contract(spec, resolved_agent_config),
        "task_declaration": _canonical(
            spec,
            include_asset_paths=True,
            include_omitted_fields=True,
        ),
    }


def checkpoint_load_semantics(
    mode: Literal["resume", "transfer"] | None,
) -> dict[str, Any]:
    """Describe exactly what a checkpoint load does to runner and environment state."""
    if mode not in {None, "resume", "transfer"}:
        raise ValueError(f"unknown checkpoint load mode {mode!r}")
    runner_state = {
        None: "fresh",
        "resume": "model, optimizer, normalizers, and learning iteration restored",
        "transfer": (
            "permissive runner state load; learning iteration reset to zero"
        ),
    }[mode]
    return {
        "mode": mode or "none",
        "runner_state": runner_state,
        "lifecycle_snapshot": "not restored",
        "environment_state": "fresh environment construction and reset",
        "common_rng_state": "not restored; initialized from the current run seed",
    }


def add_task_contract(
    manifest: Mapping[str, Any],
    spec: TaskSpec,
    *,
    engine: str | None = None,
    agent_config: Mapping[str, Any] | object | None = None,
) -> dict[str, Any]:
    """Return a compilation manifest carrying checkpoint compatibility metadata."""
    payload = dict(manifest)
    payload["portability"] = portability_report(spec, payload)
    payload["task_contract"] = task_contract(
        spec, engine=engine, agent_config=agent_config
    )
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_provenance(*, reference: str, role: str, declared: str) -> dict[str, Any]:
    path = Path(declared).expanduser().resolve()
    result: dict[str, Any] = {
        "reference": reference,
        "role": role,
        "declared": declared,
        "resolved": str(path),
        "exists": path.exists(),
    }
    if path.is_file():
        result["size_bytes"] = path.stat().st_size
        result["sha256"] = _sha256_file(path)
    elif path.is_dir():
        result["kind"] = "directory"
    return result


def _dataset_provenance(spec: TaskSpec, engine: str) -> list[dict[str, Any]]:
    datasets: list[dict[str, Any]] = []
    for reference in spec.scene.motion_references:
        resolved = reference.for_engine(engine)
        datasets.append(
            _path_provenance(
                reference=reference.name,
                role="clip",
                declared=resolved.clip,
            )
        )
        if resolved.metadata_yaml:
            datasets.append(
                _path_provenance(
                    reference=reference.name,
                    role="metadata",
                    declared=resolved.metadata_yaml,
                )
            )
        clip_root = Path(resolved.clip).expanduser()
        for selected in resolved.selected_files:
            selected_path = Path(selected).expanduser()
            if not selected_path.is_absolute():
                selected_path = clip_root / selected_path
            datasets.append(
                _path_provenance(
                    reference=reference.name,
                    role="selected_motion",
                    declared=str(selected_path),
                )
            )
    return datasets


def _installed_versions(engine: str) -> dict[str, str]:
    distributions = (
        "instinctlab",
        "instinctlab-engine-core",
        f"instinctlab-engine-{engine}",
        "instinct-rl",
        "torch",
        "isaaclab",
        "mjlab",
        "mujoco",
        "warp-lang",
    )
    versions: dict[str, str] = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _git_provenance(repository: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        root = Path(run("rev-parse", "--show-toplevel"))
        commit = run("rev-parse", "HEAD")
        dirty = bool(run("status", "--porcelain", "--untracked-files=normal"))
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"available": False}
    return {
        "available": True,
        "root": str(root),
        "commit": commit,
        "dirty": dirty,
    }


def _accelerator_provenance(device: str) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {
        "requested_device": device,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        index = torch.device(device).index if device.startswith("cuda") else None
        index = torch.cuda.current_device() if index is None else index
        properties = torch.cuda.get_device_properties(index)
        result["device"] = {
            "index": index,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        }
        try:
            query = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            result["driver_devices"] = [
                line.strip() for line in query.stdout.splitlines() if line.strip()
            ]
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            result["driver_devices"] = []
    return result


def runtime_provenance(
    spec: TaskSpec,
    *,
    engine: str,
    device: str,
    num_envs: int,
    argv: list[str] | tuple[str, ...] | None = None,
    repository: str | Path = ".",
) -> dict[str, Any]:
    """Structured source, software, hardware, and dataset identity for one run."""
    return {
        "version": _RUNTIME_PROVENANCE_VERSION,
        "command": {
            "argv": list(sys.argv if argv is None else argv),
            "engine": engine,
            "device": device,
            "num_envs": num_envs,
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "packages": _installed_versions(engine),
        "accelerator": _accelerator_provenance(device),
        "git": _git_provenance(Path(repository).resolve()),
        "datasets": _dataset_provenance(spec, engine),
    }


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    """Sort ``model_<iteration>.pt`` numerically, then fall back to the filename."""
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return (int(match.group(1)) if match else -1, path.name)


def latest_checkpoint(run_dir: str | Path, pattern: str = _DEFAULT_CHECKPOINT_PATTERN) -> Path:
    """Return the newest numeric checkpoint in one run directory.

    ``pattern`` uses the same full-match regular-expression semantics as the runner config.  This
    keeps checkpoint discovery identical for training resume and playback.
    """
    run_path = Path(run_dir).expanduser().resolve()
    matcher = re.compile(pattern)
    checkpoints = [path for path in run_path.iterdir() if path.is_file() and matcher.fullmatch(path.name)]
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint matching {pattern!r} under {run_path}")
    return max(checkpoints, key=checkpoint_sort_key)


def latest_run_checkpoint(
    log_root: str | Path,
    *,
    run_pattern: str = ".*",
    checkpoint_pattern: str = _DEFAULT_CHECKPOINT_PATTERN,
    skip_empty_runs: bool = False,
) -> Path:
    """Select a matching run, then its latest numeric checkpoint.

    Training resume treats the latest run as authoritative and reports a missing checkpoint there.
    Playback can set ``skip_empty_runs`` to walk backwards past freshly-created or interrupted run
    directories that contain no model yet.
    """
    root = Path(log_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint log root does not exist: {root}")
    matcher = re.compile(run_pattern)
    runs = sorted(path for path in root.iterdir() if path.is_dir() and matcher.fullmatch(path.name))
    if not runs:
        raise FileNotFoundError(f"no run matching {run_pattern!r} under {root}")
    if not skip_empty_runs:
        return latest_checkpoint(runs[-1], checkpoint_pattern)
    for run in reversed(runs):
        try:
            return latest_checkpoint(run, checkpoint_pattern)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"no checkpoint matching {checkpoint_pattern!r} in runs under {root}")


def validate_checkpoint_contract(
    checkpoint: str | Path,
    spec: TaskSpec,
    *,
    checkpoint_task_id: str | None = None,
    mode: Literal["resume", "transfer"] = "transfer",
    agent_config: Mapping[str, Any] | object | None = None,
) -> None:
    """Validate a strict resume or an explicitly permissive transfer load.

    Resume rejects task, robot, policy I/O, effective learning-agent, and
    training-semantic drift before environment construction. Transfer keeps
    those declarations informational and delegates tensor compatibility to the
    runner. Legacy runs therefore remain available for explicit transfer but
    cannot be called a verified resume.
    """
    if mode not in {"resume", "transfer"}:
        raise ValueError(f"unknown checkpoint load mode {mode!r}")
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    try:
        with checkpoint_path.open("rb") as handle:
            handle.read(1)
    except OSError as exc:
        raise OSError(f"checkpoint is not readable: {checkpoint_path}") from exc

    manifest_path = checkpoint_path.parent / "manifest.json"
    if not manifest_path.is_file():
        if mode == "resume":
            raise ValueError(
                f"Strict resume requires an adjacent manifest.json for {checkpoint_path}. "
                "Use explicit transfer mode for a legacy checkpoint."
            )
        warnings.warn(
            f"Checkpoint {checkpoint_path} has no adjacent manifest.json; its format version "
            "cannot be verified before the runner loads it.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    stored = manifest.get("task_contract")
    if not isinstance(stored, dict):
        if mode == "resume":
            raise ValueError(
                f"Strict resume requires task_contract metadata in {manifest_path}. "
                "Use explicit transfer mode for a legacy checkpoint."
            )
        warnings.warn(
            f"Checkpoint manifest {manifest_path} predates explicit checkpoint format metadata; "
            "the runner will validate its tensors directly.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    stored_version = stored.get("version")
    if stored_version in _LEGACY_CONTRACT_VERSIONS:
        if mode == "resume":
            raise ValueError(
                f"Strict resume requires {_CONTRACT_VERSION!r} metadata; checkpoint "
                f"{checkpoint_path} has legacy {stored_version!r}. Use explicit transfer "
                "mode if this initialization is intentional."
            )
        warnings.warn(
            f"Checkpoint manifest {manifest_path} has legacy metadata {stored_version!r} "
            "without a strict resume contract; the runner will validate transfer tensors.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    if stored_version != _CONTRACT_VERSION:
        raise ValueError(
            f"Checkpoint manifest schema version mismatch for {checkpoint_path}: "
            f"checkpoint={stored_version!r}; supported={_CONTRACT_VERSION!r}."
        )
    stored_format = stored.get("checkpoint_format_version")
    if stored_format != _CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint format version mismatch for {checkpoint_path}: "
            f"checkpoint={stored_format!r}; supported={_CHECKPOINT_FORMAT_VERSION!r}."
        )

    expected_task_id = checkpoint_task_id or spec.task_id
    if stored.get("task_id") != expected_task_id:
        if mode == "resume":
            raise ValueError(
                f"Strict resume rejected task identity drift for {checkpoint_path}: "
                f"checkpoint={stored.get('task_id')!r}; current={expected_task_id!r}."
            )
        warnings.warn(
            f"Checkpoint metadata task {stored.get('task_id')!r} differs from expected "
            f"{expected_task_id!r}; task metadata is informational and the runner will "
            "validate tensor keys and shapes.",
            RuntimeWarning,
            stacklevel=2,
        )

    if mode == "transfer":
        return

    current_agent = _agent_config(spec) if agent_config is None else agent_config
    expected = _strict_resume_contract(spec, current_agent)
    stored_resume = stored.get("strict_resume")
    if not isinstance(stored_resume, dict):
        raise ValueError(  # noqa: TRY004 - missing contract data, not caller type misuse
            f"Strict resume metadata is missing from {manifest_path}. Use explicit "
            "transfer mode if this initialization is intentional."
        )
    labels = {
        "task_id": "task identity",
        "robot": "robot schema",
        "policy_io": "policy I/O",
        "effective_agent_config": "effective agent",
        "training_semantics": "training semantics",
    }
    for field, label in labels.items():
        if stored_resume.get(field) != expected[field]:
            raise ValueError(
                f"Strict resume rejected {label} drift for {checkpoint_path}; "
                f"checkpoint and current {field!r} declarations differ. Use explicit "
                "transfer mode if this change is intentional."
            )


__all__ = [
    "add_task_contract",
    "checkpoint_load_semantics",
    "checkpoint_sort_key",
    "latest_checkpoint",
    "latest_run_checkpoint",
    "runtime_provenance",
    "task_contract",
    "validate_checkpoint_contract",
]
