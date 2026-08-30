"""Checkpoint compatibility metadata for the unified training entry points."""

from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from collections.abc import Mapping
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from instinctlab.spec import TaskSpec
from instinctlab.spec.robot import BackendAsset

_CONTRACT_VERSION = "task_spec_v1"
_DEFAULT_CHECKPOINT_PATTERN = r"model_.*\.pt"
_PUBLIC_TYPE_MODULES = {
    "instinctlab.spec.motion_reference": "instinctlab.spec.sensor",
    "instinctlab.spec.volume": "instinctlab.spec.sensor",
}


def _type_name(value: Any) -> str:
    value_type = type(value)
    # Contract v1 records the public declaration path, not its physical source file. Keeping this
    # mapping stable lets the implementation be split into cohesive modules without invalidating
    # checkpoints whose tensor contract did not change.
    module = _PUBLIC_TYPE_MODULES.get(value_type.__module__, value_type.__module__)
    return f"{module}.{value_type.__qualname__}"


def _canonical(value: Any) -> Any:
    """Convert a declaration to deterministic JSON without importing an engine."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Checkpoint contracts cannot contain non-finite value {value!r}.")
        return value
    if isinstance(value, Enum):
        return {"type": _type_name(value), "value": _canonical(value.value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BackendAsset):
        # ``asset_id`` identifies the robot. Absolute source paths vary between clones and
        # machines and must not make an otherwise identical checkpoint unloadable.
        return {
            "type": _type_name(value),
            "fields": [
                [field.name, _canonical(getattr(value, field.name))] for field in fields(value) if field.name != "path"
            ],
        }
    if is_dataclass(value) and not isinstance(value, type):

        def include(field) -> bool:
            if field.metadata.get("contract_omit", False):
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
                [field.name, _canonical(getattr(value, field.name))] for field in fields(value) if include(field)
            ],
        }
    if isinstance(value, Mapping):
        # Mapping order is part of the tensor contract for observations, actions and rewards.
        return {"mapping": [[_canonical(key), _canonical(item)] for key, item in value.items()]}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonical(item) for item in value]
        return {"set": sorted(items, key=lambda item: json.dumps(item, sort_keys=True))}
    if callable(value):
        module = getattr(value, "__module__", None)
        qualname = getattr(value, "__qualname__", None)
        if module and qualname and "<locals>" not in qualname:
            return {"callable": f"{module}.{qualname}"}
    raise TypeError(f"Task contract cannot serialize {_type_name(value)}: {value!r}.")


def task_contract(spec: TaskSpec) -> dict[str, Any]:
    """Stable, backend-independent fingerprint of a task and its tensor ordering."""
    canonical = {"version": _CONTRACT_VERSION, "spec": _canonical(spec)}
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return {
        "version": _CONTRACT_VERSION,
        "task_id": spec.task_id,
        "robot_schema_version": spec.robot.schema_version,
        "asset_id": spec.robot.asset_id,
        "joint_names": list(spec.robot.joint_names),
        "body_names": list(spec.robot.body_names),
        "hash": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    }


def add_task_contract(manifest: Mapping[str, Any], spec: TaskSpec) -> dict[str, Any]:
    """Return a compilation manifest carrying checkpoint compatibility metadata."""
    payload = dict(manifest)
    payload["task_contract"] = task_contract(spec)
    return payload


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
) -> None:
    """Validate the manifest next to a checkpoint before loading its tensors.

    Legacy runs have no contract and remain loadable because existing Isaac and InstinctMJ
    checkpoints predate the unified launcher. Spec-hash drift is recorded in the manifest
    but does not block load: the same task keeps training and playback across declaration
    edits. A different ``task_id``, contract version, or ordered joint axis still fails: policy
    inputs and outputs are shape-compatible when BFS and DFS contain the same names, so checking
    the hash alone (or only the set of names) is not enough. A Play task may pass
    the explicitly registered training task id whose policy it consumes; training
    and resume callers omit it and remain strict about their own task identity.
    """
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    manifest_path = checkpoint_path.parent / "manifest.json"
    if not manifest_path.is_file():
        warnings.warn(
            f"Checkpoint {checkpoint_path} has no adjacent manifest.json; compatibility cannot be verified. "
            "In particular, a legacy Isaac policy may use native BFS joint order rather than canonical DFS.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    stored = manifest.get("task_contract")
    if not isinstance(stored, dict):
        warnings.warn(
            f"Checkpoint manifest {manifest_path} predates task contracts; compatibility cannot be verified. "
            "In particular, a legacy Isaac policy may use native BFS joint order rather than canonical DFS.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    current = task_contract(spec)
    expected_task_id = checkpoint_task_id or current["task_id"]
    if stored.get("version") != current["version"] or stored.get("task_id") != expected_task_id:
        raise ValueError(
            f"Checkpoint task contract mismatch for {checkpoint_path}: "
            f"checkpoint task={stored.get('task_id')!r}, version={stored.get('version')!r}; "
            f"runtime task={current['task_id']!r}, expected checkpoint task={expected_task_id!r}, "
            f"version={current['version']!r}."
        )
    stored_joint_names = stored.get("joint_names")
    if stored_joint_names != current["joint_names"]:
        raise ValueError(
            f"Checkpoint canonical joint order mismatch for {checkpoint_path}: "
            f"checkpoint={stored_joint_names!r}; runtime={current['joint_names']!r}. "
            "A BFS checkpoint cannot be loaded positionally into the DFS policy interface."
        )
    if stored.get("robot_schema_version") != current["robot_schema_version"]:
        raise ValueError(
            f"Checkpoint robot joint schema mismatch for {checkpoint_path}: "
            f"checkpoint={stored.get('robot_schema_version')!r}; "
            f"runtime={current['robot_schema_version']!r}."
        )


__all__ = [
    "add_task_contract",
    "checkpoint_sort_key",
    "latest_checkpoint",
    "latest_run_checkpoint",
    "task_contract",
    "validate_checkpoint_contract",
]
