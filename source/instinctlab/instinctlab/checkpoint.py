"""Checkpoint compatibility metadata for the unified training entry points."""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from instinctlab.sim.robot_spec import BackendAsset
from instinctlab.spec import TaskSpec

_CONTRACT_VERSION = "task_spec_v1"


def _type_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


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
        return {
            "type": _type_name(value),
            "fields": [[field.name, _canonical(getattr(value, field.name))] for field in fields(value)],
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


def validate_checkpoint_contract(checkpoint: str | Path, spec: TaskSpec) -> None:
    """Validate the manifest next to a checkpoint before loading its tensors.

    Legacy runs have no contract and remain loadable because existing Isaac and InstinctMJ
    checkpoints predate the unified launcher. A present contract is never ignored.
    """
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    manifest_path = checkpoint_path.parent / "manifest.json"
    if not manifest_path.is_file():
        warnings.warn(
            f"Checkpoint {checkpoint_path} has no adjacent manifest.json; compatibility cannot be verified.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    stored = manifest.get("task_contract")
    if not isinstance(stored, dict):
        warnings.warn(
            f"Checkpoint manifest {manifest_path} predates task contracts; compatibility cannot be verified.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    current = task_contract(spec)
    if stored.get("version") != current["version"] or stored.get("hash") != current["hash"]:
        raise ValueError(
            f"Checkpoint task contract mismatch for {checkpoint_path}: "
            f"checkpoint task={stored.get('task_id')!r}, hash={str(stored.get('hash'))[:12]}; "
            f"runtime task={current['task_id']!r}, hash={current['hash'][:12]}. "
            "Use a checkpoint produced by the same TaskSpec declaration."
        )


__all__ = ["add_task_contract", "task_contract", "validate_checkpoint_contract"]
