"""Portable container and provider contract for same-engine snapshots."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


class SnapshotError(RuntimeError):
    """A snapshot is incompatible, incomplete, or cannot be restored safely."""


@runtime_checkable
class SnapshotProvider(Protocol):
    """Engine-owned capture and restore of native environment state.

    A provider must restore snapshots produced by the same provider version. It
    is intentionally not a cross-engine state translation interface.
    """

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> int: ...

    def capture(self) -> Mapping[str, Any]: ...

    def restore(self, state: Mapping[str, Any]) -> None: ...


@dataclass(frozen=True)
class EnvironmentSnapshot:
    """One complete same-engine environment checkpoint."""

    schema_version: int
    engine: str
    task_id: str
    num_envs: int
    provider_id: str
    provider_version: int
    native_state: Mapping[str, Any]
    lifecycle_state: Mapping[str, Any]
    component_states: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        """Persist without pickle, replacing the destination atomically."""
        return save_archive(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> EnvironmentSnapshot:
        """Load a schema-checked snapshot without executing serialized code."""
        value = load_archive(path)
        if not isinstance(value, dict):
            raise SnapshotError("Snapshot root must be a mapping.")
        try:
            return cls(**value)
        except (TypeError, ValueError) as exc:
            raise SnapshotError(f"Invalid snapshot document: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "engine": self.engine,
            "task_id": self.task_id,
            "num_envs": self.num_envs,
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "native_state": dict(self.native_state),
            "lifecycle_state": dict(self.lifecycle_state),
            "component_states": dict(self.component_states),
            "metadata": dict(self.metadata),
        }


def save_archive(path: str | Path, value: Any) -> Path:
    """Save a tensor tree to one atomic NPZ archive without pickle."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    document = _encode_tree(value, arrays)
    arrays["__metadata__"] = np.asarray(
        json.dumps(document, sort_keys=True, separators=(",", ":"))
    )
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        np.savez_compressed(temporary, **arrays)
    try:
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def load_archive(path: str | Path) -> Any:
    """Load a tensor tree saved by :func:`save_archive` without code execution."""
    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as archive:
            if "__metadata__" not in archive.files:
                raise SnapshotError("Lifecycle archive has no metadata document.")
            document = json.loads(str(archive["__metadata__"].item()))
            arrays = {
                name: archive[name].copy()
                for name in archive.files
                if name != "__metadata__"
            }
    except SnapshotError:
        raise
    except Exception as exc:
        raise SnapshotError(f"Cannot load lifecycle archive {source}: {exc}") from exc
    return _decode_tree(document, arrays)


def _encode_tree(value: Any, arrays: dict[str, np.ndarray]) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        torch_dtype = str(tensor.dtype).removeprefix("torch.")
        if tensor.dtype == torch.bfloat16:
            array = tensor.view(torch.uint16).numpy()
        else:
            array = tensor.numpy()
        key = f"array_{len(arrays):08d}"
        arrays[key] = array
        return {
            "__kind__": "torch",
            "array": key,
            "dtype": torch_dtype,
        }
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise SnapshotError("Object arrays are not permitted in lifecycle snapshots.")
        key = f"array_{len(arrays):08d}"
        arrays[key] = np.ascontiguousarray(value)
        return {"__kind__": "numpy", "array": key}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise SnapshotError("Snapshot mapping keys must be strings.")
        return {
            "__kind__": "mapping",
            "items": {key: _encode_tree(item, arrays) for key, item in value.items()},
        }
    if isinstance(value, tuple):
        return {
            "__kind__": "tuple",
            "items": [_encode_tree(item, arrays) for item in value],
        }
    if isinstance(value, list):
        return {
            "__kind__": "list",
            "items": [_encode_tree(item, arrays) for item in value],
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise SnapshotError(
        f"Snapshot value of type {type(value).__name__} is not safely serializable."
    )


def _decode_tree(value: Any, arrays: Mapping[str, np.ndarray]) -> Any:
    import torch

    if not isinstance(value, dict) or "__kind__" not in value:
        return value
    kind = value["__kind__"]
    if kind in {"torch", "numpy"}:
        key = value.get("array")
        if key not in arrays:
            raise SnapshotError(f"Snapshot references missing array {key!r}.")
        array = arrays[key]
        if kind == "numpy":
            return array
        dtype_name = value.get("dtype")
        dtype = getattr(torch, str(dtype_name), None)
        if not isinstance(dtype, torch.dtype):
            raise SnapshotError(f"Snapshot uses unknown torch dtype {dtype_name!r}.")
        tensor = torch.from_numpy(array.copy())
        if dtype == torch.bfloat16:
            if tensor.dtype != torch.uint16:
                raise SnapshotError("Malformed bfloat16 snapshot storage.")
            return tensor.view(torch.bfloat16)
        return tensor.to(dtype=dtype)
    items = value.get("items")
    if kind == "mapping":
        if not isinstance(items, dict):
            raise SnapshotError("Malformed snapshot mapping node.")
        return {key: _decode_tree(item, arrays) for key, item in items.items()}
    if kind in {"tuple", "list"}:
        if not isinstance(items, list):
            raise SnapshotError(f"Malformed snapshot {kind} node.")
        decoded = [_decode_tree(item, arrays) for item in items]
        return tuple(decoded) if kind == "tuple" else decoded
    raise SnapshotError(f"Snapshot contains unknown node kind {kind!r}.")


__all__ = [
    "EnvironmentSnapshot",
    "SnapshotError",
    "SnapshotProvider",
    "load_archive",
    "save_archive",
]
