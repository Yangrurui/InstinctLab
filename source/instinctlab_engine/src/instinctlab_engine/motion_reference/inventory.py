"""Deterministic file inventory for directory-backed motion references."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from instinctlab_engine.spec.sensor import MotionReferenceRef


@dataclass(frozen=True)
class MotionInventoryEntry:
    path: str
    weight: float = 1.0
    terrain_id: int | None = None


@dataclass(frozen=True)
class MotionFrameInventory:
    path: str
    frames: int
    fps: float
    duration_s: float


def _real_file(root: Path, relative: str) -> str:
    path = (root / relative).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Motion inventory entry {relative!r} is missing under {root}.")
    return str(path)


def discover_motion_inventory(ref: MotionReferenceRef) -> tuple[MotionInventoryEntry, ...]:
    """Resolve the final clip list, including metadata filtering and ordering."""
    root = Path(os.path.expanduser(ref.clip))
    if root.is_file():
        entries = [MotionInventoryEntry(str(root.resolve()))]
    elif not root.is_dir():
        raise FileNotFoundError(f"Motion dataset {ref.clip!r} resolves to missing path {root}.")
    elif ref.selected_files:
        entries = [MotionInventoryEntry(_real_file(root, relative)) for relative in ref.selected_files]
    elif ref.metadata_yaml:
        metadata_path = Path(os.path.expanduser(ref.metadata_yaml))
        if not metadata_path.is_absolute():
            metadata_path = root / metadata_path
        metadata_path = metadata_path.resolve()
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Motion metadata is missing: {metadata_path}.")
        metadata = yaml.safe_load(metadata_path.read_text()) or {}
        declared = metadata.get("motion_files") or []
        entries = [
            MotionInventoryEntry(
                _real_file(root, item["motion_file"]),
                float(item.get("weight", 1.0)),
                int(item["terrain_id"]) if "terrain_id" in item else None,
            )
            for item in declared
        ]
    else:
        paths = sorted(
            path.resolve()
            for path in root.rglob("*")
            if path.is_file() and any(path.name.endswith(suffix) for suffix in ref.supported_file_endings)
        )
        entries = [MotionInventoryEntry(str(path)) for path in paths]
    if ref.first_motion_only:
        entries = entries[:1]
    if not entries:
        raise ValueError(f"Motion dataset {root} has no files matching {ref.supported_file_endings}.")
    if any(entry.weight < 0.0 for entry in entries) or sum(entry.weight for entry in entries) <= 0.0:
        raise ValueError(f"Motion dataset {root} has invalid sampling weights.")
    return tuple(entries)


__all__ = ["MotionFrameInventory", "MotionInventoryEntry", "discover_motion_inventory"]
