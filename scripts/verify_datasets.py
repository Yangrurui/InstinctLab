#!/usr/bin/env python3
"""Verify versioned InstinctLab datasets under ``INSTINCTLAB_DATA_ROOT``."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "datasets" / "manifest.json"
SHA256 = re.compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != "instinctlab_dataset_manifest_v1":
        raise ValueError(
            f"Unsupported dataset manifest schema in {path}: "
            f"{payload.get('schema_version')!r}."
        )
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError(f"Dataset manifest {path} has no datasets.")
    return payload


def _check_file(path: Path, expected: str) -> dict[str, Any]:
    if not SHA256.fullmatch(expected):
        raise ValueError(f"Invalid SHA-256 declaration for {path}: {expected!r}.")
    if not path.is_file():
        raise FileNotFoundError(f"Dataset resource is missing: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"Dataset checksum mismatch for {path}: got {actual}, expected {expected}."
        )
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": actual,
    }


def _resolve_below(root: Path, relative: str, *, field: str) -> Path:
    """Resolve one manifest path without allowing it to leave its dataset."""
    if not relative or "\\" in relative or Path(relative).is_absolute():
        raise ValueError(f"Dataset {field} must be a portable relative path: {relative!r}.")
    root = root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Dataset {field} escapes dataset root {root}: {relative!r}."
        ) from exc
    return resolved


def _check_conversion_index(root: Path, relative: str) -> list[dict[str, Any]]:
    index_path = _resolve_below(root, relative, field="conversion_index")
    payload = json.loads(index_path.read_text())
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"Conversion index has no files: {index_path}")
    checked = []
    for item in files:
        target = item.get("target")
        checksum = item.get("target_sha256")
        if not isinstance(target, str) or not isinstance(checksum, str):
            raise TypeError(f"Malformed conversion entry in {index_path}: {item!r}")
        checked.append(
            _check_file(
                _resolve_below(root, target, field="conversion target"),
                checksum,
            )
        )
    return checked


def verify_dataset(name: str, declaration: dict[str, Any]) -> dict[str, Any]:
    from instinctlab_engine.data import resolve_data_path

    uri = declaration.get("uri")
    if not isinstance(uri, str) or not uri.startswith("dataset://"):
        raise ValueError(f"Dataset {name!r} has invalid URI {uri!r}.")
    root = resolve_data_path(uri)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset {name!r} is missing at {root} ({uri}).")

    resources = declaration.get("resources", {})
    if not isinstance(resources, dict):
        raise TypeError(f"Dataset {name!r} resources must be an object.")
    checked = []
    for relative, checksum in resources.items():
        if not isinstance(relative, str) or not isinstance(checksum, str):
            raise TypeError(f"Dataset {name!r} has a malformed resource declaration.")
        checked.append(
            _check_file(
                _resolve_below(root, relative, field="resource"),
                checksum,
            )
        )

    conversion_index = declaration.get("conversion_index")
    if conversion_index is not None:
        if not isinstance(conversion_index, str):
            raise ValueError(f"Dataset {name!r} conversion_index must be a path.")
        checked.extend(_check_conversion_index(root, conversion_index))

    return {
        "name": name,
        "declared": uri,
        "resolved": str(root),
        "status": declaration.get("status", "unspecified"),
        "resource_count": len(checked),
        "resources": checked,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset key to verify. Repeat as needed; defaults to every required dataset.",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Also verify datasets marked required=false.",
    )
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    manifest = _load_manifest(manifest_path)
    declarations = manifest["datasets"]
    if args.datasets:
        unknown = sorted(set(args.datasets) - set(declarations))
        if unknown:
            raise KeyError(f"Unknown dataset keys: {unknown}")
        selected = args.datasets
    else:
        selected = [
            name
            for name, declaration in declarations.items()
            if declaration.get("required", True) or args.include_optional
        ]

    verified = [verify_dataset(name, declarations[name]) for name in selected]
    receipt = {
        "schema_version": "instinctlab_dataset_verification_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "data_root": os.environ.get("INSTINCTLAB_DATA_ROOT", "~/Datasets"),
        "datasets": verified,
    }
    if args.receipt is not None:
        output = args.receipt.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        temporary.replace(output)
    print(
        f"Verified {len(verified)} datasets and "
        f"{sum(item['resource_count'] for item in verified)} resources "
        f"under {receipt['data_root']}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
