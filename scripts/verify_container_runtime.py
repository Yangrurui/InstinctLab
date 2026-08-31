#!/usr/bin/env python3
"""Fail closed unless an image contains the locked dual-backend runtime and wheels."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

IMAGE_DIGEST = re.compile(r".+@sha256:[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(
    path: Path, schema: str, *, schema_field: str = "schema_version"
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required runtime contract is missing: {path}")
    payload = json.loads(path.read_text())
    if payload.get(schema_field) != schema:
        raise ValueError(
            f"Unsupported schema in {path}: {payload.get(schema_field)!r}; "
            f"expected {schema!r}."
        )
    return payload


def _verify_distributions(expected: dict[str, str]) -> None:
    problems = []
    for distribution, required in expected.items():
        try:
            actual = version(distribution)
        except PackageNotFoundError:
            problems.append(f"{distribution}: missing (expected {required})")
            continue
        if actual != required:
            problems.append(f"{distribution}: {actual} (expected {required})")
    if problems:
        raise RuntimeError(
            "Container distribution lock mismatch:\n  " + "\n  ".join(problems)
        )


def _verify_external_runtime(
    lock: dict[str, Any], provenance: dict[str, Any], base_image: str
) -> None:
    if not IMAGE_DIGEST.fullmatch(base_image):
        raise ValueError(
            "INSTINCTLAB_RUNTIME_IMAGE must be an immutable image@sha256:<64 hex> reference, "
            f"got {base_image!r}."
        )
    for name, expected in lock["sources"].items():
        actual = (provenance.get("sources") or {}).get(name, {})
        if actual.get("commit") != expected["commit"]:
            raise ValueError(
                f"External runtime {name} commit is {actual.get('commit')!r}; "
                f"expected {expected['commit']!r}."
            )
        if actual.get("url") != expected["url"] or actual.get("dirty") is not False:
            raise ValueError(
                f"External runtime {name} provenance is not the clean locked source: {actual!r}."
            )


def _verify_release_artifacts(wheel_dir: Path, expected_version: str) -> dict[str, Any]:
    manifest = _load(
        wheel_dir / "SHA256SUMS.json",
        "instinctlab_release_artifacts_v1",
        schema_field="version",
    )
    if manifest.get("package_version") != expected_version:
        raise ValueError(
            f"Release artifacts are {manifest.get('package_version')!r}; "
            f"runtime lock expects {expected_version!r}."
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or len(artifacts) != 8:
        raise ValueError(
            "Release manifest must name four wheels and four source archives."
        )
    for filename, declaration in artifacts.items():
        path = wheel_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Release artifact is missing: {path}")
        actual = _sha256(path)
        if actual != declaration.get("sha256"):
            raise ValueError(
                f"Release artifact checksum mismatch for {path}: got {actual}, "
                f"expected {declaration.get('sha256')}."
            )
    wheels = sorted(name for name in artifacts if name.endswith(".whl"))
    expected_prefixes = (
        "instinctlab-",
        "instinctlab_engine_core-",
        "instinctlab_engine_isaacsim-",
        "instinctlab_engine_mjlab-",
    )
    if len(wheels) != 4 or any(
        not any(name.startswith(prefix) for name in wheels)
        for prefix in expected_prefixes
    ):
        raise ValueError(f"Release manifest has unexpected wheels: {wheels}")
    return manifest


def verify(
    *,
    lock_path: Path,
    runtime_provenance_path: Path,
    wheel_dir: Path,
    base_image: str,
) -> None:
    lock = _load(lock_path, "instinctlab_container_runtime_lock_v1")
    if platform.python_version().rsplit(".", 1)[0] != lock["python"]:
        raise RuntimeError(
            f"Container Python is {platform.python_version()}; expected {lock['python']}.x."
        )
    provenance = _load(runtime_provenance_path, "instinctlab_external_runtime_v1")
    _verify_external_runtime(lock, provenance, base_image)
    _verify_release_artifacts(wheel_dir, lock["application_version"])
    _verify_distributions(lock["distributions"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--runtime-provenance", type=Path, required=True)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--base-image", required=True)
    args = parser.parse_args()
    verify(
        lock_path=args.lock.resolve(),
        runtime_provenance_path=args.runtime_provenance.resolve(),
        wheel_dir=args.wheel_dir.resolve(),
        base_image=args.base_image,
    )
    print("Verified immutable dual-backend runtime and coordinated application wheels.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
