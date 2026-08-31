#!/usr/bin/env python3
"""Build all coordinated distributions and emit a reproducible checksum manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

from check_release import (
    APPLICATION,
    PROJECTS,
    collect_release_metadata,
    validate_release_metadata,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_REQUIREMENTS = ("build==1.2.2.post1", "twine==6.2.0")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_project(source: Path, destination: Path) -> Path:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(
            "build",
            "dist",
            "*.egg-info",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            # Maintainer-only diagnostic module is intentionally gitignored;
            # a local copy must never leak into an application artifact.
            "shadowing_probe.py",
        ),
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "dist" / "release",
        help="New or empty directory which receives wheels, sdists, and SHA256SUMS.json.",
    )
    parser.add_argument("--expected-version")
    args = parser.parse_args()
    version = validate_release_metadata(
        collect_release_metadata(),
        expected_version=args.expected_version,
    )
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"release output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="instinctlab-release-") as temporary:
        root = Path(temporary)
        stage = root / "artifacts"
        stage.mkdir()
        tools = root / "tools"
        venv.EnvBuilder(with_pip=True).create(tools)
        tool_python = tools / "bin" / "python"
        subprocess.run(
            [
                str(tool_python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                *BUILD_REQUIREMENTS,
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        projects_root = root / "projects"
        projects_root.mkdir()
        projects = [
            _copy_project(project, projects_root / project.name)
            for project in (*PROJECTS.values(), APPLICATION)
        ]
        for project in projects:
            subprocess.run(
                [
                    str(tool_python),
                    "-m",
                    "build",
                    "--wheel",
                    "--sdist",
                    "--outdir",
                    str(stage),
                    str(project),
                ],
                cwd=REPO_ROOT,
                check=True,
            )
        artifacts = sorted(
            path
            for path in stage.iterdir()
            if path.suffix == ".whl" or path.name.endswith(".tar.gz")
        )
        if len(artifacts) != 8:
            raise RuntimeError(
                f"release build produced {len(artifacts)} artifacts, expected eight: "
                f"{[path.name for path in artifacts]}"
            )
        subprocess.run(
            [
                str(tool_python),
                "-m",
                "twine",
                "check",
                *(str(path) for path in artifacts),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        for artifact in artifacts:
            shutil.copy2(artifact, output / artifact.name)

    manifest = {
        "version": "instinctlab_release_artifacts_v1",
        "package_version": version,
        "artifacts": {
            path.name: {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(output.iterdir())
            if path.suffix == ".whl" or path.name.endswith(".tar.gz")
        },
    }
    manifest_path = output / "SHA256SUMS.json"
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f"Built coordinated {version} release artifacts in {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
