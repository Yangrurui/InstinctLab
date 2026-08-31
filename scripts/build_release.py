#!/usr/bin/env python3
"""Build all coordinated distributions and emit a reproducible checksum manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
BUILD_REQUIREMENTS = (
    "build==1.2.2.post1",
    "twine==6.2.0",
    "setuptools==81.0.0",
    "wheel==0.45.1",
    "packaging==25.0",
    "toml==0.10.2",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_checkout(repository: Path = REPO_ROOT) -> str:
    """Return HEAD only when tracked and untracked source state is clean."""
    commit = _git_output(repository, "rev-parse", "HEAD")
    status = _git_output(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise RuntimeError(
            "Release builds require a clean Git checkout; reconcile these paths first:\n"
            f"{status}"
        )
    return commit


def _copy_project(source: Path, destination: Path) -> Path:
    """Copy only Git-tracked files from one project into the build stage."""
    repository = Path(_git_output(source, "rev-parse", "--show-toplevel")).resolve()
    source = source.resolve()
    try:
        project_relative = source.relative_to(repository)
    except ValueError as exc:
        raise ValueError(
            f"Project source is outside its Git repository: {source}"
        ) from exc

    encoded = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "ls-files",
            "-z",
            "--",
            project_relative.as_posix(),
        ],
        check=True,
        capture_output=True,
    ).stdout
    tracked = [Path(value.decode()) for value in encoded.split(b"\0") if value]
    if not tracked:
        raise RuntimeError(f"Project has no Git-tracked files: {source}")

    destination.mkdir(parents=True)
    for repository_relative in tracked:
        source_path = repository / repository_relative
        project_path = repository_relative.relative_to(project_relative)
        destination_path = destination / project_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.is_symlink():
            destination_path.symlink_to(os.readlink(source_path))
        else:
            shutil.copy2(source_path, destination_path)
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
    source_commit = _require_clean_checkout()
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
                    "--no-isolation",
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
        "source_commit": source_commit,
        "source_dirty": False,
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
