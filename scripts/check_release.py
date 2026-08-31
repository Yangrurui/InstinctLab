#!/usr/bin/env python3
"""Fail closed when coordinated package and public plugin versions drift."""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECTS = {
    "instinctlab-engine-core": REPO_ROOT / "source" / "instinctlab_engine",
    "instinctlab-engine-isaacsim": REPO_ROOT
    / "source"
    / "instinctlab_engine_isaacsim",
    "instinctlab-engine-mjlab": REPO_ROOT / "source" / "instinctlab_engine_mjlab",
}
APPLICATION = REPO_ROOT / "source" / "instinctlab"


def _toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _constant(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise RuntimeError(f"{path} does not define a string constant {name}")


def collect_release_metadata() -> dict:
    packages = {
        name: _toml(path / "pyproject.toml")["project"]
        for name, path in PROJECTS.items()
    }
    extension = _toml(APPLICATION / "config" / "extension.toml")["package"]
    packages["instinctlab"] = {
        "version": extension["version"],
        "requires-python": ">=3.11",
        "dependencies": [],
    }
    return {
        "packages": packages,
        "engine_core_api": _constant(
            PROJECTS["instinctlab-engine-core"]
            / "src"
            / "instinctlab_engine"
            / "plugins.py",
            "ENGINE_CORE_API_VERSION",
        ),
        "native_asset_api": _constant(
            PROJECTS["instinctlab-engine-core"]
            / "src"
            / "instinctlab_engine"
            / "assets.py",
            "NATIVE_ASSET_API_VERSION",
        ),
        "application_setup": (APPLICATION / "setup.py").read_text(),
    }


def validate_release_metadata(metadata: dict, *, expected_version: str | None = None) -> str:
    packages = metadata["packages"]
    versions = {name: values["version"] for name, values in packages.items()}
    unique_versions = set(versions.values())
    if len(unique_versions) != 1:
        raise RuntimeError(f"release package versions are not coordinated: {versions}")
    version = unique_versions.pop()
    if expected_version is not None and version != expected_version:
        raise RuntimeError(
            f"coordinated release version is {version!r}, expected {expected_version!r}"
        )
    for name, values in packages.items():
        if values["requires-python"] != ">=3.11":
            raise RuntimeError(
                f"{name} requires Python {values['requires-python']!r}, expected '>=3.11'"
            )

    core_requirement = f"instinctlab-engine-core[geometry]=={version}"
    for backend in ("instinctlab-engine-isaacsim", "instinctlab-engine-mjlab"):
        if core_requirement not in packages[backend]["dependencies"]:
            raise RuntimeError(
                f"{backend} does not require coordinated {core_requirement!r}"
            )
    setup = metadata["application_setup"]
    for requirement in (
        f"instinctlab-engine-core=={version}",
        f"instinctlab-engine-isaacsim=={version}",
        f"instinctlab-engine-mjlab=={version}",
        "instinct-rl==1.0.2",
    ):
        if f'"{requirement}"' not in setup:
            raise RuntimeError(f"application setup does not pin {requirement!r}")
    if 'python_requires=">=3.11"' not in setup:
        raise RuntimeError("application setup does not require Python >=3.11")

    public_api = ".".join(version.split(".")[:2])
    for name in ("engine_core_api", "native_asset_api"):
        api_version = metadata[name]
        if not re.fullmatch(r"\d+\.\d+", api_version):
            raise RuntimeError(f"{name} is not a major.minor API version: {api_version!r}")
        if api_version != public_api:
            raise RuntimeError(
                f"{name}={api_version!r} is not coordinated with package version {version!r}"
            )
    return version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-version",
        help="Require this exact coordinated version, for example 1.0.0 on a release tag.",
    )
    args = parser.parse_args()
    version = validate_release_metadata(
        collect_release_metadata(),
        expected_version=args.expected_version,
    )
    print(f"Release metadata is coordinated at {version}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
