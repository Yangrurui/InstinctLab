#!/usr/bin/env python3
"""Build release wheels and verify isolated engine-install combinations.

The probe runs from a temporary directory, installs only built wheels, and
filters entry-point discovery to the active virtual environment. Using
``--system-site-packages`` supplies the large simulator SDKs already installed
on a development machine without letting editable InstinctLab distributions
affect discovery.

This check covers wheel contents, backend discovery, discovery-time SDK import
isolation, native asset materialization, and the engine/task contract report.
It deliberately does not construct a simulator environment; use the live
construction checks documented in ``AGENTS.md`` for that GPU-dependent step.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECTS = (
    "instinctlab_engine",
    "instinctlab_engine_isaacsim",
    "instinctlab_engine_mjlab",
    "instinctlab",
)
MATRICES = {
    "core": ("instinctlab-engine-core",),
    "isaacsim": (
        "instinctlab-engine-core",
        "instinctlab-engine-isaacsim",
        "instinctlab",
    ),
    "mjlab": (
        "instinctlab-engine-core",
        "instinctlab-engine-mjlab",
        "instinctlab",
    ),
    "both": (
        "instinctlab-engine-core",
        "instinctlab-engine-isaacsim",
        "instinctlab-engine-mjlab",
        "instinctlab",
    ),
}
EXPECTED_ENGINES = {
    "core": (),
    "isaacsim": ("isaacsim",),
    "mjlab": ("mjlab",),
    "both": ("isaacsim", "mjlab"),
}

PROBE = r"""
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

expected_distributions = set(sys.argv[1].split(","))
expected_engines = tuple(filter(None, sys.argv[2].split(",")))
environment_root = Path(sys.prefix).resolve()

def belongs_to_environment(distribution):
    return Path(distribution.locate_file("")).resolve().is_relative_to(environment_root)

original_entry_points = metadata.entry_points

def isolated_entry_points(*args, **kwargs):
    return [
        entry_point
        for entry_point in original_entry_points(*args, **kwargs)
        if belongs_to_environment(entry_point.dist)
    ]

metadata.entry_points = isolated_entry_points
installed = {
    distribution.metadata["Name"]
    for distribution in metadata.distributions()
    if belongs_to_environment(distribution)
    and distribution.metadata["Name"].lower().startswith("instinctlab")
}
if installed != expected_distributions:
    raise AssertionError(f"installed distributions {installed}, expected {expected_distributions}")

sdk_roots = {"isaaclab", "mjlab", "mujoco", "mujoco_warp", "omni"}
before = set(sys.modules)
from instinctlab_engine import adapter, names

discovered = names()
if discovered != expected_engines:
    raise AssertionError(f"discovered engines {discovered}, expected {expected_engines}")
imported_during_discovery = {
    name.split(".", 1)[0] for name in set(sys.modules) - before
} & sdk_roots
if imported_during_discovery:
    raise AssertionError(
        f"engine discovery imported simulator SDK modules: {sorted(imported_during_discovery)}"
    )

materialized = {}
for engine in expected_engines:
    selected = adapter(engine)
    from instinctlab.tasks import registry

    task_id = "Instinct-Velocity-Flat-G1"
    robot = selected.robot_spec(registry.asset_id(task_id))
    native_path = Path(robot.asset_for(engine).path)
    if not native_path.is_file():
        raise AssertionError(f"wheel omitted native asset {native_path}")
    task = registry.spec(task_id, robot)
    report = selected.contract_report(task)
    if report["missing"]:
        raise AssertionError(f"{engine} contract has missing entries: {report['missing']}")
    materialized[engine] = {
        "task_id": task.task_id,
        "asset": str(native_path),
        "terms": len(task.mdp.terms()),
    }

print(json.dumps({"engines": discovered, "materialized": materialized}, sort_keys=True))
"""


def _run(command: list[str], *, cwd: Path) -> None:
    display = [
        "<inline-python>" if "\n" in argument else argument for argument in command
    ]
    print("+", " ".join(display), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _copy_projects(root: Path) -> list[Path]:
    copied: list[Path] = []
    ignored = shutil.ignore_patterns(
        "build",
        "dist",
        "*.egg-info",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
    )
    for project in PROJECTS:
        source = REPO_ROOT / "source" / project
        destination = root / "projects" / project
        shutil.copytree(source, destination, ignore=ignored)
        copied.append(destination)
    return copied


def _build_wheels(root: Path) -> dict[str, Path]:
    wheel_dir = root / "wheels"
    wheel_dir.mkdir()
    projects = _copy_projects(root)
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--disable-pip-version-check",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
            *(str(project) for project in projects),
        ],
        cwd=root,
    )
    wheels: dict[str, Path] = {}
    for wheel in wheel_dir.glob("*.whl"):
        distribution = wheel.name.split("-0.1.0-", 1)[0].replace("_", "-")
        wheels[distribution] = wheel
    expected = set(MATRICES["both"])
    if set(wheels) != expected:
        raise RuntimeError(
            f"built wheels {sorted(wheels)}, expected {sorted(expected)}"
        )
    return wheels


def _verify_matrix(
    root: Path,
    matrix: str,
    wheels: dict[str, Path],
) -> None:
    environment = root / "environments" / matrix
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
    python = environment / "bin" / "python"
    distributions = MATRICES[matrix]
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--ignore-installed",
            "--no-deps",
            *(str(wheels[name]) for name in distributions),
        ],
        cwd=root,
    )
    _run(
        [
            str(python),
            "-I",
            "-c",
            PROBE,
            ",".join(distributions),
            ",".join(EXPECTED_ENGINES[matrix]),
        ],
        cwd=root,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        action="append",
        choices=tuple(MATRICES),
        help="Matrix to run; repeat as needed. The default runs all four.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary wheel and environment directory for inspection.",
    )
    args = parser.parse_args()

    selected = tuple(args.matrix or MATRICES)
    if args.keep_temp:
        root = Path(tempfile.mkdtemp(prefix="instinctlab-wheel-matrix-"))
        print(f"Temporary files: {root}", flush=True)
        wheels = _build_wheels(root)
        for matrix in selected:
            _verify_matrix(root, matrix, wheels)
    else:
        with tempfile.TemporaryDirectory(prefix="instinctlab-wheel-matrix-") as temp:
            root = Path(temp)
            wheels = _build_wheels(root)
            for matrix in selected:
                _verify_matrix(root, matrix, wheels)

    print(f"Verified wheel matrices: {', '.join(selected)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
