"""Release metadata and wheel-matrix declarations stay aligned."""

from __future__ import annotations

import ast
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
ENGINE_PROJECTS = (
    "instinctlab_engine",
    "instinctlab_engine_isaacsim",
    "instinctlab_engine_mjlab",
)


def _toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def test_application_metadata_is_not_the_isaac_extension_template() -> None:
    package = _toml(SOURCE / "instinctlab/config/extension.toml")["package"]
    assert package["title"] == "InstinctLab"
    assert package["author"] == "Project Instinct"
    assert package["repository"] == "https://github.com/project-instinct/instinctlab"
    assert package["license"] == "CC-BY-NC-4.0"
    assert "template" not in package["description"].lower()
    assert _toml(SOURCE / "instinctlab/config/extension.toml")["dependencies"] == {}


def test_every_engine_distribution_declares_the_repository_license_and_owner() -> None:
    for project in ENGINE_PROJECTS:
        metadata = _toml(SOURCE / project / "pyproject.toml")["project"]
        assert metadata["license"]["text"] == "CC BY-NC 4.0"
        assert metadata["authors"] == [{"name": "Project Instinct"}]
        assert metadata["urls"]["Source"] == (
            "https://github.com/project-instinct/instinctlab"
        )


def test_application_wheel_manifest_contains_native_robot_resources() -> None:
    manifest = (SOURCE / "instinctlab/MANIFEST.in").read_text()
    assert "recursive-include instinctlab/assets/resources *" in manifest


def test_readme_factory_accepts_the_engine_normalized_robot() -> None:
    readme = (ROOT / "README.md").read_text()
    assert "def my_task(robot: RobotSpec) -> TaskSpec:" in readme
    assert "TaskSpec(robot=robot, ...)" in readme


def test_wheel_verifier_declares_all_four_install_matrices() -> None:
    source = (ROOT / "scripts/verify_wheel_matrix.py").read_text()
    tree = ast.parse(source)
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "MATRICES"
            for target in node.targets
        )
    )
    matrices = ast.literal_eval(assignment.value)
    assert tuple(matrices) == ("core", "isaacsim", "mjlab", "both")
