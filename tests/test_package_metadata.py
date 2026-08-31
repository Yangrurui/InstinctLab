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
        assert metadata["license"] == "CC-BY-NC-4.0"
        assert metadata["readme"] == "README.md"
        assert metadata["authors"] == [{"name": "Project Instinct"}]
        assert metadata["urls"]["Source"] == (
            "https://github.com/project-instinct/instinctlab"
        )


def test_application_wheel_manifest_contains_native_robot_resources() -> None:
    manifest = (SOURCE / "instinctlab/MANIFEST.in").read_text()
    assert "recursive-include instinctlab/assets/resources *" in manifest


def test_application_exposes_the_optional_deployment_verifier() -> None:
    setup = (SOURCE / "instinctlab/setup.py").read_text()

    assert '"deployment": DEPLOYMENT_REQUIRES' in setup
    assert '"onnx==1.22.0"' in setup
    assert '"onnxruntime==1.29.0"' in setup
    assert "instinctlab-verify-deployment = instinctlab.deployment_cli:main" in setup


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


def test_wheel_verifier_pins_an_isolated_build_toolchain() -> None:
    source = (ROOT / "scripts/verify_wheel_matrix.py").read_text()
    tree = ast.parse(source)
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "WHEEL_BUILD_REQUIREMENTS"
            for target in node.targets
        )
    )

    assert ast.literal_eval(assignment.value) == (
        "setuptools==81.0.0",
        "wheel==0.45.1",
        "packaging==25.0",
        "toml==0.10.2",
    )
    assert "venv.EnvBuilder(with_pip=True).create(build_environment)" in source
    assert (
        'str(build_python),\n            "-m",\n            "pip",\n            "wheel"'
        in source
    )


def test_wheel_extension_probe_tracks_the_versioned_preflight_selection() -> None:
    source = (ROOT / "scripts/verify_wheel_matrix.py").read_text()
    module = ast.parse(source)
    probe_assignment = next(
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "EXTENSION_PROBE"
            for target in node.targets
        )
    )
    probe = ast.literal_eval(probe_assignment.value)
    tree = ast.parse(probe)

    schema_check = next(
        node.test
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and "schema_version" in ast.unparse(node.test)
    )
    assert ast.literal_eval(schema_check.comparators[0]) == "preflight_v1"

    selection_check = next(
        node.test
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and "selected_components" in ast.unparse(node.test)
    )
    expected = ast.literal_eval(selection_check.comparators[0])
    assert expected == {
        "asset_id": "fixture_bot/v1",
        "articulation_asset_ids": {"robot": "fixture_bot/v1"},
        "actuator_model_ids": ["fixture.stateful.v1"],
        "actuator_groups": [
            {
                "entity": "robot",
                "name": "joint",
                "model_id": "fixture.stateful.v1",
            }
        ],
        "sensor_kinds": ["fixture.imu"],
        "terrain_kind": "fixture_plane",
        "sub_terrain_kinds": [],
    }


def test_each_backend_publishes_its_native_actuator_registrar() -> None:
    expected = {
        "instinctlab_engine_isaacsim": "isaacsim.isaaclab_pd",
        "instinctlab_engine_mjlab": "mjlab.mjlab_pd",
    }
    for project, entry_point_name in expected.items():
        project_metadata = _toml(SOURCE / project / "pyproject.toml")["project"]
        actuator_entries = project_metadata["entry-points"]["instinctlab.actuators"]
        assert tuple(actuator_entries) == (entry_point_name,)
