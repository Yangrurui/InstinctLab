"""Release automation, versions, and public compatibility policy stay coordinated."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]


def _script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def test_all_release_distributions_and_plugin_apis_are_coordinated() -> None:
    release = _script("check_release.py")

    assert release.validate_release_metadata(release.collect_release_metadata()) == "0.1.0"


def test_python_tooling_targets_the_supported_python_311() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        tools = tomllib.load(handle)["tool"]

    assert tools["pyright"]["pythonVersion"] == "3.11"
    assert tools["isort"]["py_version"] == 311


def test_release_automation_has_separate_fast_wheel_and_gpu_gates() -> None:
    workflows = ROOT / ".github" / "workflows"
    fast = (workflows / "pr-fast.yml").read_text()
    wheels = (workflows / "wheel-matrix.yml").read_text()
    gpu = (workflows / "gpu-live.yml").read_text()
    release = (workflows / "release.yml").read_text()

    assert "scripts/check_ruff_ratchet.py" in fast
    assert "pyright" in fast
    assert "scripts/verify_wheel_matrix.py" in wheels
    assert "--live-extension" not in wheels
    assert "schedule:" in gpu and "release:" in gpu
    assert "--live-extension" in gpu
    assert "self-hosted" in gpu
    assert "scripts/build_release.py" in release
    assert "gh-action-pypi-publish" in release
    assert "environment: pypi" in release


def test_release_policy_defines_versioning_deprecation_and_publication() -> None:
    policy = (ROOT / "RELEASE.md").read_text()

    assert "semantic versioning" in policy
    assert "ENGINE_CORE_API_VERSION" in policy
    assert "NATIVE_ASSET_API_VERSION" in policy
    assert "DeprecationWarning" in policy
    assert "python scripts/build_release.py" in policy
    assert "SHA256SUMS.json" in policy


def test_release_builder_uses_clean_sources_and_isolated_pinned_tools() -> None:
    builder = _script("build_release.py")

    assert builder.BUILD_REQUIREMENTS == (
        "build==1.2.2.post1",
        "twine==6.2.0",
    )
    source = (ROOT / "scripts" / "build_release.py").read_text()
    assert "shutil.copytree(" in source
    assert "venv.EnvBuilder(with_pip=True)" in source
    assert '"__pycache__"' in source
