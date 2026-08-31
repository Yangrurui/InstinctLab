"""The operator image is built from wheels on a fail-closed external runtime."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _runtime_verifier():
    path = ROOT / "scripts/verify_container_runtime.py"
    spec = importlib.util.spec_from_file_location("container_runtime_verifier", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runtime_receipt_writer():
    path = ROOT / "scripts/write_container_runtime_provenance.py"
    spec = importlib.util.spec_from_file_location("container_runtime_receipt", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dockerfile_builds_wheels_then_fail_closes_the_external_runtime() -> None:
    dockerfile = (ROOT / "docker/Dockerfile").read_text()

    assert "COPY dist/release /opt/instinctlab/release" in dockerfile
    assert "AS wheel-builder" not in dockerfile
    assert "FROM ${INSTINCTLAB_RUNTIME_IMAGE} AS runtime" in dockerfile
    assert "pip install --no-deps /opt/instinctlab/release/*.whl" in dockerfile
    assert "verify_container_runtime.py" in dockerfile
    assert "INSTINCTLAB_RUNTIME_PROVENANCE" in dockerfile
    assert "cp -r /root" not in dockerfile
    assert "DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH" not in dockerfile


def test_compose_has_no_personal_mount_or_repository_runtime_dependency() -> None:
    compose = (ROOT / "docker/docker-compose.yaml").read_text()

    assert "/public/MARS" not in compose
    assert "/root/ziwenz" not in compose
    assert "target: /datasets" in compose
    assert "read_only: true" in compose
    assert "INSTINCTLAB_HOST_DATA_ROOT:?" in compose
    assert "INSTINCTLAB_WHEEL_BUILDER_IMAGE" not in compose
    assert "source: ..\n" not in compose


def test_runtime_lock_matches_release_and_backend_source_pins() -> None:
    lock = json.loads((ROOT / "docker/runtime-lock.json").read_text())
    backend_pins = (ROOT / "source/instinctlab/config/backend_pins.toml").read_text()

    assert lock["schema_version"] == "instinctlab_container_runtime_lock_v1"
    assert lock["application_version"] == "0.1.0"
    assert lock["python"] == "3.11"
    assert lock["sources"]["isaaclab"]["commit"].startswith("f73c331738")
    assert 'commit = "f73c331738"' in backend_pins
    assert lock["sources"]["mjlab"]["tag"] == "v1.5.0"
    assert 'tag = "v1.5.0"' in backend_pins
    for distribution in (
        "instinctlab",
        "instinctlab-engine-core",
        "instinctlab-engine-isaacsim",
        "instinctlab-engine-mjlab",
    ):
        assert lock["distributions"][distribution] == "0.1.0"
    for distribution in (
        "numpy",
        "PyYAML",
        "psutil",
        "pytorch-kinematics",
        "joblib",
        "debugpy",
        "snakeviz",
        "trimesh",
        "scikit-learn",
        "opencv-python",
        "packaging",
        "pyvista",
        "coacd",
    ):
        assert distribution in lock["distributions"]
    assert set(lock["imports"]).issubset(lock["distributions"])


def test_external_runtime_requires_digest_and_exact_source_commits() -> None:
    verifier = _runtime_verifier()
    lock = json.loads((ROOT / "docker/runtime-lock.json").read_text())
    base_image = f"registry/runtime@sha256:{'a' * 64}"
    provenance = {
        "sources": {
            name: {
                "commit": declaration["commit"],
                "dirty": False,
                "url": declaration["url"],
            }
            for name, declaration in lock["sources"].items()
        },
    }

    verifier._verify_external_runtime(lock, provenance, base_image)
    with pytest.raises(ValueError, match="immutable"):
        verifier._verify_external_runtime(lock, provenance, "registry/runtime:latest")
    provenance["sources"]["mjlab"]["commit"] = "wrong"
    with pytest.raises(ValueError, match="mjlab commit"):
        verifier._verify_external_runtime(lock, provenance, base_image)


def test_external_runtime_receipt_refuses_a_dirty_source(tmp_path: Path) -> None:
    writer = _runtime_receipt_writer()
    repository = tmp_path / "backend"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Test"], check=True
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.com"],
        check=True,
    )
    (repository / "source.txt").write_text("locked\n")
    subprocess.run(["git", "-C", str(repository), "add", "source.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-q", "-m", "locked"],
        check=True,
    )
    source_url = "https://github.com/example/backend.git"
    subprocess.run(
        ["git", "-C", str(repository), "remote", "add", "origin", source_url],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected = {"url": source_url, "commit": commit}

    assert writer._source_receipt("backend", repository, expected) == {
        "url": source_url,
        "commit": commit,
        "dirty": False,
    }
    (repository / "source.txt").write_text("dirty\n")
    with pytest.raises(RuntimeError, match="dirty=True"):
        writer._source_receipt("backend", repository, expected)


def test_release_artifact_verification_checks_every_digest(tmp_path: Path) -> None:
    verifier = _runtime_verifier()
    source_commit = "a" * 40
    names = (
        "instinctlab-0.1.0-py3-none-any.whl",
        "instinctlab_engine_core-0.1.0-py3-none-any.whl",
        "instinctlab_engine_isaacsim-0.1.0-py3-none-any.whl",
        "instinctlab_engine_mjlab-0.1.0-py3-none-any.whl",
        "instinctlab-0.1.0.tar.gz",
        "instinctlab_engine_core-0.1.0.tar.gz",
        "instinctlab_engine_isaacsim-0.1.0.tar.gz",
        "instinctlab_engine_mjlab-0.1.0.tar.gz",
    )
    artifacts = {}
    for name in names:
        path = tmp_path / name
        path.write_bytes(name.encode())
        artifacts[name] = {
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (tmp_path / "SHA256SUMS.json").write_text(
        json.dumps(
            {
                "version": "instinctlab_release_artifacts_v1",
                "package_version": "0.1.0",
                "source_commit": source_commit,
                "source_dirty": False,
                "artifacts": artifacts,
            }
        )
    )

    verifier._verify_release_artifacts(tmp_path, "0.1.0", source_commit)
    (tmp_path / names[0]).write_bytes(b"drift")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verifier._verify_release_artifacts(tmp_path, "0.1.0", source_commit)


def test_release_artifact_verification_rejects_source_drift(tmp_path: Path) -> None:
    verifier = _runtime_verifier()
    (tmp_path / "SHA256SUMS.json").write_text(
        json.dumps(
            {
                "version": "instinctlab_release_artifacts_v1",
                "package_version": "0.1.0",
                "source_commit": "a" * 40,
                "source_dirty": False,
                "artifacts": {},
            }
        )
    )

    with pytest.raises(ValueError, match="source receipt"):
        verifier._verify_release_artifacts(tmp_path, "0.1.0", "b" * 40)


def test_container_import_smoke_is_locked_and_importable() -> None:
    verifier = _runtime_verifier()

    verifier._verify_imports({"packaging": "packaging"}, {"packaging": "23.0"})
    with pytest.raises(RuntimeError, match="no distribution lock"):
        verifier._verify_imports({"missing": "packaging"}, {"packaging": "23.0"})
    with pytest.raises(RuntimeError, match="invalid import name"):
        verifier._verify_imports({"packaging": "bad-name"}, {"packaging": "23.0"})


def test_container_shell_helpers_are_valid_bash() -> None:
    for name in ("docker-compose.sh", "docker-attach.sh", "docker-stop.sh"):
        subprocess.run(
            ["bash", "-n", str(ROOT / "docker" / name)],
            check=True,
            capture_output=True,
            text=True,
        )


def test_operator_helper_builds_clean_release_before_docker() -> None:
    helper = (ROOT / "docker/docker-compose.sh").read_text()

    assert "scripts/build_release.py" in helper
    assert '--output "$repository_dir/dist/release"' in helper
