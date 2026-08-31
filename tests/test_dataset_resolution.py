"""Portable dataset roots replace machine-specific task paths."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from instinctlab.tasks import registry
from instinctlab_engine.data import resolve_data_path

from tests.task_specs import task_spec

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "datasets/manifest.json"


def test_dataset_uri_resolves_under_an_isolated_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "mounted-data"
    clip = root / "collection" / "motion" / "clip.npz"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"portable clip")
    monkeypatch.setenv("INSTINCTLAB_DATA_ROOT", str(root))

    assert resolve_data_path("dataset://collection/motion/clip.npz") == clip
    assert resolve_data_path("metadata.yaml", relative_to=clip.parent) == (
        clip.parent / "metadata.yaml"
    )


@pytest.mark.parametrize(
    "uri",
    (
        "dataset:///missing-collection",
        "dataset://collection/../escape",
        "dataset://collection/%2e%2e/escape",
        "dataset://..%2fescape/file",
        "dataset://%2ftmp/escape",
        "dataset://collection%5cescape/file",
        "dataset://user:password@collection/file",
        "dataset://collection/file?variant=1",
    ),
)
def test_dataset_uri_rejects_ambiguous_or_traversing_paths(uri: str) -> None:
    with pytest.raises(ValueError):
        resolve_data_path(uri)


def test_dataset_uri_rejects_a_symlink_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "mounted-data"
    collection = root / "collection"
    outside = tmp_path / "outside"
    collection.mkdir(parents=True)
    outside.mkdir()
    (collection / "linked").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("INSTINCTLAB_DATA_ROOT", str(root))

    with pytest.raises(ValueError, match="outside"):
        resolve_data_path("dataset://collection/linked/clip.npz")


def test_manifest_verifier_runs_without_home_directory_links(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    dataset = data_root / "fixture" / "v1"
    dataset.mkdir(parents=True)
    resource = dataset / "motion.npz"
    resource.write_bytes(b"isolated dataset")
    checksum = hashlib.sha256(resource.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "instinctlab_dataset_manifest_v1",
                "datasets": {
                    "fixture": {
                        "uri": "dataset://fixture/v1",
                        "required": True,
                        "status": "test",
                        "resources": {"motion.npz": checksum},
                    }
                },
            }
        )
    )
    receipt = tmp_path / "receipt.json"
    environment = {
        **os.environ,
        "INSTINCTLAB_DATA_ROOT": str(data_root),
        "PYTHONPATH": str(REPO / "source/instinctlab_engine/src"),
    }

    subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/verify_datasets.py"),
            "--manifest",
            str(manifest),
            "--receipt",
            str(receipt),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(receipt.read_text())
    assert payload["schema_version"] == "instinctlab_dataset_verification_v1"
    assert payload["datasets"][0]["declared"] == "dataset://fixture/v1"
    assert payload["datasets"][0]["resolved"] == str(dataset)
    assert payload["datasets"][0]["resources"][0]["sha256"] == checksum


def test_manifest_verifier_rejects_a_resource_outside_the_dataset(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    dataset = data_root / "fixture"
    dataset.mkdir(parents=True)
    outside = data_root / "outside.bin"
    outside.write_bytes(b"outside")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "instinctlab_dataset_manifest_v1",
                "datasets": {
                    "fixture": {
                        "uri": "dataset://fixture",
                        "required": True,
                        "resources": {
                            "../outside.bin": hashlib.sha256(
                                outside.read_bytes()
                            ).hexdigest()
                        },
                    }
                },
            }
        )
    )
    environment = {
        **os.environ,
        "INSTINCTLAB_DATA_ROOT": str(data_root),
        "PYTHONPATH": str(REPO / "source/instinctlab_engine/src"),
    }

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/verify_datasets.py"),
            "--manifest",
            str(manifest),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "escapes dataset root" in completed.stderr


def test_active_task_resources_use_manifested_dataset_uris() -> None:
    manifest = json.loads(MANIFEST.read_text())
    roots = tuple(
        declaration["uri"].rstrip("/")
        for declaration in manifest["datasets"].values()
    )
    declared: list[str] = []

    for task_id in registry.ids():
        task = task_spec(task_id)
        for reference in task.scene.motion_references:
            declared.append(reference.clip)
            declared.extend(reference.engine_clips.values())
        terrain = task.scene.terrain
        engine_paths = terrain.params.get("engine_paths", {})
        declared.extend(engine_paths.values())
        for rigid_object in task.scene.rigid_objects:
            declared.append(rigid_object.mesh)
            declared.extend(rigid_object.engine_meshes.values())

    assert declared
    for path in declared:
        assert path.startswith("dataset://"), path
        assert any(path == root or path.startswith(f"{root}/") for root in roots), path


def test_repository_manifest_has_valid_checksum_declarations() -> None:
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["schema_version"] == "instinctlab_dataset_manifest_v1"
    for name, declaration in manifest["datasets"].items():
        assert declaration["uri"].startswith("dataset://"), name
        for relative, checksum in declaration["resources"].items():
            assert relative and not Path(relative).is_absolute(), (name, relative)
            assert len(checksum) == 64
            int(checksum, 16)
