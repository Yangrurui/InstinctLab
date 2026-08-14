from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from instinctlab.assets import ASSETS
from instinctlab.assets.registry import AssetRegistration, AssetRegistry
from instinctlab.sim.robot_spec import BackendAsset


def test_registry_resolves_g1_by_name_and_asset_id() -> None:
    by_name = ASSETS.make("unitree_g1_29dof")
    by_id = ASSETS.make_by_asset_id("popsicle_torsobase_v1")

    assert by_name.name == "unitree_g1_29dof"
    assert by_name.asset_id == "popsicle_torsobase_v1"
    assert by_name.joint_names == by_id.joint_names


def test_registry_rejects_unknown_lookups() -> None:
    with pytest.raises(KeyError):
        ASSETS.make("does_not_exist")
    with pytest.raises(KeyError):
        ASSETS.make_by_asset_id("does_not_exist")


def test_registry_rejects_conflicting_asset_id() -> None:
    registry = AssetRegistry()
    registry.register(
        AssetRegistration("robot_a", "shared_id", "instinctlab.assets.unitree_g1:make_g1_29dof_robot_spec")
    )
    with pytest.raises(ValueError, match="asset_id"):
        registry.register(
            AssetRegistration("robot_b", "shared_id", "instinctlab.assets.unitree_g1:make_g1_29dof_robot_spec")
        )


def test_backend_asset_checksum_roundtrip(tmp_path: Path) -> None:
    asset_file = tmp_path / "robot.xml"
    asset_file.write_bytes(b"<mujoco/>")
    digest = hashlib.sha256(b"<mujoco/>").hexdigest()

    asset = BackendAsset(backend="mjlab", path=str(asset_file), checksum=digest)
    asset.verify()  # matching digest is a no-op

    assert asset.compute_checksum() == digest


def test_backend_asset_checksum_mismatch_fails(tmp_path: Path) -> None:
    asset_file = tmp_path / "robot.xml"
    asset_file.write_bytes(b"<mujoco/>")

    asset = BackendAsset(backend="mjlab", path=str(asset_file), checksum="0" * 64)
    with pytest.raises(ValueError, match="checksum mismatch"):
        asset.verify()


def test_backend_asset_missing_pinned_file_fails(tmp_path: Path) -> None:
    asset = BackendAsset(backend="mjlab", path=str(tmp_path / "absent.xml"), checksum="0" * 64)
    with pytest.raises(FileNotFoundError):
        asset.verify()


def test_backend_asset_unpinned_verify_is_noop(tmp_path: Path) -> None:
    asset = BackendAsset(backend="mjlab", path=str(tmp_path / "absent.xml"), checksum=None)
    asset.verify()  # no checksum declared -> nothing to verify
