"""The installer and runtime guard must describe the same MJLab plant."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import tomllib

ROOT = Path(__file__).resolve().parents[1]
PINS_PATH = ROOT / "source" / "instinctlab" / "config" / "backend_pins.toml"


def _pins() -> dict:
    with PINS_PATH.open("rb") as stream:
        return tomllib.load(stream)


def _install_module():
    path = ROOT / "scripts" / "install.py"
    spec = importlib.util.spec_from_file_location("instinctlab_install", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mjlab_physics_stack_is_exactly_pinned() -> None:
    from instinctlab_engine_mjlab import MjlabAdapter

    cfg = _pins()["mjlab"]
    assert cfg["pypi"] == "mjlab==1.5.0"
    assert cfg["runtime"] == ["mujoco==3.10.0", "mujoco-warp==3.10.0.1", "warp-lang==1.14.0"]
    assert MjlabAdapter.SUPPORTED_VERSIONS == "==1.5.0"
    assert [f"{name}{version}" for name, version in MjlabAdapter.RUNTIME_VERSIONS.items()] == cfg["runtime"]


def test_installer_pins_runtime_before_editable_mjlab(monkeypatch, tmp_path: Path) -> None:
    install = _install_module()
    calls: list[tuple] = []
    monkeypatch.setattr(
        install,
        "_ensure_checkout",
        lambda *args, **kwargs: calls.append(("checkout", *args, kwargs)) or {},
    )
    monkeypatch.setattr(install, "_pip", lambda *args: calls.append(("pip", *args)))
    monkeypatch.setattr(install, "_pip_editable", lambda *args, **kwargs: calls.append(("editable", *args, kwargs)))

    install._install_mjlab(
        tmp_path,
        _pins(),
        force=False,
        allow_unverified=False,
    )

    assert calls[1] == ("pip", "mujoco==3.10.0", "mujoco-warp==3.10.0.1", "warp-lang==1.14.0")
    assert calls[2][0:2] == ("editable", "mjlab")


def test_existing_wrong_revision_is_refused_by_default(monkeypatch, tmp_path: Path) -> None:
    install = _install_module()
    monkeypatch.setattr(install, "_git_rev_parse", lambda repo, revision: "expected")
    monkeypatch.setattr(install, "_git_head", lambda repo: "actual")
    monkeypatch.setattr(install, "_git_dirty", lambda repo: False)

    with pytest.raises(RuntimeError, match="Refusing unverified dependency checkout"):
        install._ensure_checkout(
            "https://example.invalid/dependency.git",
            tmp_path,
            "pinned",
            allow_unverified=False,
        )


def test_existing_dirty_checkout_is_refused_by_default(monkeypatch, tmp_path: Path) -> None:
    install = _install_module()
    monkeypatch.setattr(install, "_git_rev_parse", lambda repo, revision: "same")
    monkeypatch.setattr(install, "_git_head", lambda repo: "same")
    monkeypatch.setattr(install, "_git_dirty", lambda repo: True)

    with pytest.raises(RuntimeError, match="uncommitted or untracked changes"):
        install._ensure_checkout(
            "https://example.invalid/dependency.git",
            tmp_path,
            "pinned",
            allow_unverified=False,
        )


def test_explicit_checkout_override_is_recorded(monkeypatch, tmp_path: Path) -> None:
    install = _install_module()
    monkeypatch.setattr(install, "_git_rev_parse", lambda repo, revision: "expected")
    monkeypatch.setattr(install, "_git_head", lambda repo: "actual")
    monkeypatch.setattr(install, "_git_dirty", lambda repo: True)
    report = install._ensure_checkout(
        "https://example.invalid/dependency.git",
        tmp_path,
        "pinned",
        allow_unverified=True,
    )
    output = tmp_path / "install.json"

    install._write_install_provenance(
        output,
        checkouts={"dependency": report},
        allow_unverified=True,
    )
    payload = json.loads(output.read_text())

    assert report["override_used"] is True
    assert report["actual_commit"] == "actual"
    assert report["dirty"] is True
    assert payload["allow_unverified_checkouts"] is True
    assert payload["checkouts"]["dependency"] == report
