"""The installer and runtime guard must describe the same MJLab plant."""

from __future__ import annotations

import importlib.util
from pathlib import Path

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
    monkeypatch.setattr(install, "_ensure_checkout", lambda *args: calls.append(("checkout", *args)))
    monkeypatch.setattr(install, "_pip", lambda *args: calls.append(("pip", *args)))
    monkeypatch.setattr(install, "_pip_editable", lambda *args, **kwargs: calls.append(("editable", *args, kwargs)))

    install._install_mjlab(tmp_path, _pins(), force=False)

    assert calls[1] == ("pip", "mujoco==3.10.0", "mujoco-warp==3.10.0.1", "warp-lang==1.14.0")
    assert calls[2][0:2] == ("editable", "mjlab")
