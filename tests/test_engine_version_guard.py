"""Engine launchers must fail before startup on an unverified SDK version."""

from __future__ import annotations

import pytest

from instinctlab_engine.base import require_supported_version


def test_installed_engine_versions_are_in_the_verified_ranges() -> None:
    from instinctlab.engines.isaacsim import IsaacSimAdapter
    from instinctlab.engines.mjlab import MjlabAdapter

    assert require_supported_version(
        "isaaclab", IsaacSimAdapter.SUPPORTED_VERSIONS, engine=IsaacSimAdapter.name
    ).startswith("0.54.")
    assert require_supported_version("mjlab", MjlabAdapter.SUPPORTED_VERSIONS, engine=MjlabAdapter.name) == "1.5.0"
    for distribution, supported in MjlabAdapter.RUNTIME_VERSIONS.items():
        assert require_supported_version(distribution, supported, engine=MjlabAdapter.name)


def test_unverified_version_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("instinctlab_engine.base.version", lambda _distribution: "9.0.0")
    with pytest.raises(RuntimeError, match="verified with mjlab>=1.5,<1.6"):
        require_supported_version("mjlab", ">=1.5,<1.6", engine="mjlab")


def test_mjlab_bootstrap_checks_the_complete_physics_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    from instinctlab.engines.mjlab import MjlabAdapter

    checked: list[tuple[str, str]] = []

    def record(distribution: str, supported: str, *, engine: str) -> str:
        assert engine == "mjlab"
        checked.append((distribution, supported))
        return supported.removeprefix("==")

    monkeypatch.setattr("instinctlab.engines.mjlab.adapter.require_supported_version", record)

    import sys
    from types import ModuleType

    torch_utils = ModuleType("mjlab.utils.torch")
    torch_utils.configure_torch_backends = lambda: None
    monkeypatch.setitem(sys.modules, "mjlab.utils.torch", torch_utils)

    MjlabAdapter.bootstrap(None)  # type: ignore[arg-type]

    assert checked == [
        ("mjlab", "==1.5.0"),
        ("mujoco", "==3.10.0"),
        ("mujoco-warp", "==3.10.0.1"),
        ("warp-lang", "==1.14.0"),
    ]
