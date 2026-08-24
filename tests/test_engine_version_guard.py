"""Engine launchers must fail before startup on an unverified SDK version."""

from __future__ import annotations

import pytest

from instinctlab.engines.base import require_supported_version


def test_installed_engine_versions_are_in_the_verified_ranges() -> None:
    from instinctlab.engines.isaacsim import IsaacSimAdapter
    from instinctlab.engines.mjlab import MjlabAdapter

    assert require_supported_version(
        "isaaclab", IsaacSimAdapter.SUPPORTED_VERSIONS, engine=IsaacSimAdapter.name
    ).startswith("0.54.")
    assert require_supported_version("mjlab", MjlabAdapter.SUPPORTED_VERSIONS, engine=MjlabAdapter.name).startswith(
        "1.5."
    )


def test_unverified_version_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("instinctlab.engines.base.version", lambda _distribution: "9.0.0")
    with pytest.raises(RuntimeError, match="verified with mjlab>=1.5,<1.6"):
        require_supported_version("mjlab", ">=1.5,<1.6", engine="mjlab")
