"""Shared inputs for opt-in shadowing simulator tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_DEFAULT_MOTION = Path(
    "/root/Datasets/parkour_release/parkour_motion_reference/parkour_motion_without_run_retargetted.npz"
)


def resolve_shadowing_motion() -> Path:
    path = Path(os.environ.get("INSTINCTLAB_SHADOWING_LIVE_MOTION", _DEFAULT_MOTION)).expanduser()
    if not path.is_file():
        pytest.skip("shadowing live motion is unavailable; set INSTINCTLAB_SHADOWING_LIVE_MOTION")
    return path.resolve()


__all__ = ["resolve_shadowing_motion"]
