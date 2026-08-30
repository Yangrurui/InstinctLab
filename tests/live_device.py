"""Which GPU the live tests are allowed to build on.

Repo-wide, not parkour-specific: a training run normally occupies ``cuda:0``,
so a live test that hardcodes it steals the card out from under the run. The
default is deliberately not ``cuda:0`` and there is no fallback to it -- a
machine with fewer cards should fail loudly rather than quietly take card 0.

Import-safe: no engine, no torch. Isaac Sim live tests import this before Kit
starts.
"""

from __future__ import annotations

import os

_DEFAULT_LIVE_DEVICE = "cuda:2"


def resolve_live_device() -> str:
    """GPU the live tests will construct on. Import-safe (no torch)."""
    return os.environ.get("INSTINCTLAB_LIVE_DEVICE", _DEFAULT_LIVE_DEVICE)
