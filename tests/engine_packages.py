"""Repository paths for the independently packaged engine backends."""

from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ISAACSIM_ENGINE = REPO / "source/instinctlab_engine_isaacsim/src/instinctlab_engine_isaacsim"
MJLAB_ENGINE = REPO / "source/instinctlab_engine_mjlab/src/instinctlab_engine_mjlab"
ENGINE_ROOTS = {
    "isaacsim": ISAACSIM_ENGINE,
    "mjlab": MJLAB_ENGINE,
}
