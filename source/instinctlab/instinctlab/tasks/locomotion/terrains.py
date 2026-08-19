"""Engine-free terrain intent shared by the locomotion tasks.

``rough`` is a name, not a recipe. Isaac Sim fills it from main's parkour grid; mjlab fills it
from InstinctMJ's. The two already disagree on scale and on which extra tile they carry, so the
numbers do not live here.
"""

from __future__ import annotations

from instinctlab.spec import TerrainSpec

__all__ = ["rough_terrain"]


def rough_terrain() -> TerrainSpec:
    """Reference rough ground for ``Instinct-Velocity-Rough-G1``.

    The declaration stops at the kind. Each adapter's default *is* that engine's reference.
    """
    return TerrainSpec(kind="rough")
