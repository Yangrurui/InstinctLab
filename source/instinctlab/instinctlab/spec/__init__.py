"""Engine-agnostic declaration layer.

Nothing here may import a physics engine, directly or transitively. A task is declared once in
these types and compiled to a native environment by the backend for whichever engine is running,
so anything that reaches for an engine at this level has already lost the property that makes the
declaration portable. ``tests/test_spec_isolation.py`` enforces this.
"""

from __future__ import annotations

from .entity import UNIVERSAL_KINDS, EntityRef

__all__ = ["UNIVERSAL_KINDS", "EntityRef"]
