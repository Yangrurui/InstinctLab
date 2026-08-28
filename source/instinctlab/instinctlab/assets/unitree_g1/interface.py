"""Engine-neutral routing into one Unitree G1 native configuration.

This module owns no robot, model, or actuator values. It only resolves an
explicit engine and variant to the corresponding native module.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

__all__ = ["native_module"]


def native_module(engine: str, variant: str) -> tuple[ModuleType, str]:
    """Forward one explicit engine and variant to its native G1 module."""
    return import_module(f"{__package__}.{engine}"), variant
