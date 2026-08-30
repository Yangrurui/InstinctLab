"""Native Unitree G1 configurations with an engine-neutral router."""

from instinctlab_engine.assets import register_asset
from .interface import native_module

register_asset("unitree_g1", native_module)

__all__ = ["native_module"]
