"""Robot assets. The G1 catalog is ``instinctlab.assets.unitree_g1``.

Robots are looked up by name through the engine-neutral :data:`ASSETS`
registry so task configs stay decoupled from concrete factory functions.
"""

from .registry import ASSETS, AssetRegistration, AssetRegistry

__all__ = ["ASSETS", "AssetRegistration", "AssetRegistry"]
