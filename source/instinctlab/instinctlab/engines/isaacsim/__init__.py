"""Isaac Sim backend.

Importing this module imports nothing from ``isaaclab``: the registry's decorators run at import
time so the engine's capabilities are known, while every builder body defers its imports. That is
what lets a task be checked against this engine on a machine that cannot run it.
"""

from instinctlab.compat import entity as _entity

from .adapter import IsaacSimAdapter, IsaacSimCompileCtx
from .terms import TERMS

# Decision S2: see the note in the mjlab package. ``fixed_tendon`` stays distinct from mjlab's
# ``tendon`` because the two are not known to select the same elements.
_entity.register(
    "isaacsim",
    kinds=("joint", "body", "fixed_tendon", "object_collection"),
    cfg=("isaaclab.managers", "SceneEntityCfg"),
    container=list,
)

__all__ = ["TERMS", "IsaacSimAdapter", "IsaacSimCompileCtx"]
