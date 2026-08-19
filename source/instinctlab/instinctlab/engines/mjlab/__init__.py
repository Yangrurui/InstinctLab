"""mjlab backend.

Importing this module imports nothing from ``mjlab``, the same property the Isaac Sim backend has
and for the same reason: the registry's keys have to exist for ``contract_report`` to answer, while
the builders' bodies must not, so a task can be checked against this engine anywhere.
"""

from instinctlab.compat import entity as _entity

from .adapter import MjlabAdapter, MjlabCompileCtx
from .terms import TERMS

# Decision S2: what this engine can select is declared here rather than tabulated in the shared
# layer, so an engine with selectors nobody anticipated costs a call in its own package. mjlab's
# ``tendon`` is deliberately not registered as Isaac Lab's ``fixed_tendon``; they are not known to
# select the same elements.
_entity.register(
    "mjlab",
    kinds=("joint", "body", "geom", "site", "actuator", "tendon", "camera", "light", "material", "pair"),
    cfg=("mjlab.managers.scene_entity_config", "SceneEntityCfg"),
    container=tuple,
)

__all__ = ["TERMS", "MjlabAdapter", "MjlabCompileCtx"]
