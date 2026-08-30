"""Isaac Sim backend.

Importing this module imports nothing from ``isaaclab``: the registry's decorators run at import
time so the engine's capabilities are known, while every builder body defers its imports. That is
what lets a task be checked against this engine on a machine that cannot run it.
"""

from instinctlab_engine.bridge import entity as _entity
from instinctlab_engine import register_adapter, register_sub_terrain, register_terrain

register_terrain(
    "isaacsim", "plane", "instinctlab_engine_isaacsim.terrain_builders:build_plane"
)
register_terrain(
    "isaacsim",
    "generator",
    "instinctlab_engine_isaacsim.terrain_builders:build_generator",
)
register_terrain(
    "isaacsim", "rough", "instinctlab_engine_isaacsim.terrain_builders:build_rough"
)
register_terrain(
    "isaacsim",
    "motion_matched",
    "instinctlab_engine_isaacsim.terrain_builders:build_motion_matched",
)

register_sub_terrain(
    "isaacsim",
    "pyramid_stairs",
    "instinctlab_engine_isaacsim.terrain_builders:build_standard_tile",
)
register_sub_terrain(
    "isaacsim",
    "pyramid_stairs_inv",
    "instinctlab_engine_isaacsim.terrain_builders:build_standard_tile",
)
register_sub_terrain(
    "isaacsim",
    "boxes",
    "instinctlab_engine_isaacsim.terrain_builders:build_standard_tile",
)
register_sub_terrain(
    "isaacsim",
    "random_rough",
    "instinctlab_engine_isaacsim.terrain_builders:build_standard_tile",
)
register_sub_terrain(
    "isaacsim",
    "hf_pyramid_slope",
    "instinctlab_engine_isaacsim.terrain_builders:build_standard_tile",
)
register_sub_terrain(
    "isaacsim",
    "hf_pyramid_slope_inv",
    "instinctlab_engine_isaacsim.terrain_builders:build_standard_tile",
)
register_sub_terrain(
    "isaacsim",
    "perlin_plane",
    "instinctlab_engine_isaacsim.terrain_builders:build_rough_tile",
)
register_sub_terrain(
    "isaacsim",
    "perlin_square_gap",
    "instinctlab_engine_isaacsim.terrain_builders:build_rough_tile",
)
register_sub_terrain(
    "isaacsim",
    "perlin_pyramid_stairs",
    "instinctlab_engine_isaacsim.terrain_builders:build_rough_tile",
)
register_sub_terrain(
    "isaacsim",
    "perlin_pyramid_stairs_inv",
    "instinctlab_engine_isaacsim.terrain_builders:build_rough_tile",
)
register_sub_terrain(
    "isaacsim",
    "perlin_discrete_obstacles",
    "instinctlab_engine_isaacsim.terrain_builders:build_rough_tile",
)
register_sub_terrain(
    "isaacsim",
    "perlin_random_multi_box",
    "instinctlab_engine_isaacsim.terrain_builders:build_rough_tile",
)
register_sub_terrain(
    "isaacsim",
    "perlin_pyramid_slope_inv",
    "instinctlab_engine_isaacsim.terrain_builders:build_rough_tile",
)

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


def register() -> None:
    """Register this backend through the engine-core plugin interface."""
    register_adapter(
        "isaacsim",
        "instinctlab_engine_isaacsim.adapter:IsaacSimAdapter",
    )


register()

__all__ = ["TERMS", "IsaacSimAdapter", "IsaacSimCompileCtx", "register"]
