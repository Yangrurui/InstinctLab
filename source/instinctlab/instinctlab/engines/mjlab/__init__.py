"""mjlab backend.

Importing this module imports nothing from ``mjlab``, the same property the Isaac Sim backend has
and for the same reason: the registry's keys have to exist for ``contract_report`` to answer, while
the builders' bodies must not, so a task can be checked against this engine anywhere.
"""

from instinctlab.compat import entity as _entity
from instinctlab.engines import register_sub_terrain, register_terrain

register_terrain(
    "mjlab", "plane", "instinctlab.engines.mjlab.terrain_builders:build_plane"
)
register_terrain(
    "mjlab",
    "generator",
    "instinctlab.engines.mjlab.terrain_builders:build_generator",
)
register_terrain(
    "mjlab", "rough", "instinctlab.engines.mjlab.terrain_builders:build_rough"
)
register_terrain(
    "mjlab",
    "motion_matched",
    "instinctlab.engines.mjlab.terrain_builders:build_motion_matched",
)

register_sub_terrain(
    "mjlab",
    "pyramid_stairs",
    "instinctlab.engines.mjlab.terrain_builders:build_standard_tile",
)
register_sub_terrain(
    "mjlab",
    "pyramid_stairs_inv",
    "instinctlab.engines.mjlab.terrain_builders:build_standard_tile",
)
register_sub_terrain(
    "mjlab",
    "boxes",
    "instinctlab.engines.mjlab.terrain_builders:build_standard_tile",
)
register_sub_terrain(
    "mjlab",
    "random_rough",
    "instinctlab.engines.mjlab.terrain_builders:build_standard_tile",
)
register_sub_terrain(
    "mjlab",
    "hf_pyramid_slope",
    "instinctlab.engines.mjlab.terrain_builders:build_standard_tile",
)
register_sub_terrain(
    "mjlab",
    "hf_pyramid_slope_inv",
    "instinctlab.engines.mjlab.terrain_builders:build_standard_tile",
)
register_sub_terrain(
    "mjlab",
    "perlin_plane",
    "instinctlab.engines.mjlab.terrain_builders:build_rough_tile",
)
register_sub_terrain(
    "mjlab",
    "perlin_square_gap",
    "instinctlab.engines.mjlab.terrain_builders:build_rough_tile",
)
register_sub_terrain(
    "mjlab",
    "perlin_pyramid_stairs",
    "instinctlab.engines.mjlab.terrain_builders:build_rough_tile",
)
register_sub_terrain(
    "mjlab",
    "perlin_pyramid_stairs_inv",
    "instinctlab.engines.mjlab.terrain_builders:build_rough_tile",
)
register_sub_terrain(
    "mjlab",
    "perlin_discrete_obstacles",
    "instinctlab.engines.mjlab.terrain_builders:build_rough_tile",
)
register_sub_terrain(
    "mjlab",
    "perlin_random_multi_box",
    "instinctlab.engines.mjlab.terrain_builders:build_rough_tile",
)
register_sub_terrain(
    "mjlab",
    "perlin_pyramid_slope_inv",
    "instinctlab.engines.mjlab.terrain_builders:build_rough_tile",
)

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
