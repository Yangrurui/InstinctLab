"""SDK-free registration of InstinctLab's rough generated-terrain tiles."""

from __future__ import annotations

_MULTIPLEXED_ROUGH_TILE_KINDS = (
    "perlin_plane",
    "perlin_square_gap",
    "perlin_pyramid_stairs",
    "perlin_pyramid_stairs_inv",
    "perlin_discrete_obstacles",
    "perlin_random_multi_box",
    "perlin_pyramid_slope_inv",
)
PERLIN_WAVE_KIND = "perlin_wave"
ROUGH_TILE_KINDS = (*_MULTIPLEXED_ROUGH_TILE_KINDS, PERLIN_WAVE_KIND)


def register_terrains(registry) -> None:
    """Register both native lowerings without importing either engine SDK."""
    for kind in _MULTIPLEXED_ROUGH_TILE_KINDS:
        registry.register_sub_terrain(
            "isaacsim",
            kind,
            "instinctlab_engine_isaacsim.terrain_builders:build_rough_tile",
        )
        registry.register_sub_terrain(
            "mjlab",
            kind,
            "instinctlab_engine_mjlab.terrain_builders:build_rough_tile",
        )
    registry.register_sub_terrain(
        "isaacsim",
        PERLIN_WAVE_KIND,
        "instinctlab_engine_isaacsim.terrain_builders:build_perlin_wave_tile",
    )
    registry.register_sub_terrain(
        "mjlab",
        PERLIN_WAVE_KIND,
        "instinctlab_engine_mjlab.terrain_builders:build_perlin_wave_tile",
    )


register_terrains.instinctlab_engine_api = ">=0.1,<0.2"

__all__ = ["PERLIN_WAVE_KIND", "ROUGH_TILE_KINDS", "register_terrains"]
