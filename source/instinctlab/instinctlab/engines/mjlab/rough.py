"""Lower the shared rough-terrain contract to MJLab terrain configs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from instinctlab.spec.task import SubTerrainSpec, TerrainGeneratorSpec, TerrainSpec

__all__ = ["rough_generator_cfg", "rough_importer_cfg"]


def _flat_patches(value: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    from mjlab.terrains import FlatPatchSamplingCfg

    return {name: FlatPatchSamplingCfg(**dict(params)) for name, params in value.items()}


def _perlin(value: Mapping[str, Any]) -> Any:
    from .terrains.height_field.hf_terrains_cfg import PerlinPlaneTerrainCfg

    return PerlinPlaneTerrainCfg(**dict(value))


def _sub_terrain(tile: SubTerrainSpec) -> Any:
    """Translate one engine-neutral rough tile to its MJLab implementation."""
    from .terrains.height_field.hf_terrains_cfg import (
        PerlinDiscreteObstaclesTerrainCfg,
        PerlinInvertedPyramidSlopedTerrainCfg,
        PerlinInvertedPyramidStairsTerrainCfg,
        PerlinPlaneTerrainCfg,
        PerlinPyramidStairsTerrainCfg,
        PerlinSquareGapTerrainCfg,
    )
    from .terrains.mesh_terrains_cfg import PerlinMeshRandomMultiBoxTerrainCfg

    fields = dict(tile.params)
    if flat_patches := fields.get("flat_patch_sampling"):
        fields["flat_patch_sampling"] = _flat_patches(flat_patches)
    if perlin_cfg := fields.get("perlin_cfg"):
        fields["perlin_cfg"] = _perlin(perlin_cfg)

    classes = {
        "perlin_plane": PerlinPlaneTerrainCfg,
        "perlin_square_gap": PerlinSquareGapTerrainCfg,
        "perlin_pyramid_stairs": PerlinPyramidStairsTerrainCfg,
        "perlin_pyramid_stairs_inv": PerlinInvertedPyramidStairsTerrainCfg,
        "perlin_discrete_obstacles": PerlinDiscreteObstaclesTerrainCfg,
        "perlin_random_multi_box": PerlinMeshRandomMultiBoxTerrainCfg,
        "perlin_pyramid_slope_inv": PerlinInvertedPyramidSlopedTerrainCfg,
    }
    try:
        cls = classes[tile.kind]
    except KeyError:
        raise NotImplementedError(
            f"The MJLab rough-terrain bridge has no tile {tile.kind!r}; it builds {sorted(classes)}."
        ) from None
    return cls(proportion=tile.proportion, **fields)


def rough_generator_cfg(spec: TerrainGeneratorSpec) -> Any:
    """Compile the shared recipe to the native filed generator config."""
    from .terrains.terrain_generator_cfg import FiledTerrainGeneratorCfg

    return FiledTerrainGeneratorCfg(
        seed=spec.seed,
        size=spec.size,
        border_width=spec.border_width,
        num_rows=spec.num_rows,
        num_cols=spec.num_cols,
        horizontal_scale=spec.horizontal_scale,
        vertical_scale=spec.vertical_scale,
        slope_threshold=spec.slope_threshold,
        curriculum=spec.curriculum,
        add_lights=True,
        sub_terrains={name: _sub_terrain(tile) for name, tile in spec.sub_terrains.items()},
    )


def rough_importer_cfg(spec: TerrainSpec) -> Any:
    """Build the MJLab rough importer from the shared semantic recipe."""
    if spec.generator is None:
        raise ValueError("kind='rough' needs a TerrainGeneratorSpec.")

    from .terrains.terrain_importer_cfg import TerrainImporterCfg

    return TerrainImporterCfg(
        terrain_type="hacked_generator",
        terrain_generator=rough_generator_cfg(spec.generator),
        max_init_terrain_level=spec.generator.max_init_level,
        sliding_friction=spec.dynamic_friction,
    )
