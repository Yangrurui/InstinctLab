"""Lower the shared rough-terrain contract to Isaac Lab terrain configs.

The task-owned recipe lives in :mod:`instinctlab.tasks.terrain`. Every function
here constructs an Isaac SDK config or selects an Isaac terrain class.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from instinctlab.spec.task import SubTerrainSpec, TerrainGeneratorSpec, TerrainSpec

__all__ = ["rough_generator_cfg", "rough_importer_cfg"]


def _flat_patches(value: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    from isaaclab.terrains import FlatPatchSamplingCfg

    return {name: FlatPatchSamplingCfg(**dict(params)) for name, params in value.items()}


def _perlin(value: Mapping[str, Any]) -> Any:
    import instinctlab.terrains as terrain_gen

    return terrain_gen.PerlinPlaneTerrainCfg(**dict(value))


def _sub_terrain(tile: SubTerrainSpec) -> Any:
    """Translate one engine-neutral rough tile to its Isaac implementation."""
    import instinctlab.terrains as terrain_gen

    fields = dict(tile.params)
    if flat_patches := fields.get("flat_patch_sampling"):
        fields["flat_patch_sampling"] = _flat_patches(flat_patches)
    if perlin_cfg := fields.get("perlin_cfg"):
        fields["perlin_cfg"] = _perlin(perlin_cfg)

    classes = {
        "perlin_plane": terrain_gen.PerlinPlaneTerrainCfg,
        "perlin_square_gap": terrain_gen.PerlinSquareGapTerrainCfg,
        "perlin_pyramid_stairs": terrain_gen.PerlinPyramidStairsTerrainCfg,
        "perlin_pyramid_stairs_inv": terrain_gen.PerlinInvertedPyramidStairsTerrainCfg,
        "perlin_discrete_obstacles": terrain_gen.PerlinDiscreteObstaclesTerrainCfg,
        "perlin_random_multi_box": terrain_gen.PerlinMeshRandomMultiBoxTerrainCfg,
        "perlin_pyramid_slope_inv": terrain_gen.PerlinInvertedPyramidSlopedTerrainCfg,
    }
    try:
        cls = classes[tile.kind]
    except KeyError:
        raise NotImplementedError(
            f"The Isaac rough-terrain bridge has no tile {tile.kind!r}; it builds {sorted(classes)}."
        ) from None
    return cls(proportion=tile.proportion, **fields)


def rough_generator_cfg(spec: TerrainGeneratorSpec) -> Any:
    """Compile the shared recipe to Isaac Lab's native generator config."""
    from isaaclab.terrains import TerrainGeneratorCfg

    return TerrainGeneratorCfg(
        seed=spec.seed,
        size=spec.size,
        border_width=spec.border_width,
        num_rows=spec.num_rows,
        num_cols=spec.num_cols,
        horizontal_scale=spec.horizontal_scale,
        vertical_scale=spec.vertical_scale,
        slope_threshold=spec.slope_threshold,
        use_cache=False,
        curriculum=spec.curriculum,
        sub_terrains={name: _sub_terrain(tile) for name, tile in spec.sub_terrains.items()},
    )


def rough_importer_cfg(spec: TerrainSpec) -> Any:
    """Build the native rough importer while preserving shared material values."""
    if spec.generator is None:
        raise ValueError("kind='rough' needs a TerrainGeneratorSpec.")

    from isaaclab.sim import MdlFileCfg, RigidBodyMaterialCfg
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

    from instinctlab.terrains import TerrainImporterCfg

    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=rough_generator_cfg(spec.generator),
        max_init_terrain_level=spec.generator.max_init_level,
        collision_group=-1,
        physics_material=RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=spec.static_friction,
            dynamic_friction=spec.dynamic_friction,
            restitution=spec.restitution,
        ),
        visual_material=MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
