"""mjlab's reference rough: InstinctMJ's parkour ``ROUGH_TERRAINS_CFG``.

The numbers live here, not in the task. InstinctMJ is a reference, not a dependency; the
generator and importer classes are the vendored copy under :mod:`.terrains`.
"""

from __future__ import annotations

from typing import Any

from instinctlab.spec.task import TerrainSpec

__all__ = ["rough_generator_cfg", "rough_importer_cfg"]


def _walls() -> dict[str, Any]:
    return {"wall_prob": [0.3, 0.3, 0.3, 0.3], "wall_height": 5.0, "wall_thickness": 0.05}


def rough_generator_cfg() -> Any:
    """InstinctMJ parkour's Perlin grid. Imports stay here so the module stays engine-free at rest."""
    from mjlab.terrains import FlatPatchSamplingCfg

    from .terrains.height_field.hf_terrains_cfg import (
        PerlinDiscreteObstaclesTerrainCfg,
        PerlinInvertedPyramidSlopedTerrainCfg,
        PerlinInvertedPyramidStairsTerrainCfg,
        PerlinPlaneTerrainCfg,
        PerlinPyramidStairsTerrainCfg,
        PerlinSquareGapTerrainCfg,
    )
    from .terrains.terrain_generator_cfg import FiledTerrainGeneratorCfg

    def target() -> dict[str, Any]:
        return {
            "target": FlatPatchSamplingCfg(num_patches=50, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.05)
        }

    def target_center() -> dict[str, Any]:
        return {
            "target": FlatPatchSamplingCfg(
                num_patches=50,
                patch_radius=[0.05, 0.10, 0.15, 0.20],
                max_height_diff=0.05,
                x_range=(3.7, 3.7),
                y_range=(-0.0, 0.0),
            )
        }

    def perlin() -> Any:
        return PerlinPlaneTerrainCfg(
            noise_scale=0.05,
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
        )

    return FiledTerrainGeneratorCfg(
        seed=0,
        size=(8.0, 8.0),
        border_width=3.0,
        num_rows=10,
        num_cols=20,
        horizontal_scale=0.07,
        vertical_scale=0.005,
        slope_threshold=1.0,
        curriculum=True,
        add_lights=True,
        sub_terrains={
            "perlin_rough": PerlinPlaneTerrainCfg(
                proportion=0.05,
                noise_scale=[0.0, 0.1],
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
                border_width=1.0,
                flat_patch_sampling=target(),
                **_walls(),
            ),
            "perlin_rough_stand": PerlinPlaneTerrainCfg(
                proportion=0.05,
                noise_scale=[0.0, 0.1],
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
                border_width=1.0,
                flat_patch_sampling=target(),
                **_walls(),
            ),
            "square_gaps": PerlinSquareGapTerrainCfg(
                proportion=0.10,
                gap_distance_range=(0.1, 0.7),
                gap_depth=(0.4, 0.6),
                platform_width=2.5,
                border_width=1.0,
                flat_patch_sampling=target_center(),
                **_walls(),
            ),
            "pyramid_stairs": PerlinPyramidStairsTerrainCfg(
                proportion=0.15,
                step_height_range=(0.05, 0.23),
                step_width=0.35,
                platform_width=2.5,
                border_width=1.0,
                perlin_cfg=perlin(),
                flat_patch_sampling=target_center(),
                **_walls(),
            ),
            "pyramid_stairs_high": PerlinPyramidStairsTerrainCfg(
                proportion=0.10,
                step_height_range=(0.05, 0.45),
                step_width=1.54,
                platform_width=4.0,
                border_width=1.0,
                perlin_cfg=perlin(),
                flat_patch_sampling=target_center(),
                **_walls(),
            ),
            "pyramid_stairs_inv": PerlinInvertedPyramidStairsTerrainCfg(
                proportion=0.15,
                step_height_range=(0.05, 0.23),
                step_width=0.35,
                platform_width=2.5,
                border_width=1.0,
                perlin_cfg=perlin(),
                flat_patch_sampling=target_center(),
                **_walls(),
            ),
            "pyramid_stairs_inv_high": PerlinInvertedPyramidStairsTerrainCfg(
                proportion=0.10,
                step_height_range=(0.05, 0.45),
                step_width=1.54,
                platform_width=4.0,
                border_width=1.0,
                perlin_cfg=perlin(),
                flat_patch_sampling=target_center(),
                **_walls(),
            ),
            "boxes": PerlinDiscreteObstaclesTerrainCfg(
                proportion=0.10,
                num_obstacles=20,
                obstacle_height_mode="fixed",
                obstacle_width_range=(0.8, 1.5),
                obstacle_height_range=(0.05, 0.45),
                platform_width=1.5,
                border_width=1.0,
                perlin_cfg=perlin(),
                flat_patch_sampling=target(),
                **_walls(),
            ),
            "dense_boxes": PerlinDiscreteObstaclesTerrainCfg(
                proportion=0.10,
                num_obstacles=120,
                obstacle_height_mode="fixed",
                obstacle_width_range=(0.30, 0.50),
                obstacle_height_range=(0.05, 0.45),
                platform_width=1.5,
                border_width=1.0,
                perlin_cfg=perlin(),
                flat_patch_sampling={
                    "target": FlatPatchSamplingCfg(
                        num_patches=50, patch_radius=[0.05, 0.10, 0.15], max_height_diff=0.05
                    )
                },
                **_walls(),
            ),
            "hf_pyramid_slope_inv": PerlinInvertedPyramidSlopedTerrainCfg(
                proportion=0.10,
                slope_range=(0.0, 0.7),
                platform_width=1.5,
                border_width=1.0,
                perlin_cfg=PerlinPlaneTerrainCfg(
                    noise_scale=0.00,
                    noise_frequency=20,
                    fractal_octaves=2,
                    fractal_lacunarity=2.0,
                    fractal_gain=0.25,
                    centering=True,
                ),
                flat_patch_sampling=target(),
                **_walls(),
            ),
        },
    )


def rough_importer_cfg(spec: TerrainSpec) -> Any:
    """InstinctMJ parkour's importer: ``hacked_generator`` plus ``FiledTerrainGenerator``.

    Virtual obstacles stay off. They are a parkour sensor, not the ground.
    """
    del spec
    from .terrains.terrain_importer_cfg import TerrainImporterCfg

    return TerrainImporterCfg(
        terrain_type="hacked_generator",
        terrain_generator=rough_generator_cfg(),
        max_init_terrain_level=5,
    )
