"""Isaac Sim's reference rough: main's parkour ``ROUGH_TERRAINS_CFG``.

The numbers are reconstructed here rather than imported from
``tasks/parkour/config/parkour_env_cfg.py``, which would drag the rest of that Isaac-only
tree into locomotion compile. ``tests/test_rough_g1_declaration.py`` compares the two.
"""

from __future__ import annotations

from typing import Any

from instinctlab.spec.task import TerrainSpec

__all__ = ["rough_generator_cfg", "rough_importer_cfg"]


def _walls() -> dict[str, Any]:
    return {"wall_prob": [0.3, 0.3, 0.3, 0.3], "wall_height": 5.0, "wall_thickness": 0.05}


def rough_generator_cfg() -> Any:
    """Main parkour's Perlin grid. Imports stay here so the module stays engine-free at rest."""
    from isaaclab.terrains import FlatPatchSamplingCfg, TerrainGeneratorCfg

    import instinctlab.terrains as terrain_gen

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
        return terrain_gen.PerlinPlaneTerrainCfg(
            noise_scale=0.05,
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
        )

    return TerrainGeneratorCfg(
        seed=0,
        size=(8.0, 8.0),
        border_width=3,
        num_rows=10,
        num_cols=20,
        horizontal_scale=0.05,
        vertical_scale=0.005,
        slope_threshold=1.0,
        use_cache=False,
        curriculum=True,
        sub_terrains={
            "perlin_rough": terrain_gen.PerlinPlaneTerrainCfg(
                proportion=0.05,
                noise_scale=[0.0, 0.1],
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
                flat_patch_sampling=target(),
                **_walls(),
            ),
            "perlin_rough_stand": terrain_gen.PerlinPlaneTerrainCfg(
                proportion=0.05,
                noise_scale=[0.0, 0.1],
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
                flat_patch_sampling=target(),
                **_walls(),
            ),
            "square_gaps": terrain_gen.PerlinSquareGapTerrainCfg(
                proportion=0.10,
                gap_distance_range=(0.1, 0.7),
                gap_depth=(0.4, 0.6),
                platform_width=2.5,
                border_width=1.0,
                flat_patch_sampling=target_center(),
                **_walls(),
            ),
            "pyramid_stairs": terrain_gen.PerlinPyramidStairsTerrainCfg(
                proportion=0.15,
                step_height_range=(0.05, 0.23),
                step_width=0.3,
                platform_width=2.5,
                border_width=1.0,
                perlin_cfg=perlin(),
                flat_patch_sampling=target_center(),
                **_walls(),
            ),
            "pyramid_stairs_high": terrain_gen.PerlinPyramidStairsTerrainCfg(
                proportion=0.10,
                step_height_range=(0.05, 0.45),
                step_width=1.5,
                platform_width=4.0,
                border_width=1.0,
                perlin_cfg=perlin(),
                flat_patch_sampling=target_center(),
                **_walls(),
            ),
            "pyramid_stairs_inv": terrain_gen.PerlinInvertedPyramidStairsTerrainCfg(
                proportion=0.15,
                step_height_range=(0.05, 0.23),
                step_width=0.3,
                platform_width=2.5,
                border_width=1.0,
                perlin_cfg=perlin(),
                flat_patch_sampling=target_center(),
                **_walls(),
            ),
            "pyramid_stairs_inv_high": terrain_gen.PerlinInvertedPyramidStairsTerrainCfg(
                proportion=0.10,
                step_height_range=(0.05, 0.45),
                step_width=1.5,
                platform_width=4.0,
                border_width=1.0,
                perlin_cfg=perlin(),
                flat_patch_sampling=target_center(),
                **_walls(),
            ),
            "boxes": terrain_gen.PerlinDiscreteObstaclesTerrainCfg(
                proportion=0.10,
                num_obstacles=20,
                obstacle_height_mode="fixed",
                obstacle_width_range=(0.8, 1.5),
                obstacle_height_range=(0.05, 0.45),
                platform_width=1.5,
                border_width=0.0,
                perlin_cfg=perlin(),
                flat_patch_sampling=target(),
                **_walls(),
            ),
            "mesh_boxes": terrain_gen.PerlinMeshRandomMultiBoxTerrainCfg(
                proportion=0.10,
                box_height_mean=[0.1, 0.4],
                box_height_range=0.05,
                box_length_mean=0.4,
                box_length_range=0.1,
                box_width_mean=0.4,
                box_width_range=0.1,
                platform_width=1.5,
                generation_ratio=0.3,
                no_perlin_at_obstacle=True,
                flat_patch_sampling={
                    "target": FlatPatchSamplingCfg(
                        num_patches=50, patch_radius=[0.05, 0.10, 0.15], max_height_diff=0.05
                    )
                },
                **_walls(),
            ),
            "hf_pyramid_slope_inv": terrain_gen.PerlinInvertedPyramidSlopedTerrainCfg(
                proportion=0.10,
                slope_range=(0.0, 0.7),
                platform_width=1.5,
                border_width=1.0,
                perlin_cfg=terrain_gen.PerlinPlaneTerrainCfg(
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
    """The importer main's parkour scene uses, without the virtual-obstacle sensors."""
    from isaaclab.sim import MdlFileCfg, RigidBodyMaterialCfg
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

    from instinctlab.terrains import TerrainImporterCfg

    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=rough_generator_cfg(),
        max_init_terrain_level=5,
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
