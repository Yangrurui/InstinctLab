"""mjlab's parkour/rough terrain recipe.

The numbers live here, not in the task. The generator and importer classes are the
vendored copy under :mod:`.terrains`.

The whole recipe now deliberately follows InstinctMJ
(``/root/InstinctMJ/src/instinct_mj/tasks/parkour/config/parkour_env_cfg.py``):
``horizontal_scale=0.07``, stair ``step_width`` 0.35 / 1.54, ``perlin_rough`` /
``perlin_rough_stand`` set ``border_width=1.0``, and ``boxes.border_width=1.0``.
Isaac's parkour (``engines/isaacsim/rough.py``) keeps 0.05 / 0.3 / 1.5, omits
the plane ``border_width`` kwargs (both engines' ``HfTerrainBaseCfg`` default is
0.0 — verified in the class bodies, not assumed), and uses
``boxes.border_width=0.0``. Terrain-constant parity is not being pursued.
``tests/test_rough_recipe_parity.py`` compares the two recipes by AST and fails
if they drift outside a table of those known differences.

This copy honors ``num_cols=20`` (Isaac's cumulative-proportion allocation).
Upstream mjlab / InstinctMJ ignore that field in curriculum mode and build one
column per type.

With ``border_width=1.0`` deducted, ``pyramid_stairs`` builds **6** in-field
steps on Isaac against **5** on mjlab, with central platform heights of
0.30 / 0.84 / 1.38 m against 0.25 / 0.70 / 1.15 m at difficulty 0.0 / 0.5 / 1.0.
``pyramid_stairs`` and ``pyramid_stairs_inv`` are 0.15 each, so roughly a third
of environments climb a different staircase depending on the engine.
``pyramid_stairs_high`` is unaffected — 1.50 and 1.54 round to the same pixel
geometry. Cross-engine episode-length or terrain-level curves are therefore not
comparable on the stairs terrains.

What remains unaligned, by decision:

* Slot 9 is ``dense_boxes`` (``PerlinDiscreteObstaclesTerrainCfg``) here and
  ``mesh_boxes`` (``PerlinMeshRandomMultiBoxTerrainCfg``) on Isaac. The mesh-box
  type exists only under ``instinctlab/terrains/trimesh/``; mjlab has no
  equivalent. Porting it is separate work.
* Difficulty is still mjlab's ``row / (num_rows - 1)``. Isaac uses jittered
  ``(row + U[0,1)) / num_rows`` and never hits 1.0. Duplicate columns of one
  type at the same row are therefore identical here.
* Virtual-obstacle edge cylinders are a second accepted divergence of the
  same character as the stairs step count. Isaac's Greedyconcat has no
  collinear post-merge and splits at a hardcoded 0.05 m; this recipe
  (InstinctMJ parkour) merges gaps up to 0.09 m. Isaac reads true meshes,
  including the 5 m walls; mjlab reads a repaired height-field surface.
  On the same closed box both detectors emit 12 edges — the primitive
  matches, the terrain representation does not. Measured on
  ``Instinct-Parkour-Target-G1``: about 35k vs 43k cylinders overall, 208
  vs 518 on the row-0 ``pyramid_stairs`` tile. The volume-points
  penetration penalty is therefore not comparable across engines and must
  not be read as a parity signal in a two-engine training comparison.

Measured cost of following InstinctMJ here, and an open question. A two-engine
run of ``Instinct-Parkour-Target-G1`` (256 envs, seed 42, matched at iterations
623-642) put mjlab at **0.62x** Isaac's episode length and reward on the
*aligned* terrain subset, while the *known-diverged* stairs subset came out at
0.71x / 0.77x. The gap is therefore not explained by anything documented above:
it is worst where the two engines are supposed to agree. It concentrates on the
continuous height-field terrains -- ``hf_pyramid_slope_inv`` 0.40x,
``perlin_rough`` 0.48x, ``perlin_rough_stand`` 0.49x -- and is mild on the
discretely stepped ones (stairs, ``boxes``, ``square_gaps``: 0.73-0.84x). That
pattern fits ``horizontal_scale``: a slope on a 0.07 grid is discretised into
treads 40% taller than on Isaac's 0.05. Contact overflow was ruled out
(``d.overflow`` clear at construction and mid-training; ``nacon`` 164/world
against ``nconmax=256``). The grid-resolution reading is a hypothesis, not a
measurement: the test that would settle it is a short mjlab run at
``horizontal_scale=0.05`` with nothing else changed.
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
