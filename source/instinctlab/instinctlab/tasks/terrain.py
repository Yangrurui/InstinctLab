"""Engine-neutral terrain recipes shared by task families.

The task layer owns the semantic recipe: grid layout, resolution, tile order,
and parameters that affect training.  Engine adapters lower each semantic tile
to their native terrain implementation.
"""

from __future__ import annotations

from instinctlab.spec import SubTerrainSpec, TerrainGeneratorSpec, TerrainSpec, VirtualObstacleRef

__all__ = ["rough_terrain"]


def _walls() -> dict[str, object]:
    return {
        "wall_prob": [0.3, 0.3, 0.3, 0.3],
        "wall_height": 5.0,
        "wall_thickness": 0.05,
    }


def _target_patches(*, centered: bool = False, radii: list[float] | None = None) -> dict[str, object]:
    target: dict[str, object] = {
        "num_patches": 50,
        "patch_radius": radii or [0.05, 0.10, 0.15, 0.20],
        "max_height_diff": 0.05,
    }
    if centered:
        target.update(x_range=(3.7, 3.7), y_range=(-0.0, 0.0))
    return {"target": target}


def _perlin(*, noise_scale: float = 0.05) -> dict[str, object]:
    return {
        "noise_scale": noise_scale,
        "noise_frequency": 20,
        "fractal_octaves": 2,
        "fractal_lacunarity": 2.0,
        "fractal_gain": 0.25,
        "centering": True,
    }


def _rough_generator() -> TerrainGeneratorSpec:
    """Main's effective parkour rough recipe expressed without an engine SDK."""
    return TerrainGeneratorSpec(
        seed=0,
        size=(8.0, 8.0),
        border_width=3.0,
        num_rows=10,
        num_cols=20,
        horizontal_scale=0.05,
        vertical_scale=0.005,
        slope_threshold=1.0,
        curriculum=True,
        max_init_level=5,
        sub_terrains={
            "perlin_rough": SubTerrainSpec(
                kind="perlin_plane",
                proportion=0.05,
                params={
                    "noise_scale": [0.0, 0.1],
                    "noise_frequency": 20,
                    "fractal_octaves": 2,
                    "fractal_lacunarity": 2.0,
                    "fractal_gain": 0.25,
                    "centering": True,
                    "border_width": 0.0,
                    "flat_patch_sampling": _target_patches(),
                    **_walls(),
                },
            ),
            "perlin_rough_stand": SubTerrainSpec(
                kind="perlin_plane",
                proportion=0.05,
                params={
                    "noise_scale": [0.0, 0.1],
                    "noise_frequency": 20,
                    "fractal_octaves": 2,
                    "fractal_lacunarity": 2.0,
                    "fractal_gain": 0.25,
                    "centering": True,
                    "border_width": 0.0,
                    "flat_patch_sampling": _target_patches(),
                    **_walls(),
                },
            ),
            "square_gaps": SubTerrainSpec(
                kind="perlin_square_gap",
                proportion=0.10,
                params={
                    "gap_distance_range": (0.1, 0.7),
                    "gap_depth": (0.4, 0.6),
                    "platform_width": 2.5,
                    "border_width": 1.0,
                    "flat_patch_sampling": _target_patches(centered=True),
                    **_walls(),
                },
            ),
            "pyramid_stairs": SubTerrainSpec(
                kind="perlin_pyramid_stairs",
                proportion=0.15,
                params={
                    "step_height_range": (0.05, 0.23),
                    "step_width": 0.3,
                    "platform_width": 2.5,
                    "border_width": 1.0,
                    "perlin_cfg": _perlin(),
                    "flat_patch_sampling": _target_patches(centered=True),
                    **_walls(),
                },
            ),
            "pyramid_stairs_high": SubTerrainSpec(
                kind="perlin_pyramid_stairs",
                proportion=0.10,
                params={
                    "step_height_range": (0.05, 0.45),
                    "step_width": 1.5,
                    "platform_width": 4.0,
                    "border_width": 1.0,
                    "perlin_cfg": _perlin(),
                    "flat_patch_sampling": _target_patches(centered=True),
                    **_walls(),
                },
            ),
            "pyramid_stairs_inv": SubTerrainSpec(
                kind="perlin_pyramid_stairs_inv",
                proportion=0.15,
                params={
                    "step_height_range": (0.05, 0.23),
                    "step_width": 0.3,
                    "platform_width": 2.5,
                    "border_width": 1.0,
                    "perlin_cfg": _perlin(),
                    "flat_patch_sampling": _target_patches(centered=True),
                    **_walls(),
                },
            ),
            "pyramid_stairs_inv_high": SubTerrainSpec(
                kind="perlin_pyramid_stairs_inv",
                proportion=0.10,
                params={
                    "step_height_range": (0.05, 0.45),
                    "step_width": 1.5,
                    "platform_width": 4.0,
                    "border_width": 1.0,
                    "perlin_cfg": _perlin(),
                    "flat_patch_sampling": _target_patches(centered=True),
                    **_walls(),
                },
            ),
            "boxes": SubTerrainSpec(
                kind="perlin_discrete_obstacles",
                proportion=0.10,
                params={
                    "num_obstacles": 20,
                    "obstacle_height_mode": "fixed",
                    "obstacle_width_range": (0.8, 1.5),
                    "obstacle_height_range": (0.05, 0.45),
                    "platform_width": 1.5,
                    "border_width": 0.0,
                    "perlin_cfg": _perlin(),
                    "flat_patch_sampling": _target_patches(),
                    **_walls(),
                },
            ),
            "mesh_boxes": SubTerrainSpec(
                kind="perlin_random_multi_box",
                proportion=0.10,
                params={
                    "box_height_mean": [0.1, 0.4],
                    "box_height_range": 0.05,
                    "box_length_mean": 0.4,
                    "box_length_range": 0.1,
                    "box_width_mean": 0.4,
                    "box_width_range": 0.1,
                    "platform_width": 1.5,
                    "generation_ratio": 0.3,
                    "no_perlin_at_obstacle": True,
                    "flat_patch_sampling": _target_patches(radii=[0.05, 0.10, 0.15]),
                    **_walls(),
                },
            ),
            "hf_pyramid_slope_inv": SubTerrainSpec(
                kind="perlin_pyramid_slope_inv",
                proportion=0.10,
                params={
                    "slope_range": (0.0, 0.7),
                    "platform_width": 1.5,
                    "border_width": 1.0,
                    "perlin_cfg": _perlin(noise_scale=0.0),
                    "flat_patch_sampling": _target_patches(),
                    **_walls(),
                },
            ),
        },
    )


def rough_terrain(*, virtual_obstacles: tuple[VirtualObstacleRef, ...] = ()) -> TerrainSpec:
    """Shared rough terrain used by locomotion and parkour on every engine."""
    return TerrainSpec(kind="rough", generator=_rough_generator(), virtual_obstacles=virtual_obstacles)
