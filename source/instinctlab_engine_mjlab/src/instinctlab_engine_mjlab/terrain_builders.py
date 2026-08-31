"""MJLab-native implementations registered behind semantic terrain kinds."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from instinctlab_engine.data import resolve_data_path
from instinctlab_engine.registry import TERRAIN_EXTENSIONS
from instinctlab_engine.spec.task import (
    SubTerrainSpec,
    TerrainGeneratorSpec,
    TerrainSpec,
)


def _validate_material(spec: TerrainSpec) -> None:
    if spec.restitution != 0.0:
        raise ValueError(
            "mjlab terrain cannot honor restitution; MuJoCo terrain geoms have no restitution field"
        )
    if spec.static_friction != spec.dynamic_friction:
        raise ValueError(
            "mjlab terrain has one sliding-friction coefficient and cannot honor different "
            "static_friction and dynamic_friction values"
        )


def _build_sub_terrain(tile: SubTerrainSpec, generator: TerrainGeneratorSpec) -> Any:
    builder = TERRAIN_EXTENSIONS.sub_terrain("mjlab", tile.kind)
    if builder is None:
        available = sorted(TERRAIN_EXTENSIONS.sub_terrain_kinds("mjlab"))
        raise NotImplementedError(
            f"The MJLab backend has no generated-terrain tile {tile.kind!r}; "
            f"registered tiles are {available}."
        )
    return builder(tile, generator)


def build_standard_tile(tile: SubTerrainSpec, generator: TerrainGeneratorSpec) -> Any:
    """Lower one MJLab built-in generated-terrain tile."""
    import mjlab.terrains as terrain_gen

    fields = dict(tile.params)
    if tile.kind in {"random_rough", "hf_pyramid_slope", "hf_pyramid_slope_inv"}:
        fields.setdefault("horizontal_scale", generator.horizontal_scale)
        fields.setdefault("vertical_scale", generator.vertical_scale)
    if tile.kind == "hf_pyramid_slope_inv":
        fields["inverted"] = True
        return terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=tile.proportion, **fields
        )
    classes = {
        "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg,
        "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg,
        "boxes": terrain_gen.BoxRandomGridTerrainCfg,
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg,
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg,
    }
    return classes[tile.kind](proportion=tile.proportion, **fields)


def _flat_patches(value: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    from mjlab.terrains import FlatPatchSamplingCfg

    return {
        name: FlatPatchSamplingCfg(**dict(params)) for name, params in value.items()
    }


def _perlin(value: Mapping[str, Any]) -> Any:
    from .terrains.height_field.hf_terrains_cfg import PerlinPlaneTerrainCfg

    return PerlinPlaneTerrainCfg(**dict(value))


def build_rough_tile(tile: SubTerrainSpec, generator: TerrainGeneratorSpec) -> Any:
    """Lower one shared rough-terrain tile to its MJLab implementation."""
    del generator
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
    return classes[tile.kind](proportion=tile.proportion, **fields)


def build_perlin_wave_tile(
    tile: SubTerrainSpec, generator: TerrainGeneratorSpec
) -> Any:
    """Lower the independently registered Perlin wave tile for MJLab."""
    del generator
    from .terrains.height_field.hf_terrains_cfg import PerlinWaveTerrainCfg

    fields = dict(tile.params)
    if flat_patches := fields.get("flat_patch_sampling"):
        fields["flat_patch_sampling"] = _flat_patches(flat_patches)
    if perlin_cfg := fields.get("perlin_cfg"):
        fields["perlin_cfg"] = _perlin(perlin_cfg)
    return PerlinWaveTerrainCfg(
        proportion=tile.proportion,
        **fields,
    )


def _standard_generator(spec: TerrainGeneratorSpec) -> Any:
    from .terrains.terrain_generator_cfg import FiledTerrainGeneratorCfg

    return FiledTerrainGeneratorCfg(
        seed=spec.seed,
        curriculum=spec.curriculum,
        size=spec.size,
        border_width=spec.border_width,
        num_rows=spec.num_rows,
        num_cols=spec.num_cols,
        horizontal_scale=spec.horizontal_scale,
        vertical_scale=spec.vertical_scale,
        slope_threshold=spec.slope_threshold,
        add_lights=True,
        sub_terrains={
            name: _build_sub_terrain(tile, spec)
            for name, tile in spec.sub_terrains.items()
        },
    )


def _rough_generator(spec: TerrainGeneratorSpec) -> Any:
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
        sub_terrains={
            name: _build_sub_terrain(tile, spec)
            for name, tile in spec.sub_terrains.items()
        },
    )


def build_plane(spec: TerrainSpec, profile: Mapping[str, Any]) -> Any:
    del profile
    _validate_material(spec)
    from .terrains.terrain_importer_cfg import TerrainImporterCfg

    return TerrainImporterCfg(
        terrain_type="plane", sliding_friction=spec.dynamic_friction
    )


def build_generator(spec: TerrainSpec, profile: Mapping[str, Any]) -> Any:
    del profile
    _validate_material(spec)
    if spec.generator is None:
        raise ValueError("kind='generator' needs a TerrainGeneratorSpec.")
    from .terrains.terrain_importer_cfg import TerrainImporterCfg

    return TerrainImporterCfg(
        terrain_type="generator",
        terrain_generator=_standard_generator(spec.generator),
        max_init_terrain_level=spec.generator.max_init_level,
        sliding_friction=spec.dynamic_friction,
    )


def _attach_virtual_obstacles(cfg: Any, spec: TerrainSpec) -> Any:
    """Attach the native Parkour edge-cylinder extraction.

    MJLab repairs a height-field surface and merges short collinear gaps while
    Isaac reads the generated mesh. The penetration penalty is therefore not
    comparable as a cross-engine parity signal.
    """
    if not spec.virtual_obstacles:
        return cfg
    from .terrains.virtual_obstacle.edge_cylinder_cfg import (
        GreedyconcatEdgeCylinderCfg,
    )

    obstacles: dict[str, Any] = {}
    for obstacle in spec.virtual_obstacles:
        if obstacle.kind != "greedy_edge_cylinder":
            raise NotImplementedError(
                f"mjlab has no virtual obstacle {obstacle.kind!r} for {obstacle.name!r}."
            )
        obstacles[obstacle.name] = GreedyconcatEdgeCylinderCfg(
            cylinder_radius=obstacle.cylinder_radius,
            min_points=obstacle.min_points,
            angle_threshold=obstacle.angle_threshold,
            component_workers=0,
            merge_collinear_gap=0.09,
            merge_collinear_angle_threshold=30.0,
            merge_collinear_line_distance=0.04,
        )
    cfg.virtual_obstacles = obstacles
    cfg.virtual_obstacle_source = "mesh"
    cfg.virtual_obstacle_hfield_height_threshold = 0.04
    return cfg


def build_rough(spec: TerrainSpec, profile: Mapping[str, Any]) -> Any:
    del profile
    _validate_material(spec)
    if spec.generator is None:
        raise ValueError("kind='rough' needs a TerrainGeneratorSpec.")
    from .terrains.terrain_importer_cfg import TerrainImporterCfg

    cfg = TerrainImporterCfg(
        terrain_type="hacked_generator",
        terrain_generator=_rough_generator(spec.generator),
        max_init_terrain_level=spec.generator.max_init_level,
        sliding_friction=spec.dynamic_friction,
    )
    return _attach_virtual_obstacles(cfg, spec)


def build_motion_matched(spec: TerrainSpec, profile: Mapping[str, Any]) -> Any:
    del profile
    _validate_material(spec)
    from mjlab.terrains import SubTerrainCfg

    from .motion_matched_terrain import motion_matched_terrain
    from .terrains.terrain_generator_cfg import FiledTerrainGeneratorCfg
    from .terrains.terrain_importer_cfg import TerrainImporterCfg

    @dataclass(kw_only=True)
    class MotionMatchedTerrainCfg(SubTerrainCfg):
        function = motion_matched_terrain
        path: str
        metadata_yaml: str
        crop_to_size: bool = True
        use_input_origin_frame: bool = True
        collision_coacd_threshold: float = 0.04
        collision_coacd_resolution: int = 3000
        collision_coacd_decimate: bool = False
        collision_coacd_max_ch_vertex: int = 256
        collision_coacd_log_level: str = "off"
        collision_coacd_use_disk_cache: bool = True
        collision_coacd_cache_dirname: str = ".coacd_cache"
        collision_coacd_prewarm_all: bool = True
        collision_coacd_prewarm_workers: int = 0
        collision_coacd_geom_margin: float = 0.0
        collision_coacd_z_offset: float = 0.0
        collision_coacd_auto_align_top_surface: bool = True
        collision_coacd_auto_align_resolution: float = 0.04
        collision_coacd_visualize_collision_hulls: bool = True

    path = str(resolve_data_path(spec.params["engine_paths"]["mjlab"]))
    generator = FiledTerrainGeneratorCfg(
        size=(30.0, 16.0),
        border_width=0.0,
        num_rows=3,
        num_cols=3,
        add_lights=True,
        sub_terrains={
            "motion_matched": MotionMatchedTerrainCfg(
                proportion=1.0,
                path=path,
                metadata_yaml=os.path.join(path, spec.params["metadata_yaml"]),
            )
        },
    )
    return TerrainImporterCfg(
        terrain_type="hacked_generator",
        terrain_generator=generator,
        sliding_friction=spec.dynamic_friction,
    )


__all__ = [
    "build_generator",
    "build_motion_matched",
    "build_plane",
    "build_rough",
    "build_rough_tile",
    "build_standard_tile",
]
