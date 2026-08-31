"""Isaac-native implementations registered behind semantic terrain kinds."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from instinctlab_engine.data import resolve_data_path
from instinctlab_engine.registry import TERRAIN_EXTENSIONS
from instinctlab_engine.spec.task import (
    SubTerrainSpec,
    TerrainGeneratorSpec,
    TerrainSpec,
)


def _physics_material(spec: TerrainSpec) -> Any:
    from isaaclab.sim import RigidBodyMaterialCfg

    return RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=spec.static_friction,
        dynamic_friction=spec.dynamic_friction,
        restitution=spec.restitution,
    )


def _visual_material() -> Any:
    from isaaclab.sim import MdlFileCfg
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

    return MdlFileCfg(
        mdl_path=(
            f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
            "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
        ),
        project_uvw=True,
        texture_scale=(0.25, 0.25),
    )


def _build_sub_terrain(tile: SubTerrainSpec, generator: TerrainGeneratorSpec) -> Any:
    builder = TERRAIN_EXTENSIONS.sub_terrain("isaacsim", tile.kind)
    if builder is None:
        available = sorted(TERRAIN_EXTENSIONS.sub_terrain_kinds("isaacsim"))
        raise NotImplementedError(
            f"The Isaac Sim backend has no generated-terrain tile {tile.kind!r}; "
            f"registered tiles are {available}."
        )
    return builder(tile, generator)


def build_standard_tile(tile: SubTerrainSpec, generator: TerrainGeneratorSpec) -> Any:
    """Lower one Isaac Lab built-in generated-terrain tile."""
    del generator
    import isaaclab.terrains as terrain_gen

    classes = {
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg,
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg,
        "boxes": terrain_gen.MeshRandomGridTerrainCfg,
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg,
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg,
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg,
    }
    return classes[tile.kind](proportion=tile.proportion, **tile.params)


def _flat_patches(value: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    from isaaclab.terrains import FlatPatchSamplingCfg

    return {
        name: FlatPatchSamplingCfg(**dict(params)) for name, params in value.items()
    }


def _perlin(value: Mapping[str, Any]) -> Any:
    import instinctlab_engine_isaacsim.terrains as terrain_gen

    return terrain_gen.PerlinPlaneTerrainCfg(**dict(value))


def build_rough_tile(tile: SubTerrainSpec, generator: TerrainGeneratorSpec) -> Any:
    """Lower one shared rough-terrain tile to its Isaac implementation."""
    del generator
    import instinctlab_engine_isaacsim.terrains as terrain_gen

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
    return classes[tile.kind](proportion=tile.proportion, **fields)


def build_perlin_wave_tile(
    tile: SubTerrainSpec, generator: TerrainGeneratorSpec
) -> Any:
    """Lower the independently registered Perlin wave tile for Isaac Sim."""
    del generator
    import instinctlab_engine_isaacsim.terrains as terrain_gen

    fields = dict(tile.params)
    if flat_patches := fields.get("flat_patch_sampling"):
        fields["flat_patch_sampling"] = _flat_patches(flat_patches)
    if perlin_cfg := fields.get("perlin_cfg"):
        fields["perlin_cfg"] = _perlin(perlin_cfg)
    return terrain_gen.PerlinWaveTerrainCfg(
        proportion=tile.proportion,
        **fields,
    )


def _standard_generator(spec: TerrainGeneratorSpec) -> Any:
    from isaaclab.terrains import TerrainGeneratorCfg

    return TerrainGeneratorCfg(
        seed=spec.seed,
        curriculum=spec.curriculum,
        size=spec.size,
        border_width=spec.border_width,
        num_rows=spec.num_rows,
        num_cols=spec.num_cols,
        horizontal_scale=spec.horizontal_scale,
        vertical_scale=spec.vertical_scale,
        slope_threshold=spec.slope_threshold,
        use_cache=False,
        sub_terrains={
            name: _build_sub_terrain(tile, spec)
            for name, tile in spec.sub_terrains.items()
        },
    )


def _rough_generator(spec: TerrainGeneratorSpec) -> Any:
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
        sub_terrains={
            name: _build_sub_terrain(tile, spec)
            for name, tile in spec.sub_terrains.items()
        },
    )


def build_plane(spec: TerrainSpec, profile: Mapping[str, Any]) -> Any:
    del profile
    from isaaclab.terrains import TerrainImporterCfg

    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=_physics_material(spec),
        debug_vis=False,
    )


def build_generator(spec: TerrainSpec, profile: Mapping[str, Any]) -> Any:
    del profile
    if spec.generator is None:
        raise ValueError("kind='generator' needs a TerrainGeneratorSpec.")
    from isaaclab.terrains import TerrainImporterCfg

    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=_standard_generator(spec.generator),
        max_init_terrain_level=spec.generator.max_init_level,
        collision_group=-1,
        physics_material=_physics_material(spec),
        visual_material=_visual_material(),
        debug_vis=False,
    )


def _attach_virtual_obstacles(cfg: Any, spec: TerrainSpec) -> Any:
    if not spec.virtual_obstacles:
        return cfg
    from .terrains.virtual_obstacle import GreedyconcatEdgeCylinderCfg

    obstacles: dict[str, Any] = {}
    for obstacle in spec.virtual_obstacles:
        if obstacle.kind != "greedy_edge_cylinder":
            raise NotImplementedError(
                f"Isaac Sim has no virtual obstacle {obstacle.kind!r} for {obstacle.name!r}."
            )
        obstacles[obstacle.name] = GreedyconcatEdgeCylinderCfg(
            cylinder_radius=obstacle.cylinder_radius,
            min_points=obstacle.min_points,
            angle_threshold=obstacle.angle_threshold,
        )
    cfg.virtual_obstacles = obstacles
    return cfg


def build_rough(spec: TerrainSpec, profile: Mapping[str, Any]) -> Any:
    del profile
    if spec.generator is None:
        raise ValueError("kind='rough' needs a TerrainGeneratorSpec.")
    from .terrains import TerrainImporterCfg

    cfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=_rough_generator(spec.generator),
        max_init_terrain_level=spec.generator.max_init_level,
        collision_group=-1,
        physics_material=_physics_material(spec),
        visual_material=_visual_material(),
        debug_vis=False,
    )
    return _attach_virtual_obstacles(cfg, spec)


def build_motion_matched(spec: TerrainSpec, profile: Mapping[str, Any]) -> Any:
    del profile
    from .terrains import TerrainImporterCfg
    from .terrains.terrain_generator_cfg import FiledTerrainGeneratorCfg
    from .terrains.trimesh.mesh_terrains_cfg import MotionMatchedTerrainCfg

    path = str(resolve_data_path(spec.params["engine_paths"]["isaacsim"]))
    generator = FiledTerrainGeneratorCfg(
        size=(9.0, 12.0),
        border_width=0.0,
        border_height=0.0,
        num_rows=7,
        num_cols=7,
        sub_terrains={
            "motion_matched": MotionMatchedTerrainCfg(
                proportion=1.0,
                path=path,
                metadata_yaml=os.path.join(path, spec.params["metadata_yaml"]),
            )
        },
    )
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="hacked_generator",
        terrain_generator=generator,
        collision_group=-1,
        physics_material=_physics_material(spec),
        visual_material=_visual_material(),
        debug_vis=False,
    )


__all__ = [
    "build_generator",
    "build_motion_matched",
    "build_plane",
    "build_rough",
    "build_rough_tile",
    "build_standard_tile",
]
