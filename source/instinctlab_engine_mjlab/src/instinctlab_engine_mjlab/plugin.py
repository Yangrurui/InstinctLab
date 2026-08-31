"""Lightweight MJLab plugin registration using lazy implementation paths."""

from instinctlab_engine import register_adapter, register_sub_terrain, register_terrain
from instinctlab_engine.bridge import entity as _entity

_WHOLE_TERRAINS = {
    "plane": "build_plane",
    "generator": "build_generator",
    "rough": "build_rough",
    "motion_matched": "build_motion_matched",
}

_STANDARD_TILES = (
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "boxes",
    "random_rough",
    "hf_pyramid_slope",
    "hf_pyramid_slope_inv",
)

def register() -> None:
    """Register lazy MJLab builders without importing an implementation module."""
    register_adapter(
        "mjlab",
        "instinctlab_engine_mjlab.facade:MjlabAdapter",
    )
    for kind, builder in _WHOLE_TERRAINS.items():
        register_terrain(
            "mjlab",
            kind,
            f"instinctlab_engine_mjlab.terrain_builders:{builder}",
        )
    for kind in _STANDARD_TILES:
        register_sub_terrain(
            "mjlab",
            kind,
            "instinctlab_engine_mjlab.terrain_builders:build_standard_tile",
        )
    _entity.register(
        "mjlab",
        kinds=(
            "joint",
            "body",
            "geom",
            "site",
            "actuator",
            "tendon",
            "camera",
            "light",
            "material",
            "pair",
        ),
        cfg=("mjlab.managers.scene_entity_config", "SceneEntityCfg"),
        container=tuple,
    )


register.instinctlab_engine_api = ">=0.1,<0.2"
