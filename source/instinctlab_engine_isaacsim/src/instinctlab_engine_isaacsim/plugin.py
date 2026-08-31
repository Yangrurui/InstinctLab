"""Lightweight Isaac Sim plugin registration.

Only dotted paths and selector metadata live here. Importing this registrar must
remain safe before :class:`isaaclab.app.AppLauncher` starts the simulator.
"""

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
    """Register lazy Isaac builders without importing an implementation module."""
    register_adapter(
        "isaacsim",
        "instinctlab_engine_isaacsim.facade:IsaacSimAdapter",
    )
    for kind, builder in _WHOLE_TERRAINS.items():
        register_terrain(
            "isaacsim",
            kind,
            f"instinctlab_engine_isaacsim.terrain_builders:{builder}",
        )
    for kind in _STANDARD_TILES:
        register_sub_terrain(
            "isaacsim",
            kind,
            "instinctlab_engine_isaacsim.terrain_builders:build_standard_tile",
        )
    _entity.register(
        "isaacsim",
        kinds=("joint", "body", "fixed_tendon", "object_collection"),
        cfg=("isaaclab.managers", "SceneEntityCfg"),
        container=list,
    )


register.instinctlab_engine_api = ">=0.1,<0.2"
