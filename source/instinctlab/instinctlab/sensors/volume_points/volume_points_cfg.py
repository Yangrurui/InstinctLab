from dataclasses import MISSING
from typing import Dict

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sensors import SensorBaseCfg
from isaaclab.utils import configclass

from .points_generator_cfg import PointsGeneratorCfg
from .volume_points import VolumePoints

VOLUME_POINTS_VISUALIZER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/volumePoints",
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        "sphere_penetrated": sim_utils.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    },
)


@configclass
class VolumePointsCfg(SensorBaseCfg):
    """Configuration for the volume points sensor."""

    class_type: type = VolumePoints

    filter_prim_paths_expr: list[str] = list()
    """The list of primitive paths (or expressions) to filter volume points' body with. Defaults to an empty list,
    in which case
    no filtering is applied.

    .. note::
        The expression in the list can contain the environment namespace regex ``{ENV_REGEX_NS}`` which
        will be replaced with the environment namespace.

        Example: ``{ENV_REGEX_NS}/Object`` will be replaced with ``/World/envs/env_.*/Object``.

    """

    points_generator: PointsGeneratorCfg = MISSING
    """ The points generator configuration. The generator function should be callable and accept only its cfg.
    """

    body_order: list[str] | None = None
    """Declared attach-body order. Discovery must match or the two engines sum different clouds."""

    velocity: str = "com"
    """About which point ``vel_w`` / ``points_vel_w`` are expressed.

    ``"com"`` is the legacy PhysX ``get_velocities()`` linear row (centre of
    mass). ``"attach_link"`` converts that to the link origin before the
    ``ω × r`` product — the hub quantity. Leave the default; the new-stack
    builder sets ``attach_link``. Changing the default would silently retune
    every Isaac-only parkour Gym id.
    """

    visualizer_cfg: VisualizationMarkersCfg = VOLUME_POINTS_VISUALIZER_CFG
