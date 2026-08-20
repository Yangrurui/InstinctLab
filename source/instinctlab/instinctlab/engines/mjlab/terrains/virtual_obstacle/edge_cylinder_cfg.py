from __future__ import annotations

from dataclasses import dataclass, field

from .edge_cylinder import GreedyconcatEdgeCylinder
from .virtual_obstacle_base import VirtualObstacleCfg


@dataclass(kw_only=True)
class EdgeCylinderCfg(VirtualObstacleCfg):
    angle_threshold: float = 70.0
    cylinder_radius: float = 0.2
    num_grid_cells: int = 64**3


@dataclass(kw_only=True)
class GreedyconcatEdgeCylinderCfg(EdgeCylinderCfg):
    """InstinctMJ parkour's greedy-concat detector, including the collinear post-merge."""

    class_type: type = field(default=GreedyconcatEdgeCylinder)
    adjacent_angle_threshold: float = 30.0
    point_distance_threshold: float = 0.06
    min_points: int = 5
    merge_collinear_gap: float = 0.0
    merge_collinear_angle_threshold: float = 25.0
    merge_collinear_line_distance: float | None = None
    merge_collinear_max_passes: int = 3
    merge_collinear_max_segments: int = 4000
    component_workers: int = 1
