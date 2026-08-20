"""mjlab virtual obstacles. InstinctMJ's greedy-concat detector, no Isaac types."""

from .edge_cylinder_cfg import GreedyconcatEdgeCylinderCfg
from .virtual_obstacle_base import VirtualObstacleBase, VirtualObstacleCfg

__all__ = ["GreedyconcatEdgeCylinderCfg", "VirtualObstacleBase", "VirtualObstacleCfg"]
