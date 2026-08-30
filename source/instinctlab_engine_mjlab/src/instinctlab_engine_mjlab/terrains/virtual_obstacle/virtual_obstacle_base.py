from __future__ import annotations

import torch
import trimesh
from abc import ABC, abstractmethod
from dataclasses import MISSING, dataclass


@dataclass(kw_only=True)
class VirtualObstacleCfg:
    class_type: type = MISSING


class VirtualObstacleBase(ABC):
    def __init__(self, cfg: VirtualObstacleCfg):
        self.cfg = cfg
        self.supports_edge_segment_generation = False

    @abstractmethod
    def generate(self, mesh: trimesh.Trimesh, device: torch.device | str = "cpu") -> None:
        raise NotImplementedError

    def disable_visualizer(self) -> None:
        return

    def visualize(self) -> None:
        return

    def generate_from_edge_segments(self, edge_segments, device: torch.device | str = "cpu") -> None:
        del edge_segments, device
        raise NotImplementedError("This virtual obstacle does not support edge-segment generation.")

    def debug_vis(self, visualizer) -> None:
        del visualizer
        return

    @abstractmethod
    def get_points_penetration_offset(self, points: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
