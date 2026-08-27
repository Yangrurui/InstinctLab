"""MJLab-native mesh terrain primitives used by the shared terrain bridge."""

from __future__ import annotations

import math
from dataclasses import MISSING, dataclass, field
from typing import Any

import mujoco
import numpy as np
import trimesh
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput


def _box_mesh(
    extents: tuple[float, float, float],
    position: tuple[float, float, float],
    *,
    yaw: float = 0.0,
) -> trimesh.Trimesh:
    transform = trimesh.transformations.translation_matrix(position)
    if yaw != 0.0:
        transform = transform @ trimesh.transformations.rotation_matrix(yaw, (0.0, 0.0, 1.0))
    return trimesh.creation.box(extents=extents, transform=transform)


def _sample_center_patches(
    cfg: "PerlinMeshRandomMultiBoxTerrainCfg", rng: np.random.Generator
) -> dict[str, np.ndarray] | None:
    """Sample guaranteed-clear targets on the central platform."""
    if cfg.flat_patch_sampling is None:
        return None

    patches: dict[str, np.ndarray] = {}
    center = np.asarray(cfg.size, dtype=np.float64) * 0.5
    for name, patch_cfg in cfg.flat_patch_sampling.items():
        radius = float(patch_cfg.patch_radius)
        half_extent = max(float(cfg.platform_width) * 0.5 - radius, 0.0)
        xy = rng.uniform(-half_extent, half_extent, size=(patch_cfg.num_patches, 2)) + center
        patches[name] = np.column_stack((xy, np.zeros(patch_cfg.num_patches, dtype=np.float64)))
    return patches


@dataclass(kw_only=True)
class PerlinMeshRandomMultiBoxTerrainCfg(SubTerrainCfg):
    """Native MuJoCo boxes implementing main's random-multi-box tile contract."""

    box_height_mean: tuple[float, float] | list[float] | float = MISSING
    box_height_range: float = MISSING
    box_length_mean: tuple[float, float] | list[float] | float = MISSING
    box_length_range: float = MISSING
    box_width_mean: tuple[float, float] | list[float] | float = MISSING
    box_width_range: float = MISSING
    platform_width: float = MISSING
    generation_ratio: float = MISSING
    perlin_cfg: Any | None = None
    box_perlin_cfg: Any | None = None
    no_perlin_at_obstacle: bool = False
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float | None = None
    wall_prob: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    wall_height: float = 5.0
    wall_thickness: float = 0.05

    @staticmethod
    def _mean(value: tuple[float, float] | list[float] | float, difficulty: float, half_range: float) -> float:
        if isinstance(value, (tuple, list)):
            if value[0] < half_range:
                raise ValueError(f"minimum mean {value[0]} is smaller than half-range {half_range}")
            return float(value[0] + difficulty * (value[1] - value[0]))
        if value < half_range:
            raise ValueError(f"mean {value} is smaller than half-range {half_range}")
        return float(value)

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        if self.perlin_cfg is not None or self.box_perlin_cfg is not None:
            raise NotImplementedError("MJLab random-multi-box currently supports the shared flat-ground recipe only.")

        body = spec.body("terrain")
        width, length = float(self.size[0]), float(self.size[1])
        height_mean = self._mean(self.box_height_mean, difficulty, self.box_height_range)
        length_mean = self._mean(self.box_length_mean, difficulty, self.box_length_range)
        width_mean = self._mean(self.box_width_mean, difficulty, self.box_width_range)
        num_boxes = max(1, int(self.generation_ratio * width * length / (length_mean * width_mean)))

        geometries: list[TerrainGeometry] = []
        meshes: list[trimesh.Trimesh] = []

        # A shallow native box gives the zero-height surface used by main's
        # triangle-mesh ground without relying on a degenerate MuJoCo geom.
        ground_thickness = 0.1
        ground_position = (width * 0.5, length * 0.5, -ground_thickness * 0.5)
        ground = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(width * 0.5, length * 0.5, ground_thickness * 0.5),
            pos=ground_position,
        )
        geometries.append(TerrainGeometry(geom=ground))
        meshes.append(_box_mesh((width, length, ground_thickness), ground_position))

        for _ in range(num_boxes):
            box_width = width_mean + rng.uniform(-1.0, 1.0) * self.box_width_range
            box_length = length_mean + rng.uniform(-1.0, 1.0) * self.box_length_range
            box_height = height_mean + rng.uniform(-1.0, 1.0) * self.box_height_range
            x = rng.uniform(box_width * 0.5, width - box_width * 0.5)
            y = rng.uniform(box_length * 0.5, length - box_length * 0.5)
            if (
                width * 0.5 - self.platform_width * 0.5 - box_width * 0.5
                < x
                < width * 0.5 + self.platform_width * 0.5 + box_width * 0.5
                and length * 0.5 - self.platform_width * 0.5 - box_length * 0.5
                < y
                < length * 0.5 + self.platform_width * 0.5 + box_length * 0.5
            ):
                continue

            yaw = float(rng.uniform(0.0, 2.0 * math.pi))
            position = (float(x), float(y), float(box_height * 0.5))
            geom = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(float(box_width * 0.5), float(box_length * 0.5), float(box_height * 0.5)),
                pos=position,
                quat=(math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)),
            )
            geometries.append(TerrainGeometry(geom=geom))
            meshes.append(_box_mesh((box_width, box_length, box_height), position, yaw=yaw))

        wall_specs = (
            (
                (self.wall_thickness, length, self.wall_height),
                (-self.wall_thickness * 0.5, length * 0.5, self.wall_height * 0.5),
            ),
            (
                (self.wall_thickness, length, self.wall_height),
                (width + self.wall_thickness * 0.5, length * 0.5, self.wall_height * 0.5),
            ),
            (
                (width, self.wall_thickness, self.wall_height),
                (width * 0.5, -self.wall_thickness * 0.5, self.wall_height * 0.5),
            ),
            (
                (width, self.wall_thickness, self.wall_height),
                (width * 0.5, length + self.wall_thickness * 0.5, self.wall_height * 0.5),
            ),
        )
        for probability, (extents, position) in zip(self.wall_prob, wall_specs, strict=True):
            if rng.uniform() >= probability:
                continue
            geom = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=tuple(float(value * 0.5) for value in extents),
                pos=position,
            )
            geometries.append(TerrainGeometry(geom=geom))
            meshes.append(_box_mesh(extents, position))

        output = TerrainOutput(
            origin=np.asarray((width * 0.5, length * 0.5, 0.0), dtype=np.float64),
            geometries=geometries,
            flat_patches=_sample_center_patches(self, rng),
        )
        output.instinct_surface_mesh = trimesh.util.concatenate(meshes)
        return output


__all__ = ["PerlinMeshRandomMultiBoxTerrainCfg"]
