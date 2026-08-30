"""mjlab VolumePoints: InstinctMJ's Sensor lifecycle around the portable grid.

SDK imports stay inside :func:`build_sensor` so ``contract_report`` still answers
without mjlab. Point velocity is the hub quantity ``v_link + ω × r``. mjwarp
``cvel`` linear is expressed at the free-joint subtree COM (the same point
``EntityData.body_link_vel_w`` uses), not the attach body's own subtree —
using the body's subtree silently adds ``ω × (pelvis − ankle)`` and the two
engines then scale the same penetration differently. Quaternions stay hub
``wxyz``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from instinctlab_engine.spec.sensor import VolumePointsRef

__all__ = ["build_sensor"]


def build_sensor(ref: VolumePointsRef) -> Any:
    """Native mjlab ``SensorCfg`` whose ``build`` returns the volume-points sensor."""
    import torch
    from dataclasses import dataclass

    from mjlab.entity.data import compute_velocity_from_cvel
    from mjlab.sensor import Sensor, SensorCfg
    from mjlab.utils.lab_api import math as math_utils

    bodies = tuple(ref.bodies)
    grid = torch.tensor(ref.grid.points(), dtype=torch.float32)

    @dataclass
    class VolumePointsData:
        pos_w: torch.Tensor
        quat_w: torch.Tensor
        vel_w: torch.Tensor
        ang_vel_w: torch.Tensor
        point_num_each_body: int
        points_pos_w: torch.Tensor
        points_vel_w: torch.Tensor
        penetration_offset: torch.Tensor

        @staticmethod
        def make_zero(num_envs: int, num_bodies: int, point_num_each_body: int, device="cpu"):
            return VolumePointsData(
                pos_w=torch.zeros((num_envs, num_bodies, 3), device=device),
                quat_w=torch.zeros((num_envs, num_bodies, 4), device=device),
                vel_w=torch.zeros((num_envs, num_bodies, 3), device=device),
                ang_vel_w=torch.zeros((num_envs, num_bodies, 3), device=device),
                point_num_each_body=point_num_each_body,
                points_pos_w=torch.zeros((num_envs, num_bodies, point_num_each_body, 3), device=device),
                points_vel_w=torch.zeros((num_envs, num_bodies, point_num_each_body, 3), device=device),
                penetration_offset=torch.zeros((num_envs, num_bodies, point_num_each_body, 3), device=device),
            )

    @dataclass
    class VolumePointsSensorCfg(SensorCfg):
        name: str = ref.name
        entity_name: str = ref.entity
        body_names: tuple[str, ...] = bodies
        debug_vis: bool = False
        velocity: str = "attach_link"

        def build(self) -> VolumePointsSensor:
            return VolumePointsSensor(self)

    class VolumePointsSensor(Sensor[VolumePointsData]):
        def __init__(self, cfg: VolumePointsSensorCfg):
            super().__init__()
            self.cfg = cfg
            self._volume_points_pattern: torch.Tensor | None = None
            self._sensor_data: VolumePointsData | None = None
            self._virtual_obstacles: dict[str, Any] = {}
            self.virtual_obstacles_registered = False
            self.registered_cylinder_count = 0
            self._device = "cpu"
            self._data = None
            self._num_envs = 0
            self._ALL_INDICES = torch.empty(0, dtype=torch.long)
            self._body_ids: torch.Tensor | None = None
            self._cvel_frame_ids: torch.Tensor | None = None
            self._body_names: list[str] = []
            self._num_bodies = 0

        @property
        def data(self) -> VolumePointsData:
            return super().data

        @property
        def num_bodies(self) -> int:
            return self._num_bodies

        @property
        def body_names(self) -> list[str]:
            return list(self._body_names)

        def register_virtual_obstacles(self, virtual_obstacles: dict[str, Any]) -> None:
            if not virtual_obstacles:
                raise RuntimeError(
                    f"volume-points sensor {self.cfg.name!r} was given an empty "
                    "virtual-obstacle dict. penetration_offset would stay zero."
                )
            self._virtual_obstacles = dict(virtual_obstacles)
            self.virtual_obstacles_registered = True
            self.registered_cylinder_count = _cylinder_count(self._virtual_obstacles)
            if self.registered_cylinder_count <= 0:
                raise RuntimeError(
                    f"volume-points sensor {self.cfg.name!r} registered "
                    f"{len(virtual_obstacles)} set(s) but 0 cylinders. "
                    "generate() found no edges."
                )

        def edit_spec(self, scene_spec, entities) -> None:
            del scene_spec, entities

        def initialize(self, mj_model, model, data, device: str) -> None:
            del model
            self._data = data
            self._device = device
            self._num_envs = data.nworld
            self._ALL_INDICES = torch.arange(self._num_envs, device=device, dtype=torch.long)
            body_ids, body_names = self._resolve_body_ids(mj_model)
            if not body_ids:
                raise RuntimeError(
                    "Failed to initialize volume points sensor for specified bodies."
                    f"\n\tEntity name        : {self.cfg.entity_name}"
                    f"\n\tBody patterns      : {self.cfg.body_names}"
                )
            if body_names != list(bodies):
                raise RuntimeError(
                    f"volume-points {self.cfg.name!r} resolved bodies {body_names} "
                    f"!= declared {list(bodies)}. The two engines would sum different clouds."
                )
            self._body_ids = torch.tensor(body_ids, device=device, dtype=torch.long)
            # mjwarp cvel linear lives at the kinematic-tree root subtree COM.
            root_ids = [int(mj_model.body_rootid[bid]) for bid in body_ids]
            self._cvel_frame_ids = torch.tensor(root_ids, device=device, dtype=torch.long)
            self._body_names = body_names
            self._num_bodies = len(body_ids)
            self._volume_points_pattern = grid.to(device)
            self._sensor_data = VolumePointsData.make_zero(
                num_envs=self._num_envs,
                num_bodies=self._num_bodies,
                point_num_each_body=self._volume_points_pattern.shape[0],
                device=device,
            )

        def _compute_data(self) -> VolumePointsData:
            self._refresh_volume_points(self._ALL_INDICES)
            self._refresh_penetration_offset(self._ALL_INDICES)
            return self._sensor_data

        def _refresh_volume_points(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
            env_ids = self._resolve_env_ids(env_ids)
            body_pos = self._data.xpos[env_ids][:, self._body_ids]
            body_quat = self._data.xquat[env_ids][:, self._body_ids]
            body_cvel = self._data.cvel[env_ids][:, self._body_ids]
            subtree_com = self._data.subtree_com[env_ids][:, self._cvel_frame_ids]
            body_vels = compute_velocity_from_cvel(body_pos, subtree_com, body_cvel)

            self._sensor_data.pos_w[env_ids] = body_pos
            self._sensor_data.quat_w[env_ids] = body_quat
            self._sensor_data.vel_w[env_ids] = body_vels[..., :3]
            self._sensor_data.ang_vel_w[env_ids] = body_vels[..., 3:]

            n_bodies = body_pos.shape[0] * body_pos.shape[1]
            points_pos_w = math_utils.transform_points(
                self._volume_points_pattern.unsqueeze(0).expand(n_bodies, -1, -1),
                body_pos.flatten(0, 1),
                body_quat.flatten(0, 1),
            ).reshape(*body_pos.shape[:2], self._sensor_data.point_num_each_body, 3)
            self._sensor_data.points_pos_w[env_ids] = points_pos_w
            points_vel_w = self._sensor_data.vel_w[env_ids].unsqueeze(-2).expand_as(points_pos_w).clone()
            points_vel_w += torch.linalg.cross(
                self._sensor_data.ang_vel_w[env_ids].unsqueeze(-2),
                points_pos_w - body_pos.unsqueeze(-2),
                dim=-1,
            )
            self._sensor_data.points_vel_w[env_ids] = points_vel_w

        def _refresh_penetration_offset(self, env_ids: Sequence[int] | torch.Tensor) -> None:
            env_ids = self._resolve_env_ids(env_ids)
            offset_buf = self._sensor_data.penetration_offset[env_ids]
            offset_buf[:] = 0.0
            depth_buf = torch.zeros_like(offset_buf[..., 0])
            for virtual_obstacle in self._virtual_obstacles.values():
                offset = virtual_obstacle.get_points_penetration_offset(
                    self._sensor_data.points_pos_w[env_ids].flatten(0, 2)
                ).reshape(self._sensor_data.points_pos_w[env_ids].shape)
                depth = torch.linalg.vector_norm(offset, dim=-1)
                mask = depth > depth_buf
                depth_buf[mask] = depth[mask]
                offset_buf[mask] = offset[mask]
            self._sensor_data.penetration_offset[env_ids] = offset_buf

        def debug_vis(self, visualizer) -> None:
            """Draw the foot sample cloud: green free, red inside a virtual edge."""
            if not self.cfg.debug_vis or self._sensor_data is None:
                return
            env_ids = list(visualizer.get_env_indices(self._num_envs))
            if not env_ids:
                return
            points = self._sensor_data.points_pos_w[env_ids].reshape(-1, 3)
            penetrated = (
                torch.linalg.vector_norm(self._sensor_data.penetration_offset[env_ids].reshape(-1, 3), dim=-1) > 0.0
            )
            if not bool(torch.any(penetrated)):
                points = torch.cat([points, torch.zeros_like(points[:1])], dim=0)
                penetrated = torch.cat([penetrated, torch.tensor([True], device=points.device)], dim=0)
            for point in points[~penetrated].cpu().numpy():
                visualizer.add_sphere(center=point, radius=0.01, color=(0.0, 1.0, 0.0, 1.0))
            for point in points[penetrated].cpu().numpy():
                visualizer.add_sphere(center=point, radius=0.01, color=(1.0, 0.0, 0.0, 1.0))

        def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
            if env_ids is None:
                return self._ALL_INDICES
            return torch.as_tensor(env_ids, device=self._device, dtype=torch.long)

        def _resolve_body_ids(self, mj_model) -> tuple[list[int], list[str]]:
            entity_name = self.cfg.entity_name.strip()
            body_ids: list[int] = []
            body_names: list[str] = []
            for pattern in self.cfg.body_names:
                matched: list[tuple[int, str]] = []
                for body_id in range(1, mj_model.nbody):
                    full_name = mj_model.body(body_id).name
                    if not full_name:
                        continue
                    if "/" in full_name:
                        full_entity, local_name = full_name.split("/", 1)
                    else:
                        full_entity, local_name = "", full_name
                    if entity_name and full_entity.lower() != entity_name.lower():
                        continue
                    if re.fullmatch(pattern, local_name):
                        matched.append((body_id, local_name))
                if not matched:
                    raise RuntimeError(f"volume-points {self.cfg.name!r} matched no body for {pattern!r}.")
                body_ids.extend(item[0] for item in matched)
                body_names.extend(item[1] for item in matched)
            return body_ids, body_names

    return VolumePointsSensorCfg()


def _cylinder_count(obstacles: dict[str, Any]) -> int:
    total = 0
    for obstacle in obstacles.values():
        edges = getattr(obstacle, "edges_pyt", None)
        if edges is not None:
            total += int(edges.shape[0])
            continue
        cylinders = getattr(obstacle, "cylinders", None)
        if cylinders is not None:
            total += int(getattr(cylinders, "num_cylinders", 0))
    return total
