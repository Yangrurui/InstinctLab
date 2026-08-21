"""mjlab pinhole camera aligned with InstinctMJ, not Isaac's grouped mesh targets.

Stock ``RayCastSensorCfg`` filters geom groups at the BVH kernel (default ``(0, 1, 2)``).
InstinctMJ's parkour ``NoisyGroupedRayCasterCameraCfg`` inherits that with empty
``mesh_prim_paths``, so there is no body-name mesh filter. When ``min_distance > 0``,
``GroupedRayCaster._apply_hit_filter_and_continue`` re-raycasts past hits at or inside
``min_distance`` (up to ``mesh_filter_max_hops``). That min-distance hop is kept here.
Isaac's parkour camera instead lists ``/World/ground`` plus G1 link visual prims.
"""

from __future__ import annotations

import ast
import os
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from instinctlab.compat.math import quat_apply, quat_inv, quat_mul
from instinctlab.engines.ray_alignment import refuse_unhonored_ray_alignment
from instinctlab.spec.sensor import RayCasterRef

__all__ = [
    "pinhole_camera_effective_semantics",
    "pinhole_camera_geom_groups",
    "pinhole_camera_hop_params",
    "pinhole_ray_caster",
]

_INSTINCTMJ_GROUPED_RAY_CFG = (
    Path(os.environ.get("INSTINCTMJ_ROOT", "/root/InstinctMJ"))
    / "src/instinct_mj/sensors/grouped_ray_caster/grouped_ray_caster_cfg.py"
)


def pinhole_camera_geom_groups() -> tuple[int, ...]:
    """Geom groups InstinctMJ parkour inherits from mjlab ``RayCastSensorCfg``."""
    from mjlab.sensor.raycast_sensor import RayCastSensorCfg

    field = RayCastSensorCfg.__dataclass_fields__["include_geom_groups"]
    default = field.default
    if default is None:
        raise RuntimeError(
            "mjlab RayCastSensorCfg.include_geom_groups default is None; "
            "InstinctMJ parkour expects the stock (0, 1, 2) mask."
        )
    return tuple(default)


def pinhole_camera_hop_params() -> dict[str, int | float | str]:
    """Hop limits InstinctMJ ``GroupedRayCasterCfg`` uses for min_distance continuation."""
    path = _INSTINCTMJ_GROUPED_RAY_CFG
    if not path.is_file():
        raise RuntimeError(f"InstinctMJ GroupedRayCasterCfg not found at {path}")
    max_hops: int | None = None
    epsilon: float | None = None
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.ClassDef) or node.name != "GroupedRayCasterCfg":
            continue
        for item in node.body:
            if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name):
                continue
            if item.value is None:
                continue
            if item.target.id == "mesh_filter_max_hops" and isinstance(item.value, ast.Constant):
                max_hops = int(item.value.value)
            if item.target.id == "mesh_filter_epsilon" and isinstance(item.value, ast.Constant):
                epsilon = float(item.value.value)
    if max_hops is None or epsilon is None:
        raise RuntimeError(f"{path} missing mesh_filter_max_hops or mesh_filter_epsilon defaults")
    return {
        "filter": "geom_groups_min_distance_hop",
        "hop_max": max_hops,
        "hop_epsilon_m": epsilon,
        "hop_triggers": ("distance <= min_distance",),
        "mesh_prim_paths_enabled": False,
    }


def pinhole_camera_effective_semantics(sensor: RayCasterRef) -> dict[str, Any]:
    """Effective mjlab hit semantics for manifest metadata (Isaac uses ``hit_bodies()``)."""
    groups = pinhole_camera_geom_groups()
    hop = pinhole_camera_hop_params()
    declared_bodies = sensor.hit_bodies()
    return {
        **hop,
        "include_geom_groups": groups,
        "min_distance_m": float(sensor.min_distance),
        "declared_hit_bodies_for_isaac": declared_bodies,
        "declared_hit_bodies_ignored_on_mjlab": bool(declared_bodies),
        "declared_hits_terrain": sensor.hits_terrain(),
        "group_filter_stage": "mujoco_warp_bvh_kernel",
    }


def pinhole_ray_caster(sensor: RayCasterRef) -> Any:
    """A mjlab sensor cfg that implements a pinhole :class:`RayCasterRef` under InstinctMJ semantics."""
    from mjlab.sensor import ObjRef, PinholeCameraPatternCfg
    from mjlab.sensor.raycast_sensor import RayCastSensor, RayCastSensorCfg

    if sensor.pattern.kind != "pinhole":
        raise ValueError(f"mjlab camera {sensor.name!r} has pattern.kind={sensor.pattern.kind!r}.")
    refuse_unhonored_ray_alignment(sensor)
    if sensor.miss != "infinity":
        raise ValueError(f"mjlab camera {sensor.name!r} has miss={sensor.miss!r}; the portable contract is +inf.")

    geom_groups = pinhole_camera_geom_groups()
    hop = pinhole_camera_hop_params()

    @dataclass
    class PinholeRayCastSensorCfg(RayCastSensorCfg):
        origin_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
        origin_offset_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        min_distance: float = 0.0
        image_plane_max: float = 2.5
        image_height: int = 36
        image_width: int = 64
        include_geom_groups: tuple[int, ...] | None = None

        def build(self) -> PinholeRayCastSensor:
            return PinholeRayCastSensor(self)

    class PinholeRayCastSensor(RayCastSensor):
        cfg: PinholeRayCastSensorCfg

        def initialize(self, mj_model, model, data, device: str) -> None:
            from .ray_device import ensure_warp_ray_on_device

            ensure_warp_ray_on_device(device)
            super().initialize(mj_model, model, data, device)
            self._local_offsets, self._local_directions = _world_pinhole_rays(
                width=sensor.pattern.width,
                height=sensor.pattern.height,
                horizontal_aperture=sensor.pattern.horizontal_aperture,
                vertical_aperture=sensor.pattern.vertical_aperture,
                focal_length=sensor.pattern.focal_length,
                device=device,
            )
            self._output = {
                "distance_to_image_plane": torch.zeros(
                    data.nworld, sensor.pattern.height, sensor.pattern.width, 1, device=device
                )
            }
            self._cam_pos = torch.zeros(data.nworld, 3, device=device)
            self._cam_quat = torch.zeros(data.nworld, 4, device=device)
            self._min_distance = float(self.cfg.min_distance)
            self._image_plane_max = float(self.cfg.image_plane_max)
            self._hop_epsilon = float(hop["hop_epsilon_m"])
            self._hop_max = int(hop["hop_max"])

        def prepare_rays(self) -> None:
            """Full attach-body rotation plus the world-convention offset."""
            assert self._data is not None and self._local_offsets is not None
            assert self._local_directions is not None
            if len(self._frame_infos) != 1 or self._frame_infos[0][0] != "body":
                raise RuntimeError(f"Camera {self.cfg.name!r} must attach to one body.")
            body_id = self._frame_infos[0][1]
            frame_pos = self._data.xpos[:, body_id]
            frame_quat = self._data.xquat[:, body_id]
            offset_pos = torch.as_tensor(self.cfg.origin_offset, device=frame_pos.device, dtype=frame_pos.dtype)
            offset_quat = torch.as_tensor(self.cfg.origin_offset_rot, device=frame_pos.device, dtype=frame_pos.dtype)
            cam_pos = frame_pos + quat_apply(frame_quat, offset_pos.expand_as(frame_pos))
            cam_quat = quat_mul(frame_quat, offset_quat.expand_as(frame_quat))
            self._cam_pos = cam_pos
            self._cam_quat = cam_quat

            batch = cam_pos.shape[0]
            num_rays = self._num_rays
            quat_exp = cam_quat.unsqueeze(1).expand(batch, num_rays, 4).reshape(batch * num_rays, 4)
            starts = quat_apply(
                quat_exp, self._local_offsets.unsqueeze(0).expand(batch, -1, -1).reshape(batch * num_rays, 3)
            ).view(batch, num_rays, 3)
            dirs = quat_apply(
                quat_exp, self._local_directions.unsqueeze(0).expand(batch, -1, -1).reshape(batch * num_rays, 3)
            ).view(batch, num_rays, 3)
            starts = starts + cam_pos.unsqueeze(1)

            assert self._ray_pnt is not None and self._ray_vec is not None
            import warp as wp

            wp.to_torch(self._ray_pnt).view(batch, num_rays, 3).copy_(starts)
            wp.to_torch(self._ray_vec).view(batch, num_rays, 3).copy_(dirs)
            self._cached_world_origins = starts
            self._cached_world_rays = dirs
            self._cached_frame_pos = cam_pos.unsqueeze(1)
            from instinctlab.compat.math import matrix_from_quat

            self._cached_frame_mat = matrix_from_quat(cam_quat).unsqueeze(1)

        def postprocess_rays(self) -> None:
            super().postprocess_rays()
            self._apply_min_distance_hop()
            self._write_image_plane()

        def _apply_min_distance_hop(self) -> None:
            """InstinctMJ GroupedRayCaster with empty mesh_prim_paths: BVH groups + min_distance hop."""
            if self._min_distance <= 0.0:
                return
            import warp as wp

            assert self._distances is not None and self._hit_pos_w is not None
            assert self._cached_world_origins is not None and self._cached_world_rays is not None
            distances = self._distances
            hit_pos = self._hit_pos_w
            origins = self._cached_world_origins
            rays = self._cached_world_rays
            inf = torch.full_like(distances, float("inf"))
            inf3 = torch.full_like(hit_pos, float("inf"))

            hit = distances >= 0.0
            accept = hit & (distances > self._min_distance)
            still = hit & ~accept
            final_dist = torch.where(accept, distances, inf)
            final_pos = torch.where(accept.unsqueeze(-1), hit_pos, inf3)
            if not bool(still.any()):
                self._distances = final_dist
                self._hit_pos_w = final_pos
                return

            eps = self._hop_epsilon
            traveled = torch.zeros_like(distances)
            traveled[still] = distances[still] + eps
            current = origins.clone()
            current[still] = hit_pos[still] + rays[still] * eps
            pnt = wp.to_torch(self._ray_pnt).view_as(origins)
            vec = wp.to_torch(self._ray_vec).view_as(rays)
            if self._wp_device is None:
                raise RuntimeError("Camera min_distance hop needs the sensor Warp device.")
            for _ in range(self._hop_max):
                remaining = self.cfg.max_distance - traveled
                still = still & (remaining > 0.0)
                if not bool(still.any()):
                    break
                pnt.copy_(origins)
                vec.copy_(rays)
                pnt[still] = current[still]
                if self._ctx is None:
                    raise RuntimeError("Camera min_distance hop needs a SensorContext.")
                with wp.ScopedDevice(self._wp_device):
                    self.raycast_kernel(self._ctx.render_context)
                new_dist = wp.to_torch(self._ray_dist)
                new_geom = wp.to_torch(self._ray_geomid).to(dtype=torch.long)
                new_hit = new_dist >= 0.0
                new_pos = current + rays * new_dist.clamp(min=0.0).unsqueeze(-1)
                total = traveled + new_dist
                take = still & new_hit & (total > self._min_distance) & (new_dist <= remaining)
                final_dist = torch.where(take, total, final_dist)
                final_pos = torch.where(take.unsqueeze(-1), new_pos, final_pos)
                if self._ray_geomid is not None:
                    geom_buf = wp.to_torch(self._ray_geomid).to(dtype=torch.long)
                    geom_buf = geom_buf.view_as(final_dist)
                    geom_buf = torch.where(take, new_geom.view_as(final_dist), geom_buf)
                    wp.to_torch(self._ray_geomid).copy_(geom_buf.reshape(wp.to_torch(self._ray_geomid).shape))
                still = still & new_hit & ~take
                current = torch.where(still.unsqueeze(-1), new_pos + rays * eps, current)
                traveled = torch.where(still, total + eps, traveled)
            self._distances = final_dist
            self._hit_pos_w = final_pos

        def _write_image_plane(self) -> None:
            """X of the hit in the world-convention camera frame. A miss is +inf."""
            assert self._hit_pos_w is not None
            delta = self._hit_pos_w - self._cam_pos.unsqueeze(1)
            finite = torch.isfinite(self._hit_pos_w).all(dim=-1)
            quat = quat_inv(self._cam_quat).unsqueeze(1).expand(-1, self._num_rays, -1).reshape(-1, 4)
            cam_delta = quat_apply(quat, delta.reshape(-1, 3)).view(-1, self._num_rays, 3)
            in_range = finite & (cam_delta[..., 0] <= self._image_plane_max)
            depth = torch.where(in_range, cam_delta[..., 0], torch.full_like(cam_delta[..., 0], float("inf")))
            self._output["distance_to_image_plane"] = depth.view(-1, self.cfg.image_height, self.cfg.image_width, 1)

        def _compute_data(self):
            data = super()._compute_data()
            data.output = self._output
            data.image_shape = (self.cfg.image_height, self.cfg.image_width)
            data.ray_hits_w = data.hit_pos_w
            data.pos_w = self._cam_pos
            data.quat_w_world = self._cam_quat
            return data

    return PinholeRayCastSensorCfg(
        name=sensor.name,
        frame=ObjRef(type="body", name=sensor.attach, entity=sensor.entity),
        pattern=PinholeCameraPatternCfg(
            width=sensor.pattern.width,
            height=sensor.pattern.height,
            fovy=sensor.pattern.vertical_fov_deg,
        ),
        ray_alignment="base",
        max_distance=sensor.max_distance * 2,
        exclude_parent_body=True,
        include_geom_groups=geom_groups,
        debug_vis=False,
        origin_offset=sensor.offset,
        origin_offset_rot=sensor.offset_rot,
        min_distance=sensor.min_distance,
        image_plane_max=sensor.max_distance,
        image_height=sensor.pattern.height,
        image_width=sensor.pattern.width,
    )


def _world_pinhole_rays(
    *,
    width: int,
    height: int,
    horizontal_aperture: float,
    vertical_aperture: float,
    focal_length: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Isaac / InstinctMJ pinhole: image (x-right, y-down, z-forward) to world (+X, +Y left, +Z up)."""
    f_x = width * focal_length / horizontal_aperture
    f_y = height * focal_length / vertical_aperture
    intrinsic = torch.zeros(3, 3, device=device, dtype=torch.float32)
    intrinsic[0, 0] = f_x
    intrinsic[1, 1] = f_y
    intrinsic[0, 2] = width / 2
    intrinsic[1, 2] = height / 2
    intrinsic[2, 2] = 1.0
    grid = torch.meshgrid(
        torch.arange(width, dtype=torch.int32, device=device),
        torch.arange(height, dtype=torch.int32, device=device),
        indexing="xy",
    )
    pixels = torch.vstack(list(map(torch.ravel, grid))).T
    pixels = torch.hstack([pixels, torch.ones((len(pixels), 1), device=device)])
    pixels = pixels + torch.tensor([[0.5, 0.5, 0.0]], device=device)
    cam = torch.matmul(torch.inverse(intrinsic), pixels.T)
    transform = torch.tensor([1.0, -1.0, -1.0], device=device).view(3, 1)
    cam = cam[[2, 0, 1], :] * transform
    directions = (cam / torch.norm(cam, dim=0, keepdim=True)).T.contiguous()
    starts = torch.zeros_like(directions)
    return starts, directions


def geom_groups_camera_mask(mj_model, groups: tuple[int, ...], device: str) -> torch.Tensor:
    """All geoms whose MuJoCo group is in ``groups`` — matches kernel include-list semantics."""
    ngeom = int(mj_model.ngeom)
    mask = torch.zeros(ngeom, device=device, dtype=torch.bool)
    group_arr = mj_model.geom_group
    for geom_id in range(ngeom):
        if int(group_arr[geom_id]) in groups:
            mask[geom_id] = True
    if not bool(mask.any()):
        raise RuntimeError(f"geom group mask empty for groups={groups}")
    return mask
