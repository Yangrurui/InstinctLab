"""mjlab pinhole camera that measures Isaac's quantity, not mjlab's stock default.

Stock ``RayCastSensorCfg`` generates MuJoCo-camera rays (-Z forward), starts at the
attach frame with no offset, includes geom groups ``(0, 1, 2)``, and reports a miss
as ``distance=-1``. The parkour camera is a world-convention pinhole (+X forward,
+Z up) hung from the torso by a pose offset, hits an explicit list of bodies plus
the terrain, and reports a miss as ``+inf``.

The group mask is the trap this module exists to avoid. Group 2 on the G1 is the
visual shoe; group 3 is the collision capsule. Hits are selected by body name,
then by mesh geoms of those bodies. Unlisted geoms that share a group are hopped
over rather than accepted -- a first-hit on an unlisted hand must not hide the
ground behind it, and must not look like a hit on a listed link.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Any

from instinctlab.compat.math import quat_apply, quat_inv, quat_mul
from instinctlab.engines.ray_alignment import refuse_unhonored_ray_alignment
from instinctlab.spec.sensor import RayCasterRef

__all__ = ["pinhole_ray_caster"]

_TERRAIN_BODY = "terrain"
_MAX_HOPS = 6
_HOP_EPS = 1e-4


def pinhole_ray_caster(sensor: RayCasterRef) -> Any:
    """A mjlab sensor cfg that implements a pinhole :class:`RayCasterRef`."""
    from mjlab.sensor import ObjRef, PinholeCameraPatternCfg
    from mjlab.sensor.raycast_sensor import RayCastSensor, RayCastSensorCfg

    if sensor.pattern.kind != "pinhole":
        raise ValueError(f"mjlab camera {sensor.name!r} has pattern.kind={sensor.pattern.kind!r}.")
    refuse_unhonored_ray_alignment(sensor)
    if sensor.miss != "infinity":
        raise ValueError(f"mjlab camera {sensor.name!r} has miss={sensor.miss!r}; the portable contract is +inf.")
    if not sensor.hit_bodies() and not sensor.hits_terrain():
        raise ValueError(f"mjlab camera {sensor.name!r} names nothing to hit.")

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
            mask = _camera_geom_mask(
                mj_model,
                bodies=sensor.hit_bodies(),
                include_terrain=sensor.hits_terrain(),
                device=device,
            )
            self._allowed_geom_mask = mask
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

        def prepare_rays(self) -> None:
            """Full attach-body rotation plus the world-convention offset.

            Both parkour source *codes* do this. Both source *configs* write
            ``ray_alignment="yaw"``; that field is leftover from the ray-caster
            parent and is not what the camera uses. A pinhole declared as
            ``yaw`` is refused at compile so the ignore cannot stay silent.
            """
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
            self._filter_and_continue()
            self._write_image_plane()

        def _filter_and_continue(self) -> None:
            """Keep listed hits; hop past anything else, including closer collision capsules."""
            import warp as wp

            assert self._distances is not None and self._hit_pos_w is not None
            assert self._ray_geomid is not None
            assert self._cached_world_origins is not None and self._cached_world_rays is not None
            geom_ids = wp.to_torch(self._ray_geomid).to(dtype=torch.long)
            distances = self._distances
            hit_pos = self._hit_pos_w
            origins = self._cached_world_origins
            rays = self._cached_world_rays
            inf = torch.full_like(distances, float("inf"))
            inf3 = torch.full_like(hit_pos, float("inf"))

            hit = distances >= 0.0
            allowed = torch.zeros_like(hit)
            valid_ids = geom_ids.clamp(min=0, max=self._allowed_geom_mask.numel() - 1)
            allowed[hit] = self._allowed_geom_mask[valid_ids[hit]]
            accept = hit & allowed & (distances > self._min_distance)
            still = hit & ~accept
            final_dist = torch.where(accept, distances, inf)
            final_pos = torch.where(accept.unsqueeze(-1), hit_pos, inf3)
            if not bool(still.any()):
                self._distances = final_dist
                self._hit_pos_w = final_pos
                return

            traveled = torch.zeros_like(distances)
            traveled[still] = distances[still] + _HOP_EPS
            current = origins.clone()
            current[still] = hit_pos[still] + rays[still] * _HOP_EPS
            pnt = wp.to_torch(self._ray_pnt).view_as(origins)
            vec = wp.to_torch(self._ray_vec).view_as(rays)
            if self._wp_device is None:
                raise RuntimeError("Camera hop needs the sensor Warp device.")
            for _ in range(_MAX_HOPS):
                remaining = self.cfg.max_distance - traveled
                still = still & (remaining > 0.0)
                if not bool(still.any()):
                    break
                pnt.copy_(origins)
                vec.copy_(rays)
                pnt[still] = current[still]
                if self._ctx is None:
                    raise RuntimeError("Camera hop needs a SensorContext.")
                # The in-graph launch is wrapped in Simulation.sense's ScopedDevice.
                # A hop is post-graph: without the same device, Warp falls back to
                # cuda:0 and the kernel reads cuda:2 arrays -- illegal access, no
                # Python exception until the next tensor op.
                with wp.ScopedDevice(self._wp_device):
                    self.raycast_kernel(self._ctx.render_context)
                new_dist = wp.to_torch(self._ray_dist)
                new_geom = wp.to_torch(self._ray_geomid).to(dtype=torch.long)
                new_hit = new_dist >= 0.0
                new_allowed = torch.zeros_like(new_hit)
                new_ids = new_geom.clamp(min=0, max=self._allowed_geom_mask.numel() - 1)
                new_allowed[new_hit] = self._allowed_geom_mask[new_ids[new_hit]]
                new_pos = current + rays * new_dist.clamp(min=0.0).unsqueeze(-1)
                total = traveled + new_dist
                take = still & new_hit & new_allowed & (total > self._min_distance) & (new_dist <= remaining)
                final_dist = torch.where(take, total, final_dist)
                final_pos = torch.where(take.unsqueeze(-1), new_pos, final_pos)
                still = still & new_hit & ~take
                current = torch.where(still.unsqueeze(-1), new_pos + rays * _HOP_EPS, current)
                traveled = torch.where(still, total + _HOP_EPS, traveled)
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
        # Isaac's camera casts to ``max_distance * 2`` so an off-axis ray whose
        # image-plane depth is still ≤ max is not dropped. The image-plane clip
        # below is the semantic far plane; a farther hit is +inf, not 2.5.
        max_distance=sensor.max_distance * 2,
        exclude_parent_body=True,
        include_geom_groups=None,
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


def _camera_geom_mask(
    mj_model,
    *,
    bodies: tuple[str, ...],
    include_terrain: bool,
    device: str,
) -> torch.Tensor:
    """Geoms the camera may keep: terrain body, plus mesh geoms of named bodies.

    A listed body with no mesh falls back to every geom on that body so a
    capsule-only link is still visible rather than silently absent. Group
    numbers are not consulted.
    """
    import mujoco

    wanted: set[int] = set()
    for body_id in range(int(mj_model.nbody)):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        short = name.rsplit("/", 1)[-1]
        if include_terrain and (name == _TERRAIN_BODY or short == _TERRAIN_BODY):
            wanted.add(body_id)
        if short in bodies or name in bodies:
            wanted.add(body_id)
    if not wanted:
        raise RuntimeError(
            f"Camera hit list matched no bodies. Asked for terrain={include_terrain} "
            f"bodies={list(bodies)}. A group mask is not a substitute."
        )

    ngeom = int(mj_model.ngeom)
    mesh_by_body: dict[int, list[int]] = {body_id: [] for body_id in wanted}
    all_by_body: dict[int, list[int]] = {body_id: [] for body_id in wanted}
    for geom_id in range(ngeom):
        body_id = int(mj_model.geom_bodyid[geom_id])
        if body_id not in wanted:
            continue
        all_by_body[body_id].append(geom_id)
        if int(mj_model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh_by_body[body_id].append(geom_id)

    mask = torch.zeros(ngeom, device=device, dtype=torch.bool)
    for body_id in wanted:
        chosen = mesh_by_body[body_id] or all_by_body[body_id]
        if chosen:
            mask[torch.tensor(chosen, device=device, dtype=torch.long)] = True
    if not bool(mask.any()):
        raise RuntimeError("Camera hit list resolved bodies but those bodies have no geoms.")
    return mask
