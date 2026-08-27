"""mjlab pinhole camera aligned with InstinctMJ, not Isaac's grouped mesh targets.

Stock ``RayCastSensorCfg`` filters geom groups at the BVH kernel (default ``(0, 1, 2)``).
InstinctMJ's Parkour camera inherits that mask, parent exclusion, zero update period, and six
min-distance hops. Perceptive and HOI explicitly change those native settings to ``(0, 2)``, no
parent exclusion, a 1/60 s update period, and 24 hops. The task's mjlab profile carries only these
native settings; this module resolves them without knowing which task requested the camera.

InstinctMJ keeps ``mesh_prim_paths`` empty for these cameras, so there is no body-name mesh
filter. When ``min_distance > 0``, ``GroupedRayCaster._apply_hit_filter_and_continue`` re-raycasts
past hits at or inside ``min_distance``. That min-distance hop is kept here.
Isaac's parkour camera instead lists ``/World/ground`` plus G1 link visual prims.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import MISSING, dataclass
from typing import Any

import torch

from instinctlab.compat.math import quat_apply, quat_from_euler_xyz, quat_inv, quat_mul
from instinctlab.engines.ray_alignment import refuse_unhonored_ray_alignment
from instinctlab.spec.sensor import RayCasterRef

__all__ = [
    "pinhole_camera_effective_semantics",
    "pinhole_camera_geom_groups",
    "pinhole_camera_hop_params",
    "pinhole_camera_native_settings",
    "pinhole_ray_caster",
]

_DEFAULT_FILTER_MAX_HOPS = 6
_DEFAULT_FILTER_EPSILON = 1.0e-4
_DEFAULT_UPDATE_PERIOD = 0.0


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


def pinhole_camera_hop_params() -> dict[str, Any]:
    """Local defaults matching InstinctMJ; source-parity tests protect the copied values."""
    return {
        "filter": "geom_groups_min_distance_hop",
        "hop_max": _DEFAULT_FILTER_MAX_HOPS,
        "hop_epsilon_m": _DEFAULT_FILTER_EPSILON,
        "hop_triggers": ("distance <= min_distance",),
        "mesh_prim_paths_enabled": False,
    }


def _ray_cast_sensor_default(field_name: str) -> Any:
    from mjlab.sensor.raycast_sensor import RayCastSensorCfg

    field = RayCastSensorCfg.__dataclass_fields__[field_name]
    if field.default is MISSING:
        raise RuntimeError(f"mjlab RayCastSensorCfg.{field_name} has no scalar default")
    return field.default


def pinhole_camera_native_settings(
    sensor: RayCasterRef,
    profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve InstinctMJ's native camera fields from a generic engine profile entry."""
    hop = pinhole_camera_hop_params()
    settings: dict[str, Any] = {
        "include_geom_groups": pinhole_camera_geom_groups(),
        "exclude_parent_body": bool(_ray_cast_sensor_default("exclude_parent_body")),
        "mesh_filter_max_hops": int(hop["hop_max"]),
        "mesh_filter_epsilon": float(hop["hop_epsilon_m"]),
        "update_period": _DEFAULT_UPDATE_PERIOD,
    }
    camera_profiles = {} if profile is None else profile.get("pinhole_cameras", {})
    if not isinstance(camera_profiles, Mapping):
        raise TypeError("mjlab profile 'pinhole_cameras' must map sensor names to settings")
    overrides = camera_profiles.get(sensor.name, {})
    if not isinstance(overrides, Mapping):
        raise TypeError(f"mjlab pinhole camera profile for {sensor.name!r} must be a mapping")
    unknown = set(overrides) - set(settings)
    if unknown:
        raise ValueError(f"mjlab pinhole camera {sensor.name!r} has unknown profile settings: {sorted(unknown)}")
    settings.update(overrides)

    groups = settings["include_geom_groups"]
    if groups is not None:
        groups = tuple(groups)
        if not groups or any(isinstance(group, bool) or not isinstance(group, int) for group in groups):
            raise ValueError(f"mjlab pinhole camera {sensor.name!r} has invalid geom groups {groups!r}")
        settings["include_geom_groups"] = groups
    if not isinstance(settings["exclude_parent_body"], bool):
        raise TypeError(f"mjlab pinhole camera {sensor.name!r} exclude_parent_body must be bool")
    max_hops = settings["mesh_filter_max_hops"]
    if isinstance(max_hops, bool) or not isinstance(max_hops, int) or max_hops < 1:
        raise ValueError(f"mjlab pinhole camera {sensor.name!r} has invalid mesh_filter_max_hops={max_hops!r}")
    epsilon = settings["mesh_filter_epsilon"]
    if (
        not isinstance(epsilon, (int, float))
        or isinstance(epsilon, bool)
        or not math.isfinite(epsilon)
        or epsilon <= 0.0
    ):
        raise ValueError(f"mjlab pinhole camera {sensor.name!r} has invalid mesh_filter_epsilon={epsilon!r}")
    period = settings["update_period"]
    if (
        not isinstance(period, (int, float))
        or isinstance(period, bool)
        or not math.isfinite(period)
        or period < 0.0
    ):
        raise ValueError(f"mjlab pinhole camera {sensor.name!r} has invalid update_period={period!r}")
    settings["mesh_filter_epsilon"] = float(epsilon)
    settings["update_period"] = float(period)
    return settings


def pinhole_camera_effective_semantics(
    sensor: RayCasterRef,
    profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Effective mjlab hit semantics for manifest metadata (Isaac uses ``hit_bodies()``)."""
    native = pinhole_camera_native_settings(sensor, profile)
    declared_bodies = sensor.hit_bodies()
    return {
        "filter": "geom_groups_min_distance_hop",
        "hop_max": native["mesh_filter_max_hops"],
        "hop_epsilon_m": native["mesh_filter_epsilon"],
        "hop_triggers": ("distance <= min_distance",),
        "mesh_prim_paths_enabled": False,
        "include_geom_groups": native["include_geom_groups"],
        "exclude_parent_body": native["exclude_parent_body"],
        "update_period_s": native["update_period"],
        "min_distance_m": float(sensor.min_distance),
        "declared_hit_bodies_for_isaac": declared_bodies,
        "declared_hit_bodies_ignored_on_mjlab": bool(declared_bodies),
        "declared_hits_terrain": sensor.hits_terrain(),
        "group_filter_stage": "mujoco_warp_bvh_kernel",
    }


def pinhole_ray_caster(sensor: RayCasterRef, profile: Mapping[str, Any] | None = None) -> Any:
    """A mjlab sensor cfg that implements a pinhole :class:`RayCasterRef` under InstinctMJ semantics."""
    from mjlab.sensor import ObjRef, PinholeCameraPatternCfg
    from mjlab.sensor.raycast_sensor import RayCastSensor, RayCastSensorCfg

    if sensor.pattern.kind != "pinhole":
        raise ValueError(f"mjlab camera {sensor.name!r} has pattern.kind={sensor.pattern.kind!r}.")
    refuse_unhonored_ray_alignment(sensor)
    if sensor.miss != "infinity":
        raise ValueError(f"mjlab camera {sensor.name!r} has miss={sensor.miss!r}; the portable contract is +inf.")

    native = pinhole_camera_native_settings(sensor, profile)

    @dataclass
    class PinholeRayCastSensorCfg(RayCastSensorCfg):
        origin_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
        origin_offset_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        min_distance: float = 0.0
        image_plane_max: float = 2.5
        image_height: int = 36
        image_width: int = 64
        include_geom_groups: tuple[int, ...] | None = None
        mesh_filter_max_hops: int = 6
        mesh_filter_epsilon: float = 1.0e-4
        update_period: float = 0.0

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
                    data.nworld,
                    sensor.pattern.height,
                    sensor.pattern.width,
                    1,
                    device=device,
                )
            }
            self._cam_pos = torch.zeros(data.nworld, 3, device=device)
            self._cam_quat = torch.zeros(data.nworld, 4, device=device)
            self._reported_cam_pos = torch.zeros_like(self._cam_pos)
            self._reported_cam_quat = torch.zeros_like(self._cam_quat)
            self._min_distance = float(self.cfg.min_distance)
            self._image_plane_max = float(self.cfg.image_plane_max)
            self._hop_epsilon = float(self.cfg.mesh_filter_epsilon)
            self._hop_max = int(self.cfg.mesh_filter_max_hops)
            self._update_period_s = max(float(self.cfg.update_period), 0.0)
            self._elapsed_since_refresh = torch.zeros(data.nworld, device=device)
            self._refresh_mask = torch.ones(data.nworld, dtype=torch.bool, device=device)
            if self._update_period_s > 0.0:
                self._elapsed_since_refresh.fill_(self._update_period_s)
            self._calibration_noise = torch.zeros(data.nworld, 6, device=device)
            self.frame_sequence = 0

        def update(self, dt: float) -> None:
            super().update(dt)
            if self._update_period_s > 0.0:
                self._elapsed_since_refresh += float(dt)
            self.frame_sequence += 1

        def reset(self, env_ids=None) -> None:
            super().reset(env_ids)
            if env_ids is None:
                env_ids = slice(None)
            if self._update_period_s > 0.0:
                self._elapsed_since_refresh[env_ids] = self._update_period_s
            self._refresh_mask[env_ids] = True

        def set_offset_noise(self, env_ids, pose_delta: torch.Tensor) -> None:
            """Set persistent per-environment xyz/rpy calibration error."""
            self._calibration_noise[env_ids] = pose_delta

        def _refresh_reported_pose(self, cam_pos: torch.Tensor, cam_quat: torch.Tensor) -> None:
            """Advance InstinctMJ's per-environment camera refresh clock."""
            if self._update_period_s > 0.0:
                self._refresh_mask = self._elapsed_since_refresh >= (self._update_period_s - 1.0e-8)
                refresh_ids = self._refresh_mask.nonzero(as_tuple=False).squeeze(-1)
                if refresh_ids.numel() > 0:
                    self._elapsed_since_refresh[refresh_ids] = torch.remainder(
                        self._elapsed_since_refresh[refresh_ids],
                        self._update_period_s,
                    )
                    self._reported_cam_pos[refresh_ids] = cam_pos[refresh_ids]
                    self._reported_cam_quat[refresh_ids] = cam_quat[refresh_ids]
                return
            self._refresh_mask.fill_(True)
            self._reported_cam_pos.copy_(cam_pos)
            self._reported_cam_quat.copy_(cam_quat)

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
            offset_quat = torch.as_tensor(
                self.cfg.origin_offset_rot,
                device=frame_pos.device,
                dtype=frame_pos.dtype,
            )
            cam_pos = frame_pos + quat_apply(frame_quat, offset_pos.expand_as(frame_pos))
            cam_quat = quat_mul(frame_quat, offset_quat.expand_as(frame_quat))
            delta_quat = quat_from_euler_xyz(
                self._calibration_noise[:, 3],
                self._calibration_noise[:, 4],
                self._calibration_noise[:, 5],
            )
            ray_origin_pos = cam_pos + quat_apply(cam_quat, self._calibration_noise[:, :3])
            ray_quat = quat_mul(cam_quat, delta_quat)
            # InstinctMJ randomizes the camera-local ray arrays, not the camera's
            # reported pose. Keep the base pose for reporting/projection while
            # applying xyz/rpy calibration error to ray origins and directions.
            self._cam_pos = ray_origin_pos
            self._cam_quat = cam_quat
            self._refresh_reported_pose(cam_pos, cam_quat)

            batch = cam_pos.shape[0]
            num_rays = self._num_rays
            cam_quat_exp = cam_quat.unsqueeze(1).expand(batch, num_rays, 4).reshape(batch * num_rays, 4)
            ray_quat_exp = ray_quat.unsqueeze(1).expand(batch, num_rays, 4).reshape(batch * num_rays, 4)
            starts = quat_apply(
                cam_quat_exp,
                self._local_offsets.unsqueeze(0).expand(batch, -1, -1).reshape(batch * num_rays, 3),
            ).view(batch, num_rays, 3)
            dirs = quat_apply(
                ray_quat_exp,
                self._local_directions.unsqueeze(0).expand(batch, -1, -1).reshape(batch * num_rays, 3),
            ).view(batch, num_rays, 3)
            starts = starts + ray_origin_pos.unsqueeze(1)

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
            stale_ids = None
            stale_depth = None
            if self._update_period_s > 0.0:
                stale_ids = (~self._refresh_mask).nonzero(as_tuple=False).squeeze(-1)
                if stale_ids.numel() > 0:
                    stale_depth = self._output["distance_to_image_plane"][stale_ids].clone()
            super().postprocess_rays()
            self._apply_min_distance_hop()
            self._write_image_plane()
            if stale_depth is not None:
                self._output["distance_to_image_plane"][stale_ids] = stale_depth

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
            depth = torch.where(
                in_range,
                cam_delta[..., 0],
                torch.full_like(cam_delta[..., 0], float("inf")),
            )
            self._output["distance_to_image_plane"] = depth.view(-1, self.cfg.image_height, self.cfg.image_width, 1)

        def _compute_data(self):
            data = super()._compute_data()
            data.output = self._output
            data.image_shape = (self.cfg.image_height, self.cfg.image_width)
            data.ray_hits_w = data.hit_pos_w
            data.pos_w = self._reported_cam_pos
            data.quat_w_world = self._reported_cam_quat
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
        # InstinctMJ applies max_distance to the Euclidean ray length before
        # projecting the hit onto the image plane. Extending this to 2x lets
        # oblique rays see geometry that the reference camera cannot see.
        max_distance=sensor.max_distance,
        exclude_parent_body=native["exclude_parent_body"],
        include_geom_groups=native["include_geom_groups"],
        update_period=native["update_period"],
        debug_vis=False,
        origin_offset=sensor.offset,
        origin_offset_rot=sensor.offset_rot,
        min_distance=sensor.min_distance,
        image_plane_max=sensor.max_distance,
        image_height=sensor.pattern.height,
        image_width=sensor.pattern.width,
        mesh_filter_max_hops=native["mesh_filter_max_hops"],
        mesh_filter_epsilon=native["mesh_filter_epsilon"],
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
