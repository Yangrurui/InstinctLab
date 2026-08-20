"""mjlab ray caster that measures Isaac's quantity, not mjlab's stock default.

Stock ``RayCastSensorCfg`` starts at the attach frame, uses ``max_distance=10``, and
includes geom groups ``(0, 1, 2)``. On the G1 that group mask hits the shoe visual
mesh (group 2). The Isaac parkour scanner starts twenty metres above the ankle,
casts against ``/World/ground`` only, and reports ``+inf`` on a miss. Rewards were
tuned against that. This module is the deliberate rebuild.

Terrain is identified by the terrain *body*, not by a group mask. Group ``(0, 1, 2)``
is not "terrain only" -- that is the trap. Groups of the terrain geoms are used only
as the kernel include-list, and any non-terrain geom that shares those groups is a
hard error rather than a silent first-hit.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Any

from instinctlab.spec.sensor import RayCasterRef

__all__ = ["terrain_sky_ray_caster"]

_TERRAIN_BODY = "terrain"


def terrain_sky_ray_caster(sensor: RayCasterRef) -> Any:
    """A mjlab sensor cfg that implements :class:`RayCasterRef` under Isaac semantics."""
    from mjlab.sensor import ObjRef
    from mjlab.sensor.raycast_sensor import GridPatternCfg, RayCastSensor, RayCastSensorCfg

    if sensor.hit != "terrain":
        raise ValueError(f"mjlab ray caster {sensor.name!r} has hit={sensor.hit!r}; only 'terrain' is implemented.")
    if sensor.miss != "infinity":
        raise ValueError(f"mjlab ray caster {sensor.name!r} has miss={sensor.miss!r}; the portable contract is +inf.")
    if sensor.pattern.kind != "grid":
        raise ValueError(f"mjlab ray caster {sensor.name!r} has pattern.kind={sensor.pattern.kind!r}.")

    @dataclass
    class TerrainSkyRayCastSensorCfg(RayCastSensorCfg):
        """Stock cfg plus the sky offset. Groups are filled at initialize from terrain geoms."""

        origin_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
        # Empty until initialize discovers the terrain body's groups. Hitting nothing is
        # safer than the stock (0, 1, 2) mask, which includes the robot.
        include_geom_groups: tuple[int, ...] | None = ()

        def build(self) -> TerrainSkyRayCastSensor:
            return TerrainSkyRayCastSensor(self)

    class TerrainSkyRayCastSensor(RayCastSensor):
        cfg: TerrainSkyRayCastSensorCfg

        def initialize(self, mj_model, model, data, device: str) -> None:
            from .ray_device import ensure_warp_ray_on_device

            ensure_warp_ray_on_device(device)
            mask, groups = _terrain_geom_mask(mj_model, device)
            self.cfg.include_geom_groups = groups
            self._terrain_geom_mask = mask
            super().initialize(mj_model, model, data, device)
            # Pattern offsets stay the grid. The sky origin is applied in prepare_rays
            # with the same yaw rotation Isaac uses for OffsetCfg.

        def prepare_rays(self) -> None:
            """Isaac: ``OffsetCfg`` is baked into local ray starts, then yaw-rotated.

            ``combine_frame_transforms`` in Isaac's RayCaster is the mesh-to-physics
            offset, not ``OffsetCfg``. The 20 m sky offset is rotated by yaw only, so
            it stays world-up when the foot pitches. A full-R offset would slide the
            origin by ``20 * sin(pitch)`` and silently sample the wrong tile.
            """
            assert self._data is not None and self._local_offsets is not None
            assert self._local_directions is not None

            pos_list: list[torch.Tensor] = []
            mat_list: list[torch.Tensor] = []
            for frame_type, obj_id, _ in self._frame_infos:
                if frame_type == "body":
                    pos_list.append(self._data.xpos[:, obj_id])
                    mat_list.append(self._data.xmat[:, obj_id].view(-1, 3, 3))
                elif frame_type == "site":
                    pos_list.append(self._data.site_xpos[:, obj_id])
                    mat_list.append(self._data.site_xmat[:, obj_id].view(-1, 3, 3))
                else:
                    pos_list.append(self._data.geom_xpos[:, obj_id])
                    mat_list.append(self._data.geom_xmat[:, obj_id].view(-1, 3, 3))

            frame_pos = torch.stack(pos_list, dim=1)
            frame_mat = torch.stack(mat_list, dim=1)
            batch, frames = frame_pos.shape[:2]
            rays_per = self._num_rays_per_frame

            rot_mat = self._compute_alignment_rotation(frame_mat.reshape(batch * frames, 3, 3)).reshape(
                batch, frames, 3, 3
            )
            offset = torch.as_tensor(self.cfg.origin_offset, device=frame_pos.device, dtype=frame_pos.dtype)
            local_starts = self._local_offsets + offset
            world_origins = frame_pos[:, :, None, :] + torch.einsum("bfij,nj->bfni", rot_mat, local_starts)
            world_rays = torch.einsum("bfij,nj->bfni", rot_mat, self._local_directions)

            world_origins_flat = world_origins.reshape(batch, frames * rays_per, 3)
            world_rays_flat = world_rays.reshape(batch, frames * rays_per, 3)

            assert self._ray_pnt is not None and self._ray_vec is not None
            import warp as wp

            wp.to_torch(self._ray_pnt).view(batch, self._num_rays, 3).copy_(world_origins_flat)
            wp.to_torch(self._ray_vec).view(batch, self._num_rays, 3).copy_(world_rays_flat)

            self._cached_world_origins = world_origins_flat
            self._cached_world_rays = world_rays_flat
            self._cached_frame_pos = frame_pos
            self._cached_frame_mat = frame_mat

        def postprocess_rays(self) -> None:
            super().postprocess_rays()
            assert self._distances is not None and self._hit_pos_w is not None
            import warp as wp

            assert self._ray_geomid is not None
            geom_ids = wp.to_torch(self._ray_geomid).to(dtype=torch.long)
            hit = self._distances >= 0.0
            allowed = torch.zeros_like(hit)
            valid_ids = geom_ids.clamp(min=0, max=self._terrain_geom_mask.numel() - 1)
            allowed[hit] = self._terrain_geom_mask[valid_ids[hit]]
            miss = ~allowed
            self._distances = torch.where(miss, torch.full_like(self._distances, float("inf")), self._distances)
            inf = torch.full_like(self._hit_pos_w, float("inf"))
            self._hit_pos_w = torch.where(miss.unsqueeze(-1), inf, self._hit_pos_w)

        def _compute_data(self):
            data = super()._compute_data()
            data.ray_hits_w = data.hit_pos_w
            return data

    return TerrainSkyRayCastSensorCfg(
        name=sensor.name,
        frame=ObjRef(type="body", name=sensor.attach, entity=sensor.entity),
        pattern=GridPatternCfg(
            size=sensor.pattern.size,
            resolution=sensor.pattern.resolution,
            direction=sensor.direction,
        ),
        ray_alignment=sensor.ray_alignment,
        max_distance=sensor.max_distance,
        exclude_parent_body=True,
        include_geom_groups=(),
        debug_vis=False,
        origin_offset=sensor.offset,
    )


def _terrain_geom_mask(mj_model, device: str) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Geoms whose parent body is the terrain, and the groups those geoms occupy.

    Raises if a non-terrain geom shares one of those groups: the kernel filters by
    group, so an intruder would be the first hit and look like ground.
    """
    import mujoco

    terrain_bodies: set[int] = set()
    for body_id in range(int(mj_model.nbody)):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if name == _TERRAIN_BODY or name.endswith(f"/{_TERRAIN_BODY}"):
            terrain_bodies.add(body_id)
    if not terrain_bodies:
        raise RuntimeError(
            "Ray caster hit='terrain' but the model has no body named 'terrain'. "
            "That body is how terrain geometry is identified; a group mask is not a substitute."
        )

    ngeom = int(mj_model.ngeom)
    mask = torch.zeros(ngeom, device=device, dtype=torch.bool)
    groups: set[int] = set()
    for geom_id in range(ngeom):
        if int(mj_model.geom_bodyid[geom_id]) in terrain_bodies:
            mask[geom_id] = True
            groups.add(int(mj_model.geom_group[geom_id]))
    if not bool(mask.any()):
        raise RuntimeError("The terrain body has no geoms to ray-cast against.")

    intruders: list[str] = []
    for geom_id in range(ngeom):
        group = int(mj_model.geom_group[geom_id])
        if group in groups and int(mj_model.geom_bodyid[geom_id]) not in terrain_bodies:
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
            body = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, int(mj_model.geom_bodyid[geom_id]))
            intruders.append(f"{name} (body={body!r}, group={group})")
    if intruders:
        raise RuntimeError(
            "hit='terrain' cannot use the terrain geoms' groups as a kernel mask because "
            f"these non-terrain geoms share them and would be the first hit: {intruders}. "
            "That is the (0, 1, 2) trap: a group number is not 'terrain only'."
        )
    return mask, tuple(sorted(groups))
