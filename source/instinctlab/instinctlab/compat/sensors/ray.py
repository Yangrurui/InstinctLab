"""Portable ray-hit and depth-image conventions."""

from __future__ import annotations

from typing import Any

import torch

from ..denylist import PortabilityError


def ray_hits_w(sensor: Any) -> torch.Tensor:
    """Return world-frame hit positions as ``(env, ray, 3)``; misses are ``+inf``."""
    data = sensor.data
    hits = getattr(data, "ray_hits_w", None)
    if hits is None:
        hits = getattr(data, "hit_pos_w", None)
    if hits is None:
        raise PortabilityError(
            f"{type(sensor).__name__} exposes neither ray_hits_w nor hit_pos_w; its ray output is unknown."
        )

    distances = getattr(data, "distances", None)
    if distances is None:
        return hits
    misses = distances < 0.0
    return hits.masked_fill(misses.unsqueeze(-1), float("inf"))


def ray_origin_z_w(sensor: Any) -> torch.Tensor:
    """Return one world-frame ray-origin height for every flattened hit.

    Isaac stores one ``pos_w`` per environment. MJLab stores ``frame_pos_w``
    because a sensor may contain several frames. This accessor only normalizes
    that shape; the task still decides offsets and miss values.
    """
    hits = ray_hits_w(sensor)
    data = sensor.data
    frame_positions = getattr(data, "frame_pos_w", None)
    if frame_positions is not None:
        if frame_positions.ndim != 3 or frame_positions.shape[-1] != 3:
            raise PortabilityError(
                f"Ray frame positions must be (env, frame, 3), got {tuple(frame_positions.shape)}."
            )
        num_frames = frame_positions.shape[1]
        if hits.shape[1] % num_frames:
            raise PortabilityError(
                f"{hits.shape[1]} ray hits cannot be divided among {num_frames} frames."
            )
        rays_per_frame = hits.shape[1] // num_frames
        return frame_positions[..., 2].repeat_interleave(rays_per_frame, dim=1)

    position = getattr(data, "pos_w", None)
    if position is None or position.ndim != 2 or position.shape[-1] != 3:
        shape = None if position is None else tuple(position.shape)
        raise PortabilityError(
            f"{type(sensor).__name__} has no ray-origin pos_w shaped (env, 3); got {shape}."
        )
    return position[:, 2:3].expand(-1, hits.shape[1])


def depth_image(sensor: Any) -> torch.Tensor:
    """Return distance-to-image-plane as ``(env, H, W, 1)``; misses are ``+inf``."""
    output = getattr(sensor.data, "output", None)
    if not isinstance(output, dict) or "distance_to_image_plane" not in output:
        raise PortabilityError(f"{type(sensor).__name__} has no data.output['distance_to_image_plane'].")

    image = output["distance_to_image_plane"]
    if image.ndim == 3:
        image = image.unsqueeze(-1)
    if image.ndim != 4 or image.shape[-1] != 1:
        raise PortabilityError(f"Depth image must be (env, H, W, 1), got {tuple(image.shape)}.")

    cfg = getattr(sensor, "cfg", None)
    far = getattr(cfg, "image_plane_max", None)
    if far is None:
        far = getattr(cfg, "max_distance", None)
    invalid = ~torch.isfinite(image)
    if far is not None:
        invalid |= image > float(far)
    # Keep this path entirely on the sensor device. Converting ``invalid.any()``
    # to a Python bool synchronizes CUDA once per observation, which stalls the
    # Perceptive rollout after its asynchronous ray cast.
    return image.masked_fill(invalid, float("inf"))
