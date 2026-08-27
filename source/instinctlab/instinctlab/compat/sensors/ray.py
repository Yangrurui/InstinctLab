"""Portable ray-hit and depth-image conventions."""

from __future__ import annotations

import torch
from typing import Any

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
