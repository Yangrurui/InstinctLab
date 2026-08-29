"""Engine-neutral lifecycle and visualization hooks for observation terms."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


_debug_image_sink: Callable[[str, Any], None] | None = None


def set_debug_image_sink(sink: Callable[[str, Any], None] | None) -> None:
    """Redirect observation debug images to a player-owned display."""
    global _debug_image_sink
    _debug_image_sink = sink


def show_debug_image(image: torch.Tensor, *, window_name: str) -> None:
    """Display an observation image through the player sink or a local OpenCV window."""
    if _debug_image_sink is not None:
        _debug_image_sink(window_name, image[:, -1, :, :])
        return
    panel = (
        image.permute(1, 2, 0, 3)
        .flatten(start_dim=0, end_dim=1)
        .flatten(start_dim=1, end_dim=2)
    )
    peak = float(panel.max())
    if peak <= 0:
        return
    pixels = (panel * 255.0 / peak).detach().cpu().numpy().astype("uint8")
    import cv2

    view = cv2.resize(
        pixels,
        (pixels.shape[1] * 5, pixels.shape[0] * 5),
        interpolation=cv2.INTER_AREA,
    )
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, view)
    cv2.waitKey(1)


def clear_resettable_observation_histories(
    env: Any, env_ids: Any | None = None
) -> None:
    manager = getattr(env, "observation_manager", None)
    if manager is None:
        return
    for group_cfgs in getattr(manager, "_group_obs_term_cfgs", {}).values():
        for cfg in group_cfgs:
            implementation = getattr(
                getattr(cfg, "func", None), "_impl", getattr(cfg, "func", None)
            )
            if not getattr(implementation, "clears_history_on_env_reset", False):
                continue
            selected = _env_ids(env, env_ids)
            if isinstance(selected, torch.Tensor) and selected.numel() == 0:
                continue
            implementation.clear_history(selected)


def _env_ids(env: Any, env_ids: Any | None) -> torch.Tensor | slice:
    if env_ids is None:
        return slice(None)
    if isinstance(env_ids, slice):
        return env_ids
    device = getattr(env, "device", "cpu")
    if isinstance(env_ids, torch.Tensor):
        return env_ids.reshape(-1).to(device=device, dtype=torch.long)
    if isinstance(env_ids, int):
        return torch.tensor([env_ids], device=device, dtype=torch.long)
    return torch.tensor(list(env_ids), device=device, dtype=torch.long)
