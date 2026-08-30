"""Debug-image display shared by task observations and playback frontends."""

from __future__ import annotations

from collections.abc import Callable

import torch

__all__ = ["set_debug_image_sink", "show_debug_image"]

_debug_image_sink: Callable[[str, torch.Tensor], None] | None = None


def set_debug_image_sink(
    sink: Callable[[str, torch.Tensor], None] | None,
) -> None:
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
