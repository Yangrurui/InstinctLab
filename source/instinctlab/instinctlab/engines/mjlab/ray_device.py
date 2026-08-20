"""Bind mujoco_warp's ray kernels to the simulation device.

``mujoco_warp._src.ray`` is imported with mjlab, when Warp's default device is
still ``cuda:0``. The first ``rays()`` launch then compiles that module for
cuda:0 and reads the sensor arrays that live on ``--device cuda:N`` -- illegal
access, no Python exception until the next tensor op. ``CUDA_VISIBLE_DEVICES``
would hide the first GPU rather than fix the bind.

Call :func:`ensure_warp_ray_on_device` from every mjlab ray sensor's
``initialize`` so construct-and-step works on any physical GPU.
"""

from __future__ import annotations

__all__ = ["ensure_warp_ray_on_device"]


def ensure_warp_ray_on_device(device: str) -> None:
    """Compile ``mujoco_warp._src.ray`` on ``device`` before the first sense()."""
    if not device.startswith("cuda"):
        return
    import mujoco_warp._src.ray as ray_mod
    import warp as wp

    wp_device = wp.get_device(device)
    with wp.ScopedDevice(wp_device):
        wp.load_module(ray_mod, device=wp_device)
