"""Shadowing termination terms called directly by both engines."""

from __future__ import annotations

import math
import weakref
from typing import Any

import torch

from instinctlab_engine.bridge.env import RlEnv
from instinctlab_engine.motion_reference import exhausted_envs
from instinctlab_engine.spec.sensor import MotionReferenceRef

from .observations import _name

_MAP_HALF: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def time_out(env: RlEnv) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


def dataset_exhausted(
    env: RlEnv,
    sensor: MotionReferenceRef,
    reset_without_notice: bool = False,
    print_reason: bool = False,
) -> torch.Tensor:
    motion_reference = env.scene.sensors[sensor.name]
    exhausted = exhausted_envs(motion_reference.data, motion_reference.aiming_frame_idx)
    if print_reason and exhausted.any():
        print("dataset_exhausted: ", exhausted.sum())
    if reset_without_notice:
        exhausted_ids = exhausted.nonzero(as_tuple=True)[0]
        if exhausted_ids.numel() > 0:
            motion_reference.reset(env_ids=exhausted_ids)
        exhausted[:] = False
    return exhausted


def _generator_field(generator: Any, name: str) -> Any:
    if hasattr(generator, name):
        return getattr(generator, name)
    for klass in type(generator).__mro__:
        if name in getattr(klass, "__annotations__", {}):
            raise RuntimeError(
                f"terrain generator {type(generator).__name__} declares {name!r} but the instance "
                "has no value; terrain_out_of_bounds cannot invent a map size."
            )
    raise RuntimeError(
        f"terrain generator {type(generator).__name__} has no {name!r}. "
        "terrain_out_of_bounds needs size, num_rows, num_cols and border_width."
    )


def _map_half_extents(env: RlEnv) -> tuple[float, float]:
    terrain = env.scene.terrain
    try:
        return _MAP_HALF[terrain]
    except (KeyError, TypeError):
        pass
    generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    if generator is None:
        raise RuntimeError(
            "terrain_out_of_bounds needs a generated terrain; the scene's terrain has no generator."
        )
    size = _generator_field(generator, "size")
    num_rows = _generator_field(generator, "num_rows")
    num_cols = _generator_field(generator, "num_cols")
    border_width = _generator_field(generator, "border_width")
    if size is None or num_rows is None or num_cols is None or border_width is None:
        raise RuntimeError(
            "terrain_out_of_bounds found a generator with an incomplete map extent declaration."
        )
    half_x = 0.5 * (num_rows * size[0] + 2 * border_width)
    half_y = 0.5 * (num_cols * size[1] + 2 * border_width)
    if not math.isfinite(half_x) or not math.isfinite(half_y):
        raise RuntimeError(
            f"terrain_out_of_bounds computed a non-finite map extent ({half_x}, {half_y})."
        )
    extents = (half_x, half_y)
    try:
        _MAP_HALF[terrain] = extents
    except TypeError:
        pass
    return extents


def terrain_out_of_bounds(
    env: RlEnv, distance_buffer: float, asset_cfg: Any = None
) -> torch.Tensor:
    terrain = env.scene.terrain
    generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    if generator is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    half_x, half_y = _map_half_extents(env)
    asset = env.scene[_name(asset_cfg)]
    x_out = torch.abs(asset.data.root_link_pos_w[:, 0]) > half_x - distance_buffer
    y_out = torch.abs(asset.data.root_link_pos_w[:, 1]) > half_y - distance_buffer
    return torch.logical_or(x_out, y_out)
