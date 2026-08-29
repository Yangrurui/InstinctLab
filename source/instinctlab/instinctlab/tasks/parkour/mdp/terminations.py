"""Parkour termination terms called directly by both engines."""

from __future__ import annotations

import math
import torch
import weakref
from typing import Any

from instinctlab.compat import sensors as compat_sensors
from instinctlab.compat.env import RlEnv
from instinctlab.compat.motion_reference import exhausted_envs
from instinctlab.spec.sensor import ContactSensorRef, MotionReferenceRef

from .observations import _name

_MAP_HALF: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def time_out(env: RlEnv) -> torch.Tensor:
    """Whether the episode reached its time limit. Identical on both engines."""
    return env.episode_length_buf >= env.max_episode_length


def dataset_exhausted(
    env: RlEnv,
    sensor: MotionReferenceRef,
    reset_without_notice: bool = False,
    print_reason: bool = False,
) -> torch.Tensor:
    """Reset an exhausted motion reference and optionally hide the episode termination.

    This preserves the original Isaac and InstinctMJ parkour behavior. With
    ``reset_without_notice=True`` the reference stream is resampled immediately, while the
    termination manager receives an all-false result and leaves the robot episode running.
    """
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


def illegal_contact(env: RlEnv, sensor: ContactSensorRef) -> torch.Tensor:
    """Whether any of the referenced elements is touching something it should not be.

    This one is deliberately **not** a port of either engine's version, and the reason is worth
    stating, because both originals take a newton threshold and this one does not.

    Isaac Lab thresholds the norm of ``net_forces_w_history``, which is the world-frame *normal*
    force alone -- its own docstring warns that the tangential component is excluded. mjlab
    thresholds the norm of ``force_history``, which is the full 3-D contact force, expressed in the
    contact frame unless the sensor was configured otherwise. The same threshold in newtons
    therefore means "normal load above N" on one engine and "total load including friction above N"
    on the other, and the gap between them is whatever friction happens to be carrying at that
    instant. A foot planted on a slope crosses one threshold and not the other.

    So the portable version asks each engine's own sensor whether it considers the element to be in
    contact, via the contact-duration signal that both engines maintain internally. That loses the
    ability to ignore light brushes, which is what the threshold was buying. A task that genuinely
    needs a force threshold should declare this termination per engine and write down the tolerance
    -- which is the honest version of what a shared threshold was doing anyway.
    """
    touching = compat_sensors.in_contact(env.scene.sensors[sensor.name], sensor)
    return torch.any(touching, dim=1)


def _generator_field(generator: Any, name: str) -> Any:
    """Read a terrain-generator field, refusing a missing attribute rather than a default.

    ``hasattr`` misses mjlab dataclass fields that have no class default, so the class
    annotations are consulted as well. A missing field must not become a bound that never
    fires — that is the silent-failure class this repository has already been bitten by.
    """
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
    """Half-width and half-length of the whole generated map, or a loud failure."""
    terrain = env.scene.terrain
    try:
        return _MAP_HALF[terrain]
    except (KeyError, TypeError):
        pass
    generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    if generator is None:
        raise RuntimeError(
            "terrain_out_of_bounds needs a generated terrain; the scene's terrain has no generator. "
            "A plane has no map edge, and a termination that never fires looks like a working one."
        )
    size = _generator_field(generator, "size")
    n_rows = _generator_field(generator, "num_rows")
    n_cols = _generator_field(generator, "num_cols")
    border_width = _generator_field(generator, "border_width")
    if size is None or n_rows is None or n_cols is None or border_width is None:
        raise RuntimeError(
            "terrain_out_of_bounds found a generator but size/num_rows/num_cols/border_width is "
            f"None (size={size!r}, num_rows={n_rows!r}, num_cols={n_cols!r}, "
            f"border_width={border_width!r})."
        )
    half_x = 0.5 * (n_rows * size[0] + 2 * border_width)
    half_y = 0.5 * (n_cols * size[1] + 2 * border_width)
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
    """Terminate when the actor is too close to the edge of the whole generated map.

    Both references explicitly return false for their infinite plane. Generated terrain uses
    ``size``, ``num_rows``, ``num_cols`` and ``border_width``; missing any generator field fails
    rather than silently disabling the boundary on a finite map.

    Reads ``root_link_pos_w``. Isaac Lab's original used the legacy ``root_pos_w`` alias, which
    is the same link quantity.
    """
    terrain = env.scene.terrain
    generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    if generator is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    half_x, half_y = _map_half_extents(env)
    asset = env.scene[_name(asset_cfg)]
    x_out = torch.abs(asset.data.root_link_pos_w[:, 0]) > half_x - distance_buffer
    y_out = torch.abs(asset.data.root_link_pos_w[:, 1]) > half_y - distance_buffer
    return torch.logical_or(x_out, y_out)


def bad_orientation(
    env: RlEnv, limit_angle: float, asset_cfg: Any = None
) -> torch.Tensor:
    """Terminate when the base tilts past ``limit_angle`` from upright.

    Uses ``projected_gravity_b``, the portable attitude signal. The raw gravity vectors are
    denylisted: Isaac Lab spells ``GRAVITY_VEC_W`` and follows live sim gravity, mjlab spells
    ``gravity_vec_w`` and hard-codes ``[0, 0, -1]``.
    """
    asset = env.scene[_name(asset_cfg)]
    upright_cosine = torch.clamp(-asset.data.projected_gravity_b[:, 2], -1.0, 1.0)
    return torch.acos(upright_cosine).abs() > limit_angle


def root_height_below_env_origin_minimum(
    env: RlEnv, minimum_height: float, asset_cfg: Any = None
) -> torch.Tensor:
    """Terminate when the root drops more than ``minimum_height`` below a non-positive env origin.

    Both parkour references clamp the env-origin height at zero before subtracting. Reads
    ``root_link_pos_w``; Isaac Lab's original used the legacy ``root_pos_w`` alias.
    """
    asset = env.scene[_name(asset_cfg)]
    terrain_base_height = torch.clamp(env.scene.env_origins[:, 2], max=0.0)
    return asset.data.root_link_pos_w[:, 2] - terrain_base_height < minimum_height
