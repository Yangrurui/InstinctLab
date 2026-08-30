"""Shadowing observations, imitation rewards, and failure checks.

Only public scene/data tensors are used. The two adapters lower selectors to native entity
configs, and both expose the explicit ``root_link_*`` names used below.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from instinctlab.compat import math as math_utils
from instinctlab.utils.math import quat_to_tan_norm


def depth_image(
    env, sensor, resize_shape=(18, 32), normalization_range=(0.0, 2.0), debug_vis=False
):
    """Reference camera pipeline: clamp/normalize first, then crop and resize.

    ``debug_vis`` is training-off and play-on. Viser patches it onto the live observation
    params; without this keyword MJLab's ``**term_cfg.params`` call crashes on step.
    """
    from instinctlab.compat.sensors import depth_image as read_depth

    raw = read_depth(env.scene.sensors[sensor.name]).squeeze(-1)
    lo, hi = normalization_range
    normalized = raw.clamp(lo, hi).sub(lo).div(hi - lo)
    if sensor.crop is not None:
        top, bottom, left, right = sensor.crop
        normalized = normalized[
            :, top : normalized.shape[1] - bottom, left : normalized.shape[2] - right
        ]
    processed = F.interpolate(
        normalized.unsqueeze(1), size=resize_shape, mode="bilinear", align_corners=False
    )
    if debug_vis:
        from instinctlab.compat.observation_terms import show_debug_image

        show_debug_image(processed, window_name="depth_image")
    return processed


def _name(cfg: Any, default: str) -> str:
    return cfg if isinstance(cfg, str) else getattr(cfg, "name", default)


def _ids(cfg: Any):
    ids = getattr(cfg, "body_ids", slice(None))
    return ids if ids is not None else slice(None)


def _combine(square: torch.Tensor, std: float, method: str) -> torch.Tensor:
    if method == "mean_prod":
        square = square.mean(dim=-1)
    elif method == "prod":
        square = square.sum(dim=-1)
    result = torch.exp(-square / (std * std))
    return result.sum(dim=-1) if method == "sum" else result


def link_position(env, asset_cfg, *, in_base_frame: bool = True):
    asset = env.scene[_name(asset_cfg, "robot")]
    # Both references track the link-frame origin.  Isaac Lab's legacy
    # ``body_pos_w`` currently aliases this tensor, but spelling the point
    # explicitly keeps this term out of the COM/link compatibility trap that
    # affects the corresponding velocity aliases.
    pos = asset.data.body_link_pos_w[:, _ids(asset_cfg)]
    if in_base_frame:
        inv = math_utils.subtract_frame_transforms(
            asset.data.root_link_pos_w,
            asset.data.root_link_quat_w,
        )
        pos = math_utils.transform_points(pos, *inv)
    return pos


def link_rotation(env, asset_cfg, *, in_base_frame: bool = True):
    asset = env.scene[_name(asset_cfg, "robot")]
    quat = asset.data.body_link_quat_w[:, _ids(asset_cfg)]
    if in_base_frame:
        root_inv = math_utils.quat_inv(asset.data.root_link_quat_w)
        quat = math_utils.quat_mul(
            root_inv.unsqueeze(1).expand(-1, quat.shape[1], -1), quat
        )
    return quat_to_tan_norm(quat)


def base_position_imitation(
    env, reference_cfg="motion_reference", asset_cfg="robot", std: float = 0.3
):
    reference = env.scene[_name(reference_cfg, "motion_reference")].reference_frame
    ref = reference.base_pos_w[:, 0]
    pos = env.scene[_name(asset_cfg, "robot")].data.root_link_pos_w
    reward = torch.exp(-torch.square(pos - ref).sum(dim=-1) / (std * std))
    return reward * reference.validity[:, 0]


def base_rotation_imitation(
    env,
    reference_cfg="motion_reference",
    asset_cfg="robot",
    std: float = 0.4,
    difference_type="axis_angle",
):
    reference = env.scene[_name(reference_cfg, "motion_reference")].reference_frame
    ref = reference.base_quat_w[:, 0]
    quat = env.scene[_name(asset_cfg, "robot")].data.root_link_quat_w
    if difference_type == "axis_angle":
        error = math_utils.axis_angle_from_quat(
            math_utils.quat_mul(ref, math_utils.quat_conjugate(quat))
        ).norm(dim=-1)
    elif difference_type == "box_minus":
        error = math_utils.quat_box_minus(ref, quat).norm(dim=-1)
    else:
        raise ValueError(f"Unsupported quaternion difference {difference_type!r}.")
    return torch.exp(-torch.square(error) / (std * std))


def _relative_reference(asset, buffers):
    robot_pos = asset.data.root_link_pos_w
    robot_quat = asset.data.root_link_quat_w
    ref_pos = buffers.base_pos_w[:, 0]
    ref_quat = buffers.base_quat_w[:, 0]
    delta = math_utils.yaw_quat(
        math_utils.quat_mul(robot_quat, math_utils.quat_inv(ref_quat))
    )
    base = robot_pos.clone()
    base[:, 2] = ref_pos[:, 2]
    rel_pos = base.unsqueeze(1) + math_utils.quat_apply(
        delta.unsqueeze(1).expand(-1, buffers.link_pos_w.shape[2], -1),
        buffers.link_pos_w[:, 0] - ref_pos.unsqueeze(1),
    )
    rel_quat = math_utils.quat_mul(
        delta.unsqueeze(1).expand(-1, buffers.link_quat_w.shape[2], -1),
        buffers.link_quat_w[:, 0],
    )
    return rel_pos, rel_quat


def link_position_imitation(
    env,
    reference_cfg="motion_reference",
    asset_cfg="robot",
    std=0.3,
    combine_method="mean_prod",
    in_base_frame=False,
    in_relative_world_frame=True,
):
    asset = env.scene[_name(asset_cfg, "robot")]
    buffers = env.scene[_name(reference_cfg, "motion_reference")].reference_frame
    actual = link_position(env, asset_cfg, in_base_frame=in_base_frame)
    if in_base_frame:
        target = buffers.link_pos_b[:, 0]
    elif in_relative_world_frame:
        target = _relative_reference(asset, buffers)[0]
    else:
        target = buffers.link_pos_w[:, 0]
    return _combine(torch.square(actual - target).sum(dim=-1), std, combine_method)


def link_rotation_imitation(
    env,
    reference_cfg="motion_reference",
    asset_cfg="robot",
    std=0.4,
    combine_method="mean_prod",
    in_base_frame=False,
    in_relative_world_frame=True,
):
    asset = env.scene[_name(asset_cfg, "robot")]
    buffers = env.scene[_name(reference_cfg, "motion_reference")].reference_frame
    actual = asset.data.body_link_quat_w[:, _ids(asset_cfg)]
    if in_base_frame:
        root_inv = math_utils.quat_inv(asset.data.root_link_quat_w)
        actual = math_utils.quat_mul(
            root_inv.unsqueeze(1).expand(-1, actual.shape[1], -1), actual
        )
        target = buffers.link_quat_b[:, 0]
    elif in_relative_world_frame:
        target = _relative_reference(asset, buffers)[1]
    else:
        target = buffers.link_quat_w[:, 0]
    error = math_utils.quat_error_magnitude(
        actual.reshape(-1, 4), target.reshape(-1, 4)
    ).view(actual.shape[:2])
    return _combine(torch.square(error), std, combine_method)


def _link_velocity_imitation(
    env, reference_cfg, asset_cfg, std, combine_method, angular
):
    asset = env.scene[_name(asset_cfg, "robot")]
    buffers = env.scene[_name(reference_cfg, "motion_reference")].reference_frame
    # main and InstinctMJ both use the velocity of the link-frame origin.
    # On Isaac Lab ``body_lin_vel_w`` is the COM velocity, so preferring that
    # legacy alias silently optimizes a different imitation objective.
    attr = "body_link_ang_vel_w" if angular else "body_link_lin_vel_w"
    actual = getattr(asset.data, attr)[:, _ids(asset_cfg)]
    target = buffers.link_ang_vel_w[:, 0] if angular else buffers.link_lin_vel_w[:, 0]
    return _combine(torch.square(actual - target).sum(dim=-1), std, combine_method)


def link_linear_velocity_imitation(
    env,
    reference_cfg="motion_reference",
    asset_cfg="robot",
    std=1.0,
    combine_method="mean_prod",
):
    return _link_velocity_imitation(
        env, reference_cfg, asset_cfg, std, combine_method, False
    )


def link_angular_velocity_imitation(
    env,
    reference_cfg="motion_reference",
    asset_cfg="robot",
    std=3.14,
    combine_method="mean_prod",
):
    return _link_velocity_imitation(
        env, reference_cfg, asset_cfg, std, combine_method, True
    )


def base_position_too_far(
    env,
    reference_cfg="motion_reference",
    asset_cfg="robot",
    distance_threshold=0.25,
    height_only=True,
    check_at_keyframe_threshold=-1,
    print_reason=False,
):
    reference = env.scene[_name(reference_cfg, "motion_reference")].data
    ref = reference.base_pos_w[:, 0]
    pos = env.scene[_name(asset_cfg, "robot")].data.root_link_pos_w
    diff = (pos - ref).abs() if height_only else (pos - ref).norm(dim=-1)
    return (diff[:, 2] if height_only else diff) > distance_threshold


def projected_gravity_too_far(
    env,
    reference_cfg="motion_reference",
    asset_cfg="robot",
    projected_gravity_threshold=0.8,
    z_only=False,
    check_at_keyframe_threshold=-1,
    print_reason=False,
):
    asset = env.scene[_name(asset_cfg, "robot")]
    quat = asset.data.root_link_quat_w
    reference = env.scene[_name(reference_cfg, "motion_reference")].data
    ref = reference.base_quat_w[:, 0]
    projected_gravity = asset.data.projected_gravity_b
    gravity_w = math_utils.quat_apply(quat, projected_gravity)
    diff = projected_gravity - math_utils.quat_apply_inverse(ref, gravity_w)
    return (
        diff[:, 2].abs() if z_only else diff.norm(dim=-1)
    ) > projected_gravity_threshold


def link_position_too_far(
    env,
    reference_cfg="motion_reference",
    asset_cfg="robot",
    link_ids=slice(None),
    distance_threshold=0.25,
    height_only=True,
    in_base_frame=False,
    check_at_keyframe_threshold=-1,
    print_reason=False,
):
    actual = link_position(env, asset_cfg, in_base_frame=in_base_frame)[:, link_ids]
    buffers = env.scene[_name(reference_cfg, "motion_reference")].data
    target = (buffers.link_pos_b if in_base_frame else buffers.link_pos_w)[
        :, 0, link_ids
    ]
    diff = (actual - target).abs()
    distance = diff[..., 2] if height_only else diff.norm(dim=-1)
    return distance.amax(dim=-1) > distance_threshold


class IllegalResetContact:
    """Require repeated high reset contact during the first reset steps."""

    def __init__(self, cfg, env):
        self.counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    def reset(self, env_ids=None):
        self.counter[slice(None) if env_ids is None else env_ids] = 0

    def __call__(self, env, sensor, threshold=500.0, episode_length_threshold=2):
        from instinctlab.compat.sensors import contact_force_history

        history = contact_force_history(env.scene.sensors[sensor.name], sensor)
        contacts = torch.norm(history, dim=-1).amax(dim=1).gt(threshold).any(dim=-1)
        self.counter += contacts.long()
        return (self.counter >= episode_length_threshold) & (
            env.episode_length_buf <= episode_length_threshold
        )


def undesired_contacts(env, sensor, threshold=1.0):
    """Count native-force threshold violations after normalizing only the history axes."""
    from instinctlab.compat.sensors import contact_force_history

    history = contact_force_history(env.scene.sensors[sensor.name], sensor)
    return torch.norm(history, dim=-1).amax(dim=1).gt(threshold).float().sum(dim=-1)
