"""Isaac-native event implementations used by the generic term builders."""

from __future__ import annotations

from typing import Any


def randomize_default_joint_pos(
    env: Any,
    env_ids: Any,
    asset_cfg: Any,
    offset_distribution_params: tuple[float, float] | None,
    operation: str = "add",
    distribution: str = "uniform",
) -> None:
    """Randomize native defaults and copy them onto the canonical action axis by name."""
    import torch
    from isaaclab.envs.mdp.events import _randomize_prop_by_op

    from instinctlab.utils.name_order import copy_named_columns_

    asset = env.scene[asset_cfg.name]
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)
    else:
        joint_ids = torch.tensor(
            asset_cfg.joint_ids, dtype=torch.int, device=asset.device
        )

    if offset_distribution_params is None:
        return

    positions = asset.data.default_joint_pos.to(asset.device).clone()
    positions = _randomize_prop_by_op(
        positions,
        offset_distribution_params,
        env_ids,
        joint_ids,
        operation=operation,
        distribution=distribution,
    )[env_ids][:, joint_ids]

    if isinstance(joint_ids, slice):
        asset.data.default_joint_pos[env_ids] = positions
        selected_joint_names = tuple(asset.joint_names)
    else:
        asset.data.default_joint_pos[env_ids[:, None], joint_ids] = positions
        selected_joint_names = tuple(
            asset.joint_names[index] for index in joint_ids.tolist()
        )

    action = env.action_manager.get_term("joint_pos")
    copy_named_columns_(
        action._offset,
        positions,
        env_ids,
        value_names=selected_joint_names,
        target_names=tuple(action._joint_names),
    )


def randomize_ray_offsets(
    env: Any,
    env_ids: Any,
    asset_cfg: Any,
    offset_pose_ranges: dict[str, tuple[float, float]],
    distribution: str = "uniform",
) -> None:
    """Randomize the starts and directions of an Isaac ray sensor in place."""
    import isaaclab.utils.math as math_utils
    import torch

    sensor = env.scene[asset_cfg.name]
    num_env_ids = env.scene.num_envs if env_ids is None else len(env_ids)
    ray_starts = sensor.ray_starts[env_ids]
    ray_directions = sensor.ray_directions[env_ids]

    keys = ("x", "y", "z", "roll", "pitch", "yaw")
    ranges = torch.tensor(
        [offset_pose_ranges.get(key, (0.0, 0.0)) for key in keys],
        device=ray_starts.device,
    )
    samples = (
        math_utils.sample_uniform(
            ranges[:, 0],
            ranges[:, 1],
            (num_env_ids, 6),
            device=ray_starts.device,
        )[..., None, :]
        .repeat(1, sensor.num_rays, 1)
        .flatten(0, 1)
    )
    rotations = math_utils.quat_from_euler_xyz(
        samples[..., 3], samples[..., 4], samples[..., 5]
    )
    ray_starts += samples[..., :3].reshape(ray_starts.shape)
    ray_directions = math_utils.quat_apply(
        rotations.reshape(*ray_directions.shape[:-1], 4), ray_directions
    )

    sensor.ray_starts[env_ids] = ray_starts
    sensor.ray_directions[env_ids] = ray_directions
