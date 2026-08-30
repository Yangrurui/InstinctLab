"""Locomotion observations called directly by both engines."""

from __future__ import annotations

from typing import Any

import torch

from instinctlab_engine.bridge import robot as compat_robot
from instinctlab_engine.bridge.env import RlEnv, get_command


def base_ang_vel(env: RlEnv, asset_cfg: Any = None) -> torch.Tensor:
    asset = env.scene[_name(asset_cfg)]
    return compat_robot.root_angular_velocity_b(asset)


def base_lin_vel(env: RlEnv, asset_cfg: Any = None) -> torch.Tensor:
    return env.scene[_name(asset_cfg)].data.root_link_lin_vel_b


def projected_gravity(env: RlEnv, asset_cfg: Any = None) -> torch.Tensor:
    return env.scene[_name(asset_cfg)].data.projected_gravity_b


def joint_pos_rel(env: RlEnv, asset_cfg: Any = None) -> torch.Tensor:
    asset = env.scene[_name(asset_cfg)]
    joint_ids = _joint_ids(asset_cfg)
    return (
        asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    )


def joint_vel(env: RlEnv, asset_cfg: Any = None) -> torch.Tensor:
    asset = env.scene[_name(asset_cfg)]
    return asset.data.joint_vel[:, _joint_ids(asset_cfg)]


def last_action(env: RlEnv, action_name: str | None = None) -> torch.Tensor:
    if action_name is None:
        return env.action_manager.action
    from instinctlab_engine.bridge.env import raw_action

    return raw_action(env, action_name)


def generated_commands(env: RlEnv, command_name: str) -> torch.Tensor:
    return get_command(env, command_name)


def _name(asset_cfg: Any) -> str:
    return getattr(asset_cfg, "name", "robot") if asset_cfg is not None else "robot"


def _joint_ids(asset_cfg: Any) -> Any:
    if asset_cfg is None:
        return slice(None)
    ids = getattr(asset_cfg, "joint_ids", slice(None))
    return slice(None) if ids is None else ids
