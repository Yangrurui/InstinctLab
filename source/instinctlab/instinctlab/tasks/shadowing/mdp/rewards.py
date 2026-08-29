"""Shadowing rewards called directly by both engines."""

from __future__ import annotations

from typing import Any

import torch

from instinctlab.compat import robot as compat_robot
from instinctlab.compat.env import RlEnv

from .observations import _joint_ids, _name


def action_rate_l2(env: RlEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
    )


def joint_pos_limits(env: RlEnv, asset_cfg: Any = None) -> torch.Tensor:
    asset = env.scene[_name(asset_cfg)]
    joint_ids = _joint_ids(asset_cfg)
    joint_pos = asset.data.joint_pos[:, joint_ids]
    limits = asset.data.soft_joint_pos_limits[:, joint_ids]
    out_of_limits = -(joint_pos - limits[..., 0]).clip(max=0.0)
    out_of_limits += (joint_pos - limits[..., 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def applied_torque_limits_by_ratio(
    env: RlEnv,
    asset_cfg: Any = None,
    limit_ratio: float = 0.8,
) -> torch.Tensor:
    """Penalize native effort above the task-selected fraction of native limits."""
    asset = env.scene[_name(asset_cfg)]
    joint_ids = _joint_ids(asset_cfg)
    applied = torch.abs(compat_robot.joint_applied_torque(env, asset)[:, joint_ids])
    limits = compat_robot.joint_effort_limits(env, asset, joint_ids)
    excess = torch.clamp(applied - limits * limit_ratio, min=0.0)
    return torch.sum(torch.square(excess), dim=-1)
