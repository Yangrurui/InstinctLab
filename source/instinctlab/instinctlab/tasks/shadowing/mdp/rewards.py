"""Shadowing rewards called directly by both engines."""

from __future__ import annotations

from typing import Any

import torch

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
