"""Parkour curriculum terms called directly by both engines.

Both engines' terrain objects already expose ``update_env_origins``, ``terrain_levels`` and a
generator with ``size``, so one implementation of each walk is enough. Isaac Lab's originals
hard-coded the command name and (for the distance walk) read the unqualified ``root_pos_w``
alias; these take the command as a parameter and read the hub spelling.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from instinctlab.compat.env import get_command

if TYPE_CHECKING:
    from instinctlab.compat.env import RlEnv


def terrain_levels_vel(
    env: RlEnv, env_ids: torch.Tensor, command_name: str
) -> torch.Tensor:
    """Move robots to harder or easier tiles from how far they walked this episode.

    Walked more than half a tile: climb a level. Walked less than half the commanded distance
    over the episode: drop a level. A plane has no generator and fails here rather than
    reporting a constant zero, which would look like a working curriculum that never moves.
    """
    terrain = env.scene.terrain
    generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    if generator is None:
        raise RuntimeError(
            "terrain_levels_vel needs a generated terrain; the scene's terrain has no generator. "
            "A plane has no levels to climb."
        )
    asset = env.scene["robot"]
    command = get_command(env, command_name)
    distance = torch.norm(
        asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
        dim=1,
    )
    move_up = distance > generator.size[0] / 2
    move_down = (
        distance
        < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    )
    move_down = move_down & ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def tracking_exp_vel(
    env: RlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    lin_vel_threshold: tuple = (0.3, 0.6),
    ang_vel_threshold: tuple = (0.3, 0.5),
) -> torch.Tensor:
    """Move robots to harder or easier tiles from exponential velocity-tracking scores.

    Both parkour references read ``tracking_exp_vel_xy`` / ``tracking_exp_vel_yaw`` off the
    command term and call ``terrain.update_env_origins``. Those metric keys are published by
    the pose-velocity command; if they are absent this raises with the keys that *are* there,
    rather than treating a missing score as zero and never climbing.

    A plane has no generator and fails here, for the same reason as :func:`terrain_levels_vel`.
    Isaac Lab's original hard-coded the command name ``base_velocity``; this one takes it as a
    parameter.
    """
    terrain = env.scene.terrain
    generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    if generator is None:
        raise RuntimeError(
            "tracking_exp_vel needs a generated terrain; the scene's terrain has no generator. "
            "A plane has no levels to climb."
        )
    term = env.command_manager.get_term(command_name)
    metrics = getattr(term, "metrics", None)
    if metrics is None:
        raise RuntimeError(
            f"Command {command_name!r} has no metrics; tracking_exp_vel reads "
            "tracking_exp_vel_xy and tracking_exp_vel_yaw."
        )
    needed = ("tracking_exp_vel_xy", "tracking_exp_vel_yaw")
    missing = [key for key in needed if key not in metrics]
    if missing:
        available = ", ".join(sorted(metrics)) if metrics else "none"
        raise RuntimeError(
            f"Command {command_name!r} is missing metrics {missing}. Available: {available}."
        )
    tracking_exp_vel_xy = metrics["tracking_exp_vel_xy"][env_ids]
    tracking_exp_vel_yaw = metrics["tracking_exp_vel_yaw"][env_ids]
    move_up = (tracking_exp_vel_xy > lin_vel_threshold[1]) * (
        tracking_exp_vel_yaw > ang_vel_threshold[1]
    )
    move_down = tracking_exp_vel_xy < lin_vel_threshold[0]
    move_down = move_down * ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
