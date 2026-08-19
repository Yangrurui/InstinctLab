"""Curriculum terms that run unmodified under either engine's native manager.

The only term here is the terrain-level walk: both engines' terrain objects already expose
``update_env_origins``, ``terrain_levels`` and a generator with ``size``, so one implementation
is enough. Isaac Lab's original hard-coded the command name and read the unqualified
``root_pos_w`` alias; this one takes the command as a parameter and reads the hub spelling.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from instinctlab.compat.env import get_command

if TYPE_CHECKING:
    from instinctlab.compat.env import RlEnv

__all__ = ["terrain_levels_vel"]


def terrain_levels_vel(env: RlEnv, env_ids: torch.Tensor, command_name: str) -> torch.Tensor:
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
    distance = torch.norm(asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    move_up = distance > generator.size[0] / 2
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down = move_down & ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
