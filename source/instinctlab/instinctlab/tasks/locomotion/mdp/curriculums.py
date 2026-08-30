"""Locomotion curriculum terms called directly by both engines."""

from __future__ import annotations

import torch

from instinctlab_engine.bridge.env import RlEnv, get_command


def terrain_levels_vel(
    env: RlEnv, env_ids: torch.Tensor, command_name: str
) -> torch.Tensor:
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
