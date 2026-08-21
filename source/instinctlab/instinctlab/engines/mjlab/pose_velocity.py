"""mjlab's pose-velocity command: a thin ``CommandTerm`` subclass.

Column naming lives in :mod:`instinctlab.engines.pose_velocity`. Curriculum mode now honors
``num_cols`` (via :class:`~instinctlab.engines.mjlab.terrains.terrain_generator.FiledTerrainGenerator`)
and uses the same cumulative-proportion formula as Isaac. Random mode still cannot name a
column — the shared helper returns ``None`` and the mixin raises rather than guessing.

SDK imports stay inside :func:`build_command`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from instinctlab.engines.pose_velocity import PoseVelocityMixin, column_sub_terrain_names, command_params

__all__ = ["build_command", "column_sub_terrain_names"]


def build_command(spec: Any, ctx: Any) -> Any:
    """Native mjlab ``CommandTermCfg`` whose ``build`` returns the mixin subclass."""
    from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

    fields = command_params(ctx.params(spec))

    class PoseVelocityCommand(PoseVelocityMixin, CommandTerm):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.robot = env.scene[cfg.entity_name]
            self.terrain = env.scene["terrain"]
            self._pose_velocity_setup()

        def _column_sub_terrain_names(self):
            return column_sub_terrain_names(self.terrain)

    @dataclass(kw_only=True)
    class Ranges:
        lin_vel_x: tuple[float, float] = fields["lin_vel_x"]
        lin_vel_y: tuple[float, float] = fields["lin_vel_y"]
        ang_vel_z: tuple[float, float] = fields["ang_vel_z"]

    # Class defaults must be immutable: stdlib dataclass rejects a list/dict
    # default (Isaac's ``configclass`` wraps those automatically). The parkour
    # task passes ``random_velocity_terrain=["perlin_rough_stand"]`` and a
    # velocity-box dict; those go through the constructor.
    @dataclass(kw_only=True)
    class PoseVelocityCommandCfg(CommandTermCfg):
        entity_name: str = fields["entity"]
        velocity_control_stiffness: float = fields["velocity_control_stiffness"]
        heading_control_stiffness: float = fields["heading_control_stiffness"]
        only_positive_lin_vel_x: bool = fields["only_positive_lin_vel_x"]
        ranges: Ranges
        random_velocity_terrain: list[str] | None = None
        velocity_ranges: dict | None = None
        lin_vel_threshold: float = fields["lin_vel_threshold"]
        ang_vel_threshold: float = fields["ang_vel_threshold"]
        lin_vel_metrics_std: float = fields["lin_vel_metrics_std"]
        ang_vel_metrics_std: float = fields["ang_vel_metrics_std"]
        rel_standing_envs: float = fields["rel_standing_envs"]
        target_dis_threshold: float = fields["target_dis_threshold"]
        patch_vis: bool = False

        def build(self, env):
            return PoseVelocityCommand(self, env)

    return PoseVelocityCommandCfg(
        resampling_time_range=fields["resampling_time_range"],
        debug_vis=False,
        ranges=Ranges(),
        random_velocity_terrain=fields["random_velocity_terrain"],
        velocity_ranges=fields["velocity_ranges"],
    )
