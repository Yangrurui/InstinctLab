"""Isaac Sim's pose-velocity command: a thin ``CommandTerm`` subclass.

Column naming lives in :mod:`instinctlab.engines.pose_velocity` — both engines' curriculum
grids now use Isaac Lab's cumulative-proportion formula. This module only binds the mixin
to Isaac's ``CommandTerm``.

SDK imports stay inside :func:`build_command` so ``contract_report`` still answers without Isaac.
"""

from __future__ import annotations

from typing import Any

from instinctlab.engines.pose_velocity import PoseVelocityMixin, column_sub_terrain_names, command_params

__all__ = ["build_command", "column_sub_terrain_names"]


def build_command(spec: Any, ctx: Any) -> Any:
    """Native Isaac ``CommandTermCfg`` whose ``class_type`` is the mixin subclass."""
    from isaaclab.managers import CommandTerm, CommandTermCfg
    from isaaclab.utils import configclass

    fields = command_params(ctx.params(spec))

    class PoseVelocityCommand(PoseVelocityMixin, CommandTerm):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.robot = env.scene[cfg.asset_name]
            self.terrain = env.scene["terrain"]
            self._pose_velocity_setup()

        def _column_sub_terrain_names(self):
            return column_sub_terrain_names(self.terrain)

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = fields["lin_vel_x"]
        lin_vel_y: tuple[float, float] = fields["lin_vel_y"]
        ang_vel_z: tuple[float, float] = fields["ang_vel_z"]

    @configclass
    class PoseVelocityCommandCfg(CommandTermCfg):
        class_type: type = PoseVelocityCommand
        asset_name: str = fields["entity"]
        velocity_control_stiffness: float = fields["velocity_control_stiffness"]
        heading_control_stiffness: float = fields["heading_control_stiffness"]
        only_positive_lin_vel_x: bool = fields["only_positive_lin_vel_x"]
        ranges: Ranges = Ranges()
        random_velocity_terrain: list[str] | None = fields["random_velocity_terrain"]
        velocity_ranges: dict | None = fields["velocity_ranges"]
        lin_vel_threshold: float = fields["lin_vel_threshold"]
        ang_vel_threshold: float = fields["ang_vel_threshold"]
        lin_vel_metrics_std: float = fields["lin_vel_metrics_std"]
        ang_vel_metrics_std: float = fields["ang_vel_metrics_std"]
        rel_standing_envs: float = fields["rel_standing_envs"]
        target_dis_threshold: float = fields["target_dis_threshold"]

    return PoseVelocityCommandCfg(
        resampling_time_range=fields["resampling_time_range"],
        debug_vis=False,
    )
