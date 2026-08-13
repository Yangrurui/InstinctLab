"""Canonical velocity command generator for locomotion tasks."""

from __future__ import annotations

import torch

from instinctlab.managers.unified import CommandTerm, CommandTermCfg
from instinctlab.sim.math import heading, wrap_to_pi


class UniformVelocityCommand(CommandTerm):
    def __init__(self, cfg: CommandTermCfg, env) -> None:
        super().__init__(cfg, env)
        self._command = torch.zeros((env.num_envs, 3), device=env.device)
        self._heading_target = torch.zeros(env.num_envs, device=env.device)
        self._standing = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self._heading = torch.zeros_like(self._standing)
        self._time_left = torch.zeros(env.num_envs, device=env.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def reset(self, env_ids: torch.Tensor) -> None:
        self._resample(env_ids)

    def compute(self, dt: float) -> None:
        self._time_left -= dt
        due = self._time_left <= 0.0
        if torch.any(due):
            self._resample(due.nonzero(as_tuple=False).flatten())
        if torch.any(self._heading):
            yaw = heading(self.env.scene.articulations["robot"].data.root_quat_w)
            stiffness = float(self.cfg.params.get("heading_control_stiffness", 0.5))
            angular_range = self.cfg.params["ranges"]["ang_vel_z"]
            angular_command = stiffness * wrap_to_pi(self._heading_target - yaw)
            self._command[self._heading, 2] = angular_command[self._heading].clamp(*angular_range)
        self._command[self._standing] = 0.0

    def _resample(self, env_ids: torch.Tensor) -> None:
        if not env_ids.numel():
            return
        count = int(env_ids.numel())
        ranges = self.cfg.params["ranges"]
        self._command[env_ids, 0] = self.env.rng.uniform(
            "command.base_velocity.lin_x", *ranges["lin_vel_x"], (count,)
        )
        self._command[env_ids, 1] = self.env.rng.uniform(
            "command.base_velocity.lin_y", *ranges["lin_vel_y"], (count,)
        )
        self._command[env_ids, 2] = self.env.rng.uniform(
            "command.base_velocity.ang_z", *ranges["ang_vel_z"], (count,)
        )
        self._heading_target[env_ids] = self.env.rng.uniform(
            "command.base_velocity.heading", *ranges["heading"], (count,)
        )
        self._standing[env_ids] = (
            self.env.rng.uniform("command.base_velocity.standing_mask", 0.0, 1.0, (count,))
            < float(self.cfg.params.get("rel_standing_envs", 0.0))
        )
        self._heading[env_ids] = (
            self.env.rng.uniform("command.base_velocity.heading_mask", 0.0, 1.0, (count,))
            < float(self.cfg.params.get("rel_heading_envs", 0.0))
        )
        self._heading[env_ids] &= ~self._standing[env_ids]
        self._time_left[env_ids] = self.env.rng.uniform(
            "command.base_velocity.resampling_time",
            *self.cfg.params["resampling_time_range"],
            (count,),
        )


__all__ = ["UniformVelocityCommand"]
