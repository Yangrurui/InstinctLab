"""Locomotion command generators called through either engine's generic wrapper."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import torch

from instinctlab.compat.math import wrap_to_pi
from instinctlab.compat.robot import root_linear_velocity_b


class UniformVelocityCommand:
    """Sample a body-frame SE(2) velocity command from uniform ranges."""

    def __init__(self, env: Any, params: Mapping[str, Any]) -> None:
        heading_command = bool(params["heading_command"])
        heading_range = params.get("heading")
        if heading_command and heading_range is None:
            raise ValueError("heading_command=True requires a heading range.")
        if heading_range is not None and not heading_command:
            raise ValueError("A heading range requires heading_command=True.")
        if float(params.get("init_velocity_prob", 0.0)) != 0.0:
            raise ValueError(
                "Portable UniformVelocityCommand does not write native root state; "
                "init_velocity_prob must remain 0.0."
            )

        self.cfg = SimpleNamespace(**dict(params))
        self._env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.robot = env.scene[params["entity"]]
        self.metrics = {
            "error_vel_xy": torch.zeros(self.num_envs, device=self.device),
            "error_vel_yaw": torch.zeros(self.num_envs, device=self.device),
        }
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.vel_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.heading_error = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        self.is_world_env = torch.zeros_like(self.is_heading_env)
        self.is_forward_env = torch.zeros_like(self.is_heading_env)

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    def _update_metrics(self) -> None:
        max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
        linear_velocity = root_linear_velocity_b(
            self.robot, anchor=self.cfg.metric_velocity_anchor
        )
        angular_velocity = self.robot.data.root_link_ang_vel_b
        self.metrics["error_vel_xy"] += (
            torch.norm(
                self.vel_command_b[:, :2] - linear_velocity[:, :2], dim=-1
            )
            / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - angular_velocity[:, 2])
            / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        env_ids = torch.as_tensor(
            env_ids, dtype=torch.long, device=self.device
        ).flatten()
        random = torch.empty(len(env_ids), device=self.device)
        self.vel_command_b[env_ids, 0] = random.uniform_(*self.cfg.lin_vel_x)
        self.vel_command_b[env_ids, 1] = random.uniform_(*self.cfg.lin_vel_y)
        self.vel_command_b[env_ids, 2] = random.uniform_(*self.cfg.ang_vel_z)
        if self.cfg.heading_command:
            self.heading_target[env_ids] = random.uniform_(*self.cfg.heading)
            self.is_heading_env[env_ids] = (
                random.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
            )
        self.is_standing_env[env_ids] = (
            random.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
        )

        if not self.cfg.extended_sampling:
            return
        self.is_world_env[env_ids] = (
            random.uniform_(0.0, 1.0) <= self.cfg.rel_world_envs
        )
        self.vel_command_w[env_ids] = self.vel_command_b[env_ids]
        self.is_forward_env[env_ids] = (
            random.uniform_(0.0, 1.0) <= self.cfg.rel_forward_envs
        )
        forward_ids = env_ids[self.is_forward_env[env_ids]]
        if len(forward_ids) > 0:
            self.vel_command_b[forward_ids, 0] = self.vel_command_b[
                forward_ids, 0
            ].abs().clamp(min=0.3)
            self.vel_command_b[forward_ids, 1:] = 0.0
        # InstinctMJ samples this inactive mask even when its probability is zero.
        random.uniform_(0.0, 1.0)

    def _update_command(self) -> None:
        if self.cfg.heading_command:
            self.heading_error = wrap_to_pi(
                self.heading_target - self.robot.data.heading_w
            )
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * self.heading_error[env_ids],
                min=self.cfg.ang_vel_z[0],
                max=self.cfg.ang_vel_z[1],
            )
        if self.cfg.extended_sampling and torch.any(self.is_world_env):
            env_ids = self.is_world_env.nonzero(as_tuple=False).flatten()
            heading = self.robot.data.heading_w[env_ids]
            cos_heading = torch.cos(heading)
            sin_heading = torch.sin(heading)
            velocity_x = self.vel_command_w[env_ids, 0]
            velocity_y = self.vel_command_w[env_ids, 1]
            self.vel_command_b[env_ids, 0] = (
                cos_heading * velocity_x + sin_heading * velocity_y
            )
            self.vel_command_b[env_ids, 1] = (
                -sin_heading * velocity_x + cos_heading * velocity_y
            )
        standing_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_ids] = 0.0
        self.vel_command_w[standing_ids] = 0.0
