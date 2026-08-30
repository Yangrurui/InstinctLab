"""Parkour's pose-velocity command algorithm, importable with neither engine installed.

The engine adapters supply only a generic native ``CommandTerm`` wrapper.  All
Parkour-specific sampling, terrain binding, metrics, and command arithmetic
live here and reach both managers through ``CommandTermSpec(func=...)``.

Velocity boxes are keyed by sub-terrain **name**. A name maps to a *set* of columns: both
engines' curriculum generators assign columns by Isaac Lab's cumulative-proportion formula
(``j / num_cols + 0.001``). Upstream mjlab instead emits one column per type and ignores
``num_cols``; our ``FiledTerrainGenerator`` honors the declared width so the two grids match.
The remaining naming divergence is mjlab random mode, where a column is a mix of types and
this module returns ``None`` so the command raises rather than guessing an index.

Robot state is read under the hub spellings (``root_link_*``). The Isaac reference used the
legacy COM aliases for the tracking metrics; those aliases are denylisted here, and the angular
rows are the same tensor on Isaac anyway.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from instinctlab_engine.bridge.math import (
    euler_xyz_from_quat,
    quat_apply_inverse,
    wrap_to_pi,
    yaw_quat,
)
from instinctlab_engine.bridge.terrain import column_sub_terrain_names, resolve_named_columns

__all__ = [
    "POSE_VELOCITY_PARAM_KEYS",
    "PoseVelocityCommand",
    "command_params",
]

POSE_VELOCITY_PARAM_KEYS: frozenset[str] = frozenset(
    {
        "entity",
        "asset_name",
        "entity_name",
        "resampling_time_range",
        "velocity_control_stiffness",
        "heading_control_stiffness",
        "only_positive_lin_vel_x",
        "lin_vel_x",
        "lin_vel_y",
        "ang_vel_z",
        "rel_standing_envs",
        "random_velocity_terrain",
        "velocity_ranges",
        "lin_vel_threshold",
        "ang_vel_threshold",
        "lin_vel_metrics_std",
        "ang_vel_metrics_std",
        "target_dis_threshold",
        "debug_vis",
    }
)
"""TermSpec parameter surface both engine builders accept. Extra keys are rejected."""


def _range_pair(value: Any, name: str, *, positive: bool = False) -> tuple[float, float]:
    """Return a finite ordered two-value range with a useful declaration error."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"pose_velocity {name} must be a two-value range, got {value!r}.")
    lo, hi = value
    if isinstance(lo, bool) or isinstance(hi, bool):
        raise ValueError(  # noqa: TRY004 - keep the established declaration error contract
            f"pose_velocity {name} must contain numbers, got {value!r}."
        )
    try:
        pair = (float(lo), float(hi))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"pose_velocity {name} must contain numbers, got {value!r}.") from exc
    if not all(math.isfinite(item) for item in pair) or pair[0] > pair[1]:
        raise ValueError(f"pose_velocity {name} must be finite and ordered, got {value!r}.")
    if positive and pair[0] <= 0.0:
        raise ValueError(f"pose_velocity {name} must be positive, got {value!r}.")
    return pair


def _finite_scalar(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(  # noqa: TRY004 - keep the established declaration error contract
            f"pose_velocity {name} must be numeric, got {value!r}."
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"pose_velocity {name} must be numeric, got {value!r}.") from exc
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        qualifier = f" and at least {minimum}" if minimum is not None else ""
        raise ValueError(f"pose_velocity {name} must be finite{qualifier}, got {value!r}.")
    return number


def command_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a TermSpec's params into the fields both engine configs carry.

    ``debug_vis=True`` is refused rather than stored. Training keeps the flag off (InstinctMJ
    only sets it in the play factory). Play flips the live command term; putting True in a
    TermSpec would turn drawing on for every training step.
    """
    unknown = sorted(set(params) - POSE_VELOCITY_PARAM_KEYS)
    if unknown:
        raise ValueError(f"pose_velocity does not honor {unknown}. It accepts {sorted(POSE_VELOCITY_PARAM_KEYS)}.")
    if params.get("debug_vis"):
        raise ValueError(
            "pose_velocity debug visualization is not wired through TermSpec. InstinctMJ "
            "turns it on only in the play factory; play patches the live command term. "
            "Set debug_vis=False."
        )
    required = ("resampling_time_range", "lin_vel_x", "lin_vel_y", "ang_vel_z")
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError(f"pose_velocity needs {missing}.")
    resampling_time_range = _range_pair(params["resampling_time_range"], "resampling_time_range", positive=True)
    lin_vel_x = _range_pair(params["lin_vel_x"], "lin_vel_x")
    lin_vel_y = _range_pair(params["lin_vel_y"], "lin_vel_y")
    ang_vel_z = _range_pair(params["ang_vel_z"], "ang_vel_z")
    rel_standing_envs = _finite_scalar(params.get("rel_standing_envs", 0.0), "rel_standing_envs")
    if not 0.0 <= rel_standing_envs <= 1.0:
        raise ValueError(f"pose_velocity rel_standing_envs must be in [0, 1], got {rel_standing_envs}.")

    velocity_ranges = params.get("velocity_ranges")
    if velocity_ranges is not None:
        if not isinstance(velocity_ranges, Mapping):
            raise ValueError("pose_velocity velocity_ranges must map terrain names to velocity boxes.")
        normalized_boxes: dict[str, dict[str, tuple[float, float]]] = {}
        required_box_keys = {"lin_vel_x", "lin_vel_y", "ang_vel_z"}
        for terrain_name, box in velocity_ranges.items():
            if not isinstance(terrain_name, str) or not terrain_name:
                raise ValueError(f"pose_velocity velocity_ranges has an invalid terrain name {terrain_name!r}.")
            if not isinstance(box, Mapping) or set(box) != required_box_keys:
                have = sorted(box) if isinstance(box, Mapping) else type(box).__name__
                raise ValueError(
                    f"pose_velocity velocity box {terrain_name!r} must contain exactly "
                    f"{sorted(required_box_keys)}, got {have}."
                )
            normalized_boxes[terrain_name] = {
                key: _range_pair(box[key], f"velocity_ranges[{terrain_name!r}][{key!r}]")
                for key in sorted(required_box_keys)
            }
        velocity_ranges = normalized_boxes

    random_velocity_terrain = params.get("random_velocity_terrain")
    if random_velocity_terrain is not None:
        if isinstance(random_velocity_terrain, str) or not isinstance(random_velocity_terrain, Sequence):
            raise ValueError("pose_velocity random_velocity_terrain must be a sequence of terrain names.")
        random_velocity_terrain = list(random_velocity_terrain)
        if any(not isinstance(name, str) or not name for name in random_velocity_terrain):
            raise ValueError("pose_velocity random_velocity_terrain contains an invalid terrain name.")
        if len(set(random_velocity_terrain)) != len(random_velocity_terrain):
            raise ValueError("pose_velocity random_velocity_terrain contains duplicate names.")

    entity = params.get("entity") or params.get("asset_name") or params.get("entity_name") or "robot"
    if not isinstance(entity, str) or not entity:
        raise ValueError(f"pose_velocity entity must be a non-empty string, got {entity!r}.")
    return {
        "entity": entity,
        "resampling_time_range": resampling_time_range,
        "velocity_control_stiffness": _finite_scalar(
            params.get("velocity_control_stiffness", 1.0), "velocity_control_stiffness", minimum=0.0
        ),
        "heading_control_stiffness": _finite_scalar(
            params.get("heading_control_stiffness", 1.0), "heading_control_stiffness", minimum=0.0
        ),
        "only_positive_lin_vel_x": params.get("only_positive_lin_vel_x", False),
        "lin_vel_x": lin_vel_x,
        "lin_vel_y": lin_vel_y,
        "ang_vel_z": ang_vel_z,
        "rel_standing_envs": rel_standing_envs,
        "random_velocity_terrain": random_velocity_terrain,
        "velocity_ranges": velocity_ranges,
        "lin_vel_threshold": _finite_scalar(params.get("lin_vel_threshold", 0.15), "lin_vel_threshold", minimum=0.0),
        "ang_vel_threshold": _finite_scalar(params.get("ang_vel_threshold", 0.15), "ang_vel_threshold", minimum=0.0),
        "lin_vel_metrics_std": _finite_scalar(
            params.get("lin_vel_metrics_std", 0.5), "lin_vel_metrics_std", minimum=1e-12
        ),
        "ang_vel_metrics_std": _finite_scalar(
            params.get("ang_vel_metrics_std", 0.5), "ang_vel_metrics_std", minimum=1e-12
        ),
        "target_dis_threshold": _finite_scalar(
            params.get("target_dis_threshold", 0.2), "target_dis_threshold", minimum=0.0
        ),
        "debug_vis": False,
    }


class PoseVelocityCommand:
    """Task-owned command implementation adapted by either native command manager."""

    def __init__(self, env: Any, params: Mapping[str, Any]) -> None:
        fields = command_params(params)
        ranges = SimpleNamespace(
            lin_vel_x=fields["lin_vel_x"],
            lin_vel_y=fields["lin_vel_y"],
            ang_vel_z=fields["ang_vel_z"],
        )
        self.cfg = SimpleNamespace(**fields, ranges=ranges, patch_vis=False)
        self._env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.metrics: dict[str, torch.Tensor] = {}
        self.robot = env.scene[fields["entity"]]
        self.terrain = env.scene["terrain"]
        self._pose_velocity_setup()

    def _pose_velocity_setup(self) -> None:
        device = self.device
        n_env = self.num_envs
        self.pos_command_w = torch.zeros(n_env, 3, device=device)
        self.heading_command_w = torch.zeros(n_env, device=device)
        self.pos_command_b = torch.zeros(n_env, 3, device=device)
        self.vel_command_b = torch.zeros(n_env, 3, device=device)
        self.max_command_b = torch.zeros(n_env, 3, device=device)
        self.is_standing_env = torch.zeros(n_env, dtype=torch.bool, device=device)
        self.metrics["error_vel_xy"] = torch.zeros(n_env, device=device)
        self.metrics["error_vel_yaw"] = torch.zeros(n_env, device=device)
        self.metrics["tracking_exp_vel_xy"] = torch.zeros(n_env, device=device)
        self.metrics["tracking_exp_vel_yaw"] = torch.zeros(n_env, device=device)
        # Command-mix ratios. The tracking metrics above cannot separate "tracked a zero command
        # well" from "tracked a real command well", so a distribution shift against InstinctMJ
        # looks like a tracking regression. These four split the two apart.
        self.metrics["command_nonzero_ratio"] = torch.zeros(n_env, device=device)
        self.metrics["target_near_ratio"] = torch.zeros(n_env, device=device)
        self.metrics["standing_env_ratio"] = torch.zeros(n_env, device=device)
        self.metrics["random_velocity_env_ratio"] = torch.zeros(n_env, device=device)

        self.lin_vel_x_range = torch.zeros(n_env, 2, device=device)
        self.lin_vel_y_range = torch.zeros(n_env, 2, device=device)
        self.ang_vel_z_range = torch.zeros(n_env, 2, device=device)
        self.random_lin_vel_x_range = torch.zeros(n_env, 2, device=device)
        self.random_lin_vel_y_range = torch.zeros(n_env, 2, device=device)
        self.random_ang_vel_z_range = torch.zeros(n_env, 2, device=device)
        self.random_velocity_indices = torch.zeros(n_env, dtype=torch.bool, device=device)
        self.random_lin_vel_x = torch.zeros(n_env, device=device)
        self.random_lin_vel_y = torch.zeros(n_env, device=device)
        self.random_ang_vel_z = torch.zeros(n_env, device=device)

        self._column_names = list(column_sub_terrain_names(self.terrain))
        self._bind_velocity_boxes()

        self.random_lin_vel_x_range[:, 0] = self.cfg.ranges.lin_vel_x[0]
        self.random_lin_vel_x_range[:, 1] = self.cfg.ranges.lin_vel_x[1]
        self.random_lin_vel_y_range[:, 0] = self.cfg.ranges.lin_vel_y[0]
        self.random_lin_vel_y_range[:, 1] = self.cfg.ranges.lin_vel_y[1]
        self.random_ang_vel_z_range[:, 0] = self.cfg.ranges.ang_vel_z[0]
        self.random_ang_vel_z_range[:, 1] = self.cfg.ranges.ang_vel_z[1]

        if "target" not in self.terrain.flat_patches:
            raise RuntimeError(
                "pose_velocity requires a flat patch named 'target' on the terrain. "
                f"Found: {list(self.terrain.flat_patches.keys())}"
            )
        self.valid_targets: torch.Tensor = self.terrain.flat_patches["target"]

    def _bind_velocity_boxes(self) -> None:
        requested = list(dict.fromkeys([*(self.cfg.velocity_ranges or {}), *(self.cfg.random_velocity_terrain or [])]))
        if not requested:
            return
        columns = resolve_named_columns(self._column_names, requested)
        types = self.terrain.terrain_types
        for name, box in (self.cfg.velocity_ranges or {}).items():
            for column in columns[name]:
                env_ids = types == column
                self.lin_vel_x_range[env_ids, 0] = box["lin_vel_x"][0]
                self.lin_vel_x_range[env_ids, 1] = box["lin_vel_x"][1]
                self.lin_vel_y_range[env_ids, 0] = box["lin_vel_y"][0]
                self.lin_vel_y_range[env_ids, 1] = box["lin_vel_y"][1]
                self.ang_vel_z_range[env_ids, 0] = box["ang_vel_z"][0]
                self.ang_vel_z_range[env_ids, 1] = box["ang_vel_z"][1]
        for name in self.cfg.random_velocity_terrain or []:
            for column in columns[name]:
                self.random_velocity_indices[types == column] = True

    @property
    def command(self) -> torch.Tensor:
        """Body-frame velocity command ``[vx, vy, ωz]``. Shape ``(N, 3)``."""
        return self.vel_command_b

    @property
    def pose_command(self) -> torch.Tensor:
        return torch.cat([self.pos_command_b[:, :2], self.vel_command_b[:, 0:1]], dim=1)

    def _heading_w(self) -> torch.Tensor:
        return euler_xyz_from_quat(self.robot.data.root_link_quat_w)[2]

    def _update_metrics(self) -> None:
        max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
        lin_vel = self.robot.data.root_link_lin_vel_b
        ang_vel = self.robot.data.root_link_ang_vel_b
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - lin_vel[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += torch.abs(self.vel_command_b[:, 2] - ang_vel[:, 2]) / max_command_step
        lin_vel_error = torch.sum(torch.square(self.vel_command_b[:, :2] - lin_vel[:, :2]), dim=1)
        self.metrics["tracking_exp_vel_xy"] += (
            torch.exp(-lin_vel_error / self.cfg.lin_vel_metrics_std**2) / self._env.max_episode_length
        )
        angular_vel_error = torch.square(self.vel_command_b[:, 2] - ang_vel[:, 2])
        self.metrics["tracking_exp_vel_yaw"] += (
            torch.exp(-angular_vel_error / self.cfg.ang_vel_metrics_std**2) / self._env.max_episode_length
        )

        max_episode_length = self._env.max_episode_length
        target_dist = torch.norm((self.pos_command_w - self.robot.data.root_link_pos_w[:, :3])[:, :2], dim=1)
        command_nonzero = torch.logical_or(
            torch.norm(self.vel_command_b[:, :2], dim=1) > 1e-6,
            torch.abs(self.vel_command_b[:, 2]) > 1e-6,
        )
        self.metrics["command_nonzero_ratio"] += command_nonzero.float() / max_episode_length
        self.metrics["target_near_ratio"] += (target_dist <= self.cfg.target_dis_threshold).float() / max_episode_length
        self.metrics["standing_env_ratio"] += self.is_standing_env.float() / max_episode_length
        self.metrics["random_velocity_env_ratio"] += self.random_velocity_indices.float() / max_episode_length

    def _resample_command(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        ids = torch.randint(0, self.valid_targets.shape[2], size=(len(env_ids),), device=self.device)
        self.pos_command_w[env_ids] = self.valid_targets[
            self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids], ids
        ]
        r = torch.empty(len(env_ids), device=self.device)
        self.max_command_b[env_ids, 0] = self.lin_vel_x_range[env_ids, 0] + r.uniform_(0.0, 1.0) * (
            self.lin_vel_x_range[env_ids, 1] - self.lin_vel_x_range[env_ids, 0]
        )
        self.max_command_b[env_ids, 1] = self.lin_vel_y_range[env_ids, 0] + r.uniform_(0.0, 1.0) * (
            self.lin_vel_y_range[env_ids, 1] - self.lin_vel_y_range[env_ids, 0]
        )
        self.max_command_b[env_ids, 2] = self.ang_vel_z_range[env_ids, 0] + r.uniform_(0.0, 1.0) * (
            self.ang_vel_z_range[env_ids, 1] - self.ang_vel_z_range[env_ids, 0]
        )
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

        current_batch_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        current_batch_mask[env_ids] = True
        random_velocity_env_ids = (current_batch_mask & self.random_velocity_indices).nonzero(as_tuple=False).flatten()
        if len(random_velocity_env_ids) == 0:
            return
        self.random_lin_vel_x[random_velocity_env_ids] = self.random_lin_vel_x_range[
            random_velocity_env_ids, 0
        ] + torch.rand(len(random_velocity_env_ids), device=self.device) * (
            self.random_lin_vel_x_range[random_velocity_env_ids, 1]
            - self.random_lin_vel_x_range[random_velocity_env_ids, 0]
        )
        self.random_lin_vel_y[random_velocity_env_ids] = self.random_lin_vel_y_range[
            random_velocity_env_ids, 0
        ] + torch.rand(len(random_velocity_env_ids), device=self.device) * (
            self.random_lin_vel_y_range[random_velocity_env_ids, 1]
            - self.random_lin_vel_y_range[random_velocity_env_ids, 0]
        )
        self.random_ang_vel_z[random_velocity_env_ids] = self.random_ang_vel_z_range[
            random_velocity_env_ids, 0
        ] + torch.rand(len(random_velocity_env_ids), device=self.device) * (
            self.random_ang_vel_z_range[random_velocity_env_ids, 1]
            - self.random_ang_vel_z_range[random_velocity_env_ids, 0]
        )
        # Both references apply this dead-zone to the whole tensor, not the resampled slice.
        self.random_ang_vel_z *= torch.abs(self.random_ang_vel_z) > 0.5

    def _update_command(self) -> None:
        target_vec = self.pos_command_w - self.robot.data.root_link_pos_w[:, :3]
        target_dist = torch.norm(target_vec[:, :2], dim=1)
        self.pos_command_b[:] = quat_apply_inverse(yaw_quat(self.robot.data.root_link_quat_w), target_vec)
        self.vel_command_b[:, :2] = self.pos_command_b[:, :2] * self.cfg.velocity_control_stiffness

        target_vec = self.pos_command_w - self.robot.data.root_link_pos_w
        target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
        self.heading_command_w = wrap_to_pi(target_direction - self._heading_w())
        self.vel_command_b[:, 2] = self.heading_command_w * self.cfg.heading_control_stiffness

        vx = self.vel_command_b[:, 0]
        vy = self.vel_command_b[:, 1]
        min_x = (
            -self.max_command_b[:, 0]
            if not self.cfg.only_positive_lin_vel_x
            else torch.zeros_like(self.max_command_b[:, 0])
        )
        min_y = -self.max_command_b[:, 1]
        max_x = self.max_command_b[:, 0]
        max_y = self.max_command_b[:, 1]
        eps = 1e-6
        abs_vx = vx.abs()
        abs_vy = vy.abs()

        if not self.cfg.only_positive_lin_vel_x:
            clamped_vx = torch.clamp(abs_vx, min=min_x, max=max_x)
            clamped_vy = torch.clamp(abs_vy, min=min_y, max=max_y)
            scale_x = clamped_vx / (abs_vx + eps)
            scale_y = clamped_vy / (abs_vy + eps)
            scale = torch.where(abs_vx >= abs_vy, scale_x, scale_y)
            self.vel_command_b[:, 0] = vx * scale
            self.vel_command_b[:, 1] = vy * scale
        else:
            self.vel_command_b[:, 0] = torch.clamp(vx, min=min_x, max=max_x)
            self.vel_command_b[:, 1] = torch.clamp(vy, min=min_y, max=max_y)

        self.vel_command_b[:, 2] = torch.clamp(
            self.vel_command_b[:, 2],
            self.cfg.ranges.ang_vel_z[0],
            self.cfg.ranges.ang_vel_z[1],
        )
        self.vel_command_b[:] *= (target_dist > self.cfg.target_dis_threshold).unsqueeze(-1)
        self.vel_command_b[:, :2] *= (
            (torch.norm(self.vel_command_b[:, :2], dim=1) > self.cfg.lin_vel_threshold).float().unsqueeze(-1)
        )
        self.vel_command_b[:, 2] *= (torch.abs(self.vel_command_b[:, 2]) > self.cfg.ang_vel_threshold).float()
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

        random_velocity_env_ids = self.random_velocity_indices.nonzero(as_tuple=False).flatten()
        self.vel_command_b[random_velocity_env_ids, 0] = self.random_lin_vel_x[random_velocity_env_ids]
        self.vel_command_b[random_velocity_env_ids, 1] = self.random_lin_vel_y[random_velocity_env_ids]
        self.vel_command_b[random_velocity_env_ids, 2] = self.random_ang_vel_z[random_velocity_env_ids]

    def _debug_vis_impl(self, visualizer) -> None:
        """Draw the pose target and velocity arrows. Copied from InstinctMJ's PoseVelocityCommand."""
        if self.num_envs <= 0:
            return

        pos_commands_w = self.pos_command_w.cpu().numpy()
        vel_commands_b = self.vel_command_b.cpu().numpy()
        base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
        base_quat_ws = self.robot.data.root_link_quat_w.cpu().numpy()
        lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()

        goal_radius = self.cfg.target_dis_threshold
        goal_height = 0.1
        patch_height = 0.05
        arrow_z_offset = 0.5
        arrow_scale = 0.5
        env_indices = range(self.num_envs)

        if getattr(self.cfg, "patch_vis", False):
            flat_patches = self.valid_targets.reshape(-1, 3).cpu().numpy()
            for i, patch_pos in enumerate(flat_patches):
                if np.linalg.norm(patch_pos) < 1e-6:
                    continue
                patch_start = patch_pos.copy()
                patch_end = patch_pos.copy()
                patch_end[2] += patch_height
                visualizer.add_cylinder(
                    start=patch_start,
                    end=patch_end,
                    radius=goal_radius,
                    color=(0.0, 0.0, 1.0, 0.3),
                    label=f"patch_{i}",
                )

        for batch in env_indices:
            if np.linalg.norm(base_pos_ws[batch]) < 1e-6:
                continue

            goal_pos = pos_commands_w[batch]
            goal_start = goal_pos.copy()
            goal_end = goal_pos.copy()
            goal_end[2] += goal_height
            visualizer.add_cylinder(
                start=goal_start,
                end=goal_end,
                radius=goal_radius,
                color=(1.0, 0.0, 0.0, 0.6),
                label=f"goal_{batch}",
            )

            base_pos = base_pos_ws[batch]
            arrow_start = base_pos.copy()
            arrow_start[2] += arrow_z_offset

            vel_cmd_b = vel_commands_b[batch]
            quat_w = base_quat_ws[batch]
            yaw = np.arctan2(
                2.0 * (quat_w[0] * quat_w[3] + quat_w[1] * quat_w[2]), 1.0 - 2.0 * (quat_w[2] ** 2 + quat_w[3] ** 2)
            )
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            vel_cmd_w = np.array(
                [cos_yaw * vel_cmd_b[0] - sin_yaw * vel_cmd_b[1], sin_yaw * vel_cmd_b[0] + cos_yaw * vel_cmd_b[1], 0.0]
            )

            visualizer.add_arrow(
                start=arrow_start,
                end=arrow_start + vel_cmd_w * arrow_scale,
                color=(0.1, 1.0, 0.1, 0.8),
                width=0.02,
                label=f"cmd_vel_{batch}",
            )

            lin_vel_b = lin_vel_bs[batch]
            vel_actual_w = np.array(
                [cos_yaw * lin_vel_b[0] - sin_yaw * lin_vel_b[1], sin_yaw * lin_vel_b[0] + cos_yaw * lin_vel_b[1], 0.0]
            )
            visualizer.add_arrow(
                start=arrow_start,
                end=arrow_start + vel_actual_w * arrow_scale,
                color=(0.1, 0.1, 1.0, 0.8),
                width=0.02,
                label=f"actual_vel_{batch}",
            )
