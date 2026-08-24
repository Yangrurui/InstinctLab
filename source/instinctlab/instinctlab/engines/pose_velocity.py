"""Shared pose-velocity command math, importable with neither engine installed.

Both reference implementations are the same algorithm on two ``CommandTerm`` bases. Duplicating
the arithmetic would let the copies drift the way the column-index mapping already has: training
still converges, the robot just tracks the wrong speed envelope. A third engine therefore pays
for a thin subclass, not another 350 lines.

This module lives next to ``compile.py`` rather than in ``mdp/`` because the command family is
already per-engine (field names and visualization differ) and because ``tests/test_mdp_terms.py``
scans that package as a portable term library. Putting the mixin there would either fail that
scan or force a portable-looking home onto something that is not a portable term.

Velocity boxes are keyed by sub-terrain **name**. A name maps to a *set* of columns: both
engines' curriculum generators assign columns by Isaac Lab's cumulative-proportion formula
(``j / num_cols + 0.001``). Upstream mjlab instead emits one column per type and ignores
``num_cols``; our ``FiledTerrainGenerator`` honors the declared width so the two grids match.
The remaining naming divergence is mjlab random mode, where a column is a mix of types and
this module returns ``None`` so the mixin raises rather than guessing an index.

Robot state is read under the hub spellings (``root_link_*``). The Isaac reference used the
legacy COM aliases for the tracking metrics; those aliases are denylisted here, and the angular
rows are the same tensor on Isaac anyway.
"""

from __future__ import annotations

import math
import numpy as np
import torch
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from instinctlab.compat.math import euler_xyz_from_quat, quat_apply_inverse, wrap_to_pi, yaw_quat

__all__ = [
    "POSE_VELOCITY_PARAM_KEYS",
    "PoseVelocityMixin",
    "UnresolvableTerrainColumn",
    "actual_column_count",
    "column_sub_terrain_names",
    "command_params",
    "curriculum_column_indices",
    "even_column_assignment",
    "resolve_named_columns",
    "type_share_histogram",
]


class UnresolvableTerrainColumn(RuntimeError):
    """A velocity box cannot be bound without guessing a column index."""


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
        raise ValueError(f"pose_velocity {name} must contain numbers, got {value!r}.")
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
        raise ValueError(f"pose_velocity {name} must be numeric, got {value!r}.")
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


def resolve_named_columns(
    column_names: Sequence[str | None],
    requested: Iterable[str],
) -> dict[str, tuple[int, ...]]:
    """Map each requested sub-terrain name to the columns that carry it.

    Raises if any column has no name or any requested name matches no column, listing both
    sides. An index fallback is how the references put velocity boxes on the wrong tiles.
    """
    names = list(column_names)
    unnamed = [index for index, name in enumerate(names) if not name]
    available = [name for name in names if name]
    wanted = list(requested)
    unknown = sorted(set(wanted) - set(available))
    if unnamed or unknown:
        raise UnresolvableTerrainColumn(
            "Cannot resolve velocity boxes by sub-terrain name. "
            f"Unnamable columns: {unnamed}. "
            f"Unknown names: {unknown}. "
            f"Columns: {names}. "
            f"Requested: {wanted}."
        )
    return {name: tuple(index for index, column in enumerate(names) if column == name) for name in wanted}


def curriculum_column_indices(proportions: Sequence[float], num_cols: int) -> list[int]:
    """Isaac Lab's curriculum column assignment. Do not re-derive; the ``+ 0.001`` is load-bearing.

    Copied from ``isaaclab.terrains.terrain_generator.TerrainGenerator._generate_curriculum_terrains``:
    column ``j`` is the first sub-terrain whose cumulative (normalized) proportion exceeds
    ``j / num_cols + 0.001``.
    """
    weights = np.asarray(proportions, dtype=np.float64)
    if weights.size == 0:
        raise RuntimeError("curriculum column assignment needs at least one sub-terrain proportion.")
    weights = weights / weights.sum()
    cumulative = np.cumsum(weights)
    indices: list[int] = []
    for index in range(num_cols):
        matches = np.where(index / num_cols + 0.001 < cumulative)[0]
        if matches.size == 0:
            raise RuntimeError(
                f"curriculum column {index} of {num_cols} matched no sub-terrain. "
                f"Normalized proportions: {weights.tolist()}."
            )
        indices.append(int(np.min(matches)))
    return indices


def even_column_assignment(num_envs: int, num_cols: int, device: torch.device | str | None = None) -> torch.Tensor:
    """Isaac Lab's even split of environments across columns.

    Copied from ``isaaclab.terrains.terrain_importer.TerrainImporter._compute_env_origins_curriculum``.
    Columns already encode type share (via :func:`curriculum_column_indices`), so an even split
    across columns reproduces the declared per-type proportions. Do not replace this with
    type-level weights of length ``num_cols``: that would double-count types that already occupy
    more than one column.
    """
    return torch.div(
        torch.arange(num_envs, device=device),
        (num_envs / num_cols),
        rounding_mode="floor",
    ).to(torch.long)


def type_share_histogram(
    terrain_types: torch.Tensor | Sequence[int],
    column_names: Sequence[str | None],
) -> dict[str, float]:
    """Per-name share of environments. Unnamed columns are counted under ``''``."""
    types = terrain_types.detach().cpu().tolist() if isinstance(terrain_types, torch.Tensor) else list(terrain_types)
    if not types:
        return {}
    names = list(column_names)
    counts: dict[str, int] = {}
    for column in types:
        name = names[int(column)] if 0 <= int(column) < len(names) else None
        key = name if name else ""
        counts[key] = counts.get(key, 0) + 1
    n_env = len(types)
    return {name: count / n_env for name, count in counts.items()}


def actual_column_count(terrain: Any) -> int:
    """Grid width from the terrain object, not from ``cfg.num_cols``.

    Our mjlab ``FiledTerrainGenerator`` honors ``num_cols`` in curriculum mode, so the two
    numbers should agree. Reading the built grid still catches a silent shrink if an upstream
    generator (or a future edit) starts ignoring the declaration again.
    """
    patches = getattr(terrain, "flat_patches", None)
    if patches and "target" in patches:
        return int(patches["target"].shape[1])
    origins = getattr(terrain, "terrain_origins", None)
    if origins is not None:
        return int(origins.shape[1])
    raise UnresolvableTerrainColumn(
        "Terrain has neither flat_patches['target'] nor terrain_origins, so columns cannot be named."
    )


def column_sub_terrain_names(terrain: Any) -> list[str | None]:
    """Name occupying each column under Isaac's cumulative-proportion allocation.

    Curriculum mode on both engines uses the same formula as
    :func:`curriculum_column_indices`. Random (non-curriculum) mode leaves a column as a mix
    of types; this returns ``None`` for every column so :func:`resolve_named_columns` raises
    rather than guessing. Isaac's parkour/rough grids are curriculum, so the ``None`` path is
    the mjlab-only remaining divergence.
    """
    generator = terrain.cfg.terrain_generator
    if generator is None:
        raise RuntimeError("pose_velocity needs a generated terrain; this importer has no terrain_generator.")
    names = list(generator.sub_terrains.keys())
    n_cols = actual_column_count(terrain)
    if not getattr(generator, "curriculum", False):
        return [None] * n_cols
    if n_cols != int(generator.num_cols):
        raise RuntimeError(
            f"Curriculum grid is {n_cols} columns but terrain_generator.num_cols={generator.num_cols}. "
            f"Sub-terrains: {names}."
        )
    if not names:
        raise RuntimeError("pose_velocity needs at least one named sub-terrain.")
    proportions = [generator.sub_terrains[name].proportion for name in names]
    if n_cols == len(names):
        return list(names)
    return [names[index] for index in curriculum_column_indices(proportions, n_cols)]


class PoseVelocityMixin:
    """Resample / update / metric arithmetic against ``self``.

    The subclass binds ``robot`` and ``terrain``, implements
    :meth:`_column_sub_terrain_names`, and then calls :meth:`_pose_velocity_setup`.
    Everything that does arithmetic lives here so an AST check on the subclass can
    prove it added none.
    """

    def _column_sub_terrain_names(self) -> Sequence[str | None]:
        """Sub-terrain name occupying each column, or ``None`` if that column cannot be named."""
        raise NotImplementedError

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

        self._column_names = list(self._column_sub_terrain_names())
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
