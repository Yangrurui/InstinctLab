"""Engine-neutral terrain-column queries used by tasks and native importers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch

__all__ = [
    "UnresolvableTerrainColumn",
    "actual_column_count",
    "column_sub_terrain_names",
    "curriculum_column_indices",
    "even_column_assignment",
    "resolve_named_columns",
    "type_share_histogram",
]


class UnresolvableTerrainColumn(RuntimeError):
    """A terrain-dependent task value cannot be bound without guessing a column."""


def resolve_named_columns(
    column_names: Sequence[str | None],
    requested: Iterable[str],
) -> dict[str, tuple[int, ...]]:
    """Map each requested sub-terrain name to every column that carries it."""
    names = list(column_names)
    unnamed = [index for index, name in enumerate(names) if not name]
    available = [name for name in names if name]
    wanted = list(requested)
    unknown = sorted(set(wanted) - set(available))
    if unnamed or unknown:
        raise UnresolvableTerrainColumn(
            "Cannot resolve terrain values by sub-terrain name. "
            f"Unnamable columns: {unnamed}. "
            f"Unknown names: {unknown}. "
            f"Columns: {names}. "
            f"Requested: {wanted}."
        )
    return {
        name: tuple(index for index, column in enumerate(names) if column == name)
        for name in wanted
    }


def curriculum_column_indices(
    proportions: Sequence[float], num_cols: int
) -> list[int]:
    """Isaac Lab's cumulative-proportion curriculum column assignment."""
    weights = np.asarray(proportions, dtype=np.float64)
    if weights.size == 0:
        raise RuntimeError(
            "curriculum column assignment needs at least one sub-terrain proportion."
        )
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


def even_column_assignment(
    num_envs: int,
    num_cols: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Isaac Lab's even assignment of environments across curriculum columns."""
    return torch.div(
        torch.arange(num_envs, device=device),
        (num_envs / num_cols),
        rounding_mode="floor",
    ).to(torch.long)


def type_share_histogram(
    terrain_types: torch.Tensor | Sequence[int],
    column_names: Sequence[str | None],
) -> dict[str, float]:
    """Return the environment share for every named terrain column."""
    types = (
        terrain_types.detach().cpu().tolist()
        if isinstance(terrain_types, torch.Tensor)
        else list(terrain_types)
    )
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
    """Read grid width from the built terrain rather than trusting its config."""
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
    """Return the semantic sub-terrain name occupying each built column."""
    generator = terrain.cfg.terrain_generator
    if generator is None:
        raise RuntimeError(
            "A named-column query needs a generated terrain; this importer has no terrain_generator."
        )
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
        raise RuntimeError("A named-column query needs at least one sub-terrain.")
    proportions = [generator.sub_terrains[name].proportion for name in names]
    if n_cols == len(names):
        return list(names)
    return [names[index] for index in curriculum_column_indices(proportions, n_cols)]
