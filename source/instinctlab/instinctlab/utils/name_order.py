"""Resolve tensor-axis orders from names instead of positional assumptions."""

from __future__ import annotations

from collections.abc import Sequence


class NameOrderError(ValueError):
    """A source-to-target name order cannot be resolved unambiguously."""


def resolve_name_indices(
    source_names: Sequence[str],
    target_names: Sequence[str],
    *,
    require_exact: bool = False,
) -> tuple[int, ...]:
    """Return source indices that gather a tensor into ``target_names`` order.

    The returned mapping follows ``out[i] = source[mapping[i]]``. A target may
    be a subset of the source unless ``require_exact`` is true.
    """
    source = tuple(source_names)
    target = tuple(target_names)
    if len(source) != len(set(source)):
        raise NameOrderError(f"Source names are not unique: {source}.")
    if len(target) != len(set(target)):
        raise NameOrderError(f"Target names are not unique: {target}.")

    source_index = {name: index for index, name in enumerate(source)}
    target_set = set(target)
    missing_in_source = [name for name in target if name not in source_index]
    missing_in_target = [name for name in source if name not in target_set] if require_exact else []
    if missing_in_source or missing_in_target:
        raise NameOrderError(
            "Missing in source (needed by target): "
            f"{missing_in_source}. Missing in target (present in source): {missing_in_target}."
        )
    return tuple(source_index[name] for name in target)


__all__ = ["NameOrderError", "resolve_name_indices"]
