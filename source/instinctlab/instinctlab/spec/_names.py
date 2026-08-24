"""Small name-collection helpers shared by declaration modules."""

from __future__ import annotations

from collections.abc import Sequence


def as_name_tuple(names: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize one name or an ordered name sequence to an immutable tuple."""
    if isinstance(names, str):
        return (names,)
    return tuple(names)


__all__ = ["as_name_tuple"]
