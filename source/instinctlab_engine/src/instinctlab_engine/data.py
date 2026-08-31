"""Portable dataset URIs and local data-root resolution."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

DATA_ROOT_ENV = "INSTINCTLAB_DATA_ROOT"
DATASET_SCHEME = "dataset"


def is_dataset_uri(value: str | os.PathLike[str]) -> bool:
    """Return whether ``value`` uses InstinctLab's portable dataset URI scheme."""
    return os.fspath(value).startswith(f"{DATASET_SCHEME}://")


def dataset_root() -> Path:
    """Return the configured local root for portable dataset URIs."""
    declared = os.environ.get(DATA_ROOT_ENV, "~/Datasets")
    return Path(declared).expanduser().resolve()


def _dataset_relative_path(value: str) -> Path:
    parsed = urlsplit(value)
    if parsed.scheme != DATASET_SCHEME or not parsed.netloc:
        raise ValueError(
            f"Dataset URI must have the form dataset://collection/path, got {value!r}."
        )
    if parsed.query or parsed.fragment or parsed.username or parsed.password or parsed.port:
        raise ValueError(f"Dataset URI contains unsupported URL fields: {value!r}.")
    authority = unquote(parsed.netloc)
    decoded_path = unquote(parsed.path)
    if any(separator in authority for separator in ("/", "\\")) or "\\" in decoded_path:
        raise ValueError(
            f"Dataset URI contains an encoded or non-portable path separator: {value!r}."
        )
    components = (authority, *PurePosixPath(decoded_path).parts)
    relative_parts = tuple(part for part in components if part not in {"", "/"})
    if not relative_parts or any(part in {".", ".."} for part in relative_parts):
        raise ValueError(f"Dataset URI must not contain traversal components: {value!r}.")
    return Path(*relative_parts)


def resolve_data_path(
    value: str | os.PathLike[str],
    *,
    relative_to: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve a portable dataset URI or a readable legacy filesystem path.

    ``dataset://collection/path`` is rooted at :envvar:`INSTINCTLAB_DATA_ROOT`,
    which defaults to ``~/Datasets``. Plain paths retain their historical
    ``~`` behavior. Relative plain paths can be anchored with ``relative_to``.
    Resolution does not require the target to exist so preflight can report
    missing optional resources without changing the declaration.
    """
    declared = os.fspath(value)
    if is_dataset_uri(declared):
        root = dataset_root()
        resolved = (root / _dataset_relative_path(declared)).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Dataset URI resolves outside {DATA_ROOT_ENV}: {declared!r}."
            ) from exc
        return resolved

    path = Path(declared).expanduser()
    if relative_to is not None and not path.is_absolute():
        path = Path(relative_to).expanduser() / path
    return path.resolve()


__all__ = [
    "DATASET_SCHEME",
    "DATA_ROOT_ENV",
    "dataset_root",
    "is_dataset_uri",
    "resolve_data_path",
]
