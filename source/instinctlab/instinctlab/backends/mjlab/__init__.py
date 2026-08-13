"""Lazy MJLab backend provider.

Importing this package is safe before MJLab, MuJoCo, Warp, or CUDA have been
initialized.  The engine implementation is imported only when the provider is
asked to construct a backend (or when ``MjlabBackend`` is explicitly accessed).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .simulator import MjlabBackend as MjlabBackend


class MjlabBackendProvider:
    """Construct MJLab backends without importing MJLab during registry load."""

    @staticmethod
    def add_cli_args(parser: Any) -> None:
        del parser

    @staticmethod
    def bootstrap(args: Any) -> object | None:
        del args
        return None

    @staticmethod
    def create(*, device: str, bootstrap_context: object | None = None) -> "MjlabBackend":
        del bootstrap_context
        from .simulator import MjlabBackend

        return MjlabBackend(device=device)


def __getattr__(name: str) -> object:
    if name == "MjlabBackend":
        from .simulator import MjlabBackend

        return MjlabBackend
    raise AttributeError(name)


__all__ = ["MjlabBackend", "MjlabBackendProvider"]
