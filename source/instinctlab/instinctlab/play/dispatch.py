"""Lazy viewer registration for the playback application."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

Player = Callable[..., None]
PlayerTarget = Player | str

PLAYERS: dict[tuple[str, str], PlayerTarget] = {}


def register_player(engine: str, viewer: str, player: PlayerTarget) -> None:
    """Register one viewer without importing its simulator or UI implementation."""
    if not engine.isidentifier() or not viewer.isidentifier():
        raise ValueError(f"invalid playback registration {engine!r}/{viewer!r}")
    if isinstance(player, str):
        module, separator, attribute = player.partition(":")
        if not separator or not module or not attribute.isidentifier():
            raise ValueError(f"invalid playback target {player!r}")
    elif not callable(player):
        raise TypeError("player must be callable or a 'module:attribute' path")

    key = (engine, viewer)
    existing = PLAYERS.get(key)
    if existing is not None and existing != player:
        raise ValueError(f"player {engine!r}/{viewer!r} is already registered")
    PLAYERS[key] = player


def _resolve(player: PlayerTarget) -> Player:
    if not isinstance(player, str):
        return player
    module_name, _, attribute = player.partition(":")
    resolved = getattr(import_module(module_name), attribute)
    if not callable(resolved):
        raise TypeError(f"playback target {player!r} is not callable")
    return resolved


def play(
    engine: str,
    viewer: str,
    env: Any,
    policy: Any,
    *,
    robot: Any,
    spec: Any | None = None,
    port: int = 8080,
    reload_policy: Any | None = None,
    checkpoint_dir: Any | None = None,
    strict: bool = False,
) -> None:
    """Run a registered engine/viewer combination."""
    key = (engine, viewer)
    try:
        target = PLAYERS[key]
    except KeyError:
        supported = ", ".join(f"{name}/{kind}" for name, kind in sorted(PLAYERS))
        raise ValueError(
            f"unsupported playback combination {engine!r}/{viewer!r}; registered: {supported}"
        ) from None
    _resolve(target)(
        env,
        policy,
        robot=robot,
        spec=spec,
        port=port,
        reload_policy=reload_policy,
        checkpoint_dir=checkpoint_dir,
        strict=strict,
    )
