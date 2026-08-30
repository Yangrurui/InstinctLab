"""Playback implementations for environments compiled with MJLab."""

from __future__ import annotations

from typing import Any


def play_viser(
    env: Any,
    policy: Any,
    *,
    port: int,
    reload_policy: Any | None,
    checkpoint_dir: Any | None,
    **options: Any,
) -> None:
    """Run MJLab's Viser player."""
    del options
    from .viser import play_with_viser

    play_with_viser(
        env,
        policy,
        port=port,
        reload_policy=reload_policy,
        checkpoint_dir=checkpoint_dir,
    )


def play_native(env: Any, policy: Any, **options: Any) -> None:
    """Run MJLab's native MuJoCo viewer."""
    del options
    from mjlab.viewer import NativeMujocoViewer

    NativeMujocoViewer(env, policy).run()
