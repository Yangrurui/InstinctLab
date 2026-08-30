"""Playback implementations for environments compiled with Isaac Sim."""

from __future__ import annotations

from typing import Any


def play_native(env: Any, policy: Any, **options: Any) -> None:
    """Step the policy in Isaac Sim's viewport."""
    del options
    obs = env.get_observations()
    try:
        while True:
            obs = env.step(policy(obs))[0]
    except KeyboardInterrupt:
        return


def play_viser(
    env: Any,
    policy: Any,
    *,
    spec: Any | None,
    port: int,
    reload_policy: Any | None,
    checkpoint_dir: Any | None,
    strict: bool,
    **options: Any,
) -> None:
    """Replay an Isaac policy through MJLab's Viser environment."""
    del options
    if spec is None:
        raise ValueError("Isaac Viser playback needs the task spec")

    from .viser import build_viser_env, play_with_viser

    play_env = build_viser_env(
        spec,
        num_envs=env.num_envs,
        device=str(env.device),
        strict=strict,
    )
    print(
        "[INFO] Isaac Sim has no Viser backend; playing this checkpoint in mjlab's ViserPlayViewer",
        flush=True,
    )
    try:
        play_with_viser(
            play_env,
            policy,
            port=port,
            reload_policy=reload_policy,
            checkpoint_dir=checkpoint_dir,
        )
    finally:
        play_env.close()
