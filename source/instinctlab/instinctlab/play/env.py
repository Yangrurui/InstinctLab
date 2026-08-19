"""Thin wrapper so a viewer sees a tensor from ``get_observations``.

``instinct_rl`` wrappers return ``(obs, extras)``. mjlab's ``ViserPlayViewer`` and the shared
Viser player both want the observation tensor itself. One adapter for both keeps that mismatch
out of the launcher.
"""

from __future__ import annotations

from typing import Any


class PlayEnv:
    """Present a wrapped Instinct-RL env as the viewer protocol."""

    def __init__(self, env: Any) -> None:
        self._env = env
        self.num_envs = env.num_envs

    @property
    def device(self) -> Any:
        return self._env.device

    @property
    def cfg(self) -> Any:
        return self._env.cfg

    @property
    def unwrapped(self) -> Any:
        return self._env.unwrapped

    def get_observations(self) -> Any:
        obs, _ = self._env.get_observations()
        return obs

    def step(self, actions: Any) -> tuple[Any, ...]:
        return self._env.step(actions)

    def reset(self) -> Any:
        return self._env.reset()

    def close(self) -> None:
        return self._env.close()
