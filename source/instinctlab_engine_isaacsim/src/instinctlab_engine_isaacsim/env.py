"""Isaac env hook for portable observation-history reset callbacks."""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["InstinctManagerBasedRLEnv"]


class InstinctManagerBasedRLEnv:
    """Wrap ``ManagerBasedRLEnv`` and clear opted-in histories on reset."""

    @classmethod
    def wrap(cls, base_cls: type) -> type:
        from instinctlab_engine.bridge.observation_history import (
            clear_observation_histories_on_reset,
        )

        class _Env(base_cls):  # type: ignore[misc,valid-type]
            def _reset_idx(self, env_ids: Sequence[int]):
                clear_observation_histories_on_reset(self, env_ids)
                super()._reset_idx(env_ids)
                lifecycle = getattr(self, "lifecycle", None)
                if lifecycle is not None:
                    lifecycle.on_reset(env_ids)

            def step(self, action):
                lifecycle = getattr(self, "lifecycle", None)
                if lifecycle is not None:
                    lifecycle.before_step()
                try:
                    result = super().step(action)
                except Exception:
                    if lifecycle is not None:
                        lifecycle.cancel_step()
                    raise
                if lifecycle is not None:
                    lifecycle.after_step(result[2] | result[3])
                return result

        _Env.__name__ = base_cls.__name__
        _Env.__qualname__ = base_cls.__qualname__
        return _Env
