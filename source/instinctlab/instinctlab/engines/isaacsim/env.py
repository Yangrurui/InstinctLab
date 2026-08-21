"""Isaac env hooks that keep portable depth history aligned with main's camera reset."""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["InstinctManagerBasedRLEnv"]


class InstinctManagerBasedRLEnv:
    """``ManagerBasedRLEnv`` that clears ``DelayedDepthImage`` history on episode reset."""

    @classmethod
    def wrap(cls, base_cls: type) -> type:
        from instinctlab.mdp.observations import clear_delayed_depth_history

        class _Env(base_cls):  # type: ignore[misc,valid-type]
            def _reset_idx(self, env_ids: Sequence[int]):
                clear_delayed_depth_history(self, env_ids)
                super()._reset_idx(env_ids)

        _Env.__name__ = base_cls.__name__
        _Env.__qualname__ = base_cls.__qualname__
        return _Env
