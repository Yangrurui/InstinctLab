"""Isaac env hooks that keep portable depth history aligned with main's camera reset."""

from __future__ import annotations

from typing import Any

__all__ = ["InstinctManagerBasedRLEnv"]


class InstinctManagerBasedRLEnv:
    """``ManagerBasedRLEnv`` that clears ``DelayedDepthImage`` history on episode reset."""

    @classmethod
    def wrap(cls, base_cls: type) -> type:
        from instinctlab.mdp.observations import clear_delayed_depth_history

        class _Env(base_cls):  # type: ignore[misc,valid-type]
            def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
                import torch

                env_ids = torch.arange(self.num_envs, device=self.device)
                clear_delayed_depth_history(self, env_ids)
                return super().reset(seed=seed, options=options)

        _Env.__name__ = base_cls.__name__
        _Env.__qualname__ = base_cls.__qualname__
        return _Env
