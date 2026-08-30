"""Reset bridge for observation terms that own per-environment history.

Neither native observation manager exposes a public term-reset callback. Both
currently retain the compiled term configs in ``_group_obs_term_cfgs`` with the
same shape, so each adapter calls this bridge immediately before its native
environment reset. A task implementation opts in through
``clears_history_on_env_reset`` and must provide ``clear_history(env_ids)``.

The private manager read is intentionally confined to this module. If either
SDK adds a public lifecycle hook, the adapters can drop this bridge without
changing task implementations.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch

__all__ = [
    "ResettableObservationHistory",
    "clear_observation_histories_on_reset",
]


class ResettableObservationHistory(Protocol):
    """Observation implementation that clears selected history on env reset."""

    clears_history_on_env_reset: bool

    def clear_history(self, env_ids: torch.Tensor | slice) -> None: ...


def clear_observation_histories_on_reset(
    env: Any, env_ids: Any | None = None
) -> None:
    """Clear opted-in observation histories for the environments being reset."""
    manager = getattr(env, "observation_manager", None)
    if manager is None:
        return
    selected = _env_ids(env, env_ids)
    if isinstance(selected, torch.Tensor) and selected.numel() == 0:
        return
    for group_cfgs in getattr(manager, "_group_obs_term_cfgs", {}).values():
        for cfg in group_cfgs:
            implementation = getattr(
                getattr(cfg, "func", None), "_impl", getattr(cfg, "func", None)
            )
            if not getattr(implementation, "clears_history_on_env_reset", False):
                continue
            clear_history = getattr(implementation, "clear_history", None)
            if not callable(clear_history):
                raise TypeError(
                    f"{type(implementation).__name__} opts into observation-history "
                    "reset but does not define clear_history(env_ids)."
                )
            clear_history(selected)


def _env_ids(env: Any, env_ids: Any | None) -> torch.Tensor | slice:
    if env_ids is None:
        return slice(None)
    if isinstance(env_ids, slice):
        return env_ids
    device = getattr(env, "device", "cpu")
    if isinstance(env_ids, torch.Tensor):
        return env_ids.reshape(-1).to(device=device, dtype=torch.long)
    if isinstance(env_ids, int):
        return torch.tensor([env_ids], device=device, dtype=torch.long)
    return torch.tensor(list(env_ids), device=device, dtype=torch.long)
