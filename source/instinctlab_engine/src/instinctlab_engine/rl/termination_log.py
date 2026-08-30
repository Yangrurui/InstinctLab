"""Engine-neutral termination statistics for the shared training interface."""

from __future__ import annotations

from typing import Any

import torch


class PortableTerminationLogger:
    """Report termination causes with the same dimension on every engine.

    Isaac Lab reports the fraction of environments whose most recently completed
    episode ended with each term. MJLab reports the raw number of term firings in
    the current reset batch. Both behaviors are retained inside their native
    managers; this class defines the portable ``instinct_rl`` logging contract.

    A value is the fraction of all environments whose most recently completed
    episode ended with that term. Terms are not mutually exclusive, so their sum
    may be greater than one. Environments that have not completed an episode yet
    contribute zero, matching Isaac Lab's startup behavior.
    """

    PREFIX = "Episode_Termination/"

    def __init__(self, env: Any):
        self._manager = getattr(env, "termination_manager", None)
        self._term_names = tuple(getattr(self._manager, "active_terms", ()))
        self._num_envs = int(env.num_envs)
        self._device = torch.device(env.device)
        self._last_episode_dones = torch.zeros(
            (self._num_envs, len(self._term_names)),
            device=self._device,
            dtype=torch.bool,
        )

    @property
    def enabled(self) -> bool:
        """Whether the wrapped environment exposes termination terms."""
        return self._manager is not None and bool(self._term_names)

    def reset(self) -> None:
        """Clear the portable history after an explicit full environment reset."""
        self._last_episode_dones.zero_()

    def update(self, extras: dict, dones: torch.Tensor) -> None:
        """Replace native termination log values with portable cause fractions."""
        if not self.enabled:
            return

        done_mask = dones.to(device=self._device, dtype=torch.bool).reshape(-1)
        if done_mask.shape != (self._num_envs,):
            raise ValueError(
                f"dones must have shape ({self._num_envs},), received {tuple(done_mask.shape)}"
            )

        current_terms = []
        for term_name in self._term_names:
            term_done = self._manager.get_term(term_name)
            term_done = torch.as_tensor(
                term_done, device=self._device, dtype=torch.bool
            )
            if term_done.shape != (self._num_envs,):
                raise ValueError(
                    f"termination term {term_name!r} must have shape ({self._num_envs},), "
                    f"received {tuple(term_done.shape)}"
                )
            current_terms.append(term_done)

        current_term_dones = torch.stack(current_terms, dim=1)
        self._last_episode_dones[:] = torch.where(
            done_mask.unsqueeze(1),
            current_term_dones,
            self._last_episode_dones,
        )
        cause_fractions = self._last_episode_dones.float().mean(dim=0)

        log_info = extras.setdefault("log", {})
        if not isinstance(log_info, dict):
            raise TypeError(
                f"extras['log'] must be a dict, received {type(log_info).__name__}"
            )
        for term_name, cause_fraction in zip(
            self._term_names, cause_fractions, strict=True
        ):
            log_info[self.PREFIX + term_name] = cause_fraction
