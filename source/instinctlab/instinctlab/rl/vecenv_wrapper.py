"""Instinct-RL adapter for the unified environment."""

from __future__ import annotations

import torch
from typing import Any

from instinct_rl.env import VecEnv


class InstinctRlVecEnvWrapper(VecEnv):
    def __init__(self, env: Any, *, policy_group: str = "policy", critic_group: str | None = "critic") -> None:
        self.env = env
        self.policy_group = policy_group
        self.critic_group = critic_group if critic_group in env.observation_manager.active_terms else None
        self.num_envs = env.num_envs
        self.device = torch.device(env.device)
        self.max_episode_length = env.max_episode_length
        self.num_actions = env.action_manager.total_action_dim
        self.num_rewards = env.num_rewards
        self._log_defaults: dict[str, torch.Tensor] = {}
        # instinct_rl does not reset before rollout. Observation schemas (and
        # therefore num_obs) are filled on the first compute(), which reset() does.
        self.env.reset()
        self.num_obs = self._group_flat_dim(self.policy_group)
        self.num_critic_obs = self._group_flat_dim(self.critic_group) if self.critic_group else None

    @property
    def unwrapped(self) -> Any:
        return self.env

    @property
    def cfg(self) -> Any:
        return self.env.cfg

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.env.episode_length_buf = value

    def seed(self, seed: int = -1) -> int:
        return self.env.seed(seed)

    def reset(self) -> tuple[torch.Tensor, dict]:
        observations, extras = self.env.reset()
        packed = self._pack_observations(observations)
        extras = dict(extras)
        extras["observations"] = packed
        self._stabilize_extras(extras)
        return packed["policy"], extras

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        packed = self._pack_observations(self.env.observation_manager.compute())
        return packed["policy"], {"observations": packed}

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        observations, rewards, terminated, truncated, extras = self.env.step(actions)
        packed = self._pack_observations(observations)
        rewards = rewards if rewards.ndim == 2 else rewards.unsqueeze(-1)
        dones = (terminated | truncated).to(dtype=torch.long)
        extras = dict(extras)
        extras["observations"] = packed
        if not self.env.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        self._stabilize_extras(extras)
        return packed["policy"], rewards, dones, extras

    def get_obs_segments(self, group_name: str = "policy") -> dict[str, tuple[int, ...]]:
        source = (
            self.policy_group if group_name == "policy" else self.critic_group if group_name == "critic" else group_name
        )
        if source is None:
            return {}
        names = self.env.observation_manager.active_terms[source]
        dimensions = self.env.observation_manager.group_obs_term_dim[source]
        return dict(zip(names, dimensions, strict=True))

    def get_obs_format(self) -> dict[str, dict[str, tuple[int, ...]]]:
        result = {"policy": self.get_obs_segments("policy")}
        if self.critic_group is not None:
            result["critic"] = self.get_obs_segments("critic")
        return result

    def close(self) -> None:
        self.env.close()

    def _pack_observations(self, observations: dict[str, Any]) -> dict[str, torch.Tensor]:
        group_map = {"policy": self.policy_group}
        if self.critic_group is not None:
            group_map["critic"] = self.critic_group
        packed: dict[str, torch.Tensor] = {}
        for exposed, source in group_map.items():
            value = observations[source]
            if isinstance(value, torch.Tensor):
                packed[exposed] = value.flatten(start_dim=1)
            else:
                order = self.env.observation_manager.active_terms[source]
                packed[exposed] = torch.cat([value[name].flatten(start_dim=1) for name in order], dim=1)
        return packed

    def _group_flat_dim(self, group_name: str | None) -> int:
        if group_name is None:
            return 0
        return sum(
            int(torch.tensor(shape, device="cpu").prod().item())
            for shape in self.env.observation_manager.group_obs_term_dim[group_name]
        )

    def _stabilize_extras(self, extras: dict) -> None:
        extras.setdefault("log", {})
        extras.setdefault("step", {})
        extras.setdefault("episode", {})
        for key, value in extras["log"].items():
            self._log_defaults[key] = torch.zeros_like(value) if isinstance(value, torch.Tensor) else torch.tensor(0.0)
        for key, value in self._log_defaults.items():
            extras["log"].setdefault(key, value)


__all__ = ["InstinctRlVecEnvWrapper"]
