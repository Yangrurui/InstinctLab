"""Per-terrain episode stats, opt-in from the training entry.

Both engines now lower the same declared rough-terrain recipe. Native terrain
generation and physics still differ, so the per-type curves remain useful for
diagnosing engine behavior; ``aligned`` here means that the declared tile and
parameters are shared, not that the resulting meshes or returns must match.

Attach only when asked: the per-step cost is the Python bookkeeping over
named types (measured ~2.7 ms / ~9% of a wrapped step at 16 envs), not the
overflow poll. Overflow refusal lives in
:mod:`instinctlab_engine.diagnostics.contact_overflow`
and is on by default.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Any

import torch

from instinctlab_engine.bridge.terrain import column_sub_terrain_names
from instinctlab.tasks.terrain import rough_terrain

_ROUGH_GENERATOR = rough_terrain().generator
assert _ROUGH_GENERATOR is not None
ALIGNED_TERRAINS = tuple(_ROUGH_GENERATOR.sub_terrains)
# Retained for log-schema compatibility. There are no longer task-level terrain
# exclusions; term-level differences such as virtual-obstacle extraction are
# audited separately.
EXCLUDED_TERRAINS: tuple[str, ...] = ()


def attach_terrain_split(env: Any) -> Any:
    """Wrap a VecEnv so extras['log'] carries per-terrain episode length and reward."""
    try:
        raw = env.unwrapped
        terrain = raw.scene["terrain"]
        column_names = column_sub_terrain_names(terrain)
        types = terrain.terrain_types
    except Exception as exc:  # noqa: BLE001 - training must not die on a metric
        print(f"[WARN] Per-terrain episode logging disabled: {exc}")
        return env
    print(
        "[INFO] Per-terrain episode logging on "
        f"{len({name for name in column_names if name})} named types; "
        f"shared recipe uses {list(ALIGNED_TERRAINS)}"
    )
    return TerrainSplitVecEnv(env, column_names, types)


class TerrainSplitVecEnv:
    """Delegating VecEnv that records completed episodes by sub-terrain name."""

    def __init__(self, env: Any, column_names: Sequence[str | None], terrain_types: torch.Tensor):
        object.__setattr__(self, "env", env)
        names = [name or f"col_{i}" for i, name in enumerate(column_names)]
        self._column_names = names
        self._types = terrain_types
        unique = list(dict.fromkeys(names))
        self._unique = unique
        self._lens: dict[str, deque[float]] = {name: deque(maxlen=256) for name in unique}
        self._rews: dict[str, deque[float]] = {name: deque(maxlen=256) for name in unique}
        device = env.device
        self._ep_len = torch.zeros(env.num_envs, device=device)
        self._ep_rew = torch.zeros(env.num_envs, device=device)
        self._name_index = {name: i for i, name in enumerate(unique)}
        type_ids = terrain_types.detach().long().clamp(0, len(names) - 1)
        env_names = [names[int(i)] for i in type_ids.tolist()]
        self._env_name_ids = torch.tensor(
            [self._name_index[name] for name in env_names], device=device, dtype=torch.long
        )
        self._counts = {name: env_names.count(name) for name in unique}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.env.episode_length_buf = value

    def step(self, actions: torch.Tensor):
        obs, rewards, dones, extras = self.env.step(actions)
        self._record(rewards, dones, extras)
        return obs, rewards, dones, extras

    def _record(self, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        rew = rewards.reshape(rewards.shape[0], -1).sum(dim=-1)
        self._ep_len = self._ep_len + 1
        self._ep_rew = self._ep_rew + rew
        finished = dones.reshape(-1) > 0
        if torch.any(finished):
            ids = finished.nonzero(as_tuple=False).reshape(-1)
            lengths = self._ep_len[ids].detach().cpu().tolist()
            totals = self._ep_rew[ids].detach().cpu().tolist()
            name_ids = self._env_name_ids[ids].detach().cpu().tolist()
            for length, total, name_id in zip(lengths, totals, name_ids, strict=False):
                name = self._unique[int(name_id)]
                self._lens[name].append(float(length))
                self._rews[name].append(float(total))
            self._ep_len[ids] = 0
            self._ep_rew[ids] = 0

        log = extras.setdefault("log", {})
        device = self._ep_len.device
        aligned_l: list[float] = []
        aligned_r: list[float] = []
        excluded_l: list[float] = []
        excluded_r: list[float] = []
        for name in self._unique:
            length = _mean(self._lens[name])
            reward = _mean(self._rews[name])
            log[f"Episode_Terrain/length/{name}"] = torch.tensor(length, device=device)
            log[f"Episode_Terrain/reward/{name}"] = torch.tensor(reward, device=device)
            log[f"Episode_Terrain/n_completed/{name}"] = torch.tensor(float(len(self._lens[name])), device=device)
            log[f"Episode_Terrain/n_envs/{name}"] = torch.tensor(float(self._counts[name]), device=device)
            if name in ALIGNED_TERRAINS:
                aligned_l.extend(self._lens[name])
                aligned_r.extend(self._rews[name])
            elif name in EXCLUDED_TERRAINS:
                excluded_l.extend(self._lens[name])
                excluded_r.extend(self._rews[name])
        log["Episode_Terrain/aligned_length"] = torch.tensor(_mean(aligned_l), device=device)
        log["Episode_Terrain/aligned_reward"] = torch.tensor(_mean(aligned_r), device=device)
        log["Episode_Terrain/excluded_length"] = torch.tensor(_mean(excluded_l), device=device)
        log["Episode_Terrain/excluded_reward"] = torch.tensor(_mean(excluded_r), device=device)
        log["Episode_Terrain/aligned_n_completed"] = torch.tensor(float(len(aligned_l)), device=device)
        log["Episode_Terrain/excluded_n_completed"] = torch.tensor(float(len(excluded_l)), device=device)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))
