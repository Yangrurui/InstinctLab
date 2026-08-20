"""Per-terrain episode stats and a silent contact-budget snapshot.

The parkour mix is not one task on both engines. Stairs geometry and the
virtual-obstacle sets diverge on purpose; an aggregate episode-length curve
therefore mixes comparable and non-comparable ground. This wrapper logs
length and reward per sub-terrain name so a comparison can sit on the
aligned subset.

The overflow snapshot reads ``d.overflow`` when the env exposes a mujoco_warp
data object. It never raises on a set bit -- that is a measurement, not a
guard.
"""

from __future__ import annotations

import json
import numpy as np
import os
import torch
from collections import deque
from collections.abc import Sequence
from typing import Any

from instinctlab.engines.pose_velocity import column_sub_terrain_names

# Stairs step-count and the tenth slot (dense_boxes vs mesh_boxes) are the
# documented geometry mismatches. Everything else is the comparison set.
# ``volume_points_penetration`` is excluded at the term level, not here.
ALIGNED_TERRAINS = (
    "perlin_rough",
    "perlin_rough_stand",
    "square_gaps",
    "pyramid_stairs_high",
    "pyramid_stairs_inv_high",
    "boxes",
    "hf_pyramid_slope_inv",
)
EXCLUDED_TERRAINS = (
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "dense_boxes",
    "mesh_boxes",
)


def snapshot_contact_budget(env: Any, path: str) -> dict[str, Any]:
    """Read ``d.overflow`` / contact counts if this env is mujoco_warp; else say so."""
    raw = getattr(env, "unwrapped", env)
    sim = getattr(raw, "sim", None)
    wp_data = getattr(sim, "wp_data", None)
    if wp_data is None or not hasattr(wp_data, "overflow"):
        result = {"available": False, "reason": "env has no wp_data.overflow"}
        _write_json(path, result)
        return result

    overflow = _as_numpy(wp_data.overflow).astype(np.int64, copy=False)
    flags = _decode_overflow(int(overflow.max()) if overflow.size else 0)
    n_set = int(np.count_nonzero(overflow))
    result: dict[str, Any] = {
        "available": True,
        "nworld": int(overflow.size),
        "worlds_with_overflow": n_set,
        "any_overflow": n_set > 0,
        "overflow_max": int(overflow.max()) if overflow.size else 0,
        "overflow_flags": flags,
        "overflow_per_world_nonzero": overflow[overflow != 0].tolist(),
    }
    cfg = getattr(sim, "cfg", None)
    if cfg is not None:
        result["nconmax"] = getattr(cfg, "nconmax", None)
        result["njmax"] = getattr(cfg, "njmax", None)
        result["contact_sensor_maxmatch"] = getattr(cfg, "contact_sensor_maxmatch", None)
    nacon = getattr(wp_data, "nacon", None)
    if nacon is not None:
        nacon_v = _as_numpy(nacon)
        result["nacon"] = int(nacon_v.reshape(-1)[0]) if nacon_v.size else None
    nefc = getattr(wp_data, "nefc", None)
    if nefc is not None:
        nefc_v = _as_numpy(nefc)
        result["nefc_max"] = int(nefc_v.max()) if nefc_v.size else None
        result["nefc_mean"] = float(nefc_v.mean()) if nefc_v.size else None
    _write_json(path, result)
    return result


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
        f"{len(set(n for n in column_names if n))} named types; "
        f"parity signal uses {list(ALIGNED_TERRAINS)}"
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
        self.contact_peaks = _init_contact_peaks(getattr(env, "unwrapped", env))

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
        _update_contact_peaks(self.contact_peaks, getattr(self.env, "unwrapped", self.env))
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


def _init_contact_peaks(env: Any) -> dict[str, Any] | None:
    sim = getattr(env, "sim", None)
    wp_data = getattr(sim, "wp_data", None)
    if wp_data is None or not hasattr(wp_data, "overflow"):
        return None
    return {
        "available": True,
        "any_overflow": False,
        "worlds_with_overflow_peak": 0,
        "overflow_max": 0,
        "overflow_flags": [],
        "nacon_peak": 0,
        "nefc_max_peak": 0,
        "steps_sampled": 0,
    }


def _update_contact_peaks(peaks: dict[str, Any] | None, env: Any) -> None:
    if not peaks:
        return
    wp_data = env.sim.wp_data
    overflow = _as_numpy(wp_data.overflow).astype(np.int64, copy=False)
    n_set = int(np.count_nonzero(overflow))
    mask = int(overflow.max()) if overflow.size else 0
    peaks["steps_sampled"] = int(peaks["steps_sampled"]) + 1
    peaks["worlds_with_overflow_peak"] = max(int(peaks["worlds_with_overflow_peak"]), n_set)
    peaks["overflow_max"] = max(int(peaks["overflow_max"]), mask)
    if n_set > 0:
        peaks["any_overflow"] = True
        peaks["overflow_flags"] = _decode_overflow(int(peaks["overflow_max"]))
    nacon = getattr(wp_data, "nacon", None)
    if nacon is not None:
        nacon_v = _as_numpy(nacon)
        if nacon_v.size:
            peaks["nacon_peak"] = max(int(peaks["nacon_peak"]), int(nacon_v.reshape(-1)[0]))
    nefc = getattr(wp_data, "nefc", None)
    if nefc is not None:
        nefc_v = _as_numpy(nefc)
        if nefc_v.size:
            peaks["nefc_max_peak"] = max(int(peaks["nefc_max_peak"]), int(nefc_v.max()))


def dump_contact_peaks(env: Any, path: str) -> dict[str, Any]:
    peaks = getattr(env, "contact_peaks", None)
    result = dict(peaks) if peaks else {"available": False, "reason": "no contact peak tracker"}
    _write_json(path, result)
    return result


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _as_numpy(arr: Any) -> np.ndarray:
    if hasattr(arr, "numpy"):
        out = arr.numpy()
        return out if isinstance(out, np.ndarray) else np.asarray(out)
    if hasattr(arr, "detach"):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _decode_overflow(mask: int) -> list[str]:
    if mask == 0:
        return []
    try:
        from mujoco_warp._src.types import OverflowType
    except ImportError:
        return [f"bits=0x{mask:x}"]
    return [flag.name for flag in OverflowType if mask & int(flag)]


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(f"[INFO] Contact-budget snapshot: {payload} -> {path}")
