"""Small, engine-neutral helpers for unified training lifecycle state."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DistributedRun:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def seed(self, base_seed: int) -> int:
        return base_seed + self.rank


def distributed_run(requested: bool = False, local_rank: int | None = None) -> DistributedRun:
    """Read torchrun coordinates without importing torch before Isaac bootstrap."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    resolved_local_rank = int(os.environ.get("LOCAL_RANK", local_rank if local_rank is not None else 0))
    enabled = requested or world_size > 1
    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}.")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"RANK must satisfy 0 <= RANK < WORLD_SIZE, got rank={rank}, world_size={world_size}.")
    if resolved_local_rank < 0:
        raise ValueError(f"LOCAL_RANK must be non-negative, got {resolved_local_rank}.")
    if not enabled and (rank != 0 or resolved_local_rank != 0):
        raise ValueError("RANK/LOCAL_RANK are set without distributed training.")
    return DistributedRun(enabled, rank, resolved_local_rank, world_size)


def rank_device(device: str, run: DistributedRun) -> str:
    if run.enabled and device.startswith("cuda"):
        return f"cuda:{run.local_rank}"
    return device


def initialize_process_group(run: DistributedRun) -> None:
    if not run.enabled:
        return
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=run.rank, world_size=run.world_size)


def shared_run_directory(path: str, run: DistributedRun) -> str:
    """Broadcast rank zero's timestamped path so every worker uses one run."""
    if not run.enabled:
        return path
    import torch.distributed as dist

    payload = [path if run.is_primary else None]
    dist.broadcast_object_list(payload, src=0)
    return str(payload[0])


def load_runner_checkpoint(runner: Any, checkpoint: str | Path, run: DistributedRun) -> Any:
    """Restore model, optimizer, normalizers and iteration on every DDP rank.

    `instinct_rl` deliberately makes `runner.load` rank-zero-only after a process group exists.
    That is sufficient for model broadcast but leaves resumed optimizer moments different on the
    other ranks. Unified training loads the same state on each worker before DDP wrapping.
    """
    if not run.enabled:
        return runner.load(str(checkpoint))
    if runner.cfg.get("ckpt_manipulator", False):
        raise ValueError("distributed resume does not support ckpt_manipulator")

    import torch

    state = torch.load(checkpoint, map_location=runner.device, weights_only=True)
    runner.alg.load_state_dict(state)
    for group_name, normalizer in runner.normalizers.items():
        key = f"{group_name}_normalizer_state_dict"
        if key not in state:
            raise KeyError(f"distributed checkpoint is missing {key!r}")
        normalizer.load_state_dict(state[key])
    runner.current_learning_iteration = int(state["iter"])
    return state.get("infos")


def destroy_process_group(run: DistributedRun) -> None:
    if not run.enabled:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


__all__ = [
    "DistributedRun",
    "destroy_process_group",
    "distributed_run",
    "initialize_process_group",
    "load_runner_checkpoint",
    "rank_device",
    "shared_run_directory",
]
