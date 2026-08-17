"""Process-wide task RNG, matching mjlab / Isaac Lab ``torch.manual_seed`` + ``torch.rand``."""

from __future__ import annotations

import numpy as np
import os
import random
import torch
from dataclasses import dataclass


def seed_global(seed: int) -> int:
    """Seed Python, NumPy, and Torch the same way mjlab and Isaac Lab do."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


@dataclass
class RngManager:
    """Consume the process-wide torch generator. Stream names are labels only."""

    base_seed: int
    device: torch.device | str
    schema_version: str = "global_torch_v1"

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        seed_global(self.base_seed)

    def uniform(
        self,
        stream: str,
        low: float,
        high: float,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        del stream
        if high < low:
            raise ValueError(f"uniform bounds are reversed: ({low}, {high})")
        result = torch.rand(shape, device=self.device, dtype=dtype)
        return result.mul_(high - low).add_(low)

    def integers(
        self,
        stream: str,
        low: int,
        high: int,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype = torch.int64,
    ) -> torch.Tensor:
        del stream
        if high <= low:
            raise ValueError(f"integer bounds are empty: [{low}, {high})")
        return torch.randint(low, high, shape, device=self.device, dtype=dtype)

    def normal(
        self,
        stream: str,
        mean: float,
        std: float,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        del stream
        if std < 0.0:
            raise ValueError("normal standard deviation cannot be negative")
        result = torch.randn(shape, device=self.device, dtype=dtype)
        return result.mul_(std).add_(mean)

    def state_dict(self) -> dict[str, torch.Tensor | int | str | list[torch.Tensor]]:
        payload: dict[str, torch.Tensor | int | str | list[torch.Tensor]] = {
            "base_seed": self.base_seed,
            "schema_version": self.schema_version,
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            payload["torch_cuda"] = torch.cuda.get_rng_state_all()
        return payload

    def load_state_dict(self, payload: dict[str, torch.Tensor | int | str | list[torch.Tensor]]) -> None:
        if int(payload["base_seed"]) != self.base_seed:
            raise ValueError("RNG state base_seed does not match")
        if str(payload["schema_version"]) != self.schema_version:
            raise ValueError("RNG schema version does not match")
        torch_state = payload["torch"]
        if not isinstance(torch_state, torch.Tensor):
            raise TypeError("RNG state 'torch' must be a tensor")
        torch.set_rng_state(torch_state)
        cuda_state = payload.get("torch_cuda")
        if cuda_state is not None and torch.cuda.is_available():
            if not isinstance(cuda_state, list):
                raise TypeError("RNG state 'torch_cuda' must be a list of tensors")
            torch.cuda.set_rng_state_all(cuda_state)


__all__ = ["RngManager", "seed_global"]
