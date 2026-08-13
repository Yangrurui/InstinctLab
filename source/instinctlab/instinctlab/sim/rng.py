"""Named task RNG streams isolated from simulator backends."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch


def _derive_seed(base_seed: int, stream: str) -> int:
    digest = hashlib.blake2b(f"{base_seed}:{stream}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFF_FFFF_FFFF_FFFF


@dataclass
class RngManager:
    """Own named generators so backend initialization cannot perturb task RNG."""

    base_seed: int
    device: torch.device | str
    schema_version: str = "named_streams_v1"
    _generators: dict[str, torch.Generator] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)

    def generator(self, stream: str) -> torch.Generator:
        if not stream:
            raise ValueError("RNG stream name cannot be empty")
        if stream not in self._generators:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(_derive_seed(self.base_seed, stream))
            self._generators[stream] = generator
        return self._generators[stream]

    def uniform(
        self,
        stream: str,
        low: float,
        high: float,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if high < low:
            raise ValueError(f"uniform bounds are reversed: ({low}, {high})")
        result = torch.rand(shape, generator=self.generator(stream), device=self.device, dtype=dtype)
        return result.mul_(high - low).add_(low)

    def normal(
        self,
        stream: str,
        mean: float,
        std: float,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if std < 0.0:
            raise ValueError("normal standard deviation cannot be negative")
        result = torch.randn(shape, generator=self.generator(stream), device=self.device, dtype=dtype)
        return result.mul_(std).add_(mean)

    def state_dict(self) -> dict[str, torch.Tensor | int | str]:
        payload: dict[str, torch.Tensor | int | str] = {
            "base_seed": self.base_seed,
            "schema_version": self.schema_version,
        }
        payload.update({f"stream:{name}": generator.get_state() for name, generator in self._generators.items()})
        return payload

    def load_state_dict(self, payload: dict[str, torch.Tensor | int | str]) -> None:
        if int(payload["base_seed"]) != self.base_seed:
            raise ValueError("RNG state base_seed does not match")
        if str(payload["schema_version"]) != self.schema_version:
            raise ValueError("RNG schema version does not match")
        for key, value in payload.items():
            if key.startswith("stream:"):
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"RNG state {key!r} must be a tensor")
                self.generator(key.removeprefix("stream:")).set_state(value)


__all__ = ["RngManager"]
