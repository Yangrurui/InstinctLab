from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch
from instinctlab_engine.lifecycle.snapshot import SnapshotError
from instinctlab_engine.lifecycle.state_tree import (
    capture_state_tree,
    restore_state_tree,
)


class _IgnoredEnvironment:
    def __init__(self) -> None:
        self.large_static_tensor = torch.ones(100)


class _Buffer:
    def __init__(self) -> None:
        self._buffer = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        self._active = True
        self._pointer = 4


class _Term:
    def __init__(self) -> None:
        self._env = _IgnoredEnvironment()
        self.history = _Buffer()
        self.output = torch.tensor([7.0, 8.0])


@dataclass(frozen=True)
class _FrozenConfig:
    track_air_time: bool = True
    cached_tensor: torch.Tensor = field(default_factory=lambda: torch.ones(2))


def test_state_tree_restores_nested_tensor_and_scalar_state() -> None:
    term = _Term()
    state = capture_state_tree({"term": term})
    term.history._buffer.zero_()
    term.history._active = False
    term.history._pointer = 0
    term.output.fill_(-1.0)

    restore_state_tree({"term": term}, state)

    assert term.history._buffer.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    assert term.history._active is True
    assert term.history._pointer == 4
    assert term.output.tolist() == [7.0, 8.0]
    assert "_env" not in state["term"]["attributes"]


def test_state_tree_rejects_shape_change_before_copying_any_tensor() -> None:
    term = _Term()
    state = capture_state_tree({"term": term})
    state["term"]["attributes"]["history"]["attributes"]["_buffer"][
        "value"
    ] = torch.zeros(1)
    term.output.zero_()

    with pytest.raises(SnapshotError, match="shape/dtype"):
        restore_state_tree({"term": term}, state)

    assert term.output.tolist() == [0.0, 0.0]


def test_state_tree_does_not_treat_frozen_config_scalars_as_runtime_state() -> None:
    config = _FrozenConfig()
    state = capture_state_tree({"config": config})

    assert "track_air_time" not in state["config"]["attributes"]
    assert "cached_tensor" in state["config"]["attributes"]
    restore_state_tree({"config": config}, state)
