"""Isaac Sim implementation of complete same-engine lifecycle snapshots."""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from instinctlab_engine.lifecycle.snapshot import SnapshotError
from instinctlab_engine.lifecycle.state_tree import (
    capture_state_tree,
    restore_state_tree,
    validate_state_tree,
)

_ENV_TENSORS = (
    "episode_length_buf",
    "reset_buf",
    "reset_terminated",
    "reset_time_outs",
    "reward_buf",
)
_ENV_SCALARS = ("common_step_counter", "_sim_step_counter")
_MANAGERS = (
    "action_manager",
    "command_manager",
    "observation_manager",
    "reward_manager",
    "termination_manager",
    "event_manager",
    "curriculum_manager",
)


class IsaacSimSnapshotProvider:
    """Capture native PhysX scene, manager, sensor, actuator, and RNG state."""

    provider_id = "isaacsim/manager-based-env"
    provider_version = 2

    def __init__(self, env: Any) -> None:
        self.env = env

    def capture(self) -> Mapping[str, Any]:
        return {
            "scene": self.env.scene.get_state(is_relative=False),
            "environment": self._capture_environment(),
            "components": capture_state_tree(self._component_roots()),
            "rng": _capture_rng(),
        }

    def restore(self, state: Mapping[str, Any]) -> None:
        expected = {"scene", "environment", "components", "rng"}
        if set(state) != expected:
            raise SnapshotError(
                f"Isaac snapshot fields are {sorted(state)}, expected {sorted(expected)}."
            )
        environment = state["environment"]
        if not isinstance(environment, Mapping):
            raise SnapshotError("Isaac environment snapshot is malformed.")
        self._validate_environment(environment)
        components = state["components"]
        if not isinstance(components, Mapping):
            raise SnapshotError("Isaac component snapshot is malformed.")
        roots = self._component_roots()
        validate_state_tree(roots, components)
        _validate_rng(state["rng"])

        scene_state = _move_tensors(state["scene"], self.env.device)
        self.env.scene.reset_to(scene_state, is_relative=False)
        self._restore_environment(environment)
        restore_state_tree(roots, components)
        self.env.scene.write_data_to_sim()
        self.env.sim.forward()
        _restore_rng(state["rng"])

    def _capture_environment(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name in _ENV_TENSORS:
            value = getattr(self.env, name, None)
            if isinstance(value, torch.Tensor):
                result[name] = value.clone()
        for name in _ENV_SCALARS:
            if hasattr(self.env, name):
                result[name] = int(getattr(self.env, name))
        return result

    def _validate_environment(self, state: Mapping[str, Any]) -> None:
        expected = self._capture_environment()
        if set(state) != set(expected):
            raise SnapshotError("Isaac environment buffer schema changed.")
        for name, target in expected.items():
            source = state[name]
            if isinstance(target, torch.Tensor):
                if not isinstance(source, torch.Tensor):
                    raise SnapshotError(f"Isaac environment field {name} is not a tensor.")
                if source.shape != target.shape or source.dtype != target.dtype:
                    raise SnapshotError(f"Isaac environment field {name} is incompatible.")
            elif not isinstance(source, int):
                raise SnapshotError(f"Isaac environment field {name} is not an integer.")

    def _restore_environment(self, state: Mapping[str, Any]) -> None:
        for name, value in state.items():
            target = getattr(self.env, name)
            if isinstance(target, torch.Tensor):
                target.copy_(value.to(target.device))
            else:
                setattr(self.env, name, int(value))

    def _component_roots(self) -> dict[str, object]:
        roots = {
            name: getattr(self.env, name)
            for name in _MANAGERS
            if getattr(self.env, name, None) is not None
        }
        roots["sensors"] = self.env.scene.sensors
        roots["articulation_data"] = {
            name: articulation.data
            for name, articulation in self.env.scene.articulations.items()
        }
        roots["actuators"] = {
            f"{articulation_name}/{actuator_name}": actuator
            for articulation_name, articulation in self.env.scene.articulations.items()
            for actuator_name, actuator in articulation.actuators.items()
        }
        roots["articulations"] = self.env.scene.articulations
        return roots


def _capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else (),
    }


def _restore_rng(state: Any) -> None:
    _validate_rng(state)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    cuda_states = list(state["torch_cuda"])
    if cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)


def _validate_rng(state: Any) -> None:
    if not isinstance(state, Mapping) or set(state) != {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }:
        raise SnapshotError("Isaac RNG snapshot is malformed.")
    cuda_states = list(state["torch_cuda"])
    if cuda_states and len(cuda_states) != torch.cuda.device_count():
        raise SnapshotError("Isaac CUDA RNG device count changed.")


def _move_tensors(value: Any, device: Any) -> Any:
    """Move a loaded native scene tree back to the environment device."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_tensors(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tensors(item, device) for item in value]
    return value


__all__ = ["IsaacSimSnapshotProvider"]
