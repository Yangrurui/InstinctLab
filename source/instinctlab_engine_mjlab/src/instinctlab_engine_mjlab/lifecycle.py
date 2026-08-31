"""MJLab implementation of complete same-engine lifecycle snapshots."""

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

_DYNAMIC_DATA_FIELDS = (
    "time",
    "qpos",
    "qvel",
    "act",
    "history",
    "qacc_warmstart",
    "ctrl",
    "qfrc_applied",
    "xfrc_applied",
    "eq_active",
    "mocap_pos",
    "mocap_quat",
    "userdata",
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


class MjlabSnapshotProvider:
    """Capture MJWarp integration, manager, sensor, actuator, and RNG state."""

    provider_id = "mjlab/manager-based-env"
    provider_version = 1

    def __init__(self, env: Any) -> None:
        self.env = env

    def capture(self) -> Mapping[str, Any]:
        return {
            "simulation": {
                name: getattr(self.env.sim.data, name).clone()
                for name in _DYNAMIC_DATA_FIELDS
                if hasattr(self.env.sim.data, name)
            },
            "environment": self._capture_environment(),
            "components": capture_state_tree(self._component_roots()),
            "rng": _capture_rng(),
        }

    def restore(self, state: Mapping[str, Any]) -> None:
        expected = {"simulation", "environment", "components", "rng"}
        if set(state) != expected:
            raise SnapshotError(
                f"MJLab snapshot fields are {sorted(state)}, expected {sorted(expected)}."
            )
        simulation = state["simulation"]
        environment = state["environment"]
        components = state["components"]
        if not all(
            isinstance(value, Mapping)
            for value in (simulation, environment, components)
        ):
            raise SnapshotError("MJLab snapshot contains a malformed state section.")
        self._validate_simulation(simulation)
        self._validate_environment(environment)
        roots = self._component_roots()
        validate_state_tree(roots, components)
        _validate_rng(state["rng"])

        for name, source in simulation.items():
            target = getattr(self.env.sim.data, name)
            target.copy_(source.to(target.device))
        self._restore_environment(environment)
        self.env.sim.forward()
        self.env.sim.sense()
        restore_state_tree(roots, components)
        self.env.scene.write_data_to_sim()
        _restore_rng(state["rng"])

    def _validate_simulation(self, state: Mapping[str, Any]) -> None:
        expected = {
            name
            for name in _DYNAMIC_DATA_FIELDS
            if hasattr(self.env.sim.data, name)
        }
        if set(state) != expected:
            raise SnapshotError("MJLab simulation data schema changed.")
        for name, source in state.items():
            target = getattr(self.env.sim.data, name)
            if not isinstance(source, torch.Tensor):
                raise SnapshotError(f"MJLab simulation field {name} is not a tensor.")
            if source.shape != target.shape or source.dtype != target.dtype:
                raise SnapshotError(f"MJLab simulation field {name} is incompatible.")

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
            raise SnapshotError("MJLab environment buffer schema changed.")
        for name, target in expected.items():
            source = state[name]
            if isinstance(target, torch.Tensor):
                if not isinstance(source, torch.Tensor):
                    raise SnapshotError(f"MJLab environment field {name} is not a tensor.")
                if source.shape != target.shape or source.dtype != target.dtype:
                    raise SnapshotError(f"MJLab environment field {name} is incompatible.")
            elif not isinstance(source, int):
                raise SnapshotError(f"MJLab environment field {name} is not an integer.")

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
        roots["entity_data"] = {
            name: entity.data for name, entity in self.env.scene.entities.items()
        }
        roots["actuators"] = {
            f"{entity_name}/{index}": actuator
            for entity_name, entity in self.env.scene.entities.items()
            for index, actuator in enumerate(entity.actuators)
        }
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
        raise SnapshotError("MJLab RNG snapshot is malformed.")
    cuda_states = list(state["torch_cuda"])
    if cuda_states and len(cuda_states) != torch.cuda.device_count():
        raise SnapshotError("MJLab CUDA RNG device count changed.")


__all__ = ["MjlabSnapshotProvider"]
