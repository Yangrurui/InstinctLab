from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import torch
from instinctlab_engine_isaacsim.lifecycle import IsaacSimSnapshotProvider
from instinctlab_engine_mjlab.lifecycle import (
    _DYNAMIC_DATA_FIELDS,
    MjlabSnapshotProvider,
)


class _IsaacScene:
    def __init__(self) -> None:
        self.sensors = {}
        self.articulations = {}
        self.position = torch.tensor([[1.0, 2.0]])
        self.write_count = 0

    def get_state(self, is_relative=False):
        assert is_relative is False
        return {"position": self.position.clone()}

    def reset_to(self, state, is_relative=False) -> None:
        assert is_relative is False
        self.position.copy_(state["position"])

    def write_data_to_sim(self) -> None:
        self.write_count += 1


class _Simulation:
    def __init__(self, data=None) -> None:
        self.data = data
        self.forward_count = 0
        self.sense_count = 0

    def forward(self) -> None:
        self.forward_count += 1

    def sense(self) -> None:
        self.sense_count += 1


def _environment(scene, sim):
    return SimpleNamespace(
        scene=scene,
        sim=sim,
        episode_length_buf=torch.tensor([3, 4]),
        reset_buf=torch.tensor([False, True]),
        reset_terminated=torch.tensor([False, True]),
        reset_time_outs=torch.tensor([False, False]),
        reward_buf=torch.tensor([0.5, 1.5]),
        common_step_counter=7,
        _sim_step_counter=28,
    )


def test_isaac_provider_restores_scene_buffers_and_native_forward() -> None:
    scene = _IsaacScene()
    env = _environment(scene, _Simulation())
    provider = IsaacSimSnapshotProvider(env)
    state = deepcopy(provider.capture())

    scene.position.zero_()
    env.episode_length_buf.zero_()
    env.common_step_counter = 99
    provider.restore(state)

    assert scene.position.tolist() == [[1.0, 2.0]]
    assert env.episode_length_buf.tolist() == [3, 4]
    assert env.common_step_counter == 7
    assert scene.write_count == 1
    assert env.sim.forward_count == 1


def test_mjlab_provider_restores_integration_data_buffers_and_sensing() -> None:
    data = SimpleNamespace(
        **{
            name: torch.full((2, 1), float(index), dtype=torch.float32)
            for index, name in enumerate(_DYNAMIC_DATA_FIELDS)
        }
    )
    scene = SimpleNamespace(
        sensors={},
        entities={},
        write_data_to_sim=lambda: None,
    )
    sim = _Simulation(data)
    env = _environment(scene, sim)
    provider = MjlabSnapshotProvider(env)
    state = deepcopy(provider.capture())

    data.qpos.zero_()
    env.reward_buf.zero_()
    provider.restore(state)

    qpos_value = float(_DYNAMIC_DATA_FIELDS.index("qpos"))
    assert data.qpos.tolist() == [[qpos_value], [qpos_value]]
    assert env.reward_buf.tolist() == [0.5, 1.5]
    assert sim.forward_count == 1
    assert sim.sense_count == 1
