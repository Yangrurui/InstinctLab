from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from instinctlab_engine.lifecycle import (
    EpisodeTrace,
    LifecycleRuntime,
    ReplayMismatch,
    TraceError,
    replay_trace,
)

from tests.test_spec_task import _task


class _NativeEnvironment:
    num_envs = 3
    device = "cpu"


class _CounterProvider:
    provider_id = "test/counter"
    provider_version = 1

    def __init__(self) -> None:
        self.counter = torch.zeros((), dtype=torch.long)

    def capture(self):
        return {"counter": self.counter.clone()}

    def restore(self, state) -> None:
        self.counter.copy_(state["counter"])


class _ReplayEnvironment:
    def __init__(self) -> None:
        self.device = "cpu"
        self.lifecycle = LifecycleRuntime(
            _NativeEnvironment(), _task(), engine="test"
        )
        self.provider = _CounterProvider()
        self.lifecycle.set_snapshot_provider(self.provider)

    def step(self, action):
        self.lifecycle.before_step()
        self.provider.counter += 1
        observation = action[:, :2] + self.provider.counter
        reward = action.sum(dim=1, keepdim=True) * self.provider.counter
        done = torch.full(
            (3,), self.provider.counter.item() >= 2, dtype=torch.long
        )
        timeout = torch.zeros(3, dtype=torch.bool)
        self.lifecycle.after_step(done)
        self.lifecycle.record_transition(
            actions=action,
            observations=observation,
            rewards=reward,
            dones=done,
            timeouts=timeout,
        )
        return observation, reward, done, {"time_outs": timeout}


def _record_complete_trace() -> tuple[_ReplayEnvironment, EpisodeTrace]:
    env = _ReplayEnvironment()
    env.lifecycle.start_trace(env_ids=torch.tensor([0, 2]))
    env.step(torch.arange(9, dtype=torch.float32).reshape(3, 3))
    env.step(torch.ones(3, 3))
    return env, env.lifecycle.stop_trace()


def test_episode_trace_round_trip_and_replay(tmp_path) -> None:
    env, trace = _record_complete_trace()
    path = trace.save(tmp_path / "episode.trace.npz")
    loaded = EpisodeTrace.load(path)
    env.provider.counter.fill_(99)

    report = replay_trace(env, loaded)

    assert loaded.complete is True
    assert loaded.env_ids.tolist() == [0, 2]
    assert len(loaded.steps) == 2
    assert report.matched is True
    assert report.compared_steps == 2
    assert env.provider.counter.item() == 2


def test_strict_replay_reports_first_normalized_boundary_difference() -> None:
    env, trace = _record_complete_trace()
    first = trace.steps[0]
    changed = replace(
        first,
        observation=first.observation + 0.25,
    )
    incompatible = replace(trace, steps=(changed, *trace.steps[1:]))

    with pytest.raises(ReplayMismatch, match="step 0 field observation"):
        replay_trace(env, incompatible)

    report = replay_trace(env, incompatible, strict=False)
    assert report.differences[0].max_index == (0, 0)
    assert report.differences[0].actual_at_max == pytest.approx(1.0)
    assert report.differences[0].expected_at_max == pytest.approx(1.25)

    accepted = replay_trace(
        env,
        incompatible,
        field_tolerances={"observation": (0.3, 0.0)},
    )
    assert accepted.matched is True


def test_replay_tolerances_fail_closed() -> None:
    env, trace = _record_complete_trace()

    with pytest.raises(TraceError, match="Unknown replay tolerance"):
        replay_trace(env, trace, field_tolerances={"unknown": (1.0, 0.0)})
    with pytest.raises(TraceError, match="finite and non-negative"):
        replay_trace(env, trace, atol=-1.0)


def test_trace_requires_episode_boundary_and_complete_selected_episodes() -> None:
    env = _ReplayEnvironment()
    env.lifecycle.episode_physics_tick[1] = 4
    with pytest.raises(TraceError, match="episode boundary"):
        env.lifecycle.start_trace(env_ids=[1])

    env.lifecycle.start_trace(env_ids=[0])
    env.step(torch.ones(3, 3))
    with pytest.raises(TraceError, match="not complete"):
        env.lifecycle.stop_trace()
    trace = env.lifecycle.stop_trace(require_complete=False)
    assert trace.complete is False


def test_external_reset_invalidates_an_active_trace() -> None:
    env = _ReplayEnvironment()
    env.lifecycle.start_trace(env_ids=[0])
    env.lifecycle.on_reset(torch.tensor([0]))
    with pytest.raises(TraceError, match="external reset"):
        env.lifecycle.stop_trace(require_complete=False)
