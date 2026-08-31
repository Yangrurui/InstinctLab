from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from instinctlab_engine.lifecycle import (
    ComponentContractError,
    EnvironmentSnapshot,
    LifecycleRuntime,
    SnapshotError,
)
from instinctlab_engine.spec import ComponentLifecycleSpec, LifecycleSpec

from tests.test_spec_task import _task


class _Env:
    num_envs = 3
    device = "cpu"


class _Stateful:
    def __init__(self) -> None:
        self.reset_ids = []

    def reset(self, env_ids=None) -> None:
        self.reset_ids.append(env_ids)

    def snapshot_state(self, env_ids=None):
        return {"value": torch.arange(3)}

    def restore_state(self, state, env_ids=None) -> None:
        del env_ids
        self.restored = state


class _Controller(_Stateful):
    control_dt = 0.02

    def compute(self, command):
        return command


class _SnapshotProvider:
    provider_id = "test/native"
    provider_version = 1

    def __init__(self) -> None:
        self.value = torch.tensor([1.0, 2.0, 3.0])

    def capture(self):
        return {"value": self.value.clone()}

    def restore(self, state) -> None:
        self.value.copy_(state["value"])


def _runtime() -> LifecycleRuntime:
    return LifecycleRuntime(_Env(), _task(), engine="test")


def test_runtime_advances_global_and_episode_clocks_across_partial_reset() -> None:
    runtime = _runtime()
    runtime.before_step()
    runtime.after_step(torch.zeros(3, dtype=torch.bool))

    assert runtime.reading("physics").tick == 4
    assert runtime.reading("policy").tick == 1
    assert runtime.reading("episode").tick.tolist() == [1, 1, 1]

    runtime.before_step()
    runtime.on_reset(torch.tensor([1]))
    runtime.after_step(torch.tensor([False, True, False]))

    assert runtime.reading("policy").tick == 2
    assert runtime.reading("episode").tick.tolist() == [2, 0, 2]
    assert runtime.episode_id.tolist() == [0, 1, 0]
    assert runtime.reset_count.tolist() == [0, 1, 0]


def test_runtime_accounts_for_done_when_native_env_does_not_report_reset() -> None:
    runtime = _runtime()
    runtime.before_step()
    runtime.after_step(torch.tensor([True, False, False]))

    assert runtime.reading("episode").tick.tolist() == [0, 1, 1]
    assert runtime.episode_id.tolist() == [1, 0, 0]


def test_rational_clock_reading_uses_integer_arithmetic() -> None:
    source = _task()
    task = replace(
        source,
        lifecycle=LifecycleSpec(
            clocks=(),
        ),
    )
    runtime = LifecycleRuntime(_Env(), task, engine="test")
    for _ in range(5):
        runtime.before_step()
        runtime.after_step()
    assert runtime.physics_tick == 20


def test_stateful_component_contract_is_fail_closed_and_reset_owner_is_explicit() -> None:
    source = _task()
    task = replace(
        source,
        lifecycle=LifecycleSpec(
            components={
                "reward/rewards/alive": ComponentLifecycleSpec(
                    "policy", "post_physics", "partial", "snapshot"
                )
            }
        ),
    )
    runtime = LifecycleRuntime(_Env(), task, engine="test")
    with pytest.raises(ComponentContractError, match="lacks"):
        runtime.register_component("reward/rewards/alive", object())

    component = _Stateful()
    runtime.register_component(
        "reward/rewards/alive", component, managed_reset=True
    )
    runtime.on_reset(torch.tensor([2]))
    assert len(component.reset_ids) == 1
    assert component.reset_ids[0].tolist() == [2]


def test_stateful_controller_contract_checks_compute_state_and_clock() -> None:
    source = _task()
    task = replace(
        source,
        lifecycle=LifecycleSpec(
            components={
                "controller/whole_body": ComponentLifecycleSpec(
                    "policy", "pre_step", "partial", "snapshot"
                )
            }
        ),
    )
    runtime = LifecycleRuntime(_Env(), task, engine="test")
    controller = _Controller()
    runtime.register_component(
        "controller/whole_body",
        controller,
        managed_reset=True,
    )
    assert controller.compute(torch.tensor([1.0])).tolist() == [1.0]

    controller.control_dt = 0.01
    with pytest.raises(ComponentContractError, match="declared clock period"):
        LifecycleRuntime(_Env(), task, engine="test").register_component(
            "controller/whole_body",
            controller,
        )


def test_clock_state_restore_validates_vectorization_shape() -> None:
    runtime = _runtime()
    state = runtime.state_dict()
    state["episode_id"] = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError, match="incompatible environment shapes"):
        runtime.load_state_dict(state)


def test_failed_native_step_can_be_cancelled_without_advancing_time() -> None:
    runtime = _runtime()
    runtime.before_step()
    runtime.cancel_step()
    runtime.before_step()
    runtime.after_step()
    assert runtime.policy_tick == 1


def test_snapshot_round_trip_restores_native_clock_and_component_state(tmp_path) -> None:
    source = _task()
    task = replace(
        source,
        lifecycle=LifecycleSpec(
            components={
                "reward/rewards/alive": ComponentLifecycleSpec(
                    "policy", "post_physics", "partial", "snapshot"
                )
            }
        ),
    )
    runtime = LifecycleRuntime(_Env(), task, engine="test")
    provider = _SnapshotProvider()
    component = _Stateful()
    runtime.set_snapshot_provider(provider)
    runtime.register_component("reward/rewards/alive", component)
    runtime.before_step()
    runtime.after_step()

    snapshot = runtime.snapshot(metadata={"purpose": "unit-test"})
    path = snapshot.save(tmp_path / "state.snapshot.npz")
    loaded = EnvironmentSnapshot.load(path)

    provider.value.zero_()
    runtime.physics_tick = 99
    runtime.restore(loaded)

    assert provider.value.tolist() == [1.0, 2.0, 3.0]
    assert runtime.physics_tick == 4
    assert component.restored["value"].tolist() == [0, 1, 2]
    assert loaded.metadata == {"purpose": "unit-test"}


def test_snapshot_rejects_wrong_engine_before_mutating_provider() -> None:
    runtime = _runtime()
    provider = _SnapshotProvider()
    runtime.set_snapshot_provider(provider)
    snapshot = runtime.snapshot()
    incompatible = replace(snapshot, engine="other")
    provider.value.zero_()

    with pytest.raises(SnapshotError, match="identity does not match"):
        runtime.restore(incompatible)

    assert provider.value.tolist() == [0.0, 0.0, 0.0]


def test_snapshot_requires_an_engine_provider_and_a_closed_step() -> None:
    runtime = _runtime()
    with pytest.raises(SnapshotError, match="did not attach"):
        runtime.snapshot()
    runtime.set_snapshot_provider(_SnapshotProvider())
    runtime.before_step()
    with pytest.raises(SnapshotError, match="step is open"):
        runtime.snapshot()
