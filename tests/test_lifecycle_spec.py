from __future__ import annotations

from dataclasses import replace

import pytest
from instinctlab_engine.spec import (
    ClockDomainSpec,
    ComponentLifecycleSpec,
    LifecycleSpec,
    NativeSensorRef,
)

from tests.test_spec_task import _task


def test_builtin_clocks_are_exact_physics_step_ratios() -> None:
    task = _task()
    clocks, components = task.lifecycle_contract()

    assert clocks["physics"].period_physics_steps == 1
    assert clocks["policy"].period_physics_steps == task.sim.decimation
    assert clocks["episode"].reset == "episode"
    assert components["action/joint_pos"] == ComponentLifecycleSpec(
        clock="policy",
        phase="pre_step",
        reset="partial",
        state="snapshot",
    )
    assert components["sensor/feet"].clock == "physics"
    assert components["sensor/feet"].state == "snapshot"


def test_custom_clocks_resolve_through_named_parents_without_drift() -> None:
    task = replace(
        _task(),
        lifecycle=LifecycleSpec(
            clocks=(
                ClockDomainSpec("camera", parent="policy", tick_divider=2),
                ClockDomainSpec(
                    "episode.camera",
                    parent="camera",
                    tick_divider=3,
                    reset="episode",
                ),
            )
        ),
    )
    clocks, _ = task.lifecycle_contract()

    assert clocks["camera"].period_physics_steps == 8
    assert clocks["episode.camera"].period_physics_steps == 24
    assert clocks["episode.camera"].reset == "episode"


def test_clock_cycles_and_unknown_component_overrides_fail_closed() -> None:
    cyclic = replace(
        _task(),
        lifecycle=LifecycleSpec(
            clocks=(
                ClockDomainSpec("first", parent="second"),
                ClockDomainSpec("second", parent="first"),
            )
        ),
    )
    with pytest.raises(ValueError, match="unknown parent or cycle"):
        cyclic.validate()

    unknown_component = replace(
        _task(),
        lifecycle=LifecycleSpec(
            components={
                "reward/misspelled": ComponentLifecycleSpec(
                    "policy", "pre_step", "partial", "snapshot"
                )
            }
        ),
    )
    with pytest.raises(ValueError, match="undeclared components"):
        unknown_component.validate()


def test_asynchronous_sensor_period_uses_an_exact_rational_clock() -> None:
    source = _task()
    task = replace(
        source,
        sim=replace(source.sim, physics_dt=0.006),
        scene=replace(
            source.scene,
            native_sensors=(
                NativeSensorRef(
                    name="imu", kind="imu", attach="root", update_period=0.02
                ),
            ),
        ),
    )
    clocks, components = task.lifecycle_contract()
    sensor_clock = clocks[components["sensor/imu"].clock]
    assert sensor_clock.period_numerator == 10
    assert sensor_clock.period_denominator == 3


def test_snapshot_state_requires_reset_semantics() -> None:
    with pytest.raises(ValueError, match="recoverable stateful component"):
        ComponentLifecycleSpec(
            clock="policy",
            phase="pre_step",
            reset="stateless",
            state="snapshot",
        )


def test_explicit_stateful_controller_is_a_first_class_component() -> None:
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

    _, components = task.lifecycle_contract()

    assert components["controller/whole_body"] == ComponentLifecycleSpec(
        "policy", "pre_step", "partial", "snapshot"
    )


def test_controller_rejects_stateless_or_post_step_contracts() -> None:
    source = _task()
    stateless = replace(
        source,
        lifecycle=LifecycleSpec(
            components={
                "controller/bad": ComponentLifecycleSpec(
                    "policy", "pre_step", "stateless", "stateless"
                )
            }
        ),
    )
    with pytest.raises(ValueError, match="recoverable state"):
        stateless.lifecycle_contract()

    late = replace(
        source,
        lifecycle=LifecycleSpec(
            components={
                "controller/bad": ComponentLifecycleSpec(
                    "policy", "post_step", "partial", "snapshot"
                )
            }
        ),
    )
    with pytest.raises(ValueError, match="pre_step or pre_physics"):
        late.lifecycle_contract()
