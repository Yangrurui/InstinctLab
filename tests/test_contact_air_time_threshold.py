"""Both engines must call touchdown at the same force.

The clock that produces ``current_air_time`` is identical on Isaac Lab and mjlab;
only the boolean feeding it differed. Isaac thresholds the net contact force at
``ContactSensorCfg.force_threshold`` (1 N by default). Stock mjlab uses ``found``,
which is set for any contact the solver reports at any force. Same tensor names,
same shapes, both plausible -- and ``feet_air_time`` scoring two different gaits.

Both references threshold at 1 N (main through Isaac Lab's default, InstinctMJ
through its own ``ForceThresholdContactSensor``), so the mjlab side was the one
odd behaviour of the four. The threshold now lives in the reference and each
backend maps it onto its own field.
"""

from __future__ import annotations

import torch

import pytest

from instinctlab.engines.mjlab.contact_sensor import contact_from_force, step_contact_clock
from instinctlab.spec.sensor import ContactSensorRef


def _flight_then(load: float) -> torch.Tensor:
    """Air time after one tick carrying ``load`` newtons, having flown 0.4 s."""
    force = torch.tensor([[[0.0, 0.0, load]]])
    is_contact = contact_from_force(force, threshold=1.0)
    current_air, _, _, _ = step_contact_clock(
        is_contact=is_contact,
        elapsed=torch.tensor([[0.02]]),
        current_air=torch.tensor([[0.4]]),
        last_air=torch.zeros(1, 1),
        current_contact=torch.zeros(1, 1),
        last_contact=torch.zeros(1, 1),
    )
    return current_air


def test_a_graze_below_the_threshold_does_not_end_the_flight_phase() -> None:
    """The drift itself: 0.5 N is touchdown under ``found`` and is not under 1 N."""
    assert _flight_then(0.5).item() == pytest.approx(0.42), "sub-threshold contact must keep the clock running"
    assert _flight_then(5.0).item() == 0.0, "a loaded foot must reset the air clock"


def test_the_completed_flight_time_is_what_lands_in_last_air_time() -> None:
    """``feet_air_time`` pays on ``last_air_time``, so the latch is the part that matters."""
    _, last_air, current_contact, _ = step_contact_clock(
        is_contact=torch.tensor([[True]]),
        elapsed=torch.tensor([[0.02]]),
        current_air=torch.tensor([[0.4]]),
        last_air=torch.zeros(1, 1),
        current_contact=torch.zeros(1, 1),
        last_contact=torch.zeros(1, 1),
    )
    assert last_air.item() == pytest.approx(0.42)
    assert current_contact.item() == pytest.approx(0.02)


def test_the_clock_only_latches_on_the_transition() -> None:
    """A foot already loaded must not re-latch and inflate the last flight."""
    _, last_air, _, _ = step_contact_clock(
        is_contact=torch.tensor([[True]]),
        elapsed=torch.tensor([[0.02]]),
        current_air=torch.zeros(1, 1),
        last_air=torch.tensor([[0.31]]),
        current_contact=torch.tensor([[0.06]]),
        last_contact=torch.zeros(1, 1),
    )
    assert last_air.item() == pytest.approx(0.31)


def test_a_primary_is_loaded_when_any_of_its_slots_is() -> None:
    """Matches how mjlab reduces slots for ``found``; ours must not differ there either."""
    force = torch.tensor([[[0.0, 0.0, 0.1], [0.0, 0.0, 9.0], [0.0, 0.0, 0.2], [0.0, 0.0, 0.3]]])
    assert contact_from_force(force, threshold=1.0, num_slots=2).tolist() == [[True, False]]


def test_the_threshold_is_declared_rather_than_inherited() -> None:
    """1 N is Isaac Lab's default and both references' value, but it is ours to state."""
    assert ContactSensorRef(name="c", elements=".*").air_time_force_threshold == 1.0


def test_mjlab_builds_a_sensor_that_clocks_off_force() -> None:
    pytest.importorskip("mjlab")
    from instinctlab.engines.mjlab.scene import _contact_sensor

    ref = ContactSensorRef(name="contact_forces", elements=".*", track_air_time=True, air_time_force_threshold=2.5)
    cfg = _contact_sensor(ref)

    assert (
        getattr(cfg, "force_threshold", None) == 2.5
    ), "a plain ContactSensorCfg has no threshold and mjlab would clock air time off 'found'"
    assert "force" in cfg.fields, "the clock reads net force; without the field it cannot run"
    assert type(cfg).__name__ == "ForceThresholdContactSensorCfg"


def test_the_mjlab_sensor_refuses_to_clock_without_the_force_field() -> None:
    """Silence here would mean timers pinned at zero for a whole run, as ``found`` once did."""
    pytest.importorskip("mjlab")
    from instinctlab.engines.mjlab.contact_sensor import _classes

    sensor_type, cfg_type = _classes()
    from mjlab.sensor.contact_sensor import _AirTimeState

    sensor = sensor_type.__new__(sensor_type)
    sensor._air_time_state = _AirTimeState(
        current_air_time=torch.zeros(1, 1),
        last_air_time=torch.zeros(1, 1),
        current_contact_time=torch.zeros(1, 1),
        last_contact_time=torch.zeros(1, 1),
        last_time=torch.zeros(1),
    )
    sensor._data = None
    object.__setattr__(sensor, "cfg", cfg_type(name="c", primary=None, force_threshold=1.0))
    sensor._extract_sensor_data = lambda: type("D", (), {"force": None})()  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="did not request the 'force' field"):
        sensor._update_air_time_tracking()


@pytest.mark.isaacsim
def test_isaac_carries_the_declared_threshold_onto_its_own_field() -> None:
    """Isaac already defaulted to 1 N; passing it explicitly is what keeps the two tied.

    Starts Kit itself. It did not, and since it is the only ``isaacsim`` test in this file
    there was nothing else to start one: importing ``isaaclab.sensors`` without a running
    app raises ``ModuleNotFoundError: No module named 'carb'``, so the test could not pass
    under any invocation. It looked like an environment problem rather than a broken test,
    which is how it survived being written and read.
    """
    import argparse
    import sys

    pytest.importorskip("isaaclab")
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    argv = ["--headless", "--device", "cpu"]
    previous = sys.argv
    sys.argv = [previous[0], *argv]
    try:
        AppLauncher(parser.parse_args(argv))
    finally:
        sys.argv = previous

    from instinctlab.engines.isaacsim.scene import _contact_sensor

    ref = ContactSensorRef(name="contact_forces", elements=".*", track_air_time=True, air_time_force_threshold=2.5)
    assert _contact_sensor(ref).force_threshold == 2.5
