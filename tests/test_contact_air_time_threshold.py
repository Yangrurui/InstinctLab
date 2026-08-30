"""Both engines must call touchdown at the force declared for their reference.

The clock that produces ``current_air_time`` is identical on Isaac Lab and mjlab;
only the boolean feeding it differed. Isaac thresholds the net contact force at
``ContactSensorCfg.force_threshold`` (1 N by default). Stock mjlab uses ``found``,
which is set for any contact the solver reports at any force. Same tensor names,
same shapes, both plausible -- and ``feet_air_time`` scoring two different gaits.

Locomotion and Parkour use 1 N on both references. Shadowing main explicitly
uses 10 N while InstinctMJ explicitly uses 1 N. The base threshold and the rare
per-engine override both live in ``ContactSensorRef``; each backend resolves the
same declaration onto its native field.
"""

from __future__ import annotations

import torch

import pytest

from instinctlab.compat.sensors.contact import contact_from_force, step_contact_clock
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
    """1 N is the common default; Shadowing declares its main override separately."""
    assert ContactSensorRef(name="c", elements=".*").air_time_force_threshold == 1.0


def test_contact_sensor_rejects_an_invalid_force_threshold() -> None:
    with pytest.raises(ValueError, match="invalid air_time_force_threshold"):
        ContactSensorRef(name="c", elements=".*", air_time_force_threshold=float("nan"))
    with pytest.raises(ValueError, match="invalid air_time_force_threshold"):
        ContactSensorRef(name="c", elements=".*", air_time_force_threshold=-1.0)
    with pytest.raises(ValueError, match="invalid air-time threshold override"):
        ContactSensorRef(name="c", elements=".*", engine_air_time_force_thresholds={"isaacsim": -1.0})


def test_contact_sensor_resolves_a_reference_specific_threshold() -> None:
    ref = ContactSensorRef(
        name="contact_forces",
        elements=".*",
        air_time_force_threshold=1.0,
        engine_air_time_force_thresholds={"isaacsim": 10.0},
    )
    assert ref.for_engine("isaacsim").air_time_force_threshold == 10.0
    assert ref.for_engine("mjlab").air_time_force_threshold == 1.0
    assert ref.for_engine("isaacsim").engine_air_time_force_thresholds == {}


def test_mjlab_builds_a_sensor_that_clocks_off_force() -> None:
    pytest.importorskip("mjlab")
    from instinctlab.engines.mjlab.scene import _build_contact_sensor

    ref = ContactSensorRef(
        name="contact_forces",
        elements=".*",
        track_air_time=True,
        air_time_force_threshold=10.0,
        engine_air_time_force_thresholds={"mjlab": 2.5},
    )
    cfg = _build_contact_sensor(ref)

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
    """Isaac already defaults to 1 N; passing the resolved value keeps reference choices explicit.

    Starts Kit itself. It did not, and since it is the only ``isaacsim`` test in this file
    there was nothing else to start one: importing ``isaaclab.sensors`` without a running
    app raises ``ModuleNotFoundError: No module named 'carb'``, so the test could not pass
    under any invocation. It looked like an environment problem rather than a broken test,
    which is how it survived being written and read.
    """
    pytest.importorskip("isaaclab")
    from tests.isaacsim_app import ensure_isaac_app

    ensure_isaac_app()

    from instinctlab.engines.isaacsim.scene import _build_contact_sensor

    ref = ContactSensorRef(
        name="contact_forces",
        elements=".*",
        track_air_time=True,
        air_time_force_threshold=1.0,
        engine_air_time_force_thresholds={"isaacsim": 2.5},
    )
    assert _build_contact_sensor(ref).force_threshold == 2.5
