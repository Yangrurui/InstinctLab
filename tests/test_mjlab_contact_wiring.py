"""The mjlab contact sensor actually reports contact.

Written after a run that trained for five thousand iterations against a sensor whose contact timers
never left zero. Nothing raised: mjlab accumulates air and contact time from its ``found`` field and
returns early when that field was not requested, so the compiled sensor was well-formed, the
environment stepped, the policy improved -- and ``illegal_contact`` could not fire while
``feet_air_time`` paid nothing. Episodes could only end by timing out, which is visible in a
training curve only if you have another curve to hold it against.

The static test states the requirement; the live one is the one that would have caught it, because
it asks the engine rather than the config. Both are here because the static test is cheap enough to
run everywhere and the live test is not. The live test is marked ``mjlab`` so the default suite
does not start an engine; run it with ``pytest -o addopts= -m mjlab``.
"""

from __future__ import annotations

import torch

import pytest

from instinctlab.spec.sensor import ContactSensorRef

pytest.importorskip("mjlab")


def test_the_sensor_requests_the_field_its_timers_are_built_from() -> None:
    """``track_air_time`` without ``found`` is accepted by mjlab and does nothing."""
    from instinctlab.engines.mjlab.scene import _contact_sensor

    cfg = _contact_sensor(ContactSensorRef(name="contact_forces", elements=".*", track_air_time=True))

    assert "found" in cfg.fields, (
        "mjlab derives contact and air time from 'found' and skips the update when it is absent, "
        "leaving every contact-based term reading zero for the whole run."
    )
    assert cfg.track_air_time is True


@pytest.mark.mjlab
@pytest.mark.skipif(not torch.cuda.is_available(), reason="stepping mjlab needs a GPU")
def test_standing_on_the_ground_registers_as_contact() -> None:
    """The signal every contact term reads, taken from a built environment rather than a config.

    A G1 dropped onto a plane and held at its default pose stands on its feet, so after a few steps
    the feet have accumulated contact time and are reported as touching. Both statements come from
    the same timers that ``illegal_contact`` and ``feet_air_time`` consult.
    """
    from instinctlab.compat import sensors as compat_sensors
    from instinctlab.engines.mjlab.adapter import MjlabAdapter
    from instinctlab.tasks.locomotion.config.g1 import flat_g1

    feet = ContactSensorRef(name="contact_forces", elements=".*_ankle_roll_link")
    compiled = MjlabAdapter().compile(flat_g1(), num_envs=16, device="cuda:0")
    env = compiled.make_env()
    try:
        env.reset()
        actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        for _ in range(20):
            env.step(actions)

        sensor = env.scene.sensors["contact_forces"]
        sensor_class = type(sensor).__name__
        threshold = getattr(sensor.cfg, "force_threshold", None)
        contact_time = compat_sensors.contact_time(sensor, feet)
        touching = compat_sensors.in_contact(sensor, feet)
    finally:
        env.close()

    assert torch.any(contact_time > 0.0), (
        f"a standing robot's feet accumulated no contact time (max {float(contact_time.max())}); "
        "every contact-based term reads this and would be silently dead"
    )
    assert torch.any(touching), "contact time accumulated but in_contact reported nothing"
    # The clock above must be the force-thresholded one. mjlab builds sensors through
    # cfg.build(), so a subclass that stopped being returned there would leave the stock
    # `found` clock running and every assertion in this test would still pass.
    assert (
        sensor_class == "ForceThresholdContactSensor"
    ), f"scene built a {sensor_class}; air time is back on mjlab's zero-force 'found' rule"
    assert threshold == 1.0
