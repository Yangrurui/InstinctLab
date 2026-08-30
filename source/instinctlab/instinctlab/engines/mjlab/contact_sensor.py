"""A contact sensor whose air-time clock follows the declared engine reference.

mjlab decides touchdown from ``found``: any geometric contact the solver reports,
at any force. Isaac Lab decides it from ``‖net_forces_w‖ > force_threshold`` and
defaults that to 1 N. Without an explicit threshold, the portable terms can therefore
score a different gait on mjlab -- a graze carrying no load ends the flight phase there
and does not on Isaac.

Nothing announces this. The tensors have the same names and shapes on both sides,
the numbers are plausible on both sides, and the reward stays in range.

The threshold is hub state (``ContactSensorRef.air_time_force_threshold``), not an
engine default, because the engine defaults are exactly what disagree. Each builder
first resolves the task's reference-specific value, then maps it onto its native field.
InstinctMJ reached the same place from the other direction -- its ``ForceThresholdContactSensor``
docstring calls the behaviour "InstinctLab force-threshold air-time semantics".

The clock itself lives in ``compat.sensors.contact`` so it can be tested without
mjlab installed; the subclass below only moves tensors in and out of mjlab's
state object.
"""

from __future__ import annotations

from typing import Any

from instinctlab.compat.sensors.contact import contact_from_force, step_contact_clock

__all__ = ["thresholded_contact_sensor_cfg"]


def thresholded_contact_sensor_cfg(*, force_threshold: float, **kwargs: Any) -> Any:
    """``ContactSensorCfg`` whose air-time clock reads net force, not ``found``."""
    _, cfg_type = _classes()
    return cfg_type(force_threshold=force_threshold, **kwargs)


_CACHE: tuple[type, type] | None = None


def _classes() -> tuple[type, type]:
    """Build the subclass on first use; mjlab is imported lazily on purpose."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    from dataclasses import dataclass

    from mjlab.sensor import ContactSensor, ContactSensorCfg

    class ForceThresholdContactSensor(ContactSensor):  # type: ignore[misc, valid-type]
        """mjlab contact sensor that clocks air time off net force, as Isaac Lab does."""

        def _update_air_time_tracking(self) -> None:
            state = self._air_time_state
            assert state is not None
            contact = self._extract_sensor_data()
            if contact.force is None:
                raise RuntimeError(
                    f"Contact sensor {self.cfg.name!r} clocks air time off net force but did not "
                    "request the 'force' field. Without it the timers would stay at zero all run."
                )
            assert self._data is not None
            now = self._data.time
            elapsed = (now - state.last_time).unsqueeze(-1)
            is_contact = contact_from_force(contact.force, self.cfg.force_threshold, self.cfg.num_slots)
            current_air, last_air, current_contact, last_contact = step_contact_clock(
                is_contact=is_contact,
                elapsed=elapsed,
                current_air=state.current_air_time,
                last_air=state.last_air_time,
                current_contact=state.current_contact_time,
                last_contact=state.last_contact_time,
            )
            state.current_air_time[:] = current_air
            state.last_air_time[:] = last_air
            state.current_contact_time[:] = current_contact
            state.last_contact_time[:] = last_contact
            state.last_time[:] = now

    @dataclass(kw_only=True)
    class ForceThresholdContactSensorCfg(ContactSensorCfg):  # type: ignore[misc, valid-type]
        """``ContactSensorCfg`` plus the newton threshold the air-time clock uses."""

        force_threshold: float = 1.0

        def build(self) -> Any:
            return ForceThresholdContactSensor(self)

    _CACHE = (ForceThresholdContactSensor, ForceThresholdContactSensorCfg)
    return _CACHE
