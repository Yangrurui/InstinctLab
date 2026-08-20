"""A contact sensor whose air-time clock agrees with the other engine.

mjlab decides touchdown from ``found``: any geometric contact the solver reports,
at any force. Isaac Lab decides it from ``‖net_forces_w‖ > force_threshold`` and
defaults that to 1 N. Both engines then hand the resulting durations to the same
portable terms, so ``feet_air_time`` was scoring two different gaits -- a graze
carrying no load ends the flight phase on mjlab and does not on Isaac.

Nothing announces this. The tensors have the same names and shapes on both sides,
the numbers are plausible on both sides, and the reward stays in range.

The threshold is hub state (``ContactSensorRef.air_time_force_threshold``), not an
engine default, because the engine defaults are exactly what disagree. Isaac maps
it onto its own field; this module gives mjlab the same rule. InstinctMJ reached
the same place from the other direction -- its ``ForceThresholdContactSensor``
docstring calls the behaviour "InstinctLab force-threshold air-time semantics".

The clock itself is a free function so it can be tested without mjlab installed;
the subclass below only moves tensors in and out of mjlab's state object.
"""

from __future__ import annotations

import torch
from typing import Any

__all__ = ["contact_from_force", "step_contact_clock", "thresholded_contact_sensor_cfg"]


def contact_from_force(force: torch.Tensor, threshold: float, num_slots: int = 1) -> torch.Tensor:
    """``[B, N, 3]`` net forces to a ``[B, P]`` touchdown mask.

    A primary counts as loaded when *any* of its slots carries more than the
    threshold, which is how mjlab reduces slots for the ``found`` rule too.
    """
    over = torch.linalg.vector_norm(force, dim=-1) > threshold
    if num_slots > 1:
        over = over.view(over.size(0), -1, num_slots).any(dim=-1)
    return over


def step_contact_clock(
    *,
    is_contact: torch.Tensor,
    elapsed: torch.Tensor,
    current_air: torch.Tensor,
    last_air: torch.Tensor,
    current_contact: torch.Tensor,
    last_contact: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One tick of the air/contact clock: ``(current_air, last_air, current_contact, last_contact)``.

    Same recurrence Isaac Lab and mjlab both run; only ``is_contact`` differed.
    ``last_*`` latch the duration of the phase that just ended, so they hold the
    completed flight time that ``feet_air_time`` pays on.
    """
    first_contact = (current_air > 0) & is_contact
    first_detached = (current_contact > 0) & ~is_contact
    new_last_air = torch.where(first_contact, current_air + elapsed, last_air)
    new_current_air = torch.where(~is_contact, current_air + elapsed, torch.zeros_like(current_air))
    new_last_contact = torch.where(first_detached, current_contact + elapsed, last_contact)
    new_current_contact = torch.where(is_contact, current_contact + elapsed, torch.zeros_like(current_contact))
    return new_current_air, new_last_air, new_current_contact, new_last_contact


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
