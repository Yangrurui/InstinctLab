"""Termination terms that run unmodified under either engine's native manager."""

from __future__ import annotations

import torch

from instinctlab.compat import sensors as compat_sensors
from instinctlab.compat.env import RlEnv
from instinctlab.spec.sensor import ContactSensorRef

__all__ = ["illegal_contact", "time_out"]


def time_out(env: RlEnv) -> torch.Tensor:
    """Whether the episode reached its time limit. Identical on both engines."""
    return env.episode_length_buf >= env.max_episode_length


def illegal_contact(env: RlEnv, sensor: ContactSensorRef) -> torch.Tensor:
    """Whether any of the referenced elements is touching something it should not be.

    This one is deliberately **not** a port of either engine's version, and the reason is worth
    stating, because both originals take a newton threshold and this one does not.

    Isaac Lab thresholds the norm of ``net_forces_w_history``, which is the world-frame *normal*
    force alone -- its own docstring warns that the tangential component is excluded. mjlab
    thresholds the norm of ``force_history``, which is the full 3-D contact force, expressed in the
    contact frame unless the sensor was configured otherwise. The same threshold in newtons
    therefore means "normal load above N" on one engine and "total load including friction above N"
    on the other, and the gap between them is whatever friction happens to be carrying at that
    instant. A foot planted on a slope crosses one threshold and not the other.

    So the portable version asks each engine's own sensor whether it considers the element to be in
    contact, via the contact-duration signal that both engines maintain internally. That loses the
    ability to ignore light brushes, which is what the threshold was buying. A task that genuinely
    needs a force threshold should declare this termination per engine and write down the tolerance
    -- which is the honest version of what a shared threshold was doing anyway.
    """
    touching = compat_sensors.in_contact(env.scene.sensors[sensor.name], sensor)
    return torch.any(touching, dim=1)
