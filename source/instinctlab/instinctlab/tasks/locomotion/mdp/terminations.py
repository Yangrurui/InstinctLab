"""Locomotion termination terms called directly by both engines."""

from __future__ import annotations

import torch

from instinctlab.compat import sensors as compat_sensors
from instinctlab.compat.env import RlEnv
from instinctlab.spec.sensor import ContactSensorRef


def time_out(env: RlEnv) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


def illegal_contact(env: RlEnv, sensor: ContactSensorRef) -> torch.Tensor:
    touching = compat_sensors.in_contact(env.scene.sensors[sensor.name], sensor)
    return torch.any(touching, dim=1)
