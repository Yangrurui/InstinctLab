"""mjlab motion-reference sensor: InstinctMJ's Sensor lifecycle around the portable clip.

``edit_spec`` adds no MuJoCo sensors — the reference is dataset plus FK, same as
InstinctMJ's manager. SDK imports stay inside :func:`build_motion_reference_sensor`. The clip
clock and the 50% mirror mask live in
:class:`~instinctlab.engines.motion_reference.MotionReferenceRuntime`.
"""

from __future__ import annotations

from typing import Any

from instinctlab.engines.motion_reference import MotionReferenceRuntime
from instinctlab.spec.sensor import MotionReferenceRef

__all__ = ["build_motion_reference_sensor"]


def build_motion_reference_sensor(ref: MotionReferenceRef, robot: Any) -> Any:
    """Native mjlab ``SensorCfg`` whose ``build`` returns the clip sensor."""
    import torch
    from dataclasses import dataclass

    from mjlab.sensor import Sensor, SensorCfg

    model_path = robot.asset_for("mjlab").path

    class MotionReferenceSensor(Sensor):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self._entities: dict = {}
            self._runtime = None
            self._data = None
            self.device = "cpu"

        def edit_spec(self, scene_spec, entities):
            self._entities = entities

        def initialize(self, mj_model, model, data, device: str) -> None:
            self.device = device
            entity = self._entities[ref.entity]
            num_envs = int(entity.data.default_root_state.shape[0])
            self._runtime = MotionReferenceRuntime.create(ref, model_path, num_envs, device)
            self._data = self._runtime.buffers
            self.reset()

        def reset(self, env_ids=None) -> None:
            super().reset(env_ids)
            if self._runtime is None:
                return
            if env_ids is None:
                env_ids = torch.arange(self._runtime.buffers.timestamp.shape[0], device=self.device)
            else:
                env_ids = torch.as_tensor(env_ids, device=self.device)
            self._runtime.reset(env_ids)

        def update(self, dt: float) -> None:
            super().update(dt)
            if self._runtime is None:
                return
            self._runtime.advance(dt)

        def bind_origins(self, origins: torch.Tensor) -> None:
            if self._runtime is not None:
                self._runtime.bind_origins(origins)

        @property
        def aiming_frame_idx(self):
            return self._runtime.aiming_frame_idx

        def _compute_data(self):
            return self._data

    @dataclass
    class MotionReferenceSensorCfg(SensorCfg):
        name: str = ref.name
        clip_path: str = ref.clip

        def build(self):
            return MotionReferenceSensor(self)

    return MotionReferenceSensorCfg()
