"""mjlab motion-reference sensor: InstinctMJ's Sensor lifecycle around the portable clip.

``edit_spec`` adds no MuJoCo sensors — the reference is dataset plus FK, same as
InstinctMJ's manager. SDK imports stay inside :func:`build_sensor`.
"""

from __future__ import annotations

from typing import Any

from instinctlab.engines.motion_reference import (
    envs_due_for_update,
    fill_buffers,
    load_retargetted_clip,
    lookahead_times,
    make_buffers,
    pack_motion_clip,
    sample_clip,
)
from instinctlab.spec.sensor import MotionReferenceRef

__all__ = ["build_sensor"]


def build_sensor(ref: MotionReferenceRef, robot: Any) -> Any:
    """Native mjlab ``SensorCfg`` whose ``build`` returns the clip sensor."""
    import torch
    from dataclasses import dataclass

    from mjlab.sensor import Sensor, SensorCfg

    model_path = robot.asset_for("mjlab").path
    joints = tuple(ref.joints)
    links = tuple(ref.links)

    class MotionReferenceSensor(Sensor):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self._entities: dict = {}
            self._clip = None
            self._data = None
            self._env_origins = None
            self._last_update = None
            self.device = "cpu"

        def edit_spec(self, scene_spec, entities):
            self._entities = entities

        def initialize(self, mj_model, model, data, device: str) -> None:
            self.device = device
            entity = self._entities[ref.entity]
            num_envs = int(entity.data.default_root_state.shape[0])
            raw = load_retargetted_clip(ref.clip, device=device)
            self._clip = pack_motion_clip(
                raw,
                joint_names=joints,
                link_names=links,
                model_path=model_path,
                velocity_method=ref.velocity_method,
                target_fps=ref.clip_target_fps,
                device=device,
            )
            self._data = make_buffers(num_envs, ref.num_frames, len(joints), len(links), device=device)
            self._env_origins = torch.zeros(num_envs, 3, device=device)
            self._last_update = torch.zeros(num_envs, device=device)
            self.reset()

        def reset(self, env_ids=None) -> None:
            super().reset(env_ids)
            if self._data is None:
                return
            if env_ids is None:
                env_ids = torch.arange(self._data.timestamp.shape[0], device=self.device)
            else:
                env_ids = torch.as_tensor(env_ids, device=self.device)
            lo, hi = ref.start_range
            span = torch.rand(len(env_ids), device=self.device) * (hi - lo) + lo
            self._data.start_s[env_ids] = span * self._clip.duration_s
            self._data.timestamp[env_ids] = 0.0
            self._last_update[env_ids] = 0.0
            self._refresh(env_ids)

        def update(self, dt: float) -> None:
            super().update(dt)
            if self._data is None:
                return
            self._data.timestamp = self._data.timestamp + dt
            due = envs_due_for_update(self._data.timestamp, self._last_update, ref.update_period)
            if due.numel() == 0:
                return
            self._refresh(due)
            self._last_update[due] = self._data.timestamp[due]

        def bind_origins(self, origins: torch.Tensor) -> None:
            self._env_origins = origins

        def _compute_data(self):
            return self._data

        def _refresh(self, env_ids: torch.Tensor) -> None:
            times, time_to = lookahead_times(
                self._data.timestamp[env_ids],
                self._data.start_s[env_ids],
                ref.num_frames,
                ref.frame_interval_s,
                ref.data_start_from,
            )
            fill_buffers(
                self._data,
                env_ids,
                sample_clip(self._clip, times),
                time_to,
                env_origins=self._env_origins,
            )

    @dataclass
    class MotionReferenceSensorCfg(SensorCfg):
        name: str = ref.name
        clip_path: str = ref.clip

        def build(self):
            return MotionReferenceSensor(self)

    return MotionReferenceSensorCfg()
