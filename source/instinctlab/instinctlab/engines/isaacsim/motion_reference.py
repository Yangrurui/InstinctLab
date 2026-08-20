"""Isaac Sim motion-reference sensor: a thin ``SensorBase`` around the portable clip.

SDK imports stay inside :func:`build_sensor` so ``contract_report`` still answers
without Isaac.
"""

from __future__ import annotations

from typing import Any

from instinctlab.engines.motion_reference import (
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
    """Native Isaac ``SensorBaseCfg`` whose class loads the clip at initialize."""
    import torch
    from collections.abc import Sequence

    from isaaclab.sensors import SensorBase, SensorBaseCfg
    from isaaclab.utils import configclass

    model_path = robot.asset_for("isaacsim").path
    joints = tuple(ref.joints)
    links = tuple(ref.links)

    class MotionReferenceSensor(SensorBase):
        def _initialize_impl(self):
            super()._initialize_impl()
            raw = load_retargetted_clip(ref.clip, device=self.device)
            self._clip = pack_motion_clip(
                raw,
                joint_names=joints,
                link_names=links,
                model_path=model_path,
                velocity_method=ref.velocity_method,
                target_fps=ref.clip_target_fps,
                device=self.device,
            )
            self._data = make_buffers(
                self._num_envs,
                ref.num_frames,
                len(joints),
                len(links),
                device=self.device,
            )
            self._env_origins = torch.zeros(self._num_envs, 3, device=self.device)
            self.reset()

        @property
        def data(self):
            self._update_outdated_buffers()
            return self._data

        def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
            if not hasattr(self, "_data"):
                return
            if env_ids is None:
                env_ids = torch.arange(self._num_envs, device=self.device)
            else:
                env_ids = torch.as_tensor(env_ids, device=self.device)
            super().reset(env_ids)
            lo, hi = ref.start_range
            span = torch.rand(len(env_ids), device=self.device) * (hi - lo) + lo
            self._data.start_s[env_ids] = span * self._clip.duration_s
            self._data.timestamp[env_ids] = 0.0
            self._refresh(env_ids)
            # SensorBase.reset marked us outdated. We already wrote the buffers;
            # a second refresh on the next ``.data`` access would double-count
            # exhaustion without advancing the clip.
            self._is_outdated[env_ids] = False
            self._timestamp_last_update[env_ids] = self._timestamp[env_ids]

        def bind_origins(self, origins: torch.Tensor) -> None:
            self._env_origins = origins

        def _update_buffers_impl(self, env_ids):
            env_ids = torch.as_tensor(env_ids, device=self.device)
            self._data.timestamp[env_ids] = self._timestamp[env_ids]
            self._refresh(env_ids)

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

    @configclass
    class MotionReferenceSensorCfg(SensorBaseCfg):
        class_type: type = MotionReferenceSensor
        clip_path: str = ref.clip
        update_period: float = ref.update_period

    return MotionReferenceSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/torso_link")
