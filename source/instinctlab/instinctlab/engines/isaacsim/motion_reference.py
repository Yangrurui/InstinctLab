"""Isaac Sim motion-reference sensor: a thin ``SensorBase`` around the portable clip.

SDK imports stay inside :func:`build_motion_reference_sensor` so ``contract_report`` still answers
without Isaac. The clip clock and the 50% mirror mask live in
:class:`~instinctlab.engines.motion_reference.MotionReferenceRuntime`.
"""

from __future__ import annotations

from typing import Any

from instinctlab.engines.motion_reference import MotionReferenceRuntime
from instinctlab.spec.sensor import MotionReferenceRef

__all__ = ["build_motion_reference_sensor"]


def build_motion_reference_sensor(ref: MotionReferenceRef, robot: Any) -> Any:
    """Native Isaac ``SensorBaseCfg`` whose class loads the clip at initialize."""
    import torch
    from collections.abc import Sequence

    from isaaclab.sensors import SensorBase, SensorBaseCfg
    from isaaclab.utils import configclass

    ref = ref.for_engine("isaacsim")
    model_path = robot.asset_for("isaacsim").path

    class MotionReferenceSensor(SensorBase):
        def _initialize_impl(self):
            super()._initialize_impl()
            self._runtime = MotionReferenceRuntime.create(ref, model_path, self._num_envs, self.device)
            self._data = self._runtime.buffers
            self.reset()

        @property
        def data(self):
            self._update_outdated_buffers()
            return self._data

        def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
            if not hasattr(self, "_runtime"):
                return
            if env_ids is None:
                env_ids = torch.arange(self._num_envs, device=self.device)
            else:
                env_ids = torch.as_tensor(env_ids, device=self.device)
            super().reset(env_ids)
            self._runtime.reset(env_ids)
            # SensorBase.reset marked us outdated. We already wrote the buffers;
            # a second refresh on the next ``.data`` access would double-count
            # exhaustion without advancing the clip.
            self._is_outdated[env_ids] = False
            self._timestamp_last_update[env_ids] = self._timestamp[env_ids]

        def bind_origins(self, origins: torch.Tensor) -> None:
            self._runtime.bind_origins(origins)

        @property
        def aiming_frame_idx(self):
            return self._runtime.aiming_frame_idx

        @property
        def ALL_INDICES(self):
            return torch.arange(self._num_envs, device=self.device)

        @property
        def reference_frame(self):
            return self.data

        @property
        def num_frames(self):
            return ref.num_frames

        @property
        def time_passed_from_update(self):
            return self._runtime.buffers.timestamp - self._runtime.last_update

        @property
        def time_to_aiming_frame(self):
            idx = self.aiming_frame_idx.clamp(min=0)
            return self.data.time_to_target_frame[self.ALL_INDICES, idx] - self.time_passed_from_update

        @property
        def init_reference_state(self):
            return self._runtime.init_buffers

        @property
        def joint_names(self):
            """Canonical joint order used by every tensor in the reference buffers."""
            return tuple(self._runtime.ref.joints)

        def _update_buffers_impl(self, env_ids):
            env_ids = torch.as_tensor(env_ids, device=self.device)
            self._runtime.buffers.timestamp[env_ids] = self._timestamp[env_ids]
            self._runtime.refresh_at_current_time(env_ids)

    @configclass
    class MotionReferenceSensorCfg(SensorBaseCfg):
        class_type: type = MotionReferenceSensor
        clip_path: str = ref.clip
        update_period: float = ref.update_period

    return MotionReferenceSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/torso_link")
