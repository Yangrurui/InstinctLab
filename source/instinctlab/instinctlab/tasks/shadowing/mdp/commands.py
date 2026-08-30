"""Shadowing command algorithms driven by a motion-reference sensor."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import torch

from instinctlab.compat import math as math_utils
from instinctlab.utils.math import quat_to_tan_norm
from instinctlab.utils.name_order import resolve_name_indices


def _root(data: Any, name: str) -> torch.Tensor:
    """Resolve Isaac's root names and MJLab's explicit root-link names at one boundary."""
    mj_name = name.replace("root_", "root_link_")
    for candidate in (mj_name, name):
        if hasattr(data, candidate):
            return getattr(data, candidate)
    raise AttributeError(f"robot data has neither {mj_name!r} nor {name!r}")


class MotionReferenceCommand:
    """Common lifecycle for task-owned motion-reference commands."""

    def __init__(self, env: Any, params: dict[str, Any]) -> None:
        self.cfg = SimpleNamespace(
            motion_reference=params["motion_reference"],
            entity_name=params["entity"],
            current_state_command=params.get("current_state_command", False),
            realtime_mode=params.get("realtime_mode", False),
            anchor_frame=params.get("anchor_frame", "robot"),
            in_base_frame=params.get("in_base_frame", True),
            rotation_mode=params.get("rotation_mode", "axis_angle"),
        )
        self._env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.metrics: dict[str, torch.Tensor] = {}
        self._motion = env.scene[self.cfg.motion_reference]

    @property
    def command(self):
        return self._command

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids):
        self._update_command_by_env_ids(env_ids)

    def _update_command(self):
        if self.cfg.realtime_mode:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.where(
                self._motion.time_passed_from_update < (self._env.step_dt - 1.0e-6)
            )[0]
        if env_ids.numel():
            self._update_command_by_env_ids(env_ids)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._update_command_by_env_ids(env_ids)
        return {}


class PositionReference(MotionReferenceCommand):
    def __init__(self, env: Any, params: dict[str, Any]) -> None:
        super().__init__(env, params)
        self._command = torch.zeros(
            self.num_envs, self._motion.num_frames, 3, device=self.device
        )
        self._update_command_by_env_ids(
            torch.arange(self.num_envs, device=self.device)
        )

    def _update_command_by_env_ids(self, env_ids):
        reference = self._motion.data
        if self.cfg.anchor_frame == "reference":
            current = self._motion.reference_frame
            anchor_pos = current.base_pos_w[env_ids, 0]
            anchor_quat = current.base_quat_w[env_ids, 0]
        else:
            robot = self._env.scene[self.cfg.entity_name].data
            anchor_pos = _root(robot, "root_pos_w")[env_ids]
            anchor_quat = _root(robot, "root_quat_w")[env_ids]
        inv_pos, inv_quat = math_utils.subtract_frame_transforms(
            anchor_pos, anchor_quat
        )
        self._command[env_ids] = math_utils.transform_points(
            reference.base_pos_w[env_ids], inv_pos, inv_quat
        ) * reference.validity[env_ids].unsqueeze(-1)


class RotationReference(MotionReferenceCommand):
    def __init__(self, env: Any, params: dict[str, Any]) -> None:
        super().__init__(env, params)
        dims = (
            6
            if self.cfg.rotation_mode == "tannorm"
            else (4 if self.cfg.rotation_mode == "quaternion" else 3)
        )
        self._command = torch.zeros(
            self.num_envs, self._motion.num_frames, dims, device=self.device
        )
        self._update_command_by_env_ids(
            torch.arange(self.num_envs, device=self.device)
        )

    def _update_command_by_env_ids(self, env_ids):
        reference = self._motion.data
        quat = reference.base_quat_w[env_ids]
        if self.cfg.in_base_frame:
            robot_quat = _root(
                self._env.scene[self.cfg.entity_name].data, "root_quat_w"
            )[env_ids]
            quat = math_utils.quat_mul(
                math_utils.quat_inv(robot_quat).unsqueeze(1).expand_as(quat), quat
            )
        if self.cfg.rotation_mode == "tannorm":
            value = quat_to_tan_norm(quat)
        elif self.cfg.rotation_mode == "quaternion":
            value = quat
        elif self.cfg.rotation_mode == "axis_angle":
            value = math_utils.axis_angle_from_quat(quat)
        else:
            flat = quat.flatten(0, 1)
            value = torch.stack(math_utils.euler_xyz_from_quat(flat), dim=-1).view(
                *quat.shape[:-1], 3
            )
        self._command[env_ids] = value * reference.validity[env_ids].unsqueeze(-1)


class JointPositionReference(MotionReferenceCommand):
    def __init__(self, env: Any, params: dict[str, Any]) -> None:
        super().__init__(env, params)
        reference = self._motion.data
        self._command = torch.zeros_like(reference.joint_pos)
        asset = self._env.scene[self.cfg.entity_name]
        joint_ids = torch.tensor(
            resolve_name_indices(
                asset.joint_names,
                self._motion.joint_names,
                require_exact=True,
            ),
            dtype=torch.long,
            device=self.device,
        )
        self._default = asset.data.default_joint_pos.index_select(1, joint_ids).clone()
        self._update_command_by_env_ids(
            torch.arange(self.num_envs, device=self.device)
        )

    def _update_command_by_env_ids(self, env_ids):
        reference = self._motion.data
        self._command[env_ids] = (
            reference.joint_pos[env_ids] - self._default[env_ids].unsqueeze(1)
        ) * reference.validity[env_ids].unsqueeze(-1)


class JointVelocityReference(MotionReferenceCommand):
    def __init__(self, env: Any, params: dict[str, Any]) -> None:
        super().__init__(env, params)
        reference = self._motion.data
        self._command = torch.zeros_like(reference.joint_vel)
        asset = self._env.scene[self.cfg.entity_name]
        joint_ids = torch.tensor(
            resolve_name_indices(
                asset.joint_names,
                self._motion.joint_names,
                require_exact=True,
            ),
            dtype=torch.long,
            device=self.device,
        )
        self._default = asset.data.default_joint_vel.index_select(1, joint_ids).clone()
        self._update_command_by_env_ids(
            torch.arange(self.num_envs, device=self.device)
        )

    def _update_command_by_env_ids(self, env_ids):
        reference = self._motion.data
        self._command[env_ids] = (
            reference.joint_vel[env_ids] - self._default[env_ids].unsqueeze(1)
        ) * reference.validity[env_ids].unsqueeze(-1)
