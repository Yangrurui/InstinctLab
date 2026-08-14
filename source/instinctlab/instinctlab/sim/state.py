"""Canonical simulator state containers.

All tensors exposed to tasks use WXYZ quaternions, DFS entity ordering, link
frames, and a leading environment dimension.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass


def _check_shape(name: str, value: torch.Tensor, expected: tuple[int, ...]) -> None:
    if tuple(value.shape) != expected:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {expected}")


@dataclass
class ArticulationState:
    """Mutable, backend-owned canonical articulation state."""

    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_w: torch.Tensor
    body_pos_w: torch.Tensor
    body_quat_w: torch.Tensor
    body_lin_vel_w: torch.Tensor
    body_ang_vel_w: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    joint_acc: torch.Tensor
    applied_joint_effort: torch.Tensor
    default_joint_pos: torch.Tensor
    soft_joint_pos_limits: torch.Tensor
    joint_velocity_limits: torch.Tensor
    joint_effort_limits: torch.Tensor

    @classmethod
    def allocate(
        cls,
        *,
        num_envs: int,
        num_joints: int,
        num_bodies: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> ArticulationState:
        device = torch.device(device)

        def zeros(*shape: int) -> torch.Tensor:
            return torch.zeros(shape, device=device, dtype=dtype)

        root_quat = zeros(num_envs, 4)
        root_quat[:, 0] = 1.0
        body_quat = zeros(num_envs, num_bodies, 4)
        body_quat[..., 0] = 1.0
        return cls(
            root_pos_w=zeros(num_envs, 3),
            root_quat_w=root_quat,
            root_lin_vel_w=zeros(num_envs, 3),
            root_ang_vel_w=zeros(num_envs, 3),
            body_pos_w=zeros(num_envs, num_bodies, 3),
            body_quat_w=body_quat,
            body_lin_vel_w=zeros(num_envs, num_bodies, 3),
            body_ang_vel_w=zeros(num_envs, num_bodies, 3),
            joint_pos=zeros(num_envs, num_joints),
            joint_vel=zeros(num_envs, num_joints),
            joint_acc=zeros(num_envs, num_joints),
            applied_joint_effort=zeros(num_envs, num_joints),
            default_joint_pos=zeros(num_envs, num_joints),
            soft_joint_pos_limits=zeros(num_envs, num_joints, 2),
            joint_velocity_limits=zeros(num_envs, num_joints),
            joint_effort_limits=zeros(num_envs, num_joints),
        )

    @property
    def num_envs(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def num_joints(self) -> int:
        return int(self.joint_pos.shape[1])

    @property
    def num_bodies(self) -> int:
        return int(self.body_pos_w.shape[1])

    @property
    def device(self) -> torch.device:
        return self.joint_pos.device

    def validate(self) -> None:
        n, j, b = self.num_envs, self.num_joints, self.num_bodies
        for name in ("root_pos_w", "root_lin_vel_w", "root_ang_vel_w"):
            _check_shape(name, getattr(self, name), (n, 3))
        _check_shape("root_quat_w", self.root_quat_w, (n, 4))
        for name in ("body_pos_w", "body_lin_vel_w", "body_ang_vel_w"):
            _check_shape(name, getattr(self, name), (n, b, 3))
        _check_shape("body_quat_w", self.body_quat_w, (n, b, 4))
        for name in (
            "joint_pos",
            "joint_vel",
            "joint_acc",
            "applied_joint_effort",
            "default_joint_pos",
            "joint_velocity_limits",
            "joint_effort_limits",
        ):
            _check_shape(name, getattr(self, name), (n, j))
        _check_shape("soft_joint_pos_limits", self.soft_joint_pos_limits, (n, j, 2))
        tensors = vars(self).values()
        if any(tensor.device != self.device for tensor in tensors):
            raise ValueError("all articulation state tensors must be on the same device")

    def update_joint_acceleration(
        self,
        previous_joint_velocity: torch.Tensor,
        *,
        sim_dt: float,
        env_ids: torch.Tensor | slice = slice(None),
    ) -> None:
        """Apply the canonical ``fd_v1`` acceleration definition."""
        self.joint_acc[env_ids] = (self.joint_vel[env_ids] - previous_joint_velocity[env_ids]) / sim_dt


@dataclass
class ContactState:
    """Canonical contact state; history index zero is the newest sample."""

    body_names: tuple[str, ...]
    net_forces_w: torch.Tensor
    net_forces_w_history: torch.Tensor
    contact_active: torch.Tensor
    contact_active_history: torch.Tensor
    current_air_time: torch.Tensor
    current_contact_time: torch.Tensor
    last_air_time: torch.Tensor
    last_contact_time: torch.Tensor

    @classmethod
    def allocate(
        cls,
        *,
        num_envs: int,
        body_names: tuple[str, ...],
        history_length: int,
        device: torch.device | str,
    ) -> ContactState:
        b = len(body_names)
        device = torch.device(device)
        return cls(
            body_names=body_names,
            net_forces_w=torch.zeros((num_envs, b, 3), device=device),
            net_forces_w_history=torch.zeros((num_envs, history_length, b, 3), device=device),
            contact_active=torch.zeros((num_envs, b), device=device, dtype=torch.bool),
            contact_active_history=torch.zeros((num_envs, history_length, b), device=device, dtype=torch.bool),
            current_air_time=torch.zeros((num_envs, b), device=device),
            current_contact_time=torch.zeros((num_envs, b), device=device),
            last_air_time=torch.zeros((num_envs, b), device=device),
            last_contact_time=torch.zeros((num_envs, b), device=device),
        )

    @property
    def history_length(self) -> int:
        return int(self.net_forces_w_history.shape[1])

    def update_active(self, force_threshold: float) -> None:
        self.contact_active.copy_(torch.linalg.vector_norm(self.net_forces_w, dim=-1) > force_threshold)
        if self.history_length:
            self.contact_active_history.copy_(
                torch.linalg.vector_norm(self.net_forces_w_history, dim=-1) > force_threshold
            )

    def update_air_time(self, is_contact: torch.Tensor, dt: float) -> None:
        """Advance air/contact timers with force-threshold contact semantics.

        ``is_contact`` must already apply the same threshold used by
        :meth:`update_active`. Call once per physics substep; a policy-step
        ``dt`` would miss first-contact edges inside decimation.
        """
        expected = tuple(self.contact_active.shape)
        if tuple(is_contact.shape) != expected:
            raise ValueError(f"is_contact has shape {tuple(is_contact.shape)}, expected {expected}")
        if is_contact.dtype != torch.bool:
            raise ValueError("is_contact must be a boolean tensor")
        if dt <= 0.0:
            raise ValueError("air-time dt must be positive")

        is_first_contact = (self.current_air_time > 0.0) & is_contact
        is_first_detached = (self.current_contact_time > 0.0) & ~is_contact
        self.last_air_time.copy_(torch.where(is_first_contact, self.current_air_time + dt, self.last_air_time))
        self.current_air_time.copy_(
            torch.where(~is_contact, self.current_air_time + dt, torch.zeros_like(self.current_air_time))
        )
        self.last_contact_time.copy_(
            torch.where(is_first_detached, self.current_contact_time + dt, self.last_contact_time)
        )
        self.current_contact_time.copy_(
            torch.where(is_contact, self.current_contact_time + dt, torch.zeros_like(self.current_contact_time))
        )

    def reset(self, env_ids: torch.Tensor | slice) -> None:
        for value in vars(self).values():
            if isinstance(value, torch.Tensor):
                value[env_ids] = 0

    def validate(self) -> None:
        n, b = self.net_forces_w.shape[:2]
        h = self.history_length
        _check_shape("net_forces_w", self.net_forces_w, (n, b, 3))
        _check_shape("net_forces_w_history", self.net_forces_w_history, (n, h, b, 3))
        _check_shape("contact_active", self.contact_active, (n, b))
        _check_shape("contact_active_history", self.contact_active_history, (n, h, b))
        for name in ("current_air_time", "current_contact_time", "last_air_time", "last_contact_time"):
            _check_shape(name, getattr(self, name), (n, b))
        if len(self.body_names) != b or len(set(self.body_names)) != b:
            raise ValueError("contact body_names must be unique and match the body dimension")


__all__ = ["ArticulationState", "ContactState"]
