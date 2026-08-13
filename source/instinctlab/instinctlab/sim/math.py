"""Small WXYZ quaternion helpers used by engine-neutral MDP code."""

from __future__ import annotations

import torch


def quat_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    result = quaternion.clone()
    result[..., 1:] = -result[..., 1:]
    return result


def quat_multiply(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = first.unbind(dim=-1)
    bw, bx, by, bz = second.unbind(dim=-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def quat_apply(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    vector_quaternion = torch.cat((torch.zeros_like(vector[..., :1]), vector), dim=-1)
    return quat_multiply(quat_multiply(quaternion, vector_quaternion), quat_conjugate(quaternion))[..., 1:]


def quat_apply_inverse(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat_conjugate(quaternion), vector)


def yaw_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternion.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = 0.5 * yaw
    result = torch.zeros_like(quaternion)
    result[..., 0] = torch.cos(half)
    result[..., 3] = torch.sin(half)
    return result


def heading(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + torch.pi, 2.0 * torch.pi) - torch.pi


__all__ = [
    "heading",
    "quat_apply",
    "quat_apply_inverse",
    "quat_conjugate",
    "quat_multiply",
    "wrap_to_pi",
    "yaw_quaternion",
]
