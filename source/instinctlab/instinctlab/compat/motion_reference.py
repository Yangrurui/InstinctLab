"""Engine-neutral reads from the motion-reference sensor contract.

Task MDP terms use these helpers instead of reaching into either engine's
motion-reference runtime.  Both native sensors expose the same buffer schema;
the adapters remain responsible for constructing and updating those buffers.
"""

from __future__ import annotations

from typing import Any

import torch


def clip_frame(
    buffers: Any, frame: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return root orientation, world velocities, and joints for one frame."""
    return (
        buffers.base_quat_w[:, frame],
        buffers.base_lin_vel_w[:, frame],
        buffers.base_ang_vel_w[:, frame],
        buffers.joint_pos[:, frame],
        buffers.joint_vel[:, frame],
    )


def exhausted_envs(buffers: Any, aiming_frame_idx: torch.Tensor) -> torch.Tensor:
    """Return the per-environment exhaustion mask at the current target slot."""
    num_envs = buffers.validity.shape[0]
    if aiming_frame_idx.shape != (num_envs,):
        raise ValueError(
            f"aiming_frame_idx must have shape ({num_envs},), "
            f"got {tuple(aiming_frame_idx.shape)}."
        )
    env_ids = torch.arange(num_envs, device=buffers.validity.device)
    return ~buffers.validity[env_ids, aiming_frame_idx]
