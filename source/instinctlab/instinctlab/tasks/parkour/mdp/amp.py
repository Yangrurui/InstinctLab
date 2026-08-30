"""Parkour AMP observations: one function of kinematic state, two extractors.

A discriminator that can trivially tell policy samples from reference samples
drives the style reward to a constant. Training still converges. The usual
cause is that the two branches are not the same function of state: different
joint order, different frame, different units. Both branches therefore go
through :func:`amp_obs_from_robot_like`. The policy branch reads hub robot
fields. The reference branch first promotes clip quantities onto those same
fields (:func:`robot_like_from_clip`) and then calls the same builder.

Joints leave the clip already remapped **by name** onto the canonical
depth-first order. Defaults are indexed with the compiled ``joint_ids`` of
that same named list. There are no integer joint offsets in this file.

Frames, written here so a mix-up cannot hide behind a plausible pose:

* ``projected_gravity`` — gravity direction in the **root body** frame.
* ``joint_pos_rel`` / ``joint_vel`` — canonical DFS order, minus the robot's
  defaults (also gathered in that order).
* ``base_lin_vel`` / ``base_ang_vel`` — root-link velocity in the **body**
  frame. Clip stores world velocities; :func:`robot_like_from_clip` rotates
  them with ``quat_apply_inverse``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from instinctlab.compat.env import RlEnv
from instinctlab.compat.math import quat_apply, quat_apply_inverse
from instinctlab.motion_reference import clip_frame
from instinctlab.spec.sensor import MotionReferenceRef

AMP_TERM_ORDER = (
    "projected_gravity",
    "joint_pos_rel",
    "joint_vel",
    "base_lin_vel",
    "base_ang_vel",
)
"""Discriminator concatenation order. Insertion order of the observation group."""

GRAVITY_DOWN_W = (0.0, 0.0, -1.0)
"""World-down used when the live robot's gravity is not available.

Both engines of this task run −Z gravity. The reference terms prefer the
gravity reconstructed from the robot's ``projected_gravity_b`` so a write-state
check cannot hide a live-gravity vs hardcode mismatch.
"""


def amp_obs_from_robot_like(
    data: Any, joint_ids: Any = slice(None)
) -> dict[str, torch.Tensor]:
    """Policy-side AMP builder: hub robot fields, no conversion.

    Feeding a clip that has been promoted onto these fields
    (:func:`robot_like_from_clip`) must reproduce the reference branch.
    """
    return {
        "projected_gravity": data.projected_gravity_b,
        "joint_pos_rel": data.joint_pos[:, joint_ids]
        - data.default_joint_pos[:, joint_ids],
        "joint_vel": data.joint_vel[:, joint_ids]
        - data.default_joint_vel[:, joint_ids],
        "base_lin_vel": data.root_link_lin_vel_b,
        "base_ang_vel": data.root_link_ang_vel_b,
    }


def robot_like_from_clip(
    quat_w: torch.Tensor,
    lin_vel_w: torch.Tensor,
    ang_vel_w: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    default_joint_pos: torch.Tensor,
    default_joint_vel: torch.Tensor,
    gravity_w: torch.Tensor | None = None,
) -> SimpleNamespace:
    """Promote clip quantities onto the hub fields the policy builder reads.

    Clip joints must already be in the canonical order. This function does not
    reorder them — a positional fallback would look like a remap and train.
    """
    gravity = _batch_gravity(gravity_w, quat_w)
    return SimpleNamespace(
        projected_gravity_b=quat_apply_inverse(quat_w, gravity),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        default_joint_pos=default_joint_pos,
        default_joint_vel=default_joint_vel,
        root_link_lin_vel_b=quat_apply_inverse(quat_w, lin_vel_w),
        root_link_ang_vel_b=quat_apply_inverse(quat_w, ang_vel_w),
    )


def amp_obs_from_reference(
    buffers: Any,
    default_joint_pos: torch.Tensor,
    default_joint_vel: torch.Tensor,
    gravity_w: torch.Tensor | None = None,
    frame: int = 0,
    joint_ids: Any = slice(None),
) -> dict[str, torch.Tensor]:
    """Reference branch: clip frame → hub fields → the policy builder."""
    quat_w, lin_vel_w, ang_vel_w, joint_pos, joint_vel = clip_frame(buffers, frame)
    like = robot_like_from_clip(
        quat_w,
        lin_vel_w,
        ang_vel_w,
        joint_pos,
        joint_vel,
        default_joint_pos,
        default_joint_vel,
        gravity_w,
    )
    return amp_obs_from_robot_like(like, joint_ids)


def projected_gravity_from_reference(
    env: RlEnv,
    sensor: MotionReferenceRef,
    asset_cfg: Any,
) -> torch.Tensor:
    return _reference_obs(env, sensor, asset_cfg)["projected_gravity"]


def joint_pos_rel_from_reference(
    env: RlEnv,
    sensor: MotionReferenceRef,
    asset_cfg: Any,
) -> torch.Tensor:
    return _reference_obs(env, sensor, asset_cfg)["joint_pos_rel"]


def joint_vel_rel_from_reference(
    env: RlEnv,
    sensor: MotionReferenceRef,
    asset_cfg: Any,
) -> torch.Tensor:
    return _reference_obs(env, sensor, asset_cfg)["joint_vel"]


def base_lin_vel_from_reference(
    env: RlEnv,
    sensor: MotionReferenceRef,
    asset_cfg: Any,
) -> torch.Tensor:
    return _reference_obs(env, sensor, asset_cfg)["base_lin_vel"]


def base_ang_vel_from_reference(
    env: RlEnv,
    sensor: MotionReferenceRef,
    asset_cfg: Any,
) -> torch.Tensor:
    return _reference_obs(env, sensor, asset_cfg)["base_ang_vel"]


def _reference_obs(
    env: RlEnv, sensor: MotionReferenceRef, asset_cfg: Any
) -> dict[str, torch.Tensor]:
    if asset_cfg is None:
        raise ValueError(
            "AMP reference terms need asset_cfg naming the joints in canonical order. "
            "slice(None) on a native-order default_joint_pos is the silent BFS/DFS mix-up."
        )
    robot = env.scene[_name(asset_cfg)]
    joint_ids = _joint_ids(asset_cfg)
    gravity_w = quat_apply(robot.data.root_link_quat_w, robot.data.projected_gravity_b)
    return amp_obs_from_reference(
        _motion_reference(env, sensor).reference_frame,
        robot.data.default_joint_pos[:, joint_ids],
        robot.data.default_joint_vel[:, joint_ids],
        gravity_w,
    )


def _motion_reference(env: RlEnv, sensor: MotionReferenceRef) -> Any:
    """Resolve the shared sensor contract identically on both engines."""
    return env.scene.sensors[sensor.name]


def _name(asset_cfg: Any) -> str:
    return "robot" if asset_cfg is None else asset_cfg.name


def _joint_ids(asset_cfg: Any) -> Any:
    return slice(None) if asset_cfg is None else asset_cfg.joint_ids


def _batch_gravity(
    gravity_w: torch.Tensor | None, quat_w: torch.Tensor
) -> torch.Tensor:
    n = quat_w.shape[0]
    if gravity_w is None:
        gravity = torch.zeros(n, 3, device=quat_w.device, dtype=quat_w.dtype)
        gravity[:, 2] = -1.0
        return gravity
    gravity = torch.as_tensor(gravity_w, device=quat_w.device, dtype=quat_w.dtype)
    if gravity.ndim == 1:
        gravity = gravity.unsqueeze(0).expand(n, -1)
    return gravity
