"""Which (pattern, alignment) pairs the engines actually apply.

A value the engine ignores is a silent-failure surface: the declaration is
accepted, the depth image looks plausible, and parkour keeps training. Both
camera paths ignore ``ray_alignment`` and always use the attach body's full
rotation, so a pinhole declared as ``yaw`` would compile and be wrong. Both
grid paths honour ``yaw`` and ``base``. Refuse only the combinations that
would be ignored.
"""

from __future__ import annotations

from instinctlab.spec.sensor import RayCasterRef

__all__ = ["camera_pose_for_alignment", "refuse_unhonored_ray_alignment"]


def camera_pose_for_alignment(torso_pos, torso_quat, offset, offset_rot, alignment: str):
    """Camera pose from the attach body. ``base`` uses the full R; ``yaw`` does not."""
    from instinctlab.compat.math import quat_apply, quat_mul, yaw_quat

    rot = yaw_quat(torso_quat) if alignment == "yaw" else torso_quat
    shift = offset.to(dtype=torso_pos.dtype, device=torso_pos.device)
    if shift.shape != torso_pos.shape:
        shift = shift.reshape(1, -1).expand_as(torso_pos)
    qoff = offset_rot.to(dtype=torso_quat.dtype, device=torso_quat.device)
    if qoff.shape != torso_quat.shape:
        qoff = qoff.reshape(1, -1).expand_as(torso_quat)
    return torso_pos + quat_apply(rot, shift), quat_mul(rot, qoff)


def refuse_unhonored_ray_alignment(sensor: RayCasterRef) -> None:
    """Raise if this pattern cannot honour ``sensor.ray_alignment``.

    Called from both engine builders before they lower the reference, so a
    future ``pinhole`` + ``yaw`` cannot be accepted and silently ignored.
    """
    if sensor.pattern.kind == "pinhole" and sensor.ray_alignment != "base":
        raise ValueError(
            f"Ray caster {sensor.name!r} is a pinhole with "
            f"ray_alignment={sensor.ray_alignment!r}. Both engines ignore that "
            "field on a camera and always use the attach body's full rotation; "
            f"{sensor.ray_alignment!r} would be accepted and silently ignored. "
            "Declare ray_alignment='base'."
        )
