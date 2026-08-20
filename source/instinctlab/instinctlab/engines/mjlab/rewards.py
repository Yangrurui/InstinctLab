"""Reward terms that read quantities the two engines disagree about.

Everything portable lives in ``instinctlab.mdp``. What is left here is a term whose two
implementations differ in more than spelling, and writing one portable version would have meant
picking a side and calling it the meaning.

Ported from InstinctMJ.
"""

from __future__ import annotations

import torch
from typing import Any

__all__ = [
    "CONTACT_FORCE_THRESHOLD_N",
    "applied_torque_limits_by_ratio",
    "contact_slide",
    "illegal_contact",
    "joint_torques_l2",
    "motors_power_square",
    "undesired_contacts",
]

# Full 3-D contact force after ``reduce="netforce"`` (includes friction). InstinctMJ
# parkour uses the same 1 N on this quantity. That is *not* Isaac's 1 N on
# world-frame normal load -- see engines/isaacsim/terms.py and the measured
# trigger rates recorded on the builders.
CONTACT_FORCE_THRESHOLD_N = 1.0


def _force_exceeds(env: Any, sensor: Any, threshold: float) -> torch.Tensor:
    """``(env, element)``: max-over-history ‖force‖ exceeds ``threshold``.

    Uses the hub axis order from :func:`instinctlab.compat.sensors.contact_force_history`
    so the time/element swap cannot silently invert a two-foot two-substep history.
    """
    from instinctlab.compat import sensors as sensor_compat

    history = sensor_compat.contact_force_history(env.scene.sensors[sensor.name], sensor)
    return torch.max(torch.norm(history, dim=-1), dim=1)[0] > threshold


def illegal_contact(env: Any, sensor: Any, threshold: float = CONTACT_FORCE_THRESHOLD_N) -> torch.Tensor:
    """Terminate when any referenced element's full contact force exceeds 1 N.

    InstinctMJ's parkour ``illegal_contact``. The 1 N is on ‖force‖ (friction
    included), not on Isaac's normal-only ``net_forces_w``.
    """
    return torch.any(_force_exceeds(env, sensor, threshold), dim=1)


def undesired_contacts(env: Any, sensor: Any, threshold: float = CONTACT_FORCE_THRESHOLD_N) -> torch.Tensor:
    """Count referenced elements whose full contact force exceeds 1 N.

    InstinctMJ's parkour ``undesired_contacts``. Same quantity caveat as
    :func:`illegal_contact`.
    """
    return torch.sum(_force_exceeds(env, sensor, threshold).float(), dim=1)


def contact_slide(
    env: Any,
    sensor_cfg: Any,
    asset_cfg: Any = None,
    ang_vel_penalty: bool = False,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Penalise horizontal motion of bodies that are touching something.

    Two readings here differ from the Isaac Lab version, and neither is a naming difference:

    The force history is indexed ``[env, element, step]`` where Isaac Lab has ``[env, step, body]``,
    which ``compat.sensors`` exists to hide -- but the force itself is a full three-dimensional
    contact force here and the normal component alone there, so the same threshold does not select
    the same contacts. This term keeps each engine's own quantity, which is why it is registered per
    engine instead of made portable.

    The velocity is the link's, not the centre of mass's. Isaac Lab's ``body_lin_vel_w`` is the
    centre-of-mass velocity despite the name; mjlab spells the distinction out. For a foot flat on
    the ground the two differ by the offset between the two frames crossed with the angular
    velocity, which is exactly the quantity a slide penalty is looking at.
    """
    from mjlab.managers import SceneEntityCfg

    from instinctlab.compat import sensors as sensor_compat

    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]

    forces = sensor.data.force_history[:, sensor_compat.element_ids(sensor, sensor_cfg)]
    in_contact = torch.max(torch.norm(forces, dim=-1), dim=-1)[0] > threshold

    body_vel = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * in_contact, dim=1)
    if ang_vel_penalty:
        body_ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :2]
        reward += torch.sum(body_ang_vel.norm(dim=-1) * in_contact, dim=1)
    return reward


def _unwrap_base_actuator(actuator):
    base = actuator
    while hasattr(base, "base_actuator"):
        base = base.base_actuator
    return base


def _iter_joint_stiffness(asset):
    for actuator in asset.actuators:
        if getattr(actuator, "transmission_type", "joint") != "joint":
            continue
        base = _unwrap_base_actuator(actuator)
        yield base.target_ids, base.cfg.stiffness


def joint_torques_l2(
    env: Any,
    asset_cfg: Any = None,
) -> torch.Tensor:
    """Penalise joint-space actuator force, sliced by ``joint_ids``.

    mjlab's stock term reads ``actuator_force`` (nu) and slices ``actuator_ids``. Those stay
    ``slice(None)`` when the task only named joints, so a hip/knee penalty ran on every
    actuator. ``qfrc_actuator`` is the nv quantity Isaac's ``applied_torque`` means; the two
    are not bitwise equal (Isaac excludes some passive terms, MuJoCo's ``qfrc_actuator`` can
    include gravity compensation).
    """
    if asset_cfg is None:
        from mjlab.managers import SceneEntityCfg

        asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.qfrc_actuator[:, asset_cfg.joint_ids]), dim=1)


def motors_power_square(
    env: Any,
    asset_cfg: Any = None,
    normalize_by_stiffness: bool = True,
    normalize_by_num_joints: bool = False,
) -> torch.Tensor:
    """Parkour energy: ``(qfrc_actuator * joint_vel)²``, optionally divided by stiffness.

    Ported from InstinctMJ. Not bitwise equal to Isaac's ``applied_torque * joint_vel`` --
    same denylist pair as :func:`joint_torques_l2`.
    """
    if asset_cfg is None:
        from mjlab.managers import SceneEntityCfg

        asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]
    power_j = asset.data.qfrc_actuator * asset.data.joint_vel
    if normalize_by_stiffness:
        for target_ids, stiffness in _iter_joint_stiffness(asset):
            power_j[:, target_ids] /= torch.as_tensor(stiffness, device=power_j.device, dtype=power_j.dtype)
    power_j = power_j[:, asset_cfg.joint_ids]
    power = torch.sum(torch.square(power_j), dim=-1)
    if normalize_by_num_joints:
        power = power / power_j.shape[-1]
    return power


def _effort_limits(env: Any, asset: Any, asset_cfg: Any) -> torch.Tensor:
    """Per-joint effort cap, preferring a data attribute and otherwise the model ranges.

    Isaac exposes ``joint_effort_limits``. mjlab does not; InstinctMJ reads
    ``jnt_actfrcrange`` / ``actuator_forcerange``. A stub that sets ``joint_effort_limits``
    is enough for tests that must not import mjlab's model.
    """
    limits = getattr(asset.data, "joint_effort_limits", None)
    if limits is not None:
        return limits[:, asset_cfg.joint_ids]

    from mjlab.actuator import BuiltinPdActuator

    applied = asset.data.qfrc_actuator
    joint_effort_limits = torch.zeros_like(applied)
    if isinstance(asset_cfg.joint_ids, slice):
        selected = list(range(asset.num_joints))
    else:
        selected = list(asset_cfg.joint_ids)
    selected_names = {asset.joint_names[j] for j in selected}

    actuator_forcerange = env.sim.model.actuator_forcerange
    if actuator_forcerange.ndim == 3:
        actuator_forcerange = actuator_forcerange[0]

    for actuator in asset.actuators:
        if getattr(actuator, "transmission_type", "joint") != "joint":
            continue
        for idx, joint_name in enumerate(list(actuator.target_names)):
            if joint_name not in selected_names:
                continue
            joint_id = int(actuator.target_ids[idx])
            if isinstance(_unwrap_base_actuator(actuator), BuiltinPdActuator):
                joint_id_global = int(asset.indexing.joint_ids[joint_id])
                joint_actfrcrange = env.sim.model.jnt_actfrcrange
                if joint_actfrcrange.ndim == 3:
                    effort_limit = torch.max(torch.abs(joint_actfrcrange[:, joint_id_global]), dim=-1).values
                else:
                    effort_limit = torch.max(torch.abs(joint_actfrcrange[joint_id_global]))
            else:
                ctrl_id_global = int(actuator.global_ctrl_ids[idx])
                effort_limit = torch.max(torch.abs(actuator_forcerange[ctrl_id_global]))
            joint_effort_limits[:, joint_id] = torch.maximum(joint_effort_limits[:, joint_id], effort_limit)
    return joint_effort_limits[:, asset_cfg.joint_ids]


def applied_torque_limits_by_ratio(
    env: Any,
    asset_cfg: Any = None,
    limit_ratio: float = 0.8,
) -> torch.Tensor:
    """Penalise ``qfrc_actuator`` above ``limit_ratio`` of each joint's effort cap.

    Limit sources differ across engines (Isaac ``joint_effort_limits`` vs MuJoCo forcerange),
    so the two implementations are not expected to be bitwise equal.
    """
    if asset_cfg is None:
        from mjlab.managers import SceneEntityCfg

        asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]
    applied = torch.abs(asset.data.qfrc_actuator[:, asset_cfg.joint_ids])
    limits = _effort_limits(env, asset, asset_cfg)
    return torch.sum(torch.square((applied - limits * limit_ratio).clamp(min=0.0)), dim=-1)
