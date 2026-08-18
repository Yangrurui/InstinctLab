"""Reward terms that read quantities the two engines disagree about.

Everything portable lives in ``instinctlab.mdp``. What is left here is a term whose two
implementations differ in more than spelling, and writing one portable version would have meant
picking a side and calling it the meaning.

Ported from InstinctMJ.
"""

from __future__ import annotations

import torch
from typing import Any

__all__ = ["contact_slide"]


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
