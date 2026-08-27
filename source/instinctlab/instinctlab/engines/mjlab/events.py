"""Event functions mjlab does not ship.

Two of them, and both exist for the same reason: mjlab has a neighbouring primitive that would have
compiled without complaint and randomised something else.

``reset_joints_by_offset`` adds a sampled offset to the default joint position where Isaac Lab's
``reset_joints_by_scale`` multiplies it. mjlab now ships a native offset helper, but it indexes
``soft_joint_pos_limits`` by ``env_ids`` and that tensor's leading dim is 1 -- a model constant --
so the second environment is an out-of-range read. The port below uses :func:`_rows`. ``dr.body_mass``
changes a body's mass and leaves its inertia tensor untouched, where Isaac Lab's
``randomize_rigid_body_mass`` defaults to rescaling the inertia by the same ratio, so a heavier
torso is also harder to rotate. Substituting either neighbour would leave two runs looking
comparable while training against different physics.

Both are ported from InstinctMJ, the mjlab-side reference implementation for this task.

Unlike its siblings this module imports mjlab when loaded, because a decorator has to run at
definition time. It stays lazily imported from the builders that use it, so the package as a whole
still costs nothing to import without mjlab installed.
"""

from __future__ import annotations

import math
import torch
from typing import Any

from mjlab.envs.mdp import dr as _dr
from mjlab.managers.event_manager import requires_model_fields

__all__ = [
    "randomize_body_mass",
    "reset_joints_by_offset",
    "reset_joints_by_scale",
    "uniform_mass_scale_distribution",
]


def _rows(data: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
    """Rows for ``env_ids``, whether or not the tensor has a row per environment.

    Not defensiveness for its own sake. mjlab stores joint limits once for the model, since they
    are the same in every environment, while Isaac Lab stores a copy per environment; a function
    ported between the two indexes a length-one axis by environment id and fails on the second
    environment. Terms that only ever broadcast against these tensors never notice, which is why
    this surfaces in a reset event rather than in a reward.
    """
    if data.shape[0] == 1:
        return data.expand(len(env_ids), *data.shape[1:])
    return data[env_ids]


def reset_joints_by_offset(
    env: Any,
    env_ids: torch.Tensor | None,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: Any = None,
) -> None:
    """Reset joints to their default state plus a uniform offset, clamped to the soft limits.

    Additive, not a scale. mjlab's ``reset_joints_by_scale`` multiplies; substituting it here
    is how a ±0.15 rad parkour reset becomes a 0.85–1.15 scale and still compiles.
    """
    from mjlab.managers import SceneEntityCfg

    from instinctlab.compat.math import sample_uniform

    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    asset = env.scene[asset_cfg.name]
    joint_pos = _rows(asset.data.default_joint_pos, env_ids)[:, asset_cfg.joint_ids].clone()
    joint_pos += sample_uniform(*position_range, joint_pos.shape, env.device)
    limits = _rows(asset.data.soft_joint_pos_limits, env_ids)[:, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(limits[..., 0], limits[..., 1])

    joint_vel = _rows(asset.data.default_joint_vel, env_ids)[:, asset_cfg.joint_ids].clone()
    joint_vel += sample_uniform(*velocity_range, joint_vel.shape, env.device)

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids = torch.tensor(joint_ids, device=env.device)
    asset.write_joint_state_to_sim(
        joint_pos.view(len(env_ids), -1), joint_vel.view(len(env_ids), -1), env_ids=env_ids, joint_ids=joint_ids
    )


def reset_joints_by_scale(
    env: Any,
    env_ids: torch.Tensor | None,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: Any = None,
) -> None:
    """Reset joints to their default state scaled by a uniform sample, clamped to the soft limits."""
    from mjlab.managers import SceneEntityCfg

    from instinctlab.compat.math import sample_uniform

    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    asset = env.scene[asset_cfg.name]
    joint_pos = _rows(asset.data.default_joint_pos, env_ids)[:, asset_cfg.joint_ids].clone()
    joint_pos *= sample_uniform(*position_range, joint_pos.shape, env.device)
    limits = _rows(asset.data.soft_joint_pos_limits, env_ids)[:, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(limits[..., 0], limits[..., 1])

    joint_vel = _rows(asset.data.default_joint_vel, env_ids)[:, asset_cfg.joint_ids].clone()
    joint_vel *= sample_uniform(*velocity_range, joint_vel.shape, env.device)

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids = torch.tensor(joint_ids, device=env.device)
    asset.write_joint_state_to_sim(
        joint_pos.view(len(env_ids), -1), joint_vel.view(len(env_ids), -1), env_ids=env_ids, joint_ids=joint_ids
    )


def uniform_mass_scale_distribution() -> Any:
    """A distribution uniform in the mass ratio rather than in its logarithm.

    ``dr.pseudo_inertia`` is parameterised by ``alpha``, where the mass ratio is ``exp(2*alpha)``,
    and sampling alpha uniformly would make the mass log-uniform. Isaac Lab samples the added
    kilograms uniformly, so the ratio has to be uniform to match: sample there, then map back.
    """
    from mjlab.envs.mdp import dr

    from instinctlab.compat.math import sample_uniform

    return dr.Distribution(
        name="uniform_mass_scale_to_alpha",
        sample=lambda lo, hi, shape, device: 0.5
        * torch.log(sample_uniform(torch.exp(2.0 * lo), torch.exp(2.0 * hi), shape, device=device)),
    )


@requires_model_fields(*_dr.pseudo_inertia.model_fields, recompute=_dr.pseudo_inertia.recompute)
def randomize_body_mass(
    env: Any,
    env_ids: torch.Tensor | None,
    add_range: tuple[float, float],
    asset_cfg: Any = None,
) -> None:
    """Add a uniformly sampled mass to the selected bodies, rescaling inertia to match.

    Expressed as a ratio because that is the only handle mjlab offers on mass and inertia together,
    which means each body's own default mass is needed to turn kilograms into a ratio. Bodies are
    therefore randomised one at a time.

    The declaration is forwarded from ``dr.pseudo_inertia`` rather than restated. mjlab reads which
    model fields an event will write from the event function itself, in order to give each
    environment its own copy of them beforehand; a wrapper that does not pass the declaration on
    leaves the fields shared, and the write fails on the second environment.
    """
    from mjlab.envs.mdp import dr
    from mjlab.managers import SceneEntityCfg

    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    env_ids = env_ids.to(env.device, dtype=torch.int)

    local_ids = torch.arange(asset.indexing.body_ids.numel(), device=env.device, dtype=torch.long)
    local_ids = local_ids[asset_cfg.body_ids].reshape(-1)
    distribution = uniform_mass_scale_distribution()

    for local_id, model_id in zip(local_ids.tolist(), asset.indexing.body_ids[local_ids].tolist(), strict=True):
        mass = _default_mass(env, env_ids, model_id)
        lo, hi = (mass + add_range[0]) / mass, (mass + add_range[1]) / mass
        if lo <= 0.0:
            raise ValueError(
                f"Body {local_id} weighs {mass:.3f} kg, and the range {add_range} can drive it to zero or below."
            )
        dr.pseudo_inertia(
            env=env,
            env_ids=env_ids,
            alpha_range=(0.5 * math.log(lo), 0.5 * math.log(hi)),
            distribution=distribution,
            asset_cfg=SceneEntityCfg(asset_cfg.name, body_ids=[local_id]),
        )


def _default_mass(env: Any, env_ids: torch.Tensor, model_id: int) -> float:
    """The body's default mass, which must be one number for the ratio to be well defined."""
    masses = env.sim.get_default_field("body_mass")
    masses = masses[env_ids, model_id] if "body_mass" in env.sim.per_world_default_fields else masses[model_id]
    masses = masses.reshape(-1)
    if not torch.allclose(masses, masses[0].expand_as(masses)):
        raise ValueError(
            "Randomising mass as a ratio needs one default mass per body, but this body already "
            "differs across environments."
        )
    return float(masses[0].item())
