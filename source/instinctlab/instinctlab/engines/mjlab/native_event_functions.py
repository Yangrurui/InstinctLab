"""Native MJLab event functions used by :mod:`.event_terms`.

These functions touch MuJoCo model fields or MJLab sensor state directly, so
they cannot be task-owned portable events.  Several also preserve side effects
that a nearby stock primitive would silently omit.

``reset_joints_by_offset`` adds a sampled offset to the default joint position where Isaac Lab's
``reset_joints_by_scale`` multiplies it. mjlab now ships a native offset helper, but it indexes
``soft_joint_pos_limits`` by ``env_ids`` and that tensor's leading dim is 1 -- a model constant --
so the second environment is an out-of-range read. The port below uses :func:`_rows`. ``dr.body_mass``
changes a body's mass and leaves its inertia tensor untouched, where Isaac Lab's
``randomize_rigid_body_mass`` defaults to rescaling the inertia by the same ratio, so a heavier
torso is also harder to rotate. Substituting either neighbour would leave two runs looking
comparable while training against different physics.

The implementations are ported from InstinctMJ, the MJLab-side reference for
these tasks. Default-joint randomization also updates the canonical action
offset, and ray-offset randomization uses the native sensor mutation hook.

This is deliberately separate from the SDK-free registry/lowering module: the
``requires_model_fields`` decorator has to import MJLab at definition time.
Builders import this module only when compilation reaches one of these native
functions, so listing task contracts does not load MJLab.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from mjlab.envs.mdp import dr as _dr
from mjlab.managers.event_manager import requires_model_fields

__all__ = [
    "randomize_body_mass",
    "randomize_default_joint_pos",
    "randomize_ray_offsets",
    "reset_joints_by_offset",
    "reset_joints_by_scale",
    "uniform_mass_scale_distribution",
]


def randomize_default_joint_pos(env, env_ids, asset_cfg, offset_distribution_params):
    """Randomize default joint positions and keep the position-action offset aligned."""
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = torch.as_tensor(
            env_ids, dtype=torch.long, device=env.device
        ).flatten()
    joint_ids = asset_cfg.joint_ids
    if joint_ids is None:
        joint_ids = torch.arange(
            asset.data.default_joint_pos.shape[1], device=env.device
        )
    elif isinstance(joint_ids, slice):
        joint_ids = torch.arange(
            asset.data.default_joint_pos.shape[1], device=env.device
        )[joint_ids]
    else:
        joint_ids = torch.as_tensor(
            joint_ids, dtype=torch.long, device=env.device
        ).flatten()
    index = (env_ids[:, None], joint_ids[None, :])
    target = asset.data.default_joint_pos[index]
    noise = torch.empty_like(target).uniform_(*offset_distribution_params)
    asset.data.default_joint_pos[index] = target + noise
    action = env.action_manager.get_term("joint_pos")
    action._offset[index] = asset.data.default_joint_pos[index]


def randomize_ray_offsets(env, env_ids, sensor_name, offset_pose_ranges=None):
    """Apply per-environment calibration noise to a ray sensor's frame."""
    sensor = env.scene.sensors[sensor_name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    ranges = offset_pose_ranges or {}
    keys = ("x", "y", "z", "roll", "pitch", "yaw")
    bounds = torch.tensor(
        [ranges.get(key, (0.0, 0.0)) for key in keys], device=env.device
    )
    sample = (
        torch.rand(len(env_ids), 6, device=env.device) * (bounds[:, 1] - bounds[:, 0])
        + bounds[:, 0]
    )
    if not hasattr(sensor, "set_offset_noise"):
        raise TypeError(f"sensor {sensor_name!r} does not expose set_offset_noise")
    sensor.set_offset_noise(env_ids, sample)


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
    joint_pos = _rows(asset.data.default_joint_pos, env_ids)[
        :, asset_cfg.joint_ids
    ].clone()
    joint_pos += sample_uniform(*position_range, joint_pos.shape, env.device)
    limits = _rows(asset.data.soft_joint_pos_limits, env_ids)[:, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(limits[..., 0], limits[..., 1])

    joint_vel = _rows(asset.data.default_joint_vel, env_ids)[
        :, asset_cfg.joint_ids
    ].clone()
    joint_vel += sample_uniform(*velocity_range, joint_vel.shape, env.device)

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids = torch.tensor(joint_ids, device=env.device)
    asset.write_joint_state_to_sim(
        joint_pos.view(len(env_ids), -1),
        joint_vel.view(len(env_ids), -1),
        env_ids=env_ids,
        joint_ids=joint_ids,
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
    joint_pos = _rows(asset.data.default_joint_pos, env_ids)[
        :, asset_cfg.joint_ids
    ].clone()
    joint_pos *= sample_uniform(*position_range, joint_pos.shape, env.device)
    limits = _rows(asset.data.soft_joint_pos_limits, env_ids)[:, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(limits[..., 0], limits[..., 1])

    joint_vel = _rows(asset.data.default_joint_vel, env_ids)[
        :, asset_cfg.joint_ids
    ].clone()
    joint_vel *= sample_uniform(*velocity_range, joint_vel.shape, env.device)

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids = torch.tensor(joint_ids, device=env.device)
    asset.write_joint_state_to_sim(
        joint_pos.view(len(env_ids), -1),
        joint_vel.view(len(env_ids), -1),
        env_ids=env_ids,
        joint_ids=joint_ids,
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
        sample=lambda lo, hi, shape, device: (
            0.5
            * torch.log(
                sample_uniform(
                    torch.exp(2.0 * lo), torch.exp(2.0 * hi), shape, device=device
                )
            )
        ),
    )


@requires_model_fields(
    *_dr.pseudo_inertia.model_fields, recompute=_dr.pseudo_inertia.recompute
)
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

    local_ids = torch.arange(
        asset.indexing.body_ids.numel(), device=env.device, dtype=torch.long
    )
    local_ids = local_ids[asset_cfg.body_ids].reshape(-1)
    distribution = uniform_mass_scale_distribution()

    for local_id, model_id in zip(
        local_ids.tolist(), asset.indexing.body_ids[local_ids].tolist(), strict=True
    ):
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
    masses = (
        masses[env_ids, model_id]
        if "body_mass" in env.sim.per_world_default_fields
        else masses[model_id]
    )
    masses = masses.reshape(-1)
    if not torch.allclose(masses, masses[0].expand_as(masses)):
        raise ValueError(
            "Randomising mass as a ratio needs one default mass per body, but this body already "
            "differs across environments."
        )
    return float(masses[0].item())
