"""Isaac-native event lowering.

Portable events already carry a task-owned callable and only need wrapping in
Isaac Lab's ``EventTermCfg``.  Named events live here because they translate an
engine-neutral operation onto PhysX-specific state, selectors, and parameters.

The module imports no Isaac SDK at import time.  The selected engine can build
its registry before ``AppLauncher`` starts, while each builder still imports the
native function only when compilation reaches that term.
"""

from __future__ import annotations

from typing import Any

from instinctlab_engine.registry import TermRegistry
from instinctlab_engine.spec.capability import (
    BODY_MASS_PROPERTIES,
    DR_RESTITUTION,
    DR_SLIDING_FRICTION,
    EXTERNAL_WRENCH,
    JOINT_STATE,
    ROOT_STATE,
    ROOT_VELOCITY_WRITE,
)

ISAAC_FRICTION_KEYS: frozenset[str] = frozenset(
    {
        "static_friction_range",
        "dynamic_friction_range",
        "restitution_range",
        "num_buckets",
        "make_consistent",
    }
)
"""Distribution keys ``randomize_rigid_body_material`` actually consumes."""


def randomize_default_joint_pos(
    env: Any,
    env_ids: Any,
    asset_cfg: Any,
    offset_distribution_params: tuple[float, float] | None,
    operation: str = "add",
    distribution: str = "uniform",
) -> None:
    """Randomize native defaults and copy them onto the canonical action axis by name."""
    import torch
    from isaaclab.envs.mdp.events import _randomize_prop_by_op

    from instinctlab_engine.name_order import copy_named_columns_

    asset = env.scene[asset_cfg.name]
    asset.data.default_joint_pos_nominal = torch.clone(
        asset.data.default_joint_pos[0]
    )

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)
    else:
        joint_ids = torch.tensor(
            asset_cfg.joint_ids, dtype=torch.int, device=asset.device
        )

    if offset_distribution_params is None:
        return

    positions = asset.data.default_joint_pos.to(asset.device).clone()
    positions = _randomize_prop_by_op(
        positions,
        offset_distribution_params,
        env_ids,
        joint_ids,
        operation=operation,
        distribution=distribution,
    )[env_ids][:, joint_ids]

    if isinstance(joint_ids, slice):
        asset.data.default_joint_pos[env_ids] = positions
        selected_joint_names = tuple(asset.joint_names)
    else:
        asset.data.default_joint_pos[env_ids[:, None], joint_ids] = positions
        selected_joint_names = tuple(
            asset.joint_names[index] for index in joint_ids.tolist()
        )

    action = env.action_manager.get_term("joint_pos")
    copy_named_columns_(
        action._offset,
        positions,
        env_ids,
        value_names=selected_joint_names,
        target_names=tuple(action._joint_names),
    )


def randomize_ray_offsets(
    env: Any,
    env_ids: Any,
    asset_cfg: Any,
    offset_pose_ranges: dict[str, tuple[float, float]],
    distribution: str = "uniform",
) -> None:
    """Randomize the starts and directions of an Isaac ray sensor in place."""
    import isaaclab.utils.math as math_utils
    import torch

    sensor = env.scene[asset_cfg.name]
    num_env_ids = env.scene.num_envs if env_ids is None else len(env_ids)
    ray_starts = sensor.ray_starts[env_ids]
    ray_directions = sensor.ray_directions[env_ids]

    keys = ("x", "y", "z", "roll", "pitch", "yaw")
    ranges = torch.tensor(
        [offset_pose_ranges.get(key, (0.0, 0.0)) for key in keys],
        device=ray_starts.device,
    )
    samples = (
        math_utils.sample_uniform(
            ranges[:, 0],
            ranges[:, 1],
            (num_env_ids, 6),
            device=ray_starts.device,
        )[..., None, :]
        .repeat(1, sensor.num_rays, 1)
        .flatten(0, 1)
    )
    rotations = math_utils.quat_from_euler_xyz(
        samples[..., 3], samples[..., 4], samples[..., 5]
    )
    ray_starts += samples[..., :3].reshape(ray_starts.shape)
    ray_directions = math_utils.quat_apply(
        rotations.reshape(*ray_directions.shape[:-1], 4), ray_directions
    )

    sensor.ray_starts[env_ids] = ray_starts
    sensor.ray_directions[env_ids] = ray_directions


def merge_friction_params(
    profile: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    """Overlay task-supplied friction keys, and refuse anything Isaac will not honor."""
    unknown = sorted(set(params) - ISAAC_FRICTION_KEYS)
    if unknown:
        raise ValueError(
            f"isaacsim randomize_friction does not honor {unknown}. "
            f"It honors {sorted(ISAAC_FRICTION_KEYS)}. Isaac-only keys belong in params; "
            "mjlab-only keys belong in engine_params['mjlab']."
        )
    merged = dict(profile)
    merged.update(params)
    return merged


def _event(spec, func, params: dict[str, Any]):
    """Wrap an event without losing its scheduling fields."""
    from isaaclab.managers import EventTermCfg

    return EventTermCfg(
        func=func,
        mode=spec.mode,
        interval_range_s=spec.interval_range_s,
        params=params,
    )


def _portable_event(spec, ctx):
    return _event(spec, spec.func, ctx.params(spec))


def _term_params(spec, ctx) -> dict[str, Any]:
    """Lower ``target=`` onto ``asset_cfg`` when the task used that slot."""
    params = dict(ctx.params(spec))
    if spec.target is not None and "asset_cfg" not in params:
        params["asset_cfg"] = ctx.entity(spec.target)
    return params


def _randomize_friction(spec, ctx):
    """Lower material randomization onto PhysX material buckets."""
    from isaaclab.envs.mdp import randomize_rigid_body_material

    profile = merge_friction_params(
        dict(ctx.profile.get("friction_dr", {})), ctx.params(spec)
    )
    return _event(
        spec,
        randomize_rigid_body_material,
        {"asset_cfg": ctx.entity(spec.target), **profile},
    )


def _randomize_body_mass(spec, ctx):
    from isaaclab.envs.mdp import randomize_rigid_body_mass

    params = ctx.params(spec)
    return _event(
        spec,
        randomize_rigid_body_mass,
        {
            "asset_cfg": ctx.entity(spec.target),
            "mass_distribution_params": params["add_range"],
            "operation": params["operation"],
        },
    )


def _apply_external_force_torque(spec, ctx):
    from isaaclab.envs.mdp import apply_external_force_torque

    params = ctx.params(spec)
    return _event(
        spec,
        apply_external_force_torque,
        {
            "asset_cfg": ctx.entity(spec.target),
            "force_range": params["force_range"],
            "torque_range": params["torque_range"],
        },
    )


def _reset_root_state_uniform(spec, ctx):
    from isaaclab.envs.mdp import reset_root_state_uniform

    params = ctx.params(spec)
    return _event(
        spec,
        reset_root_state_uniform,
        {
            "asset_cfg": ctx.entity(spec.target),
            "pose_range": params["pose_range"],
            "velocity_range": params["velocity_range"],
        },
    )


def _reset_joints_by_scale(spec, ctx):
    from isaaclab.envs.mdp import reset_joints_by_scale

    params = ctx.params(spec)
    return _event(
        spec,
        reset_joints_by_scale,
        {
            "asset_cfg": ctx.entity(spec.target),
            "position_range": params["position_range"],
            "velocity_range": params["velocity_range"],
        },
    )


def _reset_joints_by_offset(spec, ctx):
    """Additive joint reset; Isaac's native helper preserves that operation."""
    from isaaclab.envs.mdp import reset_joints_by_offset

    params = _term_params(spec, ctx)
    event_params = {
        "position_range": params["position_range"],
        "velocity_range": params["velocity_range"],
    }
    if "asset_cfg" in params:
        event_params["asset_cfg"] = params["asset_cfg"]
    return _event(spec, reset_joints_by_offset, event_params)


def _push_by_setting_velocity(spec, ctx):
    from isaaclab.envs.mdp import push_by_setting_velocity

    params = ctx.params(spec)
    return _event(
        spec,
        push_by_setting_velocity,
        {
            "asset_cfg": ctx.entity(spec.target),
            "velocity_range": params["velocity_range"],
        },
    )


def _randomize_joint_default(spec, ctx):
    return _event(
        spec,
        randomize_default_joint_pos,
        {
            "asset_cfg": ctx.entity(spec.target),
            "offset_distribution_params": ctx.params(spec)["range"],
            "operation": "add",
            "distribution": "uniform",
        },
    )


def _randomize_base_com(spec, ctx):
    from isaaclab.envs.mdp import randomize_rigid_body_com

    return _event(
        spec,
        randomize_rigid_body_com,
        {
            "asset_cfg": ctx.entity(spec.target),
            "com_range": ctx.params(spec)["com_range"],
        },
    )


def _randomize_actuator_gains(spec, ctx):
    from isaaclab.envs.mdp import randomize_actuator_gains

    params = ctx.params(spec)
    return _event(
        spec,
        randomize_actuator_gains,
        {
            "asset_cfg": ctx.entity(spec.target),
            "stiffness_distribution_params": params["stiffness_range"],
            "damping_distribution_params": params["damping_range"],
            "operation": params["operation"],
            "distribution": "uniform",
        },
    )


def _randomize_body_inertia(spec, ctx):
    from isaaclab.envs.mdp import randomize_rigid_body_mass

    params = ctx.params(spec)
    return _event(
        spec,
        randomize_rigid_body_mass,
        {
            "asset_cfg": ctx.entity(spec.target),
            "mass_distribution_params": params["add_range"],
            "operation": params["operation"],
        },
    )


def _randomize_ray_offsets(spec, ctx):
    from isaaclab.managers import SceneEntityCfg

    params = ctx.params(spec)
    sensor_name = params.pop("sensor_name")
    return _event(
        spec,
        randomize_ray_offsets,
        {
            "asset_cfg": SceneEntityCfg(sensor_name),
            **params,
            "distribution": "uniform",
        },
    )


def register_event_terms(terms: TermRegistry) -> None:
    """Install Isaac's generic event adapter surface into its term registry."""
    terms.portable("event")(_portable_event)
    terms.register(
        "event",
        "randomize_friction",
        _randomize_friction,
        provides=(DR_SLIDING_FRICTION, DR_RESTITUTION),
    )
    terms.register(
        "event",
        "randomize_body_mass",
        _randomize_body_mass,
        provides=(BODY_MASS_PROPERTIES,),
    )
    terms.register(
        "event",
        "apply_external_force_torque",
        _apply_external_force_torque,
        provides=(EXTERNAL_WRENCH,),
    )
    terms.register(
        "event",
        "reset_root_state_uniform",
        _reset_root_state_uniform,
        provides=(ROOT_STATE, ROOT_VELOCITY_WRITE),
    )
    terms.register(
        "event",
        "reset_joints_by_scale",
        _reset_joints_by_scale,
        provides=(JOINT_STATE,),
    )
    terms.register(
        "event",
        "reset_joints_by_offset",
        _reset_joints_by_offset,
        provides=(JOINT_STATE,),
    )
    terms.register(
        "event",
        "push_by_setting_velocity",
        _push_by_setting_velocity,
        provides=(ROOT_VELOCITY_WRITE,),
    )
    terms.register("event", "randomize_joint_default", _randomize_joint_default)
    terms.register("event", "randomize_base_com", _randomize_base_com)
    from instinctlab_engine.actuators import GAIN_RANDOMIZATION

    terms.register(
        "event",
        "randomize_actuator_gains",
        _randomize_actuator_gains,
        requires_actuator=(GAIN_RANDOMIZATION,),
    )
    terms.register("event", "randomize_body_inertia", _randomize_body_inertia)
    terms.register("event", "randomize_ray_offsets", _randomize_ray_offsets)
