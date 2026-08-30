"""MJLab-native event lowering.

Portable events already carry a task-owned callable and only need wrapping in
MJLab's ``EventTermCfg``.  Named events live here because they translate an
engine-neutral operation onto MuJoCo model fields, selectors, and distributions.

No MJLab SDK is imported at module import time.  Native imports remain inside
the builders so task declarations and shared interfaces stay SDK-free.
"""

from __future__ import annotations

import math
from typing import Any

from instinctlab.engines.registry import TermRegistry
from instinctlab.spec.capability import (
    BODY_MASS_PROPERTIES,
    DR_SLIDING_FRICTION,
    EXTERNAL_WRENCH,
    JOINT_STATE,
    ROOT_STATE,
    ROOT_VELOCITY_WRITE,
)

MJLAB_FRICTION_KEYS: frozenset[str] = frozenset(
    {"ranges", "operation", "shared_random", "distribution", "axes"}
)
MJLAB_FRICTION_ALIASES: frozenset[str] = frozenset(
    {"static_friction_range", "dynamic_friction_range"}
)


def merge_friction_params(
    profile: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    """Honor task-supplied sliding friction, and refuse keys MJLab cannot apply."""
    if "restitution_range" in params:
        raise ValueError(
            "mjlab randomize_friction cannot honor restitution_range: MuJoCo has no per-geom "
            "restitution. Put it in engine_params['isaacsim'] or drop it."
        )
    unknown = sorted(set(params) - MJLAB_FRICTION_KEYS - MJLAB_FRICTION_ALIASES)
    if unknown:
        raise ValueError(
            f"mjlab randomize_friction does not honor {unknown}. "
            f"It honors {sorted(MJLAB_FRICTION_KEYS | MJLAB_FRICTION_ALIASES)} "
            "(static/dynamic map to their union as 'ranges')."
        )
    if "ranges" in params and (params.keys() & MJLAB_FRICTION_ALIASES):
        raise ValueError(
            "mjlab randomize_friction: pass 'ranges' or static/dynamic friction, not both."
        )
    merged = dict(profile)
    for key in MJLAB_FRICTION_KEYS:
        if key in params:
            merged[key] = params[key]
    static = params.get("static_friction_range")
    dynamic = params.get("dynamic_friction_range")
    if static is not None or dynamic is not None:
        lows, highs = [], []
        if static is not None:
            lows.append(static[0])
            highs.append(static[1])
        if dynamic is not None:
            lows.append(dynamic[0])
            highs.append(dynamic[1])
        merged["ranges"] = (min(lows), max(highs))
    return merged


def _event(spec, func, params: dict[str, Any]):
    """Wrap an event without losing its scheduling fields."""
    from mjlab.managers import EventTermCfg

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


def _geoms_of(ref) -> Any:
    """Lower a whole-surface body selection to MuJoCo geoms."""
    from mjlab.managers import SceneEntityCfg

    return SceneEntityCfg(
        ref.entity if ref is not None else "robot", geom_names=(".*",)
    )


def _randomize_friction(spec, ctx):
    """Lower material randomization onto MuJoCo's single sliding coefficient."""
    from mjlab.envs.mdp import dr

    profile = merge_friction_params(
        dict(ctx.profile.get("friction_dr", {})), ctx.params(spec)
    )
    return _event(
        spec, dr.geom_friction, {"asset_cfg": _geoms_of(spec.target), **profile}
    )


def _randomize_body_mass(spec, ctx):
    from .native_event_functions import randomize_body_mass

    params = ctx.params(spec)
    if params["operation"] != "add":
        raise ValueError("mjlab randomize_body_mass only implements operation='add'.")
    return _event(
        spec,
        randomize_body_mass,
        {"asset_cfg": ctx.entity(spec.target), "add_range": params["add_range"]},
    )


def _apply_external_force_torque(spec, ctx):
    from mjlab.envs.mdp import apply_external_force_torque

    params = ctx.params(spec)
    return _event(
        spec,
        apply_external_force_torque,
        {
            "asset_cfg": ctx.entity(spec.target),
            "force_range": params.get("force_range", (0.0, 0.0)),
            "torque_range": params.get("torque_range", (0.0, 0.0)),
        },
    )


def _reset_root_state_uniform(spec, ctx):
    from mjlab.envs.mdp import reset_root_state_uniform

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
    from .native_event_functions import reset_joints_by_scale

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
    """Additive joint reset with MJLab's model-constant limit handling."""
    from .native_event_functions import reset_joints_by_offset

    params = _term_params(spec, ctx)
    event_params = {
        "position_range": params["position_range"],
        "velocity_range": params["velocity_range"],
    }
    if "asset_cfg" in params:
        event_params["asset_cfg"] = params["asset_cfg"]
    return _event(spec, reset_joints_by_offset, event_params)


def _push_by_setting_velocity(spec, ctx):
    from mjlab.envs.mdp import push_by_setting_velocity

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
    from .native_event_functions import randomize_default_joint_pos

    return _event(
        spec,
        randomize_default_joint_pos,
        {
            "asset_cfg": ctx.entity(spec.target),
            "offset_distribution_params": ctx.params(spec)["range"],
        },
    )


def _randomize_base_com(spec, ctx):
    from mjlab.envs.mdp import dr

    axes = {"x": 0, "y": 1, "z": 2}
    ranges = {axes[key]: value for key, value in ctx.params(spec)["com_range"].items()}
    return _event(
        spec,
        dr.body_ipos,
        {"asset_cfg": ctx.entity(spec.target), "ranges": ranges, "operation": "add"},
    )


def _randomize_actuator_gains(spec, ctx):
    from mjlab.envs.mdp import dr

    params = ctx.params(spec)
    return _event(
        spec,
        dr.pd_gains,
        {
            "asset_cfg": ctx.entity(spec.target),
            "kp_range": params["stiffness_range"],
            "kd_range": params["damping_range"],
            "operation": params["operation"],
        },
    )


def _randomize_body_inertia(spec, ctx):
    from mjlab.envs.mdp import dr

    from .native_event_functions import uniform_mass_scale_distribution

    params = ctx.params(spec)
    lo, hi = params["add_range"]
    return _event(
        spec,
        dr.pseudo_inertia,
        {
            "asset_cfg": ctx.entity(spec.target),
            "alpha_range": (0.5 * math.log(lo), 0.5 * math.log(hi)),
            "distribution": uniform_mass_scale_distribution(),
        },
    )


def _randomize_ray_offsets(spec, ctx):
    from .native_event_functions import randomize_ray_offsets

    return _event(spec, randomize_ray_offsets, ctx.params(spec))


def register_event_terms(terms: TermRegistry) -> None:
    """Install MJLab's generic event adapter surface into its term registry."""
    terms.portable("event")(_portable_event)
    terms.register(
        "event",
        "randomize_friction",
        _randomize_friction,
        provides=(DR_SLIDING_FRICTION,),
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
    terms.register("event", "randomize_actuator_gains", _randomize_actuator_gains)
    terms.register("event", "randomize_body_inertia", _randomize_body_inertia)
    terms.register("event", "randomize_ray_offsets", _randomize_ray_offsets)
