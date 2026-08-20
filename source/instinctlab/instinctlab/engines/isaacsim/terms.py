"""Isaac Sim's term registry, which is also its capability matrix.

Portable families get one builder each: the term already carries its function, and the builder's
whole job is to put it in the class Isaac Lab's manager expects. Per-engine families get a builder
per semantic name, and those names are what a task declares.

Commands are per-engine here even though they look portable, and that is worth recording because
the design originally assumed otherwise. The two engines' ``UniformVelocityCommandCfg`` agree on
what they mean and on most field names, then disagree on ``asset_name`` against ``entity_name`` and
on a handful of fields mjlab has and Isaac Lab does not. A shared config object would have to be
the intersection, which silently drops mjlab's. Naming the intent instead lets each side keep its
own fields, and a task states the parameters both understand.
"""

from __future__ import annotations

import inspect
import math
from typing import Any

from instinctlab.engines.registry import TermRegistry
from instinctlab.sim.capabilities import (
    BODY_MASS_PROPERTIES,
    DR_RESTITUTION,
    DR_SLIDING_FRICTION,
    EXTERNAL_WRENCH,
    JOINT_STATE,
    ROOT_STATE,
    ROOT_VELOCITY_WRITE,
)

TERMS = TermRegistry("isaacsim")

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


def merge_friction_params(profile: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Overlay task-supplied friction keys, and refuse anything this engine will not honor.

    The previous builder did ``profile.update(params)`` unconditionally. A key mjlab uses
    (``ranges``) would then sit in Isaac's event params unused -- the task asked for a range
    and got the profile's, with no error.
    """
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


def _import_cfgs() -> dict[str, Any]:
    """Isaac Lab's config classes, imported after ``AppLauncher`` has started the app."""
    from isaaclab.envs import mdp as isaac_mdp
    from isaaclab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg

    import instinctlab.envs.mdp as instinct_mdp

    return {
        "obs": ObservationTermCfg,
        "reward": RewardTermCfg,
        "done": TerminationTermCfg,
        "event": EventTermCfg,
        "mdp": isaac_mdp,
        "instinct_mdp": instinct_mdp,
    }


def _as_isaac_manager_term(cls: type) -> type:
    """Isaac Lab refuses a class term that is not a ``ManagerTermBase`` subclass.

    mjlab instantiates any class with ``(cfg=, env=)``. The delay / history buffer
    for the depth image is portable and cannot inherit Isaac's type, so this
    wrapper is the Isaac-only tax -- not a second implementation of the pipeline.
    """
    from isaaclab.managers import ManagerTermBase

    call_sig = inspect.signature(cls.__call__)

    class Wrapped(ManagerTermBase):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self._impl = cls(cfg, env)

        def reset(self, env_ids=None):
            return self._impl.reset(env_ids)

        def __call__(self, *args, **kwargs):
            return self._impl(*args, **kwargs)

    Wrapped.__call__.__signature__ = call_sig  # type: ignore[attr-defined]
    Wrapped.__name__ = cls.__name__
    Wrapped.__qualname__ = cls.__qualname__
    return Wrapped


@TERMS.portable("observation")
def _observation(spec, ctx):
    cfgs = _import_cfgs()
    func = spec.func
    if inspect.isclass(func):
        func = _as_isaac_manager_term(func)
    return cfgs["obs"](
        func=func,
        params=ctx.params(spec),
        noise=ctx.noise(spec.noise),
        scale=spec.scale,
        clip=spec.clip,
        history_length=spec.history_length,
    )


@TERMS.portable("reward")
def _reward(spec, ctx):
    return _import_cfgs()["reward"](func=spec.func, weight=spec.weight, params=ctx.params(spec))


@TERMS.portable("termination")
def _termination(spec, ctx):
    return _import_cfgs()["done"](func=spec.func, time_out=spec.time_out, params=ctx.params(spec))


@TERMS.portable("curriculum")
def _curriculum(spec, ctx):
    from isaaclab.managers import CurriculumTermCfg

    return CurriculumTermCfg(func=spec.func, params=ctx.params(spec))


def _term_params(spec, ctx):
    """Params with ``target=`` lowered onto ``asset_cfg`` when the task used that slot.

    Parkour (and flat G1) restrict joint/body terms through ``params['asset_cfg']``. A task
    that puts the same ``EntityRef`` on ``target=`` instead must not silently penalise every
    joint -- that is this repo's failure mode.
    """
    params = dict(ctx.params(spec))
    if spec.target is not None and "asset_cfg" not in params:
        params["asset_cfg"] = ctx.entity(spec.target)
    return params


def _sensor_entity(ref, ctx):
    """A ``SceneEntityCfg`` naming a declared sensor and the elements a term wants from it.

    A :class:`ContactSensorRef` is a sensor plus a subset of what it tracks, and Isaac Lab spells
    that as an entity config over the sensor whose ``body_names`` slice it. Terms written against
    the sensor rather than against the compat accessors need it in that shape.
    """
    from isaaclab.managers import SceneEntityCfg

    elements = ref.elements
    return SceneEntityCfg(ref.name, body_names=elements if isinstance(elements, str) else list(elements))


@TERMS.reward("contact_slide")
def _contact_slide(spec, ctx):
    """main's own slide penalty, kept rather than replaced. See the task's note on why."""
    cfgs = _import_cfgs()
    params = _term_params(spec, ctx)
    return cfgs["reward"](
        func=cfgs["instinct_mdp"].contact_slide,
        weight=spec.weight,
        params={**params, "sensor_cfg": _sensor_entity(params["sensor_cfg"], ctx)},
    )


@TERMS.reward("joint_acc_l2")
def _joint_acc_l2(spec, ctx):
    cfgs = _import_cfgs()
    return cfgs["reward"](func=cfgs["mdp"].joint_acc_l2, weight=spec.weight, params=_term_params(spec, ctx))


@TERMS.reward("joint_torques_l2")
def _joint_torques_l2(spec, ctx):
    cfgs = _import_cfgs()
    return cfgs["reward"](
        func=cfgs["instinct_mdp"].joint_torques_l2, weight=spec.weight, params=_term_params(spec, ctx)
    )


@TERMS.reward("motors_power_square")
def _motors_power_square(spec, ctx):
    """Parkour's energy term. Reads ``applied_torque`` (nv), which is on the denylist."""
    from instinctlab.envs.mdp.rewards.regularizations import motors_power_square

    cfgs = _import_cfgs()
    params = _term_params(spec, ctx)
    params.setdefault("normalize_by_stiffness", True)
    return cfgs["reward"](func=motors_power_square, weight=spec.weight, params=params)


@TERMS.reward("applied_torque_limits_by_ratio")
def _applied_torque_limits_by_ratio(spec, ctx):
    """Parkour's torque-limit term. Same denylist reason as ``motors_power_square``."""
    from instinctlab.envs.mdp.rewards.regularizations import applied_torque_limits_by_ratio

    cfgs = _import_cfgs()
    params = _term_params(spec, ctx)
    params.setdefault("limit_ratio", 0.8)
    return cfgs["reward"](func=applied_torque_limits_by_ratio, weight=spec.weight, params=params)


@TERMS.action("joint_position")
def _joint_position(spec, ctx):
    """Position-target action.

    Per-engine because the two drive joints through different objects, and because this is where
    decision D1 lands: the joint names come from the canonical depth-first order with
    ``preserve_order`` set, rather than from Isaac Lab's default ``[".*"]``, which resolves in
    PhysX's own breadth-first order. The resulting difference against the golden is a whitelist
    entry, and it is the reason the two engines' action vectors mean the same thing.
    """
    from instinctlab.envs.mdp import JointPositionActionCfg

    entity = ctx.entity(spec.target)
    params = ctx.params(spec)
    return JointPositionActionCfg(
        asset_name=entity.name,
        joint_names=list(entity.joint_names),
        preserve_order=spec.target.preserve_order if spec.target is not None else False,
        scale=params.get("scale", 1.0),
        use_default_offset=params.get("use_default_offset", True),
    )


@TERMS.command("pose_velocity")
def _pose_velocity(spec, ctx):
    from .pose_velocity import build_command

    return build_command(spec, ctx)


@TERMS.command("uniform_velocity")
def _uniform_velocity(spec, ctx):
    from isaaclab.envs.mdp import UniformVelocityCommandCfg

    params = ctx.params(spec)
    heading = params.get("heading", (-math.pi, math.pi))
    return UniformVelocityCommandCfg(
        asset_name=params.get("entity", "robot"),
        resampling_time_range=params["resampling_time_range"],
        rel_standing_envs=params.get("rel_standing_envs", 0.0),
        rel_heading_envs=params.get("rel_heading_envs", 1.0),
        heading_command=params.get("heading_command", False),
        heading_control_stiffness=params.get("heading_control_stiffness", 1.0),
        debug_vis=params.get("debug_vis", True),
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=params["lin_vel_x"],
            lin_vel_y=params["lin_vel_y"],
            ang_vel_z=params["ang_vel_z"],
            heading=heading,
        ),
    )


def _event(spec, func, params: dict[str, Any]):
    """One event term, with the scheduling fields every mode shares.

    Worth a helper rather than repetition: ``interval_range_s`` is only meaningful for interval
    events and is easy to forget on the builders that do not use it, and forgetting it turns a
    periodic push into one that never fires while everything still constructs.
    """
    return _import_cfgs()["event"](func=func, mode=spec.mode, interval_range_s=spec.interval_range_s, params=params)


@TERMS.event(
    "randomize_friction",
    provides=(DR_SLIDING_FRICTION, DR_RESTITUTION),
)
def _randomize_friction(spec, ctx):
    """Material randomisation: profile defaults, then task-supplied ranges.

    The scheme (64 buckets, static/dynamic/restitution) stays this engine's. The interval is
    something a task is allowed to state -- parkour wants (0.3, 1.6) rather than the flat-G1
    profile -- and a key this function will not apply is rejected rather than dropped. Selection
    is ``target=``; an Isaac-style ``asset_cfg`` in params is not a friction key.
    """
    from isaaclab.envs.mdp import randomize_rigid_body_material

    profile = merge_friction_params(dict(ctx.profile.get("friction_dr", {})), ctx.params(spec))
    return _event(spec, randomize_rigid_body_material, {"asset_cfg": ctx.entity(spec.target), **profile})


@TERMS.event("randomize_body_mass", provides=(BODY_MASS_PROPERTIES,))
def _randomize_body_mass(spec, ctx):
    from isaaclab.envs.mdp import randomize_rigid_body_mass

    params = ctx.params(spec)
    return _event(
        spec,
        randomize_rigid_body_mass,
        {
            "asset_cfg": ctx.entity(spec.target),
            "mass_distribution_params": params["add_range"],
            "operation": params.get("operation", "add"),
        },
    )


@TERMS.event("apply_external_force_torque", provides=(EXTERNAL_WRENCH,))
def _apply_external_force_torque(spec, ctx):
    from isaaclab.envs.mdp import apply_external_force_torque

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


@TERMS.event("reset_root_state_uniform", provides=(ROOT_STATE, ROOT_VELOCITY_WRITE))
def _reset_root_state_uniform(spec, ctx):
    from isaaclab.envs.mdp import reset_root_state_uniform

    params = ctx.params(spec)
    return _event(
        spec, reset_root_state_uniform, {"pose_range": params["pose_range"], "velocity_range": params["velocity_range"]}
    )


@TERMS.event("reset_joints_by_scale", provides=(JOINT_STATE,))
def _reset_joints_by_scale(spec, ctx):
    from isaaclab.envs.mdp import reset_joints_by_scale

    params = ctx.params(spec)
    return _event(
        spec,
        reset_joints_by_scale,
        {"position_range": params["position_range"], "velocity_range": params["velocity_range"]},
    )


@TERMS.event("reset_joints_by_offset", provides=(JOINT_STATE,))
def _reset_joints_by_offset(spec, ctx):
    """Additive joint reset. Isaac's native helper adds; mjlab's neighbour multiplies."""
    from isaaclab.envs.mdp import reset_joints_by_offset

    params = _term_params(spec, ctx)
    event_params = {"position_range": params["position_range"], "velocity_range": params["velocity_range"]}
    if "asset_cfg" in params:
        event_params["asset_cfg"] = params["asset_cfg"]
    return _event(spec, reset_joints_by_offset, event_params)


@TERMS.event("push_by_setting_velocity", provides=(ROOT_VELOCITY_WRITE,))
def _push_by_setting_velocity(spec, ctx):
    from isaaclab.envs.mdp import push_by_setting_velocity

    params = ctx.params(spec)
    return _event(spec, push_by_setting_velocity, {"velocity_range": params["velocity_range"]})


@TERMS.event("register_virtual_obstacles")
def _register_virtual_obstacles(spec, ctx):
    from instinctlab.mdp.events import register_virtual_obstacles

    return _event(spec, register_virtual_obstacles, ctx.params(spec))
