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
from typing import Any

from instinctlab.engines.capabilities import (
    BODY_MASS_PROPERTIES,
    DR_RESTITUTION,
    DR_SLIDING_FRICTION,
    EXTERNAL_WRENCH,
    JOINT_STATE,
    ROOT_STATE,
    ROOT_VELOCITY_WRITE,
)
from instinctlab.engines.compile import joint_position_target
from instinctlab.engines.registry import TermRegistry

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


def merge_friction_params(
    profile: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
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
    from isaaclab.managers import (
        EventTermCfg,
        ObservationTermCfg,
        RewardTermCfg,
        TerminationTermCfg,
    )

    return {
        "obs": ObservationTermCfg,
        "reward": RewardTermCfg,
        "done": TerminationTermCfg,
        "event": EventTermCfg,
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
    return _import_cfgs()["reward"](
        func=spec.func, weight=spec.weight, params=ctx.params(spec)
    )


@TERMS.portable("termination")
def _termination(spec, ctx):
    func = (
        _as_isaac_manager_term(spec.func) if inspect.isclass(spec.func) else spec.func
    )
    return _import_cfgs()["done"](
        func=func, time_out=spec.time_out, params=ctx.params(spec)
    )


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


def _ray_sensor_entity(ref):
    """A whole ray sensor selection, without contact-only body filtering."""
    from isaaclab.managers import SceneEntityCfg

    from instinctlab.spec.sensor import RayCasterRef

    if not isinstance(ref, RayCasterRef):
        raise TypeError(
            f"Isaac ray sensor term expected RayCasterRef, got {type(ref).__name__}."
        )
    return SceneEntityCfg(ref.name)


@TERMS.action("joint_position")
def _joint_position(spec, ctx):
    """Position-target action.

    Per-engine because the two drive joints through different objects, and because this is where
    decision D1 lands: the joint names come from the canonical depth-first order with
    ``preserve_order`` set, rather than from Isaac Lab's default ``[".*"]``, which resolves in
    PhysX's own breadth-first order. The resulting difference against the golden is a whitelist
    entry, and it is the reason the two engines' action vectors mean the same thing.
    """
    from isaaclab.envs.mdp import JointPositionActionCfg

    target = joint_position_target(spec, ctx)
    entity = ctx.entity(target)
    params = ctx.params(spec)
    return JointPositionActionCfg(
        asset_name=entity.name,
        joint_names=list(entity.joint_names),
        preserve_order=target.preserve_order,
        scale=params["scale"],
        use_default_offset=params["use_default_offset"],
    )


@TERMS.command("pose_velocity")
def _pose_velocity(spec, ctx):
    from .pose_velocity import build_command

    return build_command(spec, ctx)


@TERMS.command("uniform_velocity")
def _uniform_velocity(spec, ctx):
    from isaaclab.envs.mdp import UniformVelocityCommandCfg

    params = ctx.params(spec)
    heading = params["heading"]
    return UniformVelocityCommandCfg(
        asset_name=params["entity"],
        resampling_time_range=params["resampling_time_range"],
        rel_standing_envs=params["rel_standing_envs"],
        rel_heading_envs=params["rel_heading_envs"],
        heading_command=params["heading_command"],
        heading_control_stiffness=params["heading_control_stiffness"],
        debug_vis=params["debug_vis"],
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
    return _import_cfgs()["event"](
        func=func, mode=spec.mode, interval_range_s=spec.interval_range_s, params=params
    )


@TERMS.portable("event")
def _portable_event(spec, ctx):
    return _event(spec, spec.func, ctx.params(spec))


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

    profile = merge_friction_params(
        dict(ctx.profile.get("friction_dr", {})), ctx.params(spec)
    )
    return _event(
        spec,
        randomize_rigid_body_material,
        {"asset_cfg": ctx.entity(spec.target), **profile},
    )


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
            "operation": params["operation"],
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
            "force_range": params["force_range"],
            "torque_range": params["torque_range"],
        },
    )


@TERMS.event("reset_root_state_uniform", provides=(ROOT_STATE, ROOT_VELOCITY_WRITE))
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


@TERMS.event("reset_joints_by_scale", provides=(JOINT_STATE,))
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


@TERMS.event("reset_joints_by_offset", provides=(JOINT_STATE,))
def _reset_joints_by_offset(spec, ctx):
    """Additive joint reset. Isaac's native helper adds; mjlab's neighbour multiplies."""
    from isaaclab.envs.mdp import reset_joints_by_offset

    params = _term_params(spec, ctx)
    event_params = {
        "position_range": params["position_range"],
        "velocity_range": params["velocity_range"],
    }
    if "asset_cfg" in params:
        event_params["asset_cfg"] = params["asset_cfg"]
    return _event(spec, reset_joints_by_offset, event_params)


@TERMS.event("push_by_setting_velocity", provides=(ROOT_VELOCITY_WRITE,))
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


# Motion-reference commands --------------------------------------------------

_MOTION_REFERENCE_COMMAND_KINDS = {
    "motion_reference_position": "position",
    "motion_reference_rotation": "rotation",
    "motion_reference_joint_position": "joint_position",
    "motion_reference_joint_velocity": "joint_velocity",
}


def _motion_reference_command(spec, ctx):
    from .motion_reference_commands import build_command

    return build_command(_MOTION_REFERENCE_COMMAND_KINDS[spec.kind], ctx.params(spec))


for _kind in _MOTION_REFERENCE_COMMAND_KINDS:
    TERMS.register("command", _kind, _motion_reference_command)


@TERMS.observation("height_scan")
def _height_scan(spec, ctx):
    """Lower the shared ray-height observation onto Isaac's sensor-entity API."""
    from isaaclab.envs.mdp import height_scan

    params = ctx.params(spec)
    sensor = params.pop("sensor")
    return _import_cfgs()["obs"](
        func=height_scan,
        params={"sensor_cfg": _ray_sensor_entity(sensor), **params},
        noise=ctx.noise(spec.noise),
        scale=spec.scale,
        clip=spec.clip,
        history_length=spec.history_length,
    )


@TERMS.event("randomize_joint_default")
def _randomize_joint_default(spec, ctx):
    from .events import randomize_default_joint_pos

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


@TERMS.event("randomize_base_com")
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


@TERMS.event("randomize_actuator_gains")
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


@TERMS.event("randomize_body_inertia")
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


@TERMS.event("randomize_ray_offsets")
def _randomize_ray_offsets(spec, ctx):
    from isaaclab.managers import SceneEntityCfg

    from .events import randomize_ray_offsets

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
