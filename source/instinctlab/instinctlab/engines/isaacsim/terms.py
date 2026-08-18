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

import math
from typing import Any

from instinctlab.engines.registry import TermRegistry
from instinctlab.sim.capabilities import Capability

TERMS = TermRegistry("isaacsim")


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


@TERMS.portable("observation")
def _observation(spec, ctx):
    cfgs = _import_cfgs()
    return cfgs["obs"](
        func=spec.func,
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
    provides=(Capability.DR_SLIDING_FRICTION, Capability.DR_RESTITUTION),
)
def _randomize_friction(spec, ctx):
    """Material randomisation, with this engine's own scheme as the default.

    The distribution comes from the adapter's profile, not from the task, and the profile's default
    is what the golden uses: 64 buckets of per-shape materials. mjlab's default is one friction per
    environment. Neither is emulating the other, which is what "each engine keeps its own
    characteristics" means in practice -- a task that stated the scheme itself would force one
    engine to imitate the other badly.
    """
    from isaaclab.envs.mdp import randomize_rigid_body_material

    profile = dict(ctx.profile.get("friction_dr", {}))
    profile.update(ctx.params(spec))
    return _event(spec, randomize_rigid_body_material, {"asset_cfg": ctx.entity(spec.target), **profile})


@TERMS.event("randomize_body_mass", provides=(Capability.BODY_MASS_PROPERTIES,))
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


@TERMS.event("apply_external_force_torque", provides=(Capability.EXTERNAL_WRENCH,))
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


@TERMS.event("reset_root_state_uniform", provides=(Capability.ROOT_STATE, Capability.ROOT_VELOCITY_WRITE))
def _reset_root_state_uniform(spec, ctx):
    from isaaclab.envs.mdp import reset_root_state_uniform

    params = ctx.params(spec)
    return _event(
        spec, reset_root_state_uniform, {"pose_range": params["pose_range"], "velocity_range": params["velocity_range"]}
    )


@TERMS.event("reset_joints_by_scale", provides=(Capability.JOINT_STATE,))
def _reset_joints_by_scale(spec, ctx):
    from isaaclab.envs.mdp import reset_joints_by_scale

    params = ctx.params(spec)
    return _event(
        spec,
        reset_joints_by_scale,
        {"position_range": params["position_range"], "velocity_range": params["velocity_range"]},
    )


@TERMS.event("push_by_setting_velocity", provides=(Capability.ROOT_VELOCITY_WRITE,))
def _push_by_setting_velocity(spec, ctx):
    from isaaclab.envs.mdp import push_by_setting_velocity

    params = ctx.params(spec)
    return _event(spec, push_by_setting_velocity, {"velocity_range": params["velocity_range"]})
