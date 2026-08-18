"""mjlab's term registry, which is also its capability matrix.

Reading this beside ``engines/isaacsim/terms.py`` is the clearest statement of what the ``N + M``
structure bought. The two files register the same semantic names, the portable builders are the same
three lines each, and everything that differs is confined to the bodies of the per-engine builders.

Domain randomisation is where they differ most, and not superficially. Isaac Lab randomises through
event functions taking distribution parameters; mjlab randomises through a declarative ``dr`` module
whose primitives are named after the model fields they perturb. Friction is the sharpest case:
PhysX has a static and a dynamic coefficient plus restitution, drawn from a bucket pool, while
MuJoCo has one sliding coefficient and no per-geom restitution at all. Neither is emulating the
other; each takes its own scheme from its own profile.
"""

from __future__ import annotations

import math
from typing import Any

from instinctlab.engines.registry import TermRegistry
from instinctlab.sim.capabilities import Capability

TERMS = TermRegistry("mjlab")


def _cfgs() -> dict[str, Any]:
    from mjlab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg

    return {
        "obs": ObservationTermCfg,
        "reward": RewardTermCfg,
        "done": TerminationTermCfg,
        "event": EventTermCfg,
    }


@TERMS.portable("observation")
def _observation(spec, ctx):
    return _cfgs()["obs"](
        func=spec.func,
        params=ctx.params(spec),
        noise=ctx.noise(spec.noise),
        scale=spec.scale,
        clip=spec.clip,
        history_length=spec.history_length,
    )


@TERMS.portable("reward")
def _reward(spec, ctx):
    return _cfgs()["reward"](func=spec.func, weight=spec.weight, params=ctx.params(spec))


@TERMS.portable("termination")
def _termination(spec, ctx):
    return _cfgs()["done"](func=spec.func, time_out=spec.time_out, params=ctx.params(spec))


@TERMS.reward("contact_slide")
def _contact_slide(spec, ctx):
    """InstinctMJ's slide penalty, kept rather than replaced. See the task's note on why.

    The sensor reference is passed through as declared instead of being lowered to a
    ``SceneEntityCfg``. mjlab's managers resolve such a config by asking the named object for its
    element names, and a contact sensor has none to give -- it is matched against the model when it
    is built, not selected from afterwards. Element selection therefore happens inside the term,
    against the sensor itself.
    """
    from .rewards import contact_slide

    return _cfgs()["reward"](func=contact_slide, weight=spec.weight, params=ctx.params(spec))


@TERMS.reward("joint_acc_l2")
def _joint_acc_l2(spec, ctx):
    from mjlab.envs.mdp import joint_acc_l2

    return _cfgs()["reward"](func=joint_acc_l2, weight=spec.weight, params=ctx.params(spec))


@TERMS.reward("joint_torques_l2")
def _joint_torques_l2(spec, ctx):
    from mjlab.envs.mdp import joint_torques_l2

    return _cfgs()["reward"](func=joint_torques_l2, weight=spec.weight, params=ctx.params(spec))


@TERMS.action("joint_position")
def _joint_position(spec, ctx):
    """Position-target action.

    mjlab drives **actuators**, Isaac Lab drives **joints**. For a robot whose actuators are built
    one per joint they name the same things, but the selector kind differs, and this is the point
    where decision D1 is applied on this engine: the canonical depth-first joint order is passed
    explicitly with ``preserve_order`` rather than relying on ``.*``, whose expansion follows the
    model file's own order.
    """
    from mjlab.envs.mdp import JointPositionActionCfg

    target = spec.target
    params = ctx.params(spec)
    return JointPositionActionCfg(
        entity_name=target.entity if target is not None else "robot",
        actuator_names=tuple(target.joints) if target is not None and target.joints else (".*",),
        preserve_order=target.preserve_order if target is not None else False,
        scale=params.get("scale", 1.0),
        use_default_offset=params.get("use_default_offset", True),
    )


@TERMS.command("uniform_velocity")
def _uniform_velocity(spec, ctx):
    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

    params = ctx.params(spec)
    return UniformVelocityCommandCfg(
        entity_name=params.get("entity", "robot"),
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
            heading=params.get("heading", (-math.pi, math.pi)),
        ),
    )


def _event(spec, func, params: dict[str, Any]):
    return _cfgs()["event"](func=func, mode=spec.mode, interval_range_s=spec.interval_range_s, params=params)


def _geoms_of(ctx, ref) -> Any:
    """An entity selector naming geoms rather than bodies.

    MuJoCo attaches friction to geoms; PhysX attaches materials to the shapes of a body. A reference
    that says "all of the robot's surfaces" therefore has to become a geom selection here and a body
    selection there. Translating it in the builder, which knows what its own function needs, keeps
    that knowledge out of the task.
    """
    from mjlab.managers import SceneEntityCfg

    return SceneEntityCfg(ref.entity if ref is not None else "robot", geom_names=(".*",))


@TERMS.event(
    "randomize_friction",
    provides=(Capability.DR_SLIDING_FRICTION,),
)
def _randomize_friction(spec, ctx):
    """Sliding friction randomisation.

    Does not advertise ``DR_RESTITUTION``, unlike the Isaac Sim registry, because MuJoCo has no
    per-geom restitution to randomise. A task that genuinely requires it fails here at startup with
    a message naming the capability, instead of running with one third of its randomisation quietly
    absent.
    """
    from mjlab.envs.mdp import dr

    profile = dict(ctx.profile.get("friction_dr", {}))
    profile.update(ctx.params(spec))
    return _event(spec, dr.geom_friction, {"asset_cfg": _geoms_of(ctx, spec.target), **profile})


@TERMS.event("randomize_body_mass", provides=(Capability.BODY_MASS_PROPERTIES,))
def _randomize_body_mass(spec, ctx):
    from .events import randomize_body_mass

    params = ctx.params(spec)
    return _event(spec, randomize_body_mass, {"asset_cfg": ctx.entity(spec.target), "add_range": params["add_range"]})


@TERMS.event("apply_external_force_torque", provides=(Capability.EXTERNAL_WRENCH,))
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


@TERMS.event("reset_root_state_uniform", provides=(Capability.ROOT_STATE, Capability.ROOT_VELOCITY_WRITE))
def _reset_root_state_uniform(spec, ctx):
    from mjlab.envs.mdp import reset_root_state_uniform

    params = ctx.params(spec)
    return _event(
        spec,
        reset_root_state_uniform,
        {"pose_range": params["pose_range"], "velocity_range": params["velocity_range"]},
    )


@TERMS.event("reset_joints_by_scale", provides=(Capability.JOINT_STATE,))
def _reset_joints_by_scale(spec, ctx):
    from .events import reset_joints_by_scale

    params = ctx.params(spec)
    return _event(
        spec,
        reset_joints_by_scale,
        {"position_range": params["position_range"], "velocity_range": params["velocity_range"]},
    )


@TERMS.event("push_by_setting_velocity", provides=(Capability.ROOT_VELOCITY_WRITE,))
def _push_by_setting_velocity(spec, ctx):
    from mjlab.envs.mdp import push_by_setting_velocity

    params = ctx.params(spec)
    return _event(spec, push_by_setting_velocity, {"velocity_range": params["velocity_range"]})
