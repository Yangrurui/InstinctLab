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

from instinctlab.engines.compile import joint_position_target
from instinctlab.engines.registry import TermRegistry
from instinctlab.sim.capabilities import (
    BODY_MASS_PROPERTIES,
    DR_SLIDING_FRICTION,
    EXTERNAL_WRENCH,
    JOINT_STATE,
    ROOT_STATE,
    ROOT_VELOCITY_WRITE,
)

TERMS = TermRegistry("mjlab")

MJLAB_FRICTION_KEYS: frozenset[str] = frozenset({"ranges", "operation", "shared_random", "distribution", "axes"})
MJLAB_FRICTION_ALIASES: frozenset[str] = frozenset({"static_friction_range", "dynamic_friction_range"})


def merge_friction_params(profile: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Honor task-supplied sliding friction, and refuse keys this engine cannot apply.

    MuJoCo has one sliding coefficient. A task that states static and dynamic ranges is mapped
    to their union (min of lows, max of highs), which is what InstinctMJ's port does and the
    closest single interval to "the same randomisation". ``restitution_range`` is rejected
    rather than dropped: Isaac parkour randomises restitution and mjlab parkour does not, and
    silently keeping the profile (or ignoring the key) is the failure this check exists for.
    """
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
        raise ValueError("mjlab randomize_friction: pass 'ranges' or static/dynamic friction, not both.")
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


def _cfgs() -> dict[str, Any]:
    from mjlab.managers import CurriculumTermCfg, EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg

    return {
        "obs": ObservationTermCfg,
        "reward": RewardTermCfg,
        "done": TerminationTermCfg,
        "event": EventTermCfg,
        "curriculum": CurriculumTermCfg,
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


@TERMS.portable("curriculum")
def _curriculum(spec, ctx):
    return _cfgs()["curriculum"](func=spec.func, params=ctx.params(spec))


def _term_params(spec, ctx):
    """Params with ``target=`` lowered onto ``asset_cfg`` when the task used that slot.

    mjlab's stock ``joint_torques_l2`` slices ``actuator_ids``, which stay ``slice(None)``
    when the task only named joints -- so a hip/knee penalty silently ran on all 29
    actuators. Builders that go through here pass the resolved ``EntityRef`` as ``asset_cfg``.
    """
    params = dict(ctx.params(spec))
    if spec.target is not None and "asset_cfg" not in params:
        params["asset_cfg"] = ctx.entity(spec.target)
    return params


@TERMS.termination("illegal_contact")
def _illegal_contact(spec, ctx):
    """Terminate on ‖force‖ (full 3-D, friction included) above 1 N. Matches InstinctMJ."""
    from .rewards import CONTACT_FORCE_THRESHOLD_N, illegal_contact

    params = dict(ctx.params(spec))
    params.setdefault("threshold", CONTACT_FORCE_THRESHOLD_N)
    return _cfgs()["done"](func=illegal_contact, time_out=spec.time_out, params=params)


@TERMS.reward("undesired_contacts")
def _undesired_contacts(spec, ctx):
    """Count bodies whose ‖force‖ (full 3-D, friction included) exceeds 1 N."""
    from .rewards import CONTACT_FORCE_THRESHOLD_N, undesired_contacts

    params = dict(ctx.params(spec))
    params.setdefault("threshold", CONTACT_FORCE_THRESHOLD_N)
    return _cfgs()["reward"](func=undesired_contacts, weight=spec.weight, params=params)


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

    return _cfgs()["reward"](func=contact_slide, weight=spec.weight, params=_term_params(spec, ctx))


@TERMS.reward("joint_acc_l2")
def _joint_acc_l2(spec, ctx):
    from mjlab.envs.mdp import joint_acc_l2

    return _cfgs()["reward"](func=joint_acc_l2, weight=spec.weight, params=_term_params(spec, ctx))


@TERMS.reward("joint_torques_l2")
def _joint_torques_l2(spec, ctx):
    """Joint-space torque penalty. mjlab's stock term reads ``actuator_force`` (nu)."""
    from .rewards import joint_torques_l2

    return _cfgs()["reward"](func=joint_torques_l2, weight=spec.weight, params=_term_params(spec, ctx))


@TERMS.reward("motors_power_square")
def _motors_power_square(spec, ctx):
    """Parkour's energy term. Reads ``qfrc_actuator`` (nv), not ``actuator_force`` (nu)."""
    from .rewards import motors_power_square

    params = _term_params(spec, ctx)
    params.setdefault("normalize_by_stiffness", True)
    return _cfgs()["reward"](func=motors_power_square, weight=spec.weight, params=params)


@TERMS.reward("applied_torque_limits_by_ratio")
def _applied_torque_limits_by_ratio(spec, ctx):
    """Parkour's torque-limit term. Same ``qfrc_actuator`` / joint-id reasoning."""
    from .rewards import applied_torque_limits_by_ratio

    params = _term_params(spec, ctx)
    params.setdefault("limit_ratio", 0.8)
    return _cfgs()["reward"](func=applied_torque_limits_by_ratio, weight=spec.weight, params=params)


@TERMS.action("joint_position")
def _joint_position(spec, ctx):
    """Position-target action.

    mjlab drives **actuators**, Isaac Lab drives **joints**. For a robot whose actuators are built
    one per joint they name the same things, but the selector kind differs, and this is the point
    where decision D1 is applied on this engine: the canonical depth-first joint order is passed
    explicitly with ``preserve_order`` rather than relying on ``.*``, whose expansion follows the
    model file's own order.
    """
    from .actions import PreservingJointPositionActionCfg

    target = joint_position_target(spec, ctx)
    params = ctx.params(spec)
    return PreservingJointPositionActionCfg(
        entity_name=target.entity,
        actuator_names=tuple(target.joints),
        preserve_order=target.preserve_order,
        scale=params.get("scale", 1.0),
        use_default_offset=params.get("use_default_offset", True),
    )


@TERMS.command("pose_velocity")
def _pose_velocity(spec, ctx):
    from .pose_velocity import build_command

    return build_command(spec, ctx)


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
    provides=(DR_SLIDING_FRICTION,),
)
def _randomize_friction(spec, ctx):
    """Sliding friction randomisation.

    Does not advertise ``DR_RESTITUTION``, unlike the Isaac Sim registry, because MuJoCo has no
    per-geom restitution to randomise. A task that puts ``restitution_range`` in shared params is
    rejected here rather than trained without it. Isaac-only restitution belongs in
    ``engine_params['isaacsim']``. Static/dynamic ranges map to their union as ``ranges``.
    """
    from mjlab.envs.mdp import dr

    profile = merge_friction_params(dict(ctx.profile.get("friction_dr", {})), ctx.params(spec))
    return _event(spec, dr.geom_friction, {"asset_cfg": _geoms_of(ctx, spec.target), **profile})


@TERMS.event("randomize_body_mass", provides=(BODY_MASS_PROPERTIES,))
def _randomize_body_mass(spec, ctx):
    from .events import randomize_body_mass

    params = ctx.params(spec)
    return _event(
        spec,
        randomize_body_mass,
        {"asset_cfg": ctx.entity(spec.target), "add_range": params["add_range"]},
    )


@TERMS.event("apply_external_force_torque", provides=(EXTERNAL_WRENCH,))
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


@TERMS.event("reset_root_state_uniform", provides=(ROOT_STATE, ROOT_VELOCITY_WRITE))
def _reset_root_state_uniform(spec, ctx):
    from mjlab.envs.mdp import reset_root_state_uniform

    params = ctx.params(spec)
    return _event(
        spec,
        reset_root_state_uniform,
        {
            "pose_range": params["pose_range"],
            "velocity_range": params["velocity_range"],
        },
    )


@TERMS.event("reset_joints_by_scale", provides=(JOINT_STATE,))
def _reset_joints_by_scale(spec, ctx):
    from .events import reset_joints_by_scale

    params = ctx.params(spec)
    return _event(
        spec,
        reset_joints_by_scale,
        {
            "position_range": params["position_range"],
            "velocity_range": params["velocity_range"],
        },
    )


@TERMS.event("reset_joints_by_offset", provides=(JOINT_STATE,))
def _reset_joints_by_offset(spec, ctx):
    """Additive joint reset. mjlab's native helper also adds, but indexes limits per-env."""
    from .events import reset_joints_by_offset

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
    from mjlab.envs.mdp import push_by_setting_velocity

    params = ctx.params(spec)
    return _event(spec, push_by_setting_velocity, {"velocity_range": params["velocity_range"]})


@TERMS.event("register_virtual_obstacles")
def _register_virtual_obstacles(spec, ctx):
    from instinctlab.mdp.events import register_virtual_obstacles

    return _event(spec, register_virtual_obstacles, ctx.params(spec))


# Shadowing -----------------------------------------------------------------

_SHADOW_COMMAND_KINDS = {
    "shadow_position_reference": "position",
    "shadow_rotation_reference": "rotation",
    "shadow_joint_position_reference": "joint_position",
    "shadow_joint_velocity_reference": "joint_velocity",
}


def _shadow_command(spec, ctx):
    from .shadowing import build_command

    return build_command(_SHADOW_COMMAND_KINDS[spec.kind], ctx.params(spec))


for _kind in _SHADOW_COMMAND_KINDS:
    TERMS.register("command", _kind, _shadow_command)


def _shadow_entity(ctx):
    from instinctlab.spec import EntityRef

    motion = ctx.spec.scene.motion_reference("motion_reference")
    return ctx.entity(EntityRef("robot", bodies=motion.links, preserve_order=True))


def _shadow_obs(spec, ctx):
    from instinctlab.mdp import shadowing

    func = shadowing.link_position if spec.kind == "shadow_link_position" else shadowing.link_rotation
    return _cfgs()["obs"](func=func, params={"asset_cfg": _shadow_entity(ctx)})


TERMS.register("observation", "shadow_link_position", _shadow_obs)
TERMS.register("observation", "shadow_link_rotation", _shadow_obs)


@TERMS.observation("shadow_base_linear_velocity")
def _shadow_base_linear_velocity(spec, ctx):
    from mjlab.envs import mdp

    return _cfgs()["obs"](
        func=mdp.base_lin_vel,
        params=ctx.params(spec),
        noise=ctx.noise(spec.noise),
        scale=spec.scale,
        clip=spec.clip,
        history_length=spec.history_length,
    )


@TERMS.observation("shadow_depth_image")
def _shadow_depth(spec, ctx):
    from instinctlab import mdp
    from instinctlab.mdp import shadowing

    params = ctx.params(spec)
    if "history_length" in params:
        func = mdp.DelayedDepthImage
    else:
        func = shadowing.depth_image
        params.update(resize_shape=(18, 32), normalization_range=(0.0, 2.0))
    return _cfgs()["obs"](func=func, params=params)


@TERMS.observation("shadow_height_scan")
def _shadow_height(spec, ctx):
    from mjlab.envs.mdp import height_scan

    params = ctx.params(spec)
    sensor = params.pop("sensor")
    return _cfgs()["obs"](func=height_scan, params={"sensor_name": sensor.name}, clip=(-20.0, 20.0))


_SHADOW_REWARDS = {
    "shadow_base_position_gauss": "base_position_imitation",
    "shadow_base_rotation_gauss": "base_rotation_imitation",
    "shadow_link_position_gauss": "link_position_imitation",
    "shadow_link_rotation_gauss": "link_rotation_imitation",
    "shadow_link_linear_velocity_gauss": "link_linear_velocity_imitation",
    "shadow_link_angular_velocity_gauss": "link_angular_velocity_imitation",
}


def _shadow_reward(spec, ctx):
    from instinctlab.mdp import shadowing

    params = ctx.params(spec)
    params.update(reference_cfg="motion_reference", asset_cfg=_shadow_entity(ctx))
    return _cfgs()["reward"](
        func=getattr(shadowing, _SHADOW_REWARDS[spec.kind]),
        weight=spec.weight,
        params=params,
    )


for _kind in _SHADOW_REWARDS:
    TERMS.register("reward", _kind, _shadow_reward)


@TERMS.reward("shadow_undesired_contacts")
def _shadow_contact_reward(spec, ctx):
    from instinctlab.mdp.shadowing import undesired_contacts

    return _cfgs()["reward"](
        func=undesired_contacts,
        weight=spec.weight,
        params=ctx.params(spec),
    )


@TERMS.reward("shadow_torque_limit_ratio")
def _shadow_torque(spec, ctx):
    from instinctlab.spec import EntityRef

    from .rewards import applied_torque_limits_by_ratio

    entity = ctx.entity(EntityRef("robot", joints=(".*ankle.*", ".*wrist.*")))
    return _cfgs()["reward"](
        func=applied_torque_limits_by_ratio,
        weight=spec.weight,
        params={"asset_cfg": entity},
    )


_SHADOW_DONES = {
    "shadow_base_position_too_far": "base_position_too_far",
    "shadow_projected_gravity_too_far": "projected_gravity_too_far",
    "shadow_link_position_too_far": "link_position_too_far",
}


def _shadow_done(spec, ctx):
    from instinctlab.mdp import shadowing

    params = ctx.params(spec)
    params.update(reference_cfg="motion_reference", asset_cfg=_shadow_entity(ctx))
    if spec.kind == "shadow_link_position_too_far":
        motion_links = ctx.spec.scene.motion_reference("motion_reference").links
        params["link_ids"] = tuple(motion_links.index(name) for name in spec.target.bodies)
    return _cfgs()["done"](
        func=getattr(shadowing, _SHADOW_DONES[spec.kind]),
        time_out=spec.time_out,
        params=params,
    )


for _kind in _SHADOW_DONES:
    TERMS.register("termination", _kind, _shadow_done)


@TERMS.termination("shadow_illegal_reset_contact")
def _shadow_illegal_reset(spec, ctx):
    from instinctlab.mdp.shadowing import IllegalResetContact

    return _cfgs()["done"](func=IllegalResetContact, time_out=spec.time_out, params=ctx.params(spec))


@TERMS.event("randomize_joint_default")
def _shadow_joint_default(spec, ctx):
    from instinctlab.spec import EntityRef

    from .shadowing import randomize_default_joint_pos

    return _event(
        spec,
        randomize_default_joint_pos,
        {
            "asset_cfg": ctx.entity(EntityRef("robot", joints=".*")),
            "offset_distribution_params": ctx.params(spec)["range"],
        },
    )


@TERMS.event("randomize_base_com")
def _shadow_base_com(spec, ctx):
    from mjlab.envs.mdp import dr

    axes = {"x": 0, "y": 1, "z": 2}
    ranges = {axes[key]: value for key, value in ctx.params(spec)["com_range"].items()}
    return _event(
        spec,
        dr.body_ipos,
        {"asset_cfg": ctx.entity(spec.target), "ranges": ranges, "operation": "add"},
    )


@TERMS.event("randomize_actuator_gains")
def _shadow_actuator_gains(spec, ctx):
    from mjlab.envs.mdp import dr

    from instinctlab.spec import EntityRef

    params = ctx.params(spec)
    return _event(
        spec,
        dr.pd_gains,
        {
            "asset_cfg": ctx.entity(EntityRef("robot", joints=".*")),
            "kp_range": params["stiffness_range"],
            "kd_range": params["damping_range"],
            "operation": params["operation"],
        },
    )


@TERMS.event("shadow_randomize_body_inertia")
def _shadow_body_inertia(spec, ctx):
    from mjlab.envs.mdp import dr

    params = ctx.params(spec)
    lo, hi = params["add_range"]
    return _event(
        spec,
        dr.pseudo_inertia,
        {
            "asset_cfg": ctx.entity(spec.target),
            "alpha_range": (0.5 * math.log(lo), 0.5 * math.log(hi)),
        },
    )


@TERMS.event("shadow_randomize_ray_offsets")
def _shadow_ray_offsets(spec, ctx):
    from .shadowing import randomize_ray_offsets

    return _event(spec, randomize_ray_offsets, {"sensor_name": "camera", **ctx.params(spec)})


@TERMS.event("push_root_velocity")
def _shadow_push(spec, ctx):
    from mjlab.envs.mdp import push_by_setting_velocity

    return _event(
        spec,
        push_by_setting_velocity,
        {"velocity_range": ctx.params(spec)["velocity_range"]},
    )


def _shadow_runtime_event(spec, ctx):
    from instinctlab.engines import shadowing_events

    names = {
        "shadow_match_reference_origin": "match_reference_origin",
        "shadow_reset_robot_from_reference": "reset_robot_from_reference",
        "shadow_smooth_bin_failures": "smooth_bin_failures",
        "shadow_reset_objects_from_reference": "reset_objects_from_reference",
        "shadow_update_objects_from_reference": "update_objects_from_reference",
    }
    params = ctx.params(spec)
    params.setdefault("motion_reference", "motion_reference")
    return _event(spec, getattr(shadowing_events, names[spec.kind]), params)


for _kind in (
    "shadow_match_reference_origin",
    "shadow_reset_robot_from_reference",
    "shadow_smooth_bin_failures",
    "shadow_reset_objects_from_reference",
    "shadow_update_objects_from_reference",
):
    TERMS.register("event", _kind, _shadow_runtime_event)


@TERMS.curriculum("shadow_adaptive_sampling")
def _shadow_curriculum(spec, ctx):
    from instinctlab.engines.shadowing_events import adaptive_sampling

    return _cfgs()["curriculum"](func=adaptive_sampling, params=ctx.params(spec))
