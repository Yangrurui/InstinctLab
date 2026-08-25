"""Shared declarations for every G1 shadowing task.

The reference projects express the four shadowing families as native environment classes.  This
module instead describes their common contract once.  A variant only selects observations,
sensors, motion inventory and the few MDP terms that genuinely differ between families; it never
selects a simulator implementation.

Terms whose runtime is shadowing-specific use semantic ``kind`` names.  Their Isaac and MJLab
implementations live in the engine registries, alongside the existing engine-specific action and
event implementations.  This keeps task policy out of adapters without pretending that SDK
manager-term classes are portable Python objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from instinctlab import mdp
from instinctlab.assets.unitree_g1.isaacsim import G1_29DOF_LINKS, make_g1_29dof_robot_spec
from instinctlab.spec import (
    ActionTermSpec,
    AgentSpec,
    CommandTermSpec,
    ContactSensorRef,
    CurriculumTermSpec,
    DoneTermSpec,
    EntityRef,
    EventTermSpec,
    MdpSpec,
    MotionReferenceRef,
    NoiseSpec,
    ObsGroupSpec,
    ObsTermSpec,
    RayCasterRef,
    RayPatternRef,
    RewardTermSpec,
    SceneSpec,
    SimSpec,
    TaskSpec,
    TerrainSpec,
)
from instinctlab.spec.capability import Requirement

Family = Literal["whole_body", "perceptive", "perceptive_vae", "perceptive_hoi", "beyondmimic"]

MOTION_LINKS = (
    "pelvis",
    "torso_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "left_knee_link",
    "right_knee_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)

MOTION_PATHS = {
    "whole_body": "~/Xyk/Datasets/NoKov-Marslab-Motions-instinctnpz/20251016_diveroll4_single",
    "perceptive": "~/Xyk/Datasets/her_leveled",
    "perceptive_vae": "~/Xyk/Datasets/20260317_50cm_kneeClimbStep1_projectInstinct",
    "perceptive_hoi": "~/Datasets/OMOMO/retargeted_omniretarget_instinctmj_torso_v10_object_xy_align_foot_lock",
    "beyondmimic": "~/Xyk/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz",
}

RUNNERS = {
    "whole_body": "instinctlab.tasks.shadowing.whole_body.config.g1.agents.instinct_rl_ppo_cfg:G1ShadowingPPORunnerCfg",
    "perceptive": (
        "instinctlab.tasks.shadowing.perceptive.config.g1.agents.instinct_rl_ppo_cfg:G1PerceptiveShadowingPPORunnerCfg"
    ),
    "perceptive_vae": (
        "instinctlab.tasks.shadowing.perceptive.config.g1.agents.instinct_rl_vae_cfg:G1PerceptiveVaePPORunnerCfg"
    ),
    "perceptive_hoi": (
        "instinctlab.tasks.shadowing.perceptive_hoi.config.g1.agents.instinct_rl_ppo_cfg:"
        "G1PerceptiveHoiShadowingPPORunnerCfg"
    ),
    "beyondmimic": (
        "instinctlab.tasks.shadowing.beyondmimic.config.g1.agents.beyondmimic_ppo_cfg:G1BeyondMimicPPORunnerCfg"
    ),
}


@dataclass(frozen=True)
class ShadowingVariant:
    task_id: str
    family: Family
    play: bool = False
    one_motion: bool = False


def _motion_reference(variant: ShadowingVariant, joints: tuple[str, ...]) -> MotionReferenceRef:
    current_only = variant.family == "beyondmimic"
    frame_interval = 0.0 if current_only else (0.02 if variant.family == "whole_body" else 0.1)
    start_range = (0.0, 0.0) if variant.play or variant.one_motion else (0.0, 0.8)
    return MotionReferenceRef(
        name="motion_reference",
        clip=MOTION_PATHS[variant.family],
        joints=joints,
        links=MOTION_LINKS,
        num_frames=1 if current_only else 10,
        frame_interval_s=frame_interval,
        update_period=0.02,
        data_start_from="current_time",
        clip_target_fps=50.0,
        velocity_method="frontward",
        start_range=start_range,
        exhaustion="freeze_last_and_flag",
        quaternion="wxyz",
        symmetric_augmentation=None,
    )


def _camera() -> RayCasterRef:
    return RayCasterRef(
        name="camera",
        attach="torso_link",
        offset=(0.047, 0.0, 0.42),
        offset_rot=(0.9238795, 0.0, 0.3826834, 0.0),
        offset_convention="world",
        pattern=RayPatternRef(
            kind="pinhole",
            width=64,
            height=36,
            horizontal_fov_deg=89.51,
            vertical_fov_deg=58.29,
            focal_length=1.0,
        ),
        hit=("terrain", *G1_29DOF_LINKS),
        ray_alignment="base",
        miss="infinity",
        max_distance=2.5,
        min_distance=0.1,
        update_period=0.02,
    )


def _height_scanner() -> RayCasterRef:
    return RayCasterRef(
        name="height_scanner",
        attach="torso_link",
        offset=(0.0, 0.0, 20.0),
        pattern=RayPatternRef(kind="grid", resolution=0.1, size=(1.6, 1.0)),
        hit="terrain",
        ray_alignment="yaw",
        miss="infinity",
        update_period=0.02,
    )


def _scene(variant: ShadowingVariant, motion: MotionReferenceRef) -> SceneSpec:
    perceptive = variant.family in {"perceptive", "perceptive_vae", "perceptive_hoi"}
    rays: list[RayCasterRef] = []
    if perceptive:
        rays.append(_camera())
    if variant.family in {"perceptive", "perceptive_hoi"}:
        rays.append(_height_scanner())
    contacts = (
        ContactSensorRef(
            name="contact_forces" if perceptive else "undesired_contact_forces",
            elements=".*",
            track_air_time=perceptive,
            air_time_force_threshold=1.0,
            history_length=3,
        ),
    )
    return SceneSpec(
        terrain=TerrainSpec(kind="plane"),
        contact_sensors=contacts,
        ray_casters=tuple(rays),
        motion_references=(motion,),
        env_spacing=2.5 if variant.play else 4.0,
    )


def _commands(variant: ShadowingVariant) -> dict[str, CommandTermSpec]:
    names = ["position_ref_command"]
    if variant.family != "beyondmimic":
        names.append("position_b_ref_command")
    names.extend(("rotation_ref_command", "joint_pos_ref_command", "joint_vel_ref_command"))
    kinds = {
        "position_ref_command": "shadow_position_reference",
        "position_b_ref_command": "shadow_position_reference",
        "rotation_ref_command": "shadow_rotation_reference",
        "joint_pos_ref_command": "shadow_joint_position_reference",
        "joint_vel_ref_command": "shadow_joint_velocity_reference",
    }
    commands: dict[str, CommandTermSpec] = {}
    for name in names:
        params = {"motion_reference": "motion_reference", "current_state_command": False}
        if name == "position_ref_command":
            params.update(realtime_mode=True, anchor_frame="robot")
        elif name == "position_b_ref_command":
            params.update(realtime_mode=True, anchor_frame="reference")
        elif name == "rotation_ref_command":
            params.update(realtime_mode=True, in_base_frame=True, rotation_mode="tannorm")
        commands[name] = CommandTermSpec(kind=kinds[name], params=params)
    return commands


def _reference_observations() -> dict[str, ObsTermSpec]:
    return {
        "joint_pos_ref": ObsTermSpec(func=mdp.generated_commands, params={"command_name": "joint_pos_ref_command"}),
        "joint_vel_ref": ObsTermSpec(func=mdp.generated_commands, params={"command_name": "joint_vel_ref_command"}),
        "position_ref": ObsTermSpec(func=mdp.generated_commands, params={"command_name": "position_ref_command"}),
        "rotation_ref": ObsTermSpec(func=mdp.generated_commands, params={"command_name": "rotation_ref_command"}),
    }


def _proprioception(joints: EntityRef, *, corrupt: bool) -> dict[str, ObsTermSpec]:
    noise = (lambda lo, hi: NoiseSpec("uniform", lo, hi)) if corrupt else (lambda lo, hi: None)
    return {
        "projected_gravity": ObsTermSpec(func=mdp.projected_gravity, noise=noise(-0.05, 0.05)),
        "base_ang_vel": ObsTermSpec(func=mdp.base_ang_vel, noise=noise(-0.2, 0.2)),
        "joint_pos": ObsTermSpec(func=mdp.joint_pos_rel, params={"asset_cfg": joints}, noise=noise(-0.01, 0.01)),
        "joint_vel": ObsTermSpec(func=mdp.joint_vel_rel, params={"asset_cfg": joints}, noise=noise(-0.5, 0.5)),
        "last_action": ObsTermSpec(func=mdp.last_action),
    }


def _observations(variant: ShadowingVariant, joints: EntityRef, motion: MotionReferenceRef) -> dict[str, ObsGroupSpec]:
    refs = _reference_observations()
    policy: dict[str, ObsTermSpec]
    if variant.family == "perceptive_vae":
        policy = {"depth_image": ObsTermSpec(kind="shadow_depth_image", params={"sensor": _camera()})}
    else:
        policy = dict(refs)
        if variant.family in {"perceptive", "perceptive_hoi"}:
            policy["depth_image"] = ObsTermSpec(kind="shadow_depth_image", params={"sensor": _camera()})
        if variant.family == "beyondmimic":
            policy["base_lin_vel"] = ObsTermSpec(func=mdp.base_lin_vel)
    policy.update(_proprioception(joints, corrupt=True))

    critic = dict(refs)
    if variant.family != "perceptive_vae":
        critic.pop("rotation_ref", None) if variant.family in {"perceptive", "perceptive_hoi"} else None
        critic["link_pos"] = ObsTermSpec(
            kind="shadow_link_position",
            params={
                "motion_reference": motion,
                "asset_cfg": EntityRef("robot", bodies=MOTION_LINKS, preserve_order=True),
            },
        )
        critic["link_rot"] = ObsTermSpec(
            kind="shadow_link_rotation",
            params={
                "motion_reference": motion,
                "asset_cfg": EntityRef("robot", bodies=MOTION_LINKS, preserve_order=True),
            },
        )
        if variant.family in {"perceptive", "perceptive_hoi"}:
            critic["height_scan"] = ObsTermSpec(kind="shadow_height_scan", params={"sensor": _height_scanner()})
        critic["base_lin_vel"] = ObsTermSpec(func=mdp.base_lin_vel)
        critic.update(_proprioception(joints, corrupt=False))
    else:
        critic["depth_image"] = ObsTermSpec(kind="shadow_depth_image", params={"sensor": _camera()})
        critic.update(_proprioception(joints, corrupt=False))
    return {
        "policy": ObsGroupSpec(terms=policy, enable_corruption=True, concatenate_terms=False),
        "critic": ObsGroupSpec(terms=critic, enable_corruption=False, concatenate_terms=False),
    }


def _rewards(variant: ShadowingVariant) -> dict[str, dict[str, RewardTermSpec]]:
    terms = {
        "base_position_imitation_gauss": RewardTermSpec(
            kind="shadow_base_position_gauss", weight=1.0, level=Requirement.REQUIRED
        ),
        "base_rot_imitation_gauss": RewardTermSpec(
            kind="shadow_base_rotation_gauss", weight=1.0, level=Requirement.REQUIRED
        ),
        "link_pos_imitation_gauss": RewardTermSpec(
            kind="shadow_link_position_gauss", weight=1.0, level=Requirement.REQUIRED
        ),
        "link_rot_imitation_gauss": RewardTermSpec(
            kind="shadow_link_rotation_gauss", weight=1.0, level=Requirement.REQUIRED
        ),
        "link_lin_vel_imitation_gauss": RewardTermSpec(
            kind="shadow_link_linear_velocity_gauss", weight=1.0, level=Requirement.REQUIRED
        ),
        "link_ang_vel_imitation_gauss": RewardTermSpec(
            kind="shadow_link_angular_velocity_gauss", weight=1.0, level=Requirement.REQUIRED
        ),
        "action_rate_l2": RewardTermSpec(func=mdp.action_rate_l2, weight=-0.1, level=Requirement.REQUIRED),
        "joint_limit": RewardTermSpec(func=mdp.joint_pos_limits, weight=-10.0, level=Requirement.REQUIRED),
        "undesired_contacts": RewardTermSpec(kind="shadow_undesired_contacts", weight=-1.0, level=Requirement.REQUIRED),
    }
    if variant.family in {"perceptive", "perceptive_vae", "perceptive_hoi"}:
        terms["applied_torque_limits_by_ratio"] = RewardTermSpec(
            kind="shadow_torque_limit_ratio", weight=-0.1, level=Requirement.REQUIRED
        )
    return {"rewards": terms}


def _events(variant: ShadowingVariant) -> dict[str, EventTermSpec]:
    events = {
        "physics_material": EventTermSpec(kind="randomize_friction", mode="startup", params={"ranges": (0.3, 1.6)}),
        "add_joint_default_pos": EventTermSpec(
            kind="randomize_joint_default", mode="startup", params={"range": (-0.01, 0.01)}
        ),
        "base_com": EventTermSpec(kind="randomize_base_com", mode="startup", params={"range": (-0.02, 0.02)}),
    }
    perceptive = variant.family in {"perceptive", "perceptive_vae", "perceptive_hoi"}
    if perceptive:
        events.update(
            randomize_ray_offsets=EventTermSpec(kind="shadow_randomize_ray_offsets", mode="startup"),
            randomize_actuator_gains=EventTermSpec(kind="randomize_actuator_gains", mode="startup"),
            randomize_rigid_body_mass=EventTermSpec(kind="randomize_body_mass", mode="startup"),
        )
    else:
        events["push_robot"] = EventTermSpec(
            kind="push_root_velocity",
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={"velocity_range": (-0.5, 0.5)},
        )
    if variant.family == "perceptive_hoi":
        events["reset_robot"] = EventTermSpec(kind="shadow_reset_robot_from_reference", mode="reset")
        events["reset_rigid_objects_state_by_reference"] = EventTermSpec(
            kind="shadow_reset_objects_from_reference", mode="reset"
        )
        events["update_rigid_objects_state_by_reference"] = EventTermSpec(
            kind="shadow_update_objects_from_reference", mode="interval", interval_range_s=(0.02, 0.02)
        )
    else:
        events["match_motion_ref_with_scene"] = EventTermSpec(kind="shadow_match_reference_origin", mode="reset")
        events["reset_robot"] = EventTermSpec(kind="shadow_reset_robot_from_reference", mode="reset")
    events["bin_fail_counter_smoothing"] = EventTermSpec(
        kind="shadow_smooth_bin_failures", mode="interval", interval_range_s=(0.02, 0.02)
    )
    if variant.play:
        for name in ("push_robot", "bin_fail_counter_smoothing"):
            events.pop(name, None)
    return events


def _terminations(variant: ShadowingVariant, motion: MotionReferenceRef) -> dict[str, DoneTermSpec]:
    terms = {
        "time_out": DoneTermSpec(func=mdp.time_out, time_out=True),
        "base_pos_too_far": DoneTermSpec(kind="shadow_base_position_too_far"),
        "base_pg_too_far": DoneTermSpec(kind="shadow_projected_gravity_too_far"),
        "link_pos_too_far": DoneTermSpec(kind="shadow_link_position_too_far"),
        "dataset_exhausted": DoneTermSpec(
            func=mdp.dataset_exhausted,
            params={"reference_cfg": motion, "reset_without_notice": variant.family == "perceptive_vae"},
            time_out=True,
        ),
        "out_of_border": DoneTermSpec(func=mdp.terrain_out_of_bounds, time_out=True),
    }
    if variant.family in {"perceptive", "perceptive_vae", "perceptive_hoi"}:
        terms = {
            "time_out": terms.pop("time_out"),
            "illegal_reset_contact": DoneTermSpec(kind="shadow_illegal_reset_contact"),
            **terms,
        }
    return terms


def build_shadowing_task(variant: ShadowingVariant) -> TaskSpec:
    robot = make_g1_29dof_robot_spec().overridden(actuator_delay=(0, 2))
    joints = EntityRef("robot", joints=tuple(robot.joint_names), preserve_order=True)
    motion = _motion_reference(variant, tuple(robot.joint_names))
    curriculum = (
        {} if variant.play else {"beyond_adaptive_sampling": CurriculumTermSpec(kind="shadow_adaptive_sampling")}
    )
    agent_overrides = {"experiment_name": "g1_perceptive_shadowing_one_motion"} if variant.one_motion else {}
    reference_num_envs = {
        "whole_body": {"isaacsim": 4096, "mjlab": 2048},
        "perceptive": {"isaacsim": 1024, "mjlab": 1024},
        "perceptive_vae": {"isaacsim": 4096, "mjlab": 4096},
        "perceptive_hoi": {"isaacsim": 4096, "mjlab": 4096},
        "beyondmimic": {"isaacsim": 4096, "mjlab": 4096},
    }[variant.family]
    return TaskSpec(
        task_id=variant.task_id,
        robot=robot,
        scene=_scene(variant, motion),
        sim=SimSpec(
            physics_dt=0.005,
            decimation=4,
            episode_length_s=10.0,
            profiles={
                "isaacsim": {},
                "mjlab": {
                    "iterations": 10,
                    "ls_iterations": 20,
                    "njmax": 1200,
                    "nconmax": None,
                },
            },
        ),
        mdp=MdpSpec(
            observations=_observations(variant, joints, motion),
            actions={"joint_pos": ActionTermSpec(kind="joint_position", target=joints, params={"scale": 0.5})},
            commands=_commands(variant),
            rewards=_rewards(variant),
            events=_events(variant),
            curriculum=curriculum,
            terminations=_terminations(variant, motion),
        ),
        agent=AgentSpec(runner=RUNNERS[variant.family], overrides=agent_overrides),
        engines=("isaacsim", "mjlab"),
        engine_extras={
            "isaacsim": {
                "shadowing_family": variant.family,
                "play": variant.play,
                "reference_num_envs": reference_num_envs["isaacsim"],
            },
            "mjlab": {
                "shadowing_family": variant.family,
                "play": variant.play,
                "reference_num_envs": reference_num_envs["mjlab"],
            },
        },
    )


__all__ = ["MOTION_LINKS", "ShadowingVariant", "build_shadowing_task"]
