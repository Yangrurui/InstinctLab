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
    RigidObjectRef,
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
    "whole_body": {
        "isaacsim": "~/Datasets/NoKov-Marslab-Motions-instinctnpz/20251016_diveroll4_single",
        # Both references consume the same released clip.  InstinctMJ's source spells this as a
        # developer-specific ~/Xyk path; bind the engine to the portable compatibility root used
        # on this server instead of requiring that private directory layout.
        "mjlab": "~/Datasets/NoKov-Marslab-Motions-instinctnpz/20251016_diveroll4_single",
    },
    "perceptive": {
        "isaacsim": "~/Datasets/deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
        "mjlab": "~/Datasets/deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
    },
    "perceptive_vae": {
        "isaacsim": "~/Datasets/NoKov-Marslab-Motions-instinctnpz/20251116_50cm_kneeClimbStep1",
        "mjlab": "~/Xyk/Datasets/20260317_50cm_kneeClimbStep1_projectInstinct",
    },
    "perceptive_hoi": {
        "isaacsim": "/localhdd/Datasets/OMOMO/retargeted",
        "mjlab": "~/Datasets/OMOMO/retargeted_omniretarget_instinctmj_torso_v10_object_xy_align_foot_lock",
    },
    "beyondmimic": {
        "isaacsim": "~/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz",
        "mjlab": "~/Xyk/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz",
    },
}

BEYONDMIMIC_SELECTED_MOTION = "sprint1_subject2_retargetted.npz"

HOI_OBJECT_SCALES = {
    "floorlamp": 1.55 * 0.3793,
    "largebox": 1.55 * 0.3486,
    "whitechair": 1.55 * 0.3129,
    "trashcan": 1.55 * 0.2326,
    "smalltable": 1.55 * 0.0162,
    "suitcase": 1.55 * 0.3672,
}


def _hoi_objects() -> tuple[RigidObjectRef, ...]:
    return tuple(
        RigidObjectRef(
            name=name,
            mesh=f"/localhdd/Datasets/OMOMO/data/captured_objects/{name}_cleaned_simplified.obj",
            engine_meshes={
                "isaacsim": f"/localhdd/Datasets/OMOMO/data/captured_objects/{name}_cleaned_simplified.obj",
                "mjlab": f"~/Datasets/OMOMO/data/captured_objects/{name}_cleaned_simplified.obj",
            },
            scale=(scale, scale, scale),
        )
        for name, scale in HOI_OBJECT_SCALES.items()
    )


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
    start_range = (0.0, 0.8) if variant.family in {"whole_body", "beyondmimic"} else (0.0, 0.0)
    if variant.play and variant.family != "whole_body":
        start_range = (0.0, 0.0)
    terrain = variant.family in {"perceptive", "perceptive_vae"}
    binned = not variant.one_motion and not (variant.play and variant.family != "whole_body")
    return MotionReferenceRef(
        name="motion_reference",
        clip=MOTION_PATHS[variant.family]["isaacsim"],
        engine_clips=MOTION_PATHS[variant.family],
        joints=joints,
        links=MOTION_LINKS,
        scene_objects=(
            (
                "floorlamp",
                "largebox",
                "whitechair",
                "trashcan",
                "smalltable",
                "suitcase",
            )
            if variant.family == "perceptive_hoi"
            else ()
        ),
        num_frames=1 if current_only else 10,
        frame_interval_s=frame_interval,
        update_period=0.02,
        data_start_from="current_time",
        clip_target_fps=50.0,
        velocity_method="frontbackward",
        start_range=start_range,
        dataset_kind=("terrain" if terrain else ("omomo" if variant.family == "perceptive_hoi" else "retargetted")),
        metadata_yaml="metadata.yaml" if terrain else None,
        selected_files=(BEYONDMIMIC_SELECTED_MOTION,) if variant.family == "beyondmimic" else (),
        first_motion_only=variant.one_motion,
        sampling_strategy=(
            "concat_motion_bins"
            if variant.family in {"perceptive", "perceptive_vae", "perceptive_hoi"}
            and not variant.one_motion
            and not variant.play
            else "independent"
        ),
        motion_bin_length_s=1.0 if binned else None,
        ensure_link_below_zero_ground=variant.family == "perceptive",
        motion_start_height_offset=0.1 if variant.family == "perceptive" else 0.0,
        # main disables the terrain-motion lift for Isaac Perceptive, while
        # InstinctMJ intentionally keeps it to reduce early terrain contacts.
        engine_overrides=(
            {
                "isaacsim": {
                    "ensure_link_below_zero_ground": False,
                    "motion_start_height_offset": 0.0,
                }
            }
            if variant.family == "perceptive"
            else {}
        ),
        exhaustion="freeze_last_and_flag",
        quaternion="wxyz",
        symmetric_augmentation=None,
    )


def _camera(*, include_objects: bool = False) -> RayCasterRef:
    return RayCasterRef(
        name="camera",
        attach="torso_link",
        offset=(0.0487988662332928, 0.015, 0.4378029937970051),
        offset_rot=(0.9135367613482678, 0.004363309284746571, 0.4067366430758002, 0.0),
        offset_convention="world",
        pattern=RayPatternRef(
            kind="pinhole",
            # Both references render 27x48, crop two pixels on every edge, then
            # resize the policy image to 18x32.
            width=48,
            height=27,
            horizontal_fov_deg=87.0,
            vertical_fov_deg=58.0,
            focal_length=1.0,
        ),
        hit=(
            "terrain",
            *G1_29DOF_LINKS,
            *(HOI_OBJECT_SCALES if include_objects else ()),
        ),
        ray_alignment="base",
        miss="infinity",
        max_distance=1.0e6,
        min_distance=0.05,
        crop=(2, 2, 2, 2),
        update_period=1.0 / 60.0,
    )


def _height_scanner() -> RayCasterRef:
    return RayCasterRef(
        name="height_scanner",
        mode="terrain_height",
        attach="torso_link",
        offset=(0.0, 0.0, 20.0),
        pattern=RayPatternRef(kind="grid", resolution=0.1, size=(1.6, 1.0)),
        hit="terrain",
        ray_alignment="yaw",
        miss="infinity",
        max_distance=1.0e6,
        engine_max_distances={"isaacsim": 1.0e6, "mjlab": 5.0},
        update_period=0.02,
    )


_NON_SUPPORT_CONTACTS = (
    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
)


def _contact_sensor(variant: ShadowingVariant, *, undesired_subset: bool = False) -> ContactSensorRef:
    perceptive = variant.family in {"perceptive", "perceptive_vae", "perceptive_hoi"}
    return ContactSensorRef(
        name="contact_forces" if perceptive else "undesired_contact_forces",
        elements=_NON_SUPPORT_CONTACTS if undesired_subset else ".*",
        track_air_time=perceptive,
        air_time_force_threshold=1.0,
        history_length=3,
    )


def _scene(variant: ShadowingVariant, motion: MotionReferenceRef) -> SceneSpec:
    perceptive = variant.family in {"perceptive", "perceptive_vae", "perceptive_hoi"}
    rays: list[RayCasterRef] = []
    if perceptive:
        rays.append(_camera(include_objects=variant.family == "perceptive_hoi"))
    if variant.family in {"perceptive", "perceptive_hoi"}:
        rays.append(_height_scanner())
    contacts = (_contact_sensor(variant),)
    terrain = TerrainSpec(kind="plane")
    if variant.family in {"perceptive", "perceptive_vae"}:
        terrain = TerrainSpec(
            kind="shadow_motion_matched",
            params={
                "engine_paths": MOTION_PATHS[variant.family],
                "metadata_yaml": "metadata.yaml",
            },
        )
    return SceneSpec(
        terrain=terrain,
        contact_sensors=contacts,
        ray_casters=tuple(rays),
        motion_references=(motion,),
        rigid_objects=_hoi_objects() if variant.family == "perceptive_hoi" else (),
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
        params = {
            "motion_reference": "motion_reference",
            "current_state_command": False,
        }
        if name == "position_ref_command":
            params.update(realtime_mode=True, anchor_frame="robot")
        elif name == "position_b_ref_command":
            params.update(realtime_mode=True, anchor_frame="reference")
        elif name == "rotation_ref_command":
            params.update(realtime_mode=True, in_base_frame=True, rotation_mode="tannorm")
        commands[name] = CommandTermSpec(kind=kinds[name], params=params)
    return commands


def _reference_observations(*, policy: bool, has_anchor_command: bool = True) -> dict[str, ObsTermSpec]:
    return {
        "joint_pos_ref": ObsTermSpec(
            func=mdp.generated_commands,
            params={"command_name": "joint_pos_ref_command"},
        ),
        "joint_vel_ref": ObsTermSpec(
            func=mdp.generated_commands,
            params={"command_name": "joint_vel_ref_command"},
        ),
        "position_ref": ObsTermSpec(
            func=mdp.generated_commands,
            params={
                "command_name": "position_b_ref_command" if policy and has_anchor_command else "position_ref_command"
            },
            noise=NoiseSpec("uniform", -0.25, 0.25) if policy else None,
        ),
        "rotation_ref": ObsTermSpec(
            func=mdp.generated_commands,
            params={"command_name": "rotation_ref_command"},
            noise=NoiseSpec("uniform", -0.05, 0.05) if policy else None,
        ),
    }


def _proprioception(
    joints: EntityRef, *, corrupt: bool, history_length: int, include_projected_gravity: bool = True
) -> dict[str, ObsTermSpec]:
    noise = (lambda lo, hi: NoiseSpec("uniform", lo, hi)) if corrupt else (lambda lo, hi: None)
    terms = {
        "base_ang_vel": ObsTermSpec(func=mdp.base_ang_vel, noise=noise(-0.2, 0.2), history_length=history_length),
        "joint_pos": ObsTermSpec(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": joints},
            noise=noise(-0.01, 0.01),
            history_length=history_length,
        ),
        "joint_vel": ObsTermSpec(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": joints},
            noise=noise(-0.5, 0.5),
            history_length=history_length,
        ),
        "last_action": ObsTermSpec(func=mdp.last_action, history_length=history_length),
    }
    if include_projected_gravity:
        # Keep main's declaration order: gravity precedes the other proprioception.
        return {
            "projected_gravity": ObsTermSpec(
                func=mdp.projected_gravity,
                noise=noise(-0.05, 0.05),
                history_length=history_length,
            ),
            **terms,
        }
    return terms


def _observations(variant: ShadowingVariant, joints: EntityRef, motion: MotionReferenceRef) -> dict[str, ObsGroupSpec]:
    policy_refs = _reference_observations(policy=True, has_anchor_command=variant.family != "beyondmimic")
    critic_refs = _reference_observations(policy=False)
    history_length = 8 if variant.family in {"perceptive", "perceptive_vae", "perceptive_hoi"} else 0
    policy: dict[str, ObsTermSpec]
    if variant.family == "perceptive_vae":
        policy = {
            "depth_image": ObsTermSpec(
                kind="shadow_depth_image",
                params={
                    "sensor": _camera(include_objects=variant.family == "perceptive_hoi"),
                    "history_length": 10,
                    "history_skip_frames": 3,
                    "num_output_frames": 4,
                    "resize_shape": (18, 32),
                    "normalization_range": (0.0, 2.0),
                },
            )
        }
    else:
        policy = dict(policy_refs)
        if variant.family in {"perceptive", "perceptive_hoi"}:
            policy["depth_image"] = ObsTermSpec(
                kind="shadow_depth_image",
                params={"sensor": _camera(include_objects=variant.family == "perceptive_hoi")},
            )
        if variant.family == "beyondmimic":
            policy["base_lin_vel"] = ObsTermSpec(kind="shadow_base_linear_velocity")
    policy.update(_proprioception(joints, corrupt=True, history_length=history_length))

    critic = dict(critic_refs)
    if variant.family != "perceptive_vae":
        (
            critic.pop("rotation_ref", None)
            if variant.family
            in {
                "perceptive",
                "perceptive_hoi",
            }
            else None
        )
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
        critic["base_lin_vel"] = ObsTermSpec(kind="shadow_base_linear_velocity", history_length=history_length)
        critic.update(
            _proprioception(
                joints, corrupt=False, history_length=history_length, include_projected_gravity=False
            )
        )
    else:
        critic["depth_image"] = ObsTermSpec(
            kind="shadow_depth_image", params={"sensor": _camera(include_objects=False)}
        )
        critic.update(
            _proprioception(
                joints, corrupt=False, history_length=history_length, include_projected_gravity=False
            )
        )
    return {
        "policy": ObsGroupSpec(terms=policy, enable_corruption=True, concatenate_terms=False),
        "critic": ObsGroupSpec(terms=critic, enable_corruption=False, concatenate_terms=False),
    }


def _rewards(variant: ShadowingVariant) -> dict[str, dict[str, RewardTermSpec]]:
    terms = {
        "base_position_imitation_gauss": RewardTermSpec(
            kind="shadow_base_position_gauss",
            weight=0.5,
            params={"std": 0.3},
            level=Requirement.REQUIRED,
        ),
        "base_rot_imitation_gauss": RewardTermSpec(
            kind="shadow_base_rotation_gauss",
            weight=0.5,
            params={"std": 0.4, "difference_type": "axis_angle"},
            level=Requirement.REQUIRED,
        ),
        "link_pos_imitation_gauss": RewardTermSpec(
            kind="shadow_link_position_gauss",
            weight=1.0,
            params={
                "combine_method": "mean_prod",
                "in_base_frame": False,
                "in_relative_world_frame": True,
                "std": 0.3,
            },
            level=Requirement.REQUIRED,
        ),
        "link_rot_imitation_gauss": RewardTermSpec(
            kind="shadow_link_rotation_gauss",
            weight=1.0,
            params={
                "combine_method": "mean_prod",
                "in_base_frame": False,
                "in_relative_world_frame": True,
                "std": 0.4,
            },
            level=Requirement.REQUIRED,
        ),
        "link_lin_vel_imitation_gauss": RewardTermSpec(
            kind="shadow_link_linear_velocity_gauss",
            weight=1.0,
            params={"combine_method": "mean_prod", "std": 1.0},
            level=Requirement.REQUIRED,
        ),
        "link_ang_vel_imitation_gauss": RewardTermSpec(
            kind="shadow_link_angular_velocity_gauss",
            weight=1.0,
            params={"combine_method": "mean_prod", "std": 3.14},
            level=Requirement.REQUIRED,
        ),
        "action_rate_l2": RewardTermSpec(func=mdp.action_rate_l2, weight=-0.1, level=Requirement.REQUIRED),
        "joint_limit": RewardTermSpec(func=mdp.joint_pos_limits, weight=-10.0, level=Requirement.REQUIRED),
        "undesired_contacts": RewardTermSpec(
            kind="shadow_undesired_contacts",
            weight=-0.1,
            params={
                "threshold": 1.0,
                "sensor": _contact_sensor(variant, undesired_subset=True),
            },
            level=Requirement.REQUIRED,
        ),
    }
    if variant.family in {"perceptive", "perceptive_vae", "perceptive_hoi"}:
        terms["applied_torque_limits_by_ratio"] = RewardTermSpec(
            kind="shadow_torque_limit_ratio", weight=-0.05, level=Requirement.REQUIRED
        )
    return {"rewards": terms}


def _events(variant: ShadowingVariant) -> dict[str, EventTermSpec]:
    perceptive = variant.family in {"perceptive", "perceptive_vae", "perceptive_hoi"}
    isaac_material = {
        "static_friction_range": (1.25, 2.0) if perceptive else (0.3, 1.6),
        "dynamic_friction_range": (1.2, 1.8) if perceptive else (0.3, 1.2),
        "restitution_range": (0.0, 0.5),
        "num_buckets": 64,
    }
    if perceptive:
        isaac_material["make_consistent"] = True
    mj_material = (
        {
            "ranges": {0: (1.25, 2.0), 1: (1.2, 1.8), 2: (0.0, 0.5)},
            "operation": "abs",
            "distribution": "uniform",
        }
        if perceptive
        else {"ranges": (0.3, 1.6), "operation": "abs", "shared_random": True}
    )
    events = {
        "physics_material": EventTermSpec(
            kind="randomize_friction",
            mode="startup",
            target=EntityRef("robot", bodies=".*"),
            engine_params={
                "isaacsim": isaac_material,
                "mjlab": mj_material,
            },
        ),
        "add_joint_default_pos": EventTermSpec(
            kind="randomize_joint_default",
            mode="startup",
            params={"range": (-0.01, 0.01)},
        ),
        "base_com": EventTermSpec(
            kind="randomize_base_com",
            mode="startup",
            target=EntityRef("robot", bodies=("torso_link",)),
            params={
                "com_range": {
                    "x": (-0.025, 0.025),
                    "y": (-0.05, 0.05),
                    "z": (-0.05, 0.05),
                }
            },
        ),
    }
    if perceptive:
        events.update(
            randomize_ray_offsets=EventTermSpec(
                kind="shadow_randomize_ray_offsets",
                mode="startup",
                params={
                    "offset_pose_ranges": {
                        "x": (-0.01, 0.01),
                        "y": (-0.01, 0.01),
                        "z": (-0.01, 0.01),
                        "roll": (-0.034906585, 0.034906585),
                        "pitch": (-0.174532925, 0.174532925),
                        "yaw": (-0.034906585, 0.034906585),
                    }
                },
            ),
            randomize_actuator_gains=EventTermSpec(
                kind="randomize_actuator_gains",
                mode="startup",
                params={
                    "stiffness_range": (0.8, 1.2),
                    "damping_range": (0.9, 1.1),
                    "operation": "scale",
                },
            ),
            randomize_rigid_body_mass=EventTermSpec(
                kind="shadow_randomize_body_inertia",
                mode="startup",
                target=EntityRef(
                    "robot",
                    bodies=("torso_link", ".*ankle.*", ".*wrist.*"),
                ),
                params={"add_range": (0.8, 1.2), "operation": "scale"},
            ),
        )
    else:
        events["push_robot"] = EventTermSpec(
            kind="push_root_velocity",
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.2, 0.2),
                    "roll": (-0.52, 0.52),
                    "pitch": (-0.52, 0.52),
                    "yaw": (-0.78, 0.78),
                }
            },
        )
    reset_params = {
        "position_offset": (0.0, 0.0, 0.0),
        "dof_vel_ratio": 1.0,
        "base_lin_vel_ratio": 1.0,
        "base_ang_vel_ratio": 1.0,
        "randomize_joint_pos_range": (-0.1, 0.1),
    }
    if perceptive:
        reset_params.update(
            randomize_pose_range={
                "x": (-0.15, 0.15),
                "y": (-0.15, 0.15),
                "z": (0.0, 0.0),
            },
            randomize_velocity_range={},
        )
    else:
        reset_params.update(
            randomize_pose_range={
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.01, 0.01),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.2, 0.2),
            },
            randomize_velocity_range={
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            },
        )
    if variant.family == "perceptive_hoi":
        events["reset_robot"] = EventTermSpec(
            kind="shadow_reset_robot_from_reference",
            mode="reset",
            params=reset_params,
            engine_params={"isaacsim": {"root_velocity_frame": "com"}},
        )
        events["reset_rigid_objects_state_by_reference"] = EventTermSpec(
            kind="shadow_reset_objects_from_reference",
            mode="reset",
            engine_params={"isaacsim": {"root_velocity_frame": "com"}},
        )
        events["update_rigid_objects_state_by_reference"] = EventTermSpec(
            kind="shadow_update_objects_from_reference",
            mode="interval",
            interval_range_s=(0.02, 0.02),
            params={"invalid_object_pos": (0.0, 0.0, -1.0)},
            engine_params={"isaacsim": {"root_velocity_frame": "com"}},
        )
    else:
        events["match_motion_ref_with_scene"] = EventTermSpec(kind="shadow_match_reference_origin", mode="startup")
        events["reset_robot"] = EventTermSpec(
            kind="shadow_reset_robot_from_reference",
            mode="reset",
            params=reset_params,
            engine_params={"isaacsim": {"root_velocity_frame": "com"}},
        )
    events["bin_fail_counter_smoothing"] = EventTermSpec(
        kind="shadow_smooth_bin_failures",
        mode="interval",
        interval_range_s=(0.02, 0.02),
    )
    if variant.play:
        for name in ("push_robot", "bin_fail_counter_smoothing"):
            events.pop(name, None)
    return events


def _terminations(variant: ShadowingVariant, motion: MotionReferenceRef) -> dict[str, DoneTermSpec]:
    terms = {
        "time_out": DoneTermSpec(func=mdp.time_out, time_out=True),
        "base_pos_too_far": DoneTermSpec(
            kind="shadow_base_position_too_far",
            params={
                "distance_threshold": 0.25,
                "check_at_keyframe_threshold": -1,
                "print_reason": False,
                "height_only": True,
            },
        ),
        "base_pg_too_far": DoneTermSpec(
            kind="shadow_projected_gravity_too_far",
            params={
                "projected_gravity_threshold": 0.8,
                "check_at_keyframe_threshold": -1,
                "z_only": False,
                "print_reason": False,
            },
        ),
        "link_pos_too_far": DoneTermSpec(
            kind="shadow_link_position_too_far",
            params={
                "distance_threshold": 0.25,
                "in_base_frame": False,
                "check_at_keyframe_threshold": -1,
                "height_only": True,
                "print_reason": False,
            },
            target=EntityRef(
                "motion_reference",
                bodies=(
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ),
                preserve_order=True,
            ),
        ),
        "dataset_exhausted": DoneTermSpec(
            func=mdp.dataset_exhausted,
            params={
                "sensor": motion,
                "reset_without_notice": variant.family == "perceptive_vae",
            },
            time_out=True,
        ),
        "out_of_border": DoneTermSpec(
            func=mdp.terrain_out_of_bounds,
            params={"distance_buffer": 0.1},
            time_out=True,
        ),
    }
    if variant.family in {"perceptive", "perceptive_vae", "perceptive_hoi"}:
        terms = {
            "time_out": terms.pop("time_out"),
            "illegal_reset_contact": DoneTermSpec(
                kind="shadow_illegal_reset_contact",
                params={
                    "sensor": _contact_sensor(variant, undesired_subset=True),
                    "threshold": 500.0,
                    "episode_length_threshold": 2,
                },
            ),
            **terms,
        }
    if variant.family == "perceptive_hoi":
        terms.pop("out_of_border")
    return terms


def build_shadowing_task(variant: ShadowingVariant) -> TaskSpec:
    # Both final G1 references replace their base robot with the BeyondMimic plant. On Isaac this
    # is ImplicitPD; on MJLab it is BuiltinPD. Neither final override uses the delayed tables.
    # main's effective shadowing ArticulationCfg leaves UrdfFileCfg's
    # ``merge_fixed_joints=True`` default in force.  The shared catalog keeps fixed joints for
    # locomotion/MJCF inventory purposes, so shadowing carries this Isaac-only plant override.
    robot = make_g1_29dof_robot_spec().overridden(
        import_options={"isaacsim": {"merge_fixed_joints": True}},
    )
    joints = EntityRef("robot", joints=tuple(robot.joint_names), preserve_order=True)
    action_scale = {joint.name: joint.action_scale for joint in robot.joint_properties}
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
    isaac_profile = {
        # All effective main shadowing configs set the terrain material as the
        # simulation default and double Isaac Lab's rigid-patch budget, including
        # the plane tasks.  At 4096 envs these are plant settings, not tuning hints.
        "use_terrain_physics_material": True,
        "gpu_max_rigid_patch_count": 10 * 2**15,
    }
    if variant.family in {"perceptive", "perceptive_vae", "perceptive_hoi"}:
        # Effective main Perceptive/HOI configs also raise both contact storage
        # and the collision stack from Isaac Lab's much smaller defaults.
        isaac_profile.update(
            gpu_max_rigid_contact_count=2**27,
            gpu_collision_stack_size=2**27,
        )
    mjlab_profile = {
        "iterations": 10,
        "ls_iterations": 20,
        "njmax": 1200,
        "nconmax": None,
    }
    if variant.family == "perceptive":
        mjlab_profile.update(
            njmax=700,
            nconmax=128,
            contact_sensor_maxmatch=128,
            ccd_iterations=128,
            jacobian="sparse",
        )
    elif variant.family == "perceptive_vae":
        mjlab_profile.update(
            njmax=512,
            nconmax=128,
            contact_sensor_maxmatch=128,
            ccd_iterations=128,
            jacobian="sparse",
        )
    elif variant.family == "perceptive_hoi":
        mjlab_profile.update(
            njmax=700,
            nconmax=256,
            contact_sensor_maxmatch=256,
            ccd_iterations=128,
            jacobian="sparse",
        )
    elif variant.family == "beyondmimic":
        mjlab_profile.update(
            njmax=None if variant.play else 350,
            nconmax=None if variant.play else 100,
            contact_sensor_maxmatch=500 if variant.play else 100,
            ccd_iterations=80,
        )
    return TaskSpec(
        task_id=variant.task_id,
        robot=robot,
        scene=_scene(variant, motion),
        sim=SimSpec(
            physics_dt=0.005,
            decimation=4,
            episode_length_s=10.0,
            profiles={
                "isaacsim": isaac_profile,
                "mjlab": mjlab_profile,
            },
        ),
        mdp=MdpSpec(
            observations=_observations(variant, joints, motion),
            actions={
                "joint_pos": ActionTermSpec(
                    kind="joint_position",
                    target=joints,
                    params={"scale": action_scale, "use_default_offset": True},
                )
            },
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
