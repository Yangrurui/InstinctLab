"""Proprioceptive parkour target-following for G1, declared once for every engine.

This is the core of ``Instinct-Parkour-Target-G1``: the ten-sub-terrain procedural curriculum
grid, a terrain-aware pose-velocity command, parkour's reward set, terminations and curriculum.
Nothing here imports an engine.

The ankle height scanners, ``feet_at_plane``, the depth camera, the motion-reference
clip, the AMP observation groups, the ankle VolumePoints cloud, virtual-obstacle
registration, and ``volume_points_penetration`` are in.

AMP uses ``amp_policy`` / ``amp_reference`` — the same function of kinematic state, two
extractors. The agent is WasabiPPO with a 4-expert MoE; ``depth_image`` is routed through
a Conv2d encoder (``component_names=["depth_image"]``), not flattened into the MLP.
``dataset_exhausted`` stays out: with ``reset_without_notice=True`` it reports 0.
Exhaustion is visible on the sensor (``validity``, ``exhausted_count``) instead.
"""

from __future__ import annotations

import math
from pathlib import Path

from instinctlab import mdp
from instinctlab.assets.unitree_g1.isaacsim import G1_29DOF_LINKS, make_g1_29dof_robot_spec
from instinctlab.sim.robot_spec import RobotSpec
from instinctlab.spec import (
    ActionTermSpec,
    AgentSpec,
    CommandTermSpec,
    ContactSensorRef,
    CurriculumTermSpec,
    DoneTermSpec,
    EntityRef,
    EventTermSpec,
    Grid3dPointsRef,
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
    VirtualObstacleRef,
    VolumePointsRef,
)
from instinctlab.spec.capability import Requirement

_PARKOUR_DIR = Path(__file__).resolve().parents[2]
_SHOE_URDF = _PARKOUR_DIR / "urdf" / "g1_29dof_torsoBase_popsicle_with_shoe.urdf"
_SHOE_XML = _PARKOUR_DIR / "mjcf" / "g1_29dof_torsoBase_popsicle_with_shoe.xml"

ROBOT = EntityRef("robot", bodies=".*")
COMMAND = "base_velocity"

# One broad contact sensor, sliced by terms. That is what lets the same declaration work
# against Isaac Lab's one prim-path sensor and mjlab's several narrow ones.
FEET_CONTACT = ContactSensorRef(name="contact_forces", elements=".*_ankle_roll_link")
TORSO_CONTACT = ContactSensorRef(name="contact_forces", elements="torso_link")
UNDESIRED_CONTACT = ContactSensorRef(name="contact_forces", elements="(?!.*_ankle_roll_link).*")

_FOOT_PATTERN = RayPatternRef(kind="grid", resolution=0.12, size=(0.12, 0.0))
LEFT_HEIGHT_SCANNER = RayCasterRef(
    name="left_height_scanner",
    attach="left_ankle_roll_link",
    offset=(0.04, 0.0, 20.0),
    pattern=_FOOT_PATTERN,
    hit="terrain",
    ray_alignment="yaw",
    miss="infinity",
)
RIGHT_HEIGHT_SCANNER = RayCasterRef(
    name="right_height_scanner",
    attach="right_ankle_roll_link",
    offset=(0.04, 0.0, 20.0),
    pattern=_FOOT_PATTERN,
    hit="terrain",
    ray_alignment="yaw",
    miss="infinity",
)

# Isaac parkour adds these links via ``get_link_prim_targets(G1_29DOF_LINKS)``.
# Named bodies, not a geom-group mask: on the G1, group 2 is the visual shoe
# and group 3 is the collision capsule.
DEPTH_CAMERA = RayCasterRef(
    name="camera",
    attach="torso_link",
    offset=(0.0487988662332928, 0.01, 0.4378029937970051),
    offset_rot=(0.9135367613482678, 0.004363309284746571, 0.4067366430758002, 0.0),
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
    crop=(18, 0, 16, 16),
    update_period=0.02,
)

# G1 shoe sole, attach-body local frame. Both sources retune z after the
# generic VolumePointsCfg defaults (-0.04, 0.0).
LEG_VOLUME_POINTS = VolumePointsRef(
    name="leg_volume_points",
    attach=("left_ankle_roll_link", "right_ankle_roll_link"),
    grid=Grid3dPointsRef(
        x_min=-0.025,
        x_max=0.12,
        x_num=10,
        y_min=-0.03,
        y_max=0.03,
        y_num=5,
        z_min=-0.063,
        z_max=-0.023,
        z_num=2,
    ),
)
PARKOUR_EDGE_CYLINDERS = VirtualObstacleRef(
    name="edges",
    kind="greedy_edge_cylinder",
    cylinder_radius=0.05,
    min_points=2,
)

# Same clip and links as the legacy AMP task. Joints are named in the canonical
# depth-first order (D1); the published npz is legs-first and is remapped by name.
PARKOUR_MOTION_CLIP = "~/Datasets/parkour_release/parkour_motion_reference/parkour_motion_without_run_retargetted.npz"
PARKOUR_MOTION_LINKS = (
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

_OBSTACLE_VEL = {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)}
SHARED_VELOCITY_RANGES = {
    "perlin_rough": {"lin_vel_x": (0.45, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
    "perlin_rough_stand": {"lin_vel_x": (0.0, 0.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)},
    "square_gaps": dict(_OBSTACLE_VEL),
    "pyramid_stairs": dict(_OBSTACLE_VEL),
    "pyramid_stairs_high": dict(_OBSTACLE_VEL),
    "pyramid_stairs_inv": dict(_OBSTACLE_VEL),
    "pyramid_stairs_inv_high": dict(_OBSTACLE_VEL),
    "boxes": dict(_OBSTACLE_VEL),
    "hf_pyramid_slope_inv": dict(_OBSTACLE_VEL),
}


def _canonical_joints(robot: RobotSpec) -> EntityRef:
    """Every joint, named explicitly, in the canonical depth-first order. Decision D1 lives here.

    A lone ``".*"`` selects the same twenty-nine joints, so it is tempting to write. It does not
    order them: ``preserve_order`` orders a selection by the *patterns* it was given, and one
    pattern leaves the entity's own order in place -- PhysX's breadth-first walk on Isaac Sim, the
    model file's order on mjlab, neither of which is the depth-first order this project declares.
    Naming the joints is what makes the joint axis mean the same thing on both engines.

    Both the action term and the two joint observations take this, because pinning one without the
    other would leave a policy whose inputs and outputs are indexed differently per engine.
    """
    return EntityRef("robot", joints=tuple(robot.joint_names), preserve_order=True)


def _action_scale(robot: RobotSpec) -> dict[str, float]:
    """Per-joint action scale, taken from the robot catalog rather than from the engine."""
    return {joint.name: joint.action_scale for joint in robot.joint_properties}


def _velocity_limits(robot: RobotSpec) -> tuple[float, ...]:
    """Per-joint velocity caps in catalog order, the source ``joint_vel_limits`` reads."""
    return tuple(joint.velocity_limit for joint in robot.joint_properties)


def _policy_observations(*, corrupt: bool, joints: EntityRef) -> ObsGroupSpec:
    """Proprioception plus the depth camera; the critic also sees linear velocity.

    One function for both groups because they differ in exactly two ways -- the critic sees base
    linear velocity and no noise -- and writing them separately is how the two silently drift apart
    in term order. The MoE encoder takes ``depth_image`` out of the flat MLP (8×18×32 → 128);
    proprioception stays 768 / 792.
    """
    noise = (lambda lo, hi: NoiseSpec("uniform", lo, hi)) if corrupt else (lambda lo, hi: None)
    terms = {
        "base_ang_vel": ObsTermSpec(func=mdp.base_ang_vel, noise=noise(-0.2, 0.2), scale=0.25, history_length=8),
        "projected_gravity": ObsTermSpec(func=mdp.projected_gravity, noise=noise(-0.05, 0.05), history_length=8),
        "velocity_commands": ObsTermSpec(
            func=mdp.generated_commands, params={"command_name": COMMAND}, history_length=8
        ),
        "joint_pos": ObsTermSpec(
            func=mdp.joint_pos_rel, noise=noise(-0.01, 0.01), params={"asset_cfg": joints}, history_length=8
        ),
        "joint_vel": ObsTermSpec(
            func=mdp.joint_vel_rel,
            noise=noise(-0.5, 0.5),
            scale=0.05,
            params={"asset_cfg": joints},
            history_length=8,
        ),
        "actions": ObsTermSpec(func=mdp.last_action, history_length=8),
        "depth_image": ObsTermSpec(
            func=mdp.DelayedDepthImage,
            params={
                "sensor": DEPTH_CAMERA,
                "history_skip_frames": 5,
                "num_output_frames": 8,
                "delayed_frame_ranges": (0, 1),
                "history_length": 37,
                "blur_kernel_size": 3,
                "blur_sigma": 1.0,
            },
            history_length=0,
            level=Requirement.REQUIRED,
        ),
    }
    if not corrupt:
        terms = {"base_lin_vel": ObsTermSpec(func=mdp.base_lin_vel, history_length=8), **terms}
    return ObsGroupSpec(terms=terms, enable_corruption=corrupt, concatenate_terms=False)


AMP_HISTORY = 10


def _amp_policy_observations(joints: EntityRef) -> ObsGroupSpec:
    """Live-robot AMP state. Same terms, same frames, same order as the reference group.

    History is the observation manager's last 10 env steps, not the clip's 10-frame
    look-ahead. Joints are named in the canonical depth-first order (D1).
    """
    return ObsGroupSpec(
        terms={
            "projected_gravity": ObsTermSpec(func=mdp.projected_gravity, history_length=AMP_HISTORY),
            "joint_pos_rel": ObsTermSpec(
                func=mdp.joint_pos_rel, params={"asset_cfg": joints}, history_length=AMP_HISTORY
            ),
            "joint_vel": ObsTermSpec(
                func=mdp.joint_vel_rel, scale=0.05, params={"asset_cfg": joints}, history_length=AMP_HISTORY
            ),
            "base_lin_vel": ObsTermSpec(func=mdp.base_lin_vel, history_length=AMP_HISTORY),
            "base_ang_vel": ObsTermSpec(func=mdp.base_ang_vel, history_length=AMP_HISTORY),
        },
        enable_corruption=False,
        concatenate_terms=False,
    )


def _amp_reference_observations(joints: EntityRef, sensor: MotionReferenceRef) -> ObsGroupSpec:
    """Clip AMP state. The same function as :func:`_amp_policy_observations` of the clip."""
    params = {"sensor": sensor, "asset_cfg": joints}
    return ObsGroupSpec(
        terms={
            "projected_gravity": ObsTermSpec(
                func=mdp.projected_gravity_from_reference, params=params, history_length=AMP_HISTORY
            ),
            "joint_pos_rel": ObsTermSpec(
                func=mdp.joint_pos_rel_from_reference, params=params, history_length=AMP_HISTORY
            ),
            "joint_vel": ObsTermSpec(
                func=mdp.joint_vel_rel_from_reference, scale=0.05, params=params, history_length=AMP_HISTORY
            ),
            "base_lin_vel": ObsTermSpec(
                func=mdp.base_lin_vel_from_reference, params=params, history_length=AMP_HISTORY
            ),
            "base_ang_vel": ObsTermSpec(
                func=mdp.base_ang_vel_from_reference, params=params, history_length=AMP_HISTORY
            ),
        },
        enable_corruption=False,
        concatenate_terms=False,
    )


def _rewards(joints: EntityRef, velocity_limits: tuple[float, ...]) -> dict[str, RewardTermSpec]:
    legs = EntityRef("robot", joints=(".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"))
    feet = EntityRef("robot", bodies=".*_ankle_roll_link")
    return {
        "track_lin_vel_xy_exp": RewardTermSpec(
            func=mdp.track_lin_vel_xy_exp, weight=2.0, params={"command_name": COMMAND, "std": 0.5}
        ),
        "track_ang_vel_z_exp": RewardTermSpec(
            func=mdp.track_ang_vel_z_exp, weight=2.0, params={"command_name": COMMAND, "std": 0.5}
        ),
        "heading_error": RewardTermSpec(func=mdp.heading_error, weight=-1.0, params={"command_name": COMMAND}),
        "dont_wait": RewardTermSpec(func=mdp.dont_wait, weight=-0.5, params={"command_name": COMMAND}),
        "is_alive": RewardTermSpec(func=mdp.is_alive, weight=3.0),
        "stand_still": RewardTermSpec(
            func=mdp.stand_still_when_idle, weight=-0.3, params={"command_name": COMMAND, "offset": 4.0}
        ),
        "volume_points_penetration": RewardTermSpec(
            func=mdp.volume_points_penetration,
            weight=-4.0,
            params={"sensor": LEG_VOLUME_POINTS},
            level=Requirement.REQUIRED,
        ),
        # Portable on purpose. Isaac's ContactSensor already gates air-time at 1 N
        # (cfg.force_threshold default). mjlab's timer uses ``found``. The
        # acceptance-run near-zero is explained by early base_contact, not by
        # this term's own criterion -- see engines/*/terms.py.
        "feet_air_time": RewardTermSpec(
            func=mdp.feet_air_time,
            weight=0.5,
            params={"command_name": COMMAND, "sensor": FEET_CONTACT, "vel_threshold": 0.15},
        ),
        "feet_slide": RewardTermSpec(
            kind="contact_slide",
            weight=-0.4,
            params={"sensor_cfg": FEET_CONTACT, "asset_cfg": feet, "threshold": 1.0},
            level=Requirement.REQUIRED,
        ),
        "joint_deviation_hip": RewardTermSpec(
            func=mdp.joint_deviation_square,
            weight=-0.5,
            params={"asset_cfg": EntityRef("robot", joints=(".*_hip_yaw_joint", ".*_hip_roll_joint"))},
        ),
        "ang_vel_xy_l2": RewardTermSpec(func=mdp.ang_vel_xy_l2, weight=-0.05),
        "dof_torques_l2": RewardTermSpec(
            kind="joint_torques_l2",
            weight=-1.5e-7,
            params={"asset_cfg": legs},
            level=Requirement.REQUIRED,
        ),
        "dof_acc_l2": RewardTermSpec(
            kind="joint_acc_l2",
            weight=-1.25e-7,
            params={"asset_cfg": EntityRef("robot", joints=".*")},
            level=Requirement.REQUIRED,
        ),
        "dof_vel_l2": RewardTermSpec(
            func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": EntityRef("robot", joints=".*")}
        ),
        "action_rate_l2": RewardTermSpec(func=mdp.action_rate_l2, weight=-0.005),
        "flat_orientation_l2": RewardTermSpec(func=mdp.flat_orientation_l2, weight=-3.0),
        "pelvis_orientation_l2": RewardTermSpec(
            func=mdp.link_orientation, weight=-3.0, params={"asset_cfg": EntityRef("robot", bodies="pelvis")}
        ),
        "feet_flat_ori": RewardTermSpec(
            func=mdp.feet_orientation_contact, weight=-0.4, params={"sensor": FEET_CONTACT, "asset_cfg": feet}
        ),
        "feet_at_plane": RewardTermSpec(
            func=mdp.feet_at_plane,
            weight=-0.1,
            params={
                "sensor": FEET_CONTACT,
                "left_scanner": LEFT_HEIGHT_SCANNER,
                "right_scanner": RIGHT_HEIGHT_SCANNER,
                "asset_cfg": feet,
                "height_offset": 0.058,
            },
            level=Requirement.REQUIRED,
        ),
        "feet_close_xy": RewardTermSpec(
            func=mdp.feet_close_xy_gauss,
            weight=0.4,
            params={"threshold": 0.12, "std": math.sqrt(0.05), "asset_cfg": feet},
        ),
        "energy": RewardTermSpec(
            kind="motors_power_square",
            weight=-5e-5,
            params={"asset_cfg": legs, "normalize_by_stiffness": True},
            level=Requirement.REQUIRED,
        ),
        "freeze_upper_body": RewardTermSpec(
            func=mdp.joint_deviation_l1,
            weight=-0.004,
            params={
                "asset_cfg": EntityRef("robot", joints=(".*_shoulder_.*", ".*_elbow_.*", ".*_wrist.*", "waist_.*"))
            },
        ),
        "dof_pos_limits": RewardTermSpec(
            func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": EntityRef("robot", joints=".*")}
        ),
        "dof_vel_limits": RewardTermSpec(
            func=mdp.joint_vel_limits,
            weight=-1.0,
            params={"soft_ratio": 0.9, "limits": velocity_limits, "asset_cfg": joints},
        ),
        "torque_limits": RewardTermSpec(
            kind="applied_torque_limits_by_ratio",
            weight=-0.01,
            params={"asset_cfg": EntityRef("robot", joints=".*"), "limit_ratio": 0.8},
            level=Requirement.REQUIRED,
        ),
        "undesired_contacts": RewardTermSpec(
            kind="undesired_contacts",
            weight=-1.0,
            params={"sensor": UNDESIRED_CONTACT},
            level=Requirement.REQUIRED,
        ),
    }


def _events() -> dict[str, EventTermSpec]:
    """Parkour's events. No push, no base-mass randomization -- those are locomotion's."""
    return {
        "physics_material": EventTermSpec(
            kind="randomize_friction",
            mode="startup",
            target=ROBOT,
            params={"static_friction_range": (0.3, 1.6), "dynamic_friction_range": (0.3, 1.6)},
            engine_params={
                "isaacsim": {
                    "restitution_range": (0.05, 0.5),
                    "num_buckets": 64,
                    "make_consistent": True,
                }
            },
        ),
        "reset_base": EventTermSpec(
            kind="reset_root_state_uniform",
            mode="reset",
            params={
                "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
                "velocity_range": {
                    "x": (-0.2, 0.2),
                    "y": (-0.2, 0.2),
                    "z": (-0.2, 0.2),
                    "roll": (-0.2, 0.2),
                    "pitch": (-0.2, 0.2),
                    "yaw": (-0.2, 0.2),
                },
            },
        ),
        "reset_robot_joints": EventTermSpec(
            kind="reset_joints_by_offset",
            mode="reset",
            params={"position_range": (-0.15, 0.15), "velocity_range": (0.0, 0.0)},
        ),
        "register_virtual_obstacles": EventTermSpec(
            kind="register_virtual_obstacles",
            mode="startup",
            params={"sensor": LEG_VOLUME_POINTS},
            level=Requirement.REQUIRED,
        ),
    }


def _commands() -> dict[str, CommandTermSpec]:
    """Terrain-aware target-following velocity. The tenth sub-terrain is named per engine."""
    return {
        COMMAND: CommandTermSpec(
            kind="pose_velocity",
            level=Requirement.REQUIRED,
            params={
                "entity": "robot",
                "resampling_time_range": (8.0, 12.0),
                "lin_vel_x": (0.0, 0.0),
                "lin_vel_y": (0.0, 0.0),
                "ang_vel_z": (-1.0, 1.0),
                "velocity_control_stiffness": 2.0,
                "heading_control_stiffness": 2.0,
                "only_positive_lin_vel_x": True,
                "rel_standing_envs": 0.05,
                "random_velocity_terrain": ["perlin_rough_stand"],
                "velocity_ranges": SHARED_VELOCITY_RANGES,
                "lin_vel_threshold": 0.0,
                "ang_vel_threshold": 0.0,
                "lin_vel_metrics_std": 0.5,
                "ang_vel_metrics_std": 0.5,
                "target_dis_threshold": 0.4,
            },
            engine_params={
                "isaacsim": {"velocity_ranges": {"mesh_boxes": dict(_OBSTACLE_VEL)}},
                "mjlab": {"velocity_ranges": {"dense_boxes": dict(_OBSTACLE_VEL)}},
            },
        )
    }


def parkour_g1_robot() -> RobotSpec:
    """Catalog G1 plus the four main-parkour plant overrides. Does not touch the factory.

    Main's ``G1ParkourEnvCfg`` copied the popsicle ArticulationCfg and then changed
    the URDF, spawn z, ``merge_fixed_joints``, and the delayed-actuator table. The
    catalog stays the flat/rough plant; this is the copy the task holds so both
    adapters read one ``RobotSpec`` instead of a second override bag someone can
    forget to apply.
    """
    return make_g1_29dof_robot_spec().overridden(
        default_root_pos=(0.0, 0.0, 0.9),
        actuator_delay=(0, 2),
        asset_paths={"isaacsim": str(_SHOE_URDF), "mjlab": str(_SHOE_XML)},
        import_options={"isaacsim": {"merge_fixed_joints": True}},
    )


def parkour_target_g1() -> TaskSpec:
    """The proprioceptive parkour task."""
    robot = parkour_g1_robot()
    joints = _canonical_joints(robot)
    motion_reference = MotionReferenceRef(
        name="motion_reference",
        clip=PARKOUR_MOTION_CLIP,
        joints=tuple(robot.joint_names),
        links=PARKOUR_MOTION_LINKS,
        num_frames=10,
        frame_interval_s=0.02,
        update_period=0.02,
        data_start_from="one_frame_interval",
        clip_target_fps=50.0,
        velocity_method="frontward",
        start_range=(0.0, 0.9),
        exhaustion="freeze_last_and_flag",
        quaternion="wxyz",
    )
    return TaskSpec(
        task_id="Instinct-Parkour-Target-G1",
        robot=robot,
        scene=SceneSpec(
            terrain=TerrainSpec(kind="rough", virtual_obstacles=(PARKOUR_EDGE_CYLINDERS,)),
            contact_sensors=(
                ContactSensorRef(name="contact_forces", elements=".*", track_air_time=True, history_length=3),
            ),
            ray_casters=(LEFT_HEIGHT_SCANNER, RIGHT_HEIGHT_SCANNER, DEPTH_CAMERA),
            motion_references=(motion_reference,),
            volume_points=(LEG_VOLUME_POINTS,),
            env_spacing=2.5,
        ),
        sim=SimSpec(physics_dt=0.005, decimation=4, episode_length_s=20.0),
        mdp=MdpSpec(
            observations={
                "policy": _policy_observations(corrupt=True, joints=joints),
                "critic": _policy_observations(corrupt=False, joints=joints),
                "amp_policy": _amp_policy_observations(joints),
                "amp_reference": _amp_reference_observations(joints, motion_reference),
            },
            actions={
                "joint_pos": ActionTermSpec(
                    kind="joint_position",
                    target=joints,
                    params={"scale": _action_scale(robot), "use_default_offset": True},
                )
            },
            commands=_commands(),
            rewards={"rewards": _rewards(joints, _velocity_limits(robot))},
            terminations={
                "time_out": DoneTermSpec(func=mdp.time_out, time_out=True),
                "terrain_out_of_bounds": DoneTermSpec(
                    func=mdp.terrain_out_of_bounds, time_out=True, params={"distance_buffer": 2.0}
                ),
                "base_contact": DoneTermSpec(kind="illegal_contact", params={"sensor": TORSO_CONTACT}),
                "bad_orientation": DoneTermSpec(func=mdp.bad_orientation, params={"limit_angle": 1.0}),
                "root_height": DoneTermSpec(
                    func=mdp.root_height_below_env_origin_minimum, params={"minimum_height": 0.5}
                ),
            },
            events=_events(),
            curriculum={
                "terrain_levels": CurriculumTermSpec(
                    func=mdp.tracking_exp_vel,
                    params={
                        "command_name": COMMAND,
                        "lin_vel_threshold": (0.3, 0.6),
                        "ang_vel_threshold": (0.0, 0.0),
                    },
                    level=Requirement.REQUIRED,
                )
            },
        ),
        agent=AgentSpec(
            runner="instinctlab.tasks.parkour.config.g1.agents.instinct_rl_parkour_cfg:G1ParkourTargetPPORunnerCfg"
        ),
        engines=("isaacsim", "mjlab"),
    )
