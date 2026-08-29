"""Robot-independent Parkour task configuration."""

from __future__ import annotations

from instinctlab.engines.assets import RobotSpec
from instinctlab.spec import (
    ActionTermSpec,
    AgentSpec,
    CommandTermSpec,
    ContactSensorRef,
    CurriculumTermSpec,
    DoneTermSpec,
    EntityRef,
    EventTermSpec,
    MotionReferenceRef,
    NoiseSpec,
    ObsGroupSpec,
    ObsTermSpec,
    RayCasterRef,
    RewardTermSpec,
    SceneSpec,
    SimSpec,
    VirtualObstacleRef,
    VolumePointsRef,
)
from instinctlab.spec.capability import Requirement
from instinctlab.tasks.parkour.mdp import (
    amp,
    curriculums,
    events,
    observations,
    terminations,
)
from instinctlab.tasks.terrain import rough_terrain


class ParkourPolicyObsCfg:
    def __init__(self, robot: RobotSpec, depth_camera: RayCasterRef) -> None:
        self.base_ang_vel = ObsTermSpec(
            func=observations.base_ang_vel,
            noise=NoiseSpec("uniform", -0.2, 0.2),
            scale=0.25,
            history_length=8,
        )
        self.projected_gravity = ObsTermSpec(
            func=observations.projected_gravity,
            noise=NoiseSpec("uniform", -0.05, 0.05),
            history_length=8,
        )
        self.velocity_commands = ObsTermSpec(
            func=observations.generated_commands,
            params={"command_name": "base_velocity"},
            history_length=8,
        )
        self.joint_pos = ObsTermSpec(
            func=observations.joint_pos_rel,
            params={
                "asset_cfg": EntityRef(
                    "robot",
                    joints=tuple(robot.joint_names),
                    preserve_order=True,
                )
            },
            noise=NoiseSpec("uniform", -0.01, 0.01),
            history_length=8,
        )
        self.joint_vel = ObsTermSpec(
            func=observations.joint_vel_rel,
            params={
                "asset_cfg": EntityRef(
                    "robot",
                    joints=tuple(robot.joint_names),
                    preserve_order=True,
                )
            },
            noise=NoiseSpec("uniform", -0.5, 0.5),
            scale=0.05,
            history_length=8,
        )
        self.actions = ObsTermSpec(func=observations.last_action, history_length=8)
        self.depth_image = ObsTermSpec(
            func=observations.DelayedDepthImage,
            params={
                "sensor": depth_camera,
                "history_skip_frames": 5,
                "num_output_frames": 8,
                "delayed_frame_ranges": (0, 1),
                "history_length": 37,
                "blur_kernel_size": 3,
                "blur_sigma": 1.0,
            },
            history_length=0,
            level=Requirement.REQUIRED,
        )


class ParkourCriticObsCfg:
    def __init__(self, robot: RobotSpec, depth_camera: RayCasterRef) -> None:
        self.base_lin_vel = ObsTermSpec(
            func=observations.base_lin_vel, history_length=8
        )
        self.base_ang_vel = ObsTermSpec(
            func=observations.base_ang_vel, scale=0.25, history_length=8
        )
        self.projected_gravity = ObsTermSpec(
            func=observations.projected_gravity, history_length=8
        )
        self.velocity_commands = ObsTermSpec(
            func=observations.generated_commands,
            params={"command_name": "base_velocity"},
            history_length=8,
        )
        self.joint_pos = ObsTermSpec(
            func=observations.joint_pos_rel,
            params={
                "asset_cfg": EntityRef(
                    "robot",
                    joints=tuple(robot.joint_names),
                    preserve_order=True,
                )
            },
            history_length=8,
        )
        self.joint_vel = ObsTermSpec(
            func=observations.joint_vel_rel,
            params={
                "asset_cfg": EntityRef(
                    "robot",
                    joints=tuple(robot.joint_names),
                    preserve_order=True,
                )
            },
            scale=0.05,
            history_length=8,
        )
        self.actions = ObsTermSpec(func=observations.last_action, history_length=8)
        self.depth_image = ObsTermSpec(
            func=observations.DelayedDepthImage,
            params={
                "sensor": depth_camera,
                "history_skip_frames": 5,
                "num_output_frames": 8,
                "delayed_frame_ranges": (0, 1),
                "history_length": 37,
                "blur_kernel_size": 3,
                "blur_sigma": 1.0,
            },
            history_length=0,
            level=Requirement.REQUIRED,
        )


class ParkourAmpPolicyObsCfg:
    def __init__(self, robot: RobotSpec) -> None:
        self.projected_gravity = ObsTermSpec(
            func=observations.projected_gravity, history_length=10
        )
        self.joint_pos_rel = ObsTermSpec(
            func=observations.joint_pos_rel,
            params={
                "asset_cfg": EntityRef(
                    "robot",
                    joints=tuple(robot.joint_names),
                    preserve_order=True,
                )
            },
            history_length=10,
        )
        self.joint_vel = ObsTermSpec(
            func=observations.joint_vel_rel,
            params={
                "asset_cfg": EntityRef(
                    "robot",
                    joints=tuple(robot.joint_names),
                    preserve_order=True,
                )
            },
            scale=0.05,
            history_length=10,
        )
        self.base_lin_vel = ObsTermSpec(
            func=observations.base_lin_vel, history_length=10
        )
        self.base_ang_vel = ObsTermSpec(
            func=observations.base_ang_vel, history_length=10
        )


class ParkourAmpReferenceObsCfg:
    def __init__(self, robot: RobotSpec, motion_reference: MotionReferenceRef) -> None:
        self.projected_gravity = ObsTermSpec(
            func=amp.projected_gravity_from_reference,
            params={
                "sensor": motion_reference,
                "asset_cfg": EntityRef(
                    "robot",
                    joints=tuple(robot.joint_names),
                    preserve_order=True,
                ),
            },
            history_length=10,
        )
        self.joint_pos_rel = ObsTermSpec(
            func=amp.joint_pos_rel_from_reference,
            params={
                "sensor": motion_reference,
                "asset_cfg": EntityRef(
                    "robot",
                    joints=tuple(robot.joint_names),
                    preserve_order=True,
                ),
            },
            history_length=10,
        )
        self.joint_vel = ObsTermSpec(
            func=amp.joint_vel_rel_from_reference,
            params={
                "sensor": motion_reference,
                "asset_cfg": EntityRef(
                    "robot",
                    joints=tuple(robot.joint_names),
                    preserve_order=True,
                ),
            },
            scale=0.05,
            history_length=10,
        )
        self.base_lin_vel = ObsTermSpec(
            func=amp.base_lin_vel_from_reference,
            params={
                "sensor": motion_reference,
                "asset_cfg": EntityRef(
                    "robot",
                    joints=tuple(robot.joint_names),
                    preserve_order=True,
                ),
            },
            history_length=10,
        )
        self.base_ang_vel = ObsTermSpec(
            func=amp.base_ang_vel_from_reference,
            params={
                "sensor": motion_reference,
                "asset_cfg": EntityRef(
                    "robot",
                    joints=tuple(robot.joint_names),
                    preserve_order=True,
                ),
            },
            history_length=10,
        )


class ParkourEnvCfg:
    def __init__(
        self,
        robot: RobotSpec,
        motion_reference: MotionReferenceRef,
        depth_camera: RayCasterRef,
        left_height_scanner: RayCasterRef,
        right_height_scanner: RayCasterRef,
        leg_volume_points: VolumePointsRef,
        rewards: dict[str, dict[str, RewardTermSpec]],
        torso_contact: ContactSensorRef,
        agent: AgentSpec,
    ) -> None:
        self.robot = robot
        self.scene = SceneSpec(
            terrain=rough_terrain(
                virtual_obstacles=(
                    VirtualObstacleRef(
                        name="edges",
                        kind="greedy_edge_cylinder",
                        cylinder_radius=0.05,
                        min_points=2,
                    ),
                )
            ),
            contact_sensors=(
                ContactSensorRef(
                    name="contact_forces",
                    elements=".*",
                    track_air_time=True,
                    history_length=3,
                ),
            ),
            ray_casters=(left_height_scanner, right_height_scanner, depth_camera),
            motion_references=(motion_reference,),
            volume_points=(leg_volume_points,),
            env_spacing=2.5,
        )
        self.sim = SimSpec(
            physics_dt=0.005,
            decimation=4,
            episode_length_s=20.0,
            profiles={
                "mjlab": {
                    "contact_sensor_maxmatch": 128,
                    "ccd_iterations": 128,
                    "pinhole_cameras": {
                        "camera": {
                            "include_geom_groups": (0, 1, 2),
                            "exclude_parent_body": True,
                            "mesh_filter_max_hops": 6,
                            "mesh_filter_epsilon": 1.0e-4,
                            "update_period": 0.0,
                        }
                    },
                }
            },
        )
        self.commands = {
            "base_velocity": CommandTermSpec(
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
                    "velocity_ranges": {
                        "perlin_rough": {
                            "lin_vel_x": (0.45, 1.0),
                            "lin_vel_y": (0.0, 0.0),
                            "ang_vel_z": (-1.0, 1.0),
                        },
                        "perlin_rough_stand": {
                            "lin_vel_x": (0.0, 0.0),
                            "lin_vel_y": (0.0, 0.0),
                            "ang_vel_z": (0.0, 0.0),
                        },
                        "square_gaps": {
                            "lin_vel_x": (0.45, 0.8),
                            "lin_vel_y": (0.0, 0.0),
                            "ang_vel_z": (-1.0, 1.0),
                        },
                        "pyramid_stairs": {
                            "lin_vel_x": (0.45, 0.8),
                            "lin_vel_y": (0.0, 0.0),
                            "ang_vel_z": (-1.0, 1.0),
                        },
                        "pyramid_stairs_high": {
                            "lin_vel_x": (0.45, 0.8),
                            "lin_vel_y": (0.0, 0.0),
                            "ang_vel_z": (-1.0, 1.0),
                        },
                        "pyramid_stairs_inv": {
                            "lin_vel_x": (0.45, 0.8),
                            "lin_vel_y": (0.0, 0.0),
                            "ang_vel_z": (-1.0, 1.0),
                        },
                        "pyramid_stairs_inv_high": {
                            "lin_vel_x": (0.45, 0.8),
                            "lin_vel_y": (0.0, 0.0),
                            "ang_vel_z": (-1.0, 1.0),
                        },
                        "boxes": {
                            "lin_vel_x": (0.45, 0.8),
                            "lin_vel_y": (0.0, 0.0),
                            "ang_vel_z": (-1.0, 1.0),
                        },
                        "mesh_boxes": {
                            "lin_vel_x": (0.45, 0.8),
                            "lin_vel_y": (0.0, 0.0),
                            "ang_vel_z": (-1.0, 1.0),
                        },
                        "hf_pyramid_slope_inv": {
                            "lin_vel_x": (0.45, 0.8),
                            "lin_vel_y": (0.0, 0.0),
                            "ang_vel_z": (-1.0, 1.0),
                        },
                    },
                    "lin_vel_threshold": 0.0,
                    "ang_vel_threshold": 0.0,
                    "lin_vel_metrics_std": 0.5,
                    "ang_vel_metrics_std": 0.5,
                    "target_dis_threshold": 0.4,
                },
            )
        }
        policy = ParkourPolicyObsCfg(robot, depth_camera)
        critic = ParkourCriticObsCfg(robot, depth_camera)
        amp_policy = ParkourAmpPolicyObsCfg(robot)
        amp_reference = ParkourAmpReferenceObsCfg(robot, motion_reference)
        self.observations = {
            "policy": ObsGroupSpec(
                terms=dict(vars(policy)),
                concatenate_terms=False,
                enable_corruption=True,
            ),
            "critic": ObsGroupSpec(
                terms=dict(vars(critic)),
                concatenate_terms=False,
                enable_corruption=False,
            ),
            "amp_policy": ObsGroupSpec(
                terms=dict(vars(amp_policy)),
                concatenate_terms=False,
                enable_corruption=False,
            ),
            "amp_reference": ObsGroupSpec(
                terms=dict(vars(amp_reference)),
                concatenate_terms=False,
                enable_corruption=False,
            ),
        }
        self.actions = {
            "joint_pos": ActionTermSpec(
                kind="joint_position",
                target=EntityRef(
                    "robot",
                    joints=tuple(robot.joint_names),
                    preserve_order=True,
                ),
                params={
                    "scale": {
                        joint.name: joint.action_scale
                        for joint in robot.joint_properties
                    },
                    "use_default_offset": True,
                },
            )
        }
        self.rewards = rewards
        self.curriculum = {
            "terrain_levels": CurriculumTermSpec(
                func=curriculums.tracking_exp_vel,
                params={
                    "command_name": "base_velocity",
                    "lin_vel_threshold": (0.3, 0.6),
                    "ang_vel_threshold": (0.0, 0.0),
                },
                level=Requirement.REQUIRED,
            )
        }
        self.terminations = {
            "time_out": DoneTermSpec(func=terminations.time_out, time_out=True),
            "terrain_out_of_bounds": DoneTermSpec(
                func=terminations.terrain_out_of_bounds,
                time_out=True,
                params={"distance_buffer": 2.0},
            ),
            "base_contact": DoneTermSpec(
                kind="illegal_contact",
                params={"sensor": torso_contact, "threshold": 1.0},
            ),
            "bad_orientation": DoneTermSpec(
                func=terminations.bad_orientation,
                params={"limit_angle": 1.0},
            ),
            "root_height": DoneTermSpec(
                func=terminations.root_height_below_env_origin_minimum,
                params={"minimum_height": 0.5},
            ),
            "dataset_exhausted": DoneTermSpec(
                func=terminations.dataset_exhausted,
                time_out=True,
                params={
                    "sensor": motion_reference,
                    "print_reason": False,
                    "reset_without_notice": True,
                },
            ),
        }
        self.events = {
            "physics_material": EventTermSpec(
                kind="randomize_friction",
                mode="startup",
                target=EntityRef("robot", bodies=".*"),
                params={
                    "static_friction_range": (0.3, 1.6),
                    "dynamic_friction_range": (0.3, 1.6),
                },
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
                target=EntityRef("robot"),
                params={
                    "pose_range": {
                        "x": (-0.1, 0.1),
                        "y": (-0.1, 0.1),
                        "yaw": (-0.1, 0.1),
                    },
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
            "register_virtual_obstacles": EventTermSpec(
                func=events.register_virtual_obstacles,
                mode="startup",
                params={"sensor": leg_volume_points},
                level=Requirement.REQUIRED,
            ),
            "reset_robot_joints": EventTermSpec(
                kind="reset_joints_by_offset",
                mode="reset",
                target=EntityRef("robot", joints=".*"),
                params={
                    "position_range": (-0.15, 0.15),
                    "velocity_range": (0.0, 0.0),
                },
            ),
        }
        self.agent = agent
