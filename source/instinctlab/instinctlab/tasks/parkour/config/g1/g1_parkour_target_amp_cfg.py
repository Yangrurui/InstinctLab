"""Explicit G1 configuration for the robot-independent Parkour task."""

from __future__ import annotations

import math

from instinctlab.engines.assets import RobotSpec
from instinctlab.spec import (
    AgentSpec,
    ContactSensorRef,
    EntityRef,
    Grid3dPointsRef,
    MdpSpec,
    MotionReferenceRef,
    RayCasterRef,
    RayPatternRef,
    RewardTermSpec,
    SymmetricAugmentationSpec,
    TaskSpec,
    VolumePointsRef,
)
from instinctlab.spec.capability import Requirement
from instinctlab.tasks.parkour.config.parkour_env_cfg import (
    ParkourEnvCfg,
)
from instinctlab.tasks.parkour.mdp import rewards


class G1ParkourRewardsCfg:
    def __init__(self, robot: RobotSpec) -> None:
        self.track_lin_vel_xy_exp = RewardTermSpec(
            func=rewards.track_lin_vel_xy_exp,
            weight=2.0,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        self.track_ang_vel_z_exp = RewardTermSpec(
            func=rewards.track_ang_vel_z_exp,
            weight=2.0,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        self.heading_error = RewardTermSpec(
            func=rewards.heading_error,
            weight=-1.0,
            params={"command_name": "base_velocity"},
        )
        self.dont_wait = RewardTermSpec(
            func=rewards.dont_wait,
            weight=-0.5,
            params={"command_name": "base_velocity"},
        )
        self.is_alive = RewardTermSpec(func=rewards.is_alive, weight=3.0)
        self.stand_still = RewardTermSpec(
            func=rewards.stand_still_when_idle,
            weight=-0.3,
            params={"command_name": "base_velocity", "offset": 4.0},
        )
        self.volume_points_penetration = RewardTermSpec(
            func=rewards.volume_points_penetration,
            weight=-8.0,
            params={
                "sensor": VolumePointsRef(
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
            },
            level=Requirement.REQUIRED,
        )
        self.feet_air_time = RewardTermSpec(
            func=rewards.feet_air_time,
            weight=0.5,
            params={
                "command_name": "base_velocity",
                "sensor": ContactSensorRef(
                    name="contact_forces",
                    elements=".*_ankle_roll_link",
                ),
                "vel_threshold": 0.15,
            },
        )
        self.feet_slide = RewardTermSpec(
            func=rewards.contact_slide,
            weight=-0.4,
            params={
                "sensor_cfg": ContactSensorRef(
                    name="contact_forces",
                    elements=".*_ankle_roll_link",
                ),
                "asset_cfg": EntityRef("robot", bodies=".*_ankle_roll_link"),
                "threshold": 1.0,
            },
            level=Requirement.REQUIRED,
        )
        self.joint_deviation_hip = RewardTermSpec(
            func=rewards.joint_deviation_square,
            weight=-0.5,
            params={
                "asset_cfg": EntityRef(
                    "robot",
                    joints=(".*_hip_yaw_joint", ".*_hip_roll_joint"),
                )
            },
        )
        self.ang_vel_xy_l2 = RewardTermSpec(func=rewards.ang_vel_xy_l2, weight=-0.05)
        self.dof_torques_l2 = RewardTermSpec(
            func=rewards.joint_torques_l2,
            weight=-1.5e-7,
            params={
                "asset_cfg": EntityRef(
                    "robot",
                    joints=(".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"),
                )
            },
            level=Requirement.REQUIRED,
        )
        self.dof_acc_l2 = RewardTermSpec(
            func=rewards.joint_acc_l2,
            weight=-1.25e-7,
            params={"asset_cfg": EntityRef("robot", joints=".*")},
            level=Requirement.REQUIRED,
        )
        self.dof_vel_l2 = RewardTermSpec(
            func=rewards.joint_vel_l2,
            weight=-1e-4,
            params={"asset_cfg": EntityRef("robot", joints=".*")},
        )
        self.action_rate_l2 = RewardTermSpec(func=rewards.action_rate_l2, weight=-0.005)
        self.flat_orientation_l2 = RewardTermSpec(
            func=rewards.flat_orientation_l2, weight=-3.0
        )
        self.pelvis_orientation_l2 = RewardTermSpec(
            func=rewards.link_orientation,
            weight=-3.0,
            params={"asset_cfg": EntityRef("robot", bodies="pelvis")},
        )
        self.feet_flat_ori = RewardTermSpec(
            func=rewards.feet_orientation_contact,
            weight=-0.4,
            params={
                "sensor": ContactSensorRef(
                    name="contact_forces",
                    elements=".*_ankle_roll_link",
                ),
                "asset_cfg": EntityRef("robot", bodies=".*_ankle_roll_link"),
            },
        )
        self.feet_at_plane = RewardTermSpec(
            func=rewards.feet_at_plane,
            weight=-0.1,
            params={
                "sensor": ContactSensorRef(
                    name="contact_forces",
                    elements=".*_ankle_roll_link",
                ),
                "left_scanner": RayCasterRef(
                    name="left_height_scanner",
                    attach="left_ankle_roll_link",
                    mode="terrain_height",
                    offset=(0.04, 0.0, 20.0),
                    pattern=RayPatternRef(
                        kind="grid",
                        resolution=0.12,
                        size=(0.12, 0.0),
                    ),
                    hit="terrain",
                    ray_alignment="yaw",
                    miss="infinity",
                    engine_max_distances={"mjlab": 10.0},
                ),
                "right_scanner": RayCasterRef(
                    name="right_height_scanner",
                    attach="right_ankle_roll_link",
                    mode="terrain_height",
                    offset=(0.04, 0.0, 20.0),
                    pattern=RayPatternRef(
                        kind="grid",
                        resolution=0.12,
                        size=(0.12, 0.0),
                    ),
                    hit="terrain",
                    ray_alignment="yaw",
                    miss="infinity",
                    engine_max_distances={"mjlab": 10.0},
                ),
                "asset_cfg": EntityRef("robot", bodies=".*_ankle_roll_link"),
                "height_offset": 0.058,
            },
            level=Requirement.REQUIRED,
        )
        self.feet_close_xy = RewardTermSpec(
            func=rewards.feet_close_xy_gauss,
            weight=0.4,
            params={
                "threshold": 0.12,
                "std": math.sqrt(0.05),
                "asset_cfg": EntityRef("robot", bodies=".*_ankle_roll_link"),
            },
        )
        self.energy = RewardTermSpec(
            func=rewards.motors_power_square,
            weight=-5e-5,
            params={
                "asset_cfg": EntityRef(
                    "robot",
                    joints=(".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"),
                ),
                "normalize_by_stiffness": True,
            },
            level=Requirement.REQUIRED,
        )
        self.freeze_upper_body = RewardTermSpec(
            func=rewards.joint_deviation_l1,
            weight=-0.004,
            params={
                "asset_cfg": EntityRef(
                    "robot",
                    joints=(
                        ".*_shoulder_.*",
                        ".*_elbow_.*",
                        ".*_wrist.*",
                        "waist_.*",
                    ),
                )
            },
        )
        self.dof_pos_limits = RewardTermSpec(
            func=rewards.joint_pos_limits,
            weight=-1.0,
            params={"asset_cfg": EntityRef("robot", joints=".*")},
        )
        self.dof_vel_limits = RewardTermSpec(
            func=rewards.joint_vel_limits,
            weight=-1.0,
            params={
                "soft_ratio": 0.9,
                "limits": tuple(
                    joint.velocity_limit for joint in robot.joint_properties
                ),
                "asset_cfg": EntityRef(
                    "robot",
                    joints=tuple(robot.joint_names),
                    preserve_order=True,
                ),
            },
        )
        self.torque_limits = RewardTermSpec(
            func=rewards.applied_torque_limits_by_ratio,
            weight=-0.01,
            params={
                "asset_cfg": EntityRef("robot", joints=".*"),
                "limit_ratio": 0.8,
            },
            level=Requirement.REQUIRED,
        )
        self.undesired_contacts = RewardTermSpec(
            func=rewards.undesired_contacts_by_force,
            weight=-1.0,
            params={
                "sensor": ContactSensorRef(
                    name="contact_forces",
                    elements="(?!.*_ankle_roll_link).*",
                ),
                "threshold": 1.0,
            },
            level=Requirement.REQUIRED,
        )


class G1ParkourEnvCfg(ParkourEnvCfg):
    def __init__(self, robot: RobotSpec) -> None:
        motion_reference = MotionReferenceRef(
            name="motion_reference",
            clip=(
                "~/Datasets/parkour_release/parkour_motion_reference/"
                "parkour_motion_without_run_retargetted.npz"
            ),
            joints=tuple(robot.joint_names),
            links=(
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
            ),
            num_frames=10,
            frame_interval_s=0.02,
            update_period=0.02,
            data_start_from="one_frame_interval",
            clip_target_fps=50.0,
            velocity_method="frontward",
            start_range=(0.0, 0.9),
            exhaustion="freeze_last_and_flag",
            quaternion="wxyz",
            symmetric_augmentation=SymmetricAugmentationSpec.from_left_right(
                tuple(robot.joint_names),
                (
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
                ),
            ),
        )
        left_height_scanner = RayCasterRef(
            name="left_height_scanner",
            attach="left_ankle_roll_link",
            mode="terrain_height",
            offset=(0.04, 0.0, 20.0),
            pattern=RayPatternRef(kind="grid", resolution=0.12, size=(0.12, 0.0)),
            hit="terrain",
            ray_alignment="yaw",
            miss="infinity",
            engine_max_distances={"mjlab": 10.0},
        )
        right_height_scanner = RayCasterRef(
            name="right_height_scanner",
            attach="right_ankle_roll_link",
            mode="terrain_height",
            offset=(0.04, 0.0, 20.0),
            pattern=RayPatternRef(kind="grid", resolution=0.12, size=(0.12, 0.0)),
            hit="terrain",
            ray_alignment="yaw",
            miss="infinity",
            engine_max_distances={"mjlab": 10.0},
        )
        depth_camera = RayCasterRef(
            name="camera",
            attach="torso_link",
            offset=(0.0487988662332928, 0.01, 0.4378029937970051),
            offset_rot=(
                0.9135367613482678,
                0.004363309284746571,
                0.4067366430758002,
                0.0,
            ),
            offset_convention="world",
            pattern=RayPatternRef(
                kind="pinhole",
                width=64,
                height=36,
                horizontal_fov_deg=89.51,
                vertical_fov_deg=58.29,
                focal_length=1.0,
            ),
            hit=(
                "terrain",
                "torso_link",
                "waist_roll_link",
                "waist_yaw_link",
                "pelvis",
                "left_hip_pitch_link",
                "left_hip_roll_link",
                "left_hip_yaw_link",
                "left_knee_link",
                "left_ankle_pitch_link",
                "left_ankle_roll_link",
                "right_hip_pitch_link",
                "right_hip_roll_link",
                "right_hip_yaw_link",
                "right_knee_link",
                "right_ankle_pitch_link",
                "right_ankle_roll_link",
                "left_shoulder_pitch_link",
                "left_shoulder_roll_link",
                "left_shoulder_yaw_link",
                "left_elbow_link",
                "left_wrist_roll_link",
                "left_wrist_pitch_link",
                "left_wrist_yaw_link",
                "right_shoulder_pitch_link",
                "right_shoulder_roll_link",
                "right_shoulder_yaw_link",
                "right_elbow_link",
                "right_wrist_roll_link",
                "right_wrist_pitch_link",
                "right_wrist_yaw_link",
            ),
            ray_alignment="base",
            miss="infinity",
            max_distance=2.5,
            min_distance=0.1,
            crop=(18, 0, 16, 16),
            update_period=0.02,
        )
        leg_volume_points = VolumePointsRef(
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
        super().__init__(
            robot=robot,
            motion_reference=motion_reference,
            depth_camera=depth_camera,
            left_height_scanner=left_height_scanner,
            right_height_scanner=right_height_scanner,
            leg_volume_points=leg_volume_points,
            rewards={"rewards": dict(vars(G1ParkourRewardsCfg(robot)))},
            torso_contact=ContactSensorRef(
                name="contact_forces",
                elements="torso_link",
            ),
            agent=AgentSpec(
                runner=(
                    "instinctlab.tasks.parkour.config.g1.agents.instinct_rl_amp_cfg:"
                    "G1ParkourTargetPPORunnerCfg"
                )
            ),
        )


def parkour_target_g1(robot: RobotSpec) -> TaskSpec:
    """Convert the complete G1 Parkour config at the registry boundary."""
    config = G1ParkourEnvCfg(robot)
    return TaskSpec(
        task_id="Instinct-Parkour-Target-G1",
        robot=config.robot,
        scene=config.scene,
        sim=config.sim,
        mdp=MdpSpec(
            commands=config.commands,
            observations=config.observations,
            actions=config.actions,
            rewards=config.rewards,
            curriculum=config.curriculum,
            terminations=config.terminations,
            events=config.events,
        ),
        agent=config.agent,
        engines=("isaacsim", "mjlab"),
    )
