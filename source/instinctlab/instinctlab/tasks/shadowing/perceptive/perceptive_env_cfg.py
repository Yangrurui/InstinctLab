"""Robot-independent Perceptive Shadowing task configuration."""

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
    TerrainSpec,
)
from instinctlab.spec.capability import Requirement
from instinctlab.tasks.shadowing.mdp import (
    events as shadowing_events,
)
from instinctlab.tasks.shadowing.mdp import (
    observations,
    rewards,
    terminations,
)
from instinctlab.tasks.shadowing.mdp import (
    terms as shadowing_terms,
)


class PerceptiveCommandsCfg:
    def __init__(self) -> None:
        self.position_ref_command = CommandTermSpec(
            kind="motion_reference_position",
            params={
                "motion_reference": "motion_reference",
                "entity": "robot",
                "current_state_command": False,
                "realtime_mode": True,
                "anchor_frame": "robot",
            },
        )
        self.position_b_ref_command = CommandTermSpec(
            kind="motion_reference_position",
            params={
                "motion_reference": "motion_reference",
                "entity": "robot",
                "current_state_command": False,
                "realtime_mode": True,
                "anchor_frame": "reference",
            },
        )
        self.rotation_ref_command = CommandTermSpec(
            kind="motion_reference_rotation",
            params={
                "motion_reference": "motion_reference",
                "entity": "robot",
                "current_state_command": False,
                "realtime_mode": True,
                "in_base_frame": True,
                "rotation_mode": "tannorm",
            },
        )
        self.joint_pos_ref_command = CommandTermSpec(
            kind="motion_reference_joint_position",
            params={
                "motion_reference": "motion_reference",
                "entity": "robot",
                "current_state_command": False,
            },
        )
        self.joint_vel_ref_command = CommandTermSpec(
            kind="motion_reference_joint_velocity",
            params={
                "motion_reference": "motion_reference",
                "entity": "robot",
                "current_state_command": False,
            },
        )


class PerceptiveActionsCfg:
    def __init__(self, robot: RobotSpec) -> None:
        self.joint_pos = ActionTermSpec(
            kind="joint_position",
            target=EntityRef(
                "robot",
                joints=tuple(robot.joint_names),
                preserve_order=True,
            ),
            params={
                "scale": {
                    joint.name: joint.action_scale for joint in robot.joint_properties
                },
                "use_default_offset": True,
            },
        )


class PerceptivePolicyObsCfg:
    def __init__(self, robot: RobotSpec, camera: RayCasterRef) -> None:
        self.joint_pos_ref = ObsTermSpec(
            func=observations.generated_commands,
            params={"command_name": "joint_pos_ref_command"},
        )
        self.joint_vel_ref = ObsTermSpec(
            func=observations.generated_commands,
            params={"command_name": "joint_vel_ref_command"},
        )
        self.position_ref = ObsTermSpec(
            func=observations.generated_commands,
            params={"command_name": "position_b_ref_command"},
            noise=NoiseSpec("uniform", -0.25, 0.25),
        )
        self.rotation_ref = ObsTermSpec(
            func=observations.generated_commands,
            params={"command_name": "rotation_ref_command"},
            noise=NoiseSpec("uniform", -0.05, 0.05),
        )
        self.depth_image = ObsTermSpec(
            func=shadowing_terms.depth_image,
            params={
                "sensor": camera,
                "resize_shape": (18, 32),
                "normalization_range": (0.0, 2.0),
            },
        )
        self.projected_gravity = ObsTermSpec(
            func=observations.projected_gravity,
            noise=NoiseSpec("uniform", -0.05, 0.05),
            history_length=8,
        )
        self.base_ang_vel = ObsTermSpec(
            func=observations.base_ang_vel,
            noise=NoiseSpec("uniform", -0.2, 0.2),
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
            history_length=8,
        )
        self.last_action = ObsTermSpec(
            func=observations.last_action,
            history_length=8,
        )


class PerceptiveCriticObsCfg:
    def __init__(
        self,
        robot: RobotSpec,
        motion_reference: MotionReferenceRef,
        height_scanner: RayCasterRef,
    ) -> None:
        self.joint_pos_ref = ObsTermSpec(
            func=observations.generated_commands,
            params={"command_name": "joint_pos_ref_command"},
        )
        self.joint_vel_ref = ObsTermSpec(
            func=observations.generated_commands,
            params={"command_name": "joint_vel_ref_command"},
        )
        self.position_ref = ObsTermSpec(
            func=observations.generated_commands,
            params={"command_name": "position_ref_command"},
        )
        self.link_pos = ObsTermSpec(
            func=shadowing_terms.link_position,
            params={
                "asset_cfg": EntityRef(
                    "robot",
                    bodies=motion_reference.links,
                    preserve_order=True,
                ),
            },
        )
        self.link_rot = ObsTermSpec(
            func=shadowing_terms.link_rotation,
            params={
                "asset_cfg": EntityRef(
                    "robot",
                    bodies=motion_reference.links,
                    preserve_order=True,
                ),
            },
        )
        self.height_scan = ObsTermSpec(
            kind="height_scan",
            params={"sensor": height_scanner},
            clip=(-20.0, 20.0),
        )
        self.base_lin_vel = ObsTermSpec(
            func=observations.base_lin_vel,
            history_length=8,
        )
        self.base_ang_vel = ObsTermSpec(
            func=observations.base_ang_vel,
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
            history_length=8,
        )
        self.last_action = ObsTermSpec(
            func=observations.last_action,
            history_length=8,
        )


class PerceptiveVaePolicyObsCfg:
    def __init__(self, robot: RobotSpec, camera: RayCasterRef) -> None:
        self.depth_image = ObsTermSpec(
            func=observations.DelayedDepthImage,
            params={
                "sensor": camera,
                "history_length": 10,
                "history_skip_frames": 3,
                "num_output_frames": 4,
                "delayed_frame_ranges": (0, 0),
                "resize_shape": (18, 32),
                "normalization_range": (0.0, 2.0),
            },
        )
        self.projected_gravity = ObsTermSpec(
            func=observations.projected_gravity,
            noise=NoiseSpec("uniform", -0.05, 0.05),
            history_length=8,
        )
        self.base_ang_vel = ObsTermSpec(
            func=observations.base_ang_vel,
            noise=NoiseSpec("uniform", -0.2, 0.2),
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
            history_length=8,
        )
        self.last_action = ObsTermSpec(
            func=observations.last_action,
            history_length=8,
        )


class PerceptiveVaeCriticObsCfg:
    def __init__(self, robot: RobotSpec, camera: RayCasterRef) -> None:
        self.joint_pos_ref = ObsTermSpec(
            func=observations.generated_commands,
            params={"command_name": "joint_pos_ref_command"},
        )
        self.joint_vel_ref = ObsTermSpec(
            func=observations.generated_commands,
            params={"command_name": "joint_vel_ref_command"},
        )
        self.position_ref = ObsTermSpec(
            func=observations.generated_commands,
            params={"command_name": "position_b_ref_command"},
            noise=NoiseSpec("uniform", -0.25, 0.25),
        )
        self.rotation_ref = ObsTermSpec(
            func=observations.generated_commands,
            params={"command_name": "rotation_ref_command"},
            noise=NoiseSpec("uniform", -0.05, 0.05),
        )
        self.depth_image = ObsTermSpec(
            func=shadowing_terms.depth_image,
            params={
                "sensor": camera,
                "resize_shape": (18, 32),
                "normalization_range": (0.0, 2.0),
            },
        )
        self.projected_gravity = ObsTermSpec(
            func=observations.projected_gravity,
            history_length=8,
        )
        self.base_ang_vel = ObsTermSpec(
            func=observations.base_ang_vel,
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
            history_length=8,
        )
        self.last_action = ObsTermSpec(
            func=observations.last_action,
            history_length=8,
        )


class PerceptiveEventsCfg:
    def __init__(self) -> None:
        self.physics_material = EventTermSpec(
            kind="randomize_friction",
            mode="startup",
            target=EntityRef("robot", bodies=".*"),
            engine_params={
                "isaacsim": {
                    "static_friction_range": (1.25, 2.0),
                    "dynamic_friction_range": (1.2, 1.8),
                    "restitution_range": (0.0, 0.5),
                    "num_buckets": 64,
                    "make_consistent": True,
                },
                "mjlab": {
                    "ranges": {0: (1.25, 2.0), 1: (1.2, 1.8), 2: (0.0, 0.5)},
                    "operation": "abs",
                    "distribution": "uniform",
                },
            },
        )
        self.add_joint_default_pos = EventTermSpec(
            kind="randomize_joint_default",
            mode="startup",
            target=EntityRef("robot", joints=".*"),
            params={"range": (-0.01, 0.01)},
        )
        self.base_com = EventTermSpec(
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
        )
        self.randomize_ray_offsets = EventTermSpec(
            kind="randomize_ray_offsets",
            mode="startup",
            params={
                "sensor_name": "camera",
                "offset_pose_ranges": {
                    "x": (-0.01, 0.01),
                    "y": (-0.01, 0.01),
                    "z": (-0.01, 0.01),
                    "roll": (-0.034906585, 0.034906585),
                    "pitch": (-0.174532925, 0.174532925),
                    "yaw": (-0.034906585, 0.034906585),
                },
            },
        )
        self.randomize_actuator_gains = EventTermSpec(
            kind="randomize_actuator_gains",
            mode="startup",
            target=EntityRef("robot", joints=".*"),
            params={
                "stiffness_range": (0.8, 1.2),
                "damping_range": (0.9, 1.1),
                "operation": "scale",
            },
        )
        self.randomize_rigid_body_mass = EventTermSpec(
            kind="randomize_body_inertia",
            mode="startup",
            target=EntityRef(
                "robot",
                bodies=("torso_link", ".*ankle.*", ".*wrist.*"),
            ),
            params={"add_range": (0.8, 1.2), "operation": "scale"},
        )
        self.match_motion_ref_with_scene = EventTermSpec(
            func=shadowing_events.match_reference_origin,
            mode="startup",
            params={"motion_reference": "motion_reference"},
        )
        self.reset_robot = EventTermSpec(
            func=shadowing_events.reset_robot_from_reference,
            mode="reset",
            params={
                "motion_reference": "motion_reference",
                "position_offset": (0.0, 0.0, 0.0),
                "dof_vel_ratio": 1.0,
                "base_lin_vel_ratio": 1.0,
                "base_ang_vel_ratio": 1.0,
                "randomize_joint_pos_range": (-0.1, 0.1),
                "randomize_pose_range": {
                    "x": (-0.15, 0.15),
                    "y": (-0.15, 0.15),
                    "z": (0.0, 0.0),
                },
                "randomize_velocity_range": {},
            },
            engine_params={"isaacsim": {"root_velocity_frame": "com"}},
        )


class PerceptiveAdaptiveEventsCfg:
    def __init__(self) -> None:
        self.physics_material = EventTermSpec(
            kind="randomize_friction",
            mode="startup",
            target=EntityRef("robot", bodies=".*"),
            engine_params={
                "isaacsim": {
                    "static_friction_range": (1.25, 2.0),
                    "dynamic_friction_range": (1.2, 1.8),
                    "restitution_range": (0.0, 0.5),
                    "num_buckets": 64,
                    "make_consistent": True,
                },
                "mjlab": {
                    "ranges": {0: (1.25, 2.0), 1: (1.2, 1.8), 2: (0.0, 0.5)},
                    "operation": "abs",
                    "distribution": "uniform",
                },
            },
        )
        self.add_joint_default_pos = EventTermSpec(
            kind="randomize_joint_default",
            mode="startup",
            target=EntityRef("robot", joints=".*"),
            params={"range": (-0.01, 0.01)},
        )
        self.base_com = EventTermSpec(
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
        )
        self.randomize_ray_offsets = EventTermSpec(
            kind="randomize_ray_offsets",
            mode="startup",
            params={
                "sensor_name": "camera",
                "offset_pose_ranges": {
                    "x": (-0.01, 0.01),
                    "y": (-0.01, 0.01),
                    "z": (-0.01, 0.01),
                    "roll": (-0.034906585, 0.034906585),
                    "pitch": (-0.174532925, 0.174532925),
                    "yaw": (-0.034906585, 0.034906585),
                },
            },
        )
        self.randomize_actuator_gains = EventTermSpec(
            kind="randomize_actuator_gains",
            mode="startup",
            target=EntityRef("robot", joints=".*"),
            params={
                "stiffness_range": (0.8, 1.2),
                "damping_range": (0.9, 1.1),
                "operation": "scale",
            },
        )
        self.randomize_rigid_body_mass = EventTermSpec(
            kind="randomize_body_inertia",
            mode="startup",
            target=EntityRef(
                "robot",
                bodies=("torso_link", ".*ankle.*", ".*wrist.*"),
            ),
            params={"add_range": (0.8, 1.2), "operation": "scale"},
        )
        self.match_motion_ref_with_scene = EventTermSpec(
            func=shadowing_events.match_reference_origin,
            mode="startup",
            params={"motion_reference": "motion_reference"},
        )
        self.reset_robot = EventTermSpec(
            func=shadowing_events.reset_robot_from_reference,
            mode="reset",
            params={
                "motion_reference": "motion_reference",
                "position_offset": (0.0, 0.0, 0.0),
                "dof_vel_ratio": 1.0,
                "base_lin_vel_ratio": 1.0,
                "base_ang_vel_ratio": 1.0,
                "randomize_joint_pos_range": (-0.1, 0.1),
                "randomize_pose_range": {
                    "x": (-0.15, 0.15),
                    "y": (-0.15, 0.15),
                    "z": (0.0, 0.0),
                },
                "randomize_velocity_range": {},
            },
            engine_params={"isaacsim": {"root_velocity_frame": "com"}},
        )
        self.bin_fail_counter_smoothing = EventTermSpec(
            func=shadowing_events.smooth_bin_failures,
            mode="interval",
            interval_range_s=(0.02, 0.02),
            params={"motion_reference": "motion_reference"},
        )


class PerceptivePlayEventsCfg:
    def __init__(self) -> None:
        self.randomize_ray_offsets = EventTermSpec(
            kind="randomize_ray_offsets",
            mode="startup",
            params={
                "sensor_name": "camera",
                "offset_pose_ranges": {
                    "x": (-0.01, 0.01),
                    "y": (-0.01, 0.01),
                    "z": (-0.01, 0.01),
                    "roll": (-0.034906585, 0.034906585),
                    "pitch": (-0.174532925, 0.174532925),
                    "yaw": (-0.034906585, 0.034906585),
                },
            },
        )
        self.randomize_actuator_gains = EventTermSpec(
            kind="randomize_actuator_gains",
            mode="startup",
            target=EntityRef("robot", joints=".*"),
            params={
                "stiffness_range": (0.8, 1.2),
                "damping_range": (0.9, 1.1),
                "operation": "scale",
            },
        )
        self.randomize_rigid_body_mass = EventTermSpec(
            kind="randomize_body_inertia",
            mode="startup",
            target=EntityRef(
                "robot",
                bodies=("torso_link", ".*ankle.*", ".*wrist.*"),
            ),
            params={"add_range": (0.8, 1.2), "operation": "scale"},
        )
        self.match_motion_ref_with_scene = EventTermSpec(
            func=shadowing_events.match_reference_origin,
            mode="startup",
            params={"motion_reference": "motion_reference"},
        )
        self.reset_robot = EventTermSpec(
            func=shadowing_events.reset_robot_from_reference,
            mode="reset",
            params={
                "motion_reference": "motion_reference",
                "position_offset": (0.0, 0.0, 0.0),
                "dof_vel_ratio": 1.0,
                "base_lin_vel_ratio": 1.0,
                "base_ang_vel_ratio": 1.0,
                "randomize_joint_pos_range": (0.0, 0.0),
                "randomize_pose_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "randomize_velocity_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
            },
            engine_params={"isaacsim": {"root_velocity_frame": "com"}},
        )


class PerceptiveRewardsCfg:
    def __init__(self, motion_reference: MotionReferenceRef) -> None:
        self.base_position_imitation_gauss = RewardTermSpec(
            func=shadowing_terms.base_position_imitation,
            weight=0.5,
            params={
                "reference_cfg": motion_reference,
                "asset_cfg": EntityRef(
                    "robot",
                    bodies=motion_reference.links,
                    preserve_order=True,
                ),
                "std": 0.3,
            },
            level=Requirement.REQUIRED,
        )
        self.base_rot_imitation_gauss = RewardTermSpec(
            func=shadowing_terms.base_rotation_imitation,
            weight=0.5,
            params={
                "reference_cfg": motion_reference,
                "asset_cfg": EntityRef(
                    "robot",
                    bodies=motion_reference.links,
                    preserve_order=True,
                ),
                "std": 0.4,
                "difference_type": "axis_angle",
            },
            level=Requirement.REQUIRED,
        )
        self.link_pos_imitation_gauss = RewardTermSpec(
            func=shadowing_terms.link_position_imitation,
            weight=1.0,
            params={
                "reference_cfg": motion_reference,
                "asset_cfg": EntityRef(
                    "robot",
                    bodies=motion_reference.links,
                    preserve_order=True,
                ),
                "combine_method": "mean_prod",
                "in_base_frame": False,
                "in_relative_world_frame": True,
                "std": 0.3,
            },
            level=Requirement.REQUIRED,
        )
        self.link_rot_imitation_gauss = RewardTermSpec(
            func=shadowing_terms.link_rotation_imitation,
            weight=1.0,
            params={
                "reference_cfg": motion_reference,
                "asset_cfg": EntityRef(
                    "robot",
                    bodies=motion_reference.links,
                    preserve_order=True,
                ),
                "combine_method": "mean_prod",
                "in_base_frame": False,
                "in_relative_world_frame": True,
                "std": 0.4,
            },
            level=Requirement.REQUIRED,
        )
        self.link_lin_vel_imitation_gauss = RewardTermSpec(
            func=shadowing_terms.link_linear_velocity_imitation,
            weight=1.0,
            params={
                "reference_cfg": motion_reference,
                "asset_cfg": EntityRef(
                    "robot",
                    bodies=motion_reference.links,
                    preserve_order=True,
                ),
                "combine_method": "mean_prod",
                "std": 1.0,
            },
            level=Requirement.REQUIRED,
        )
        self.link_ang_vel_imitation_gauss = RewardTermSpec(
            func=shadowing_terms.link_angular_velocity_imitation,
            weight=1.0,
            params={
                "reference_cfg": motion_reference,
                "asset_cfg": EntityRef(
                    "robot",
                    bodies=motion_reference.links,
                    preserve_order=True,
                ),
                "combine_method": "mean_prod",
                "std": 3.14,
            },
            level=Requirement.REQUIRED,
        )
        self.action_rate_l2 = RewardTermSpec(
            func=rewards.action_rate_l2,
            weight=-0.1,
            level=Requirement.REQUIRED,
        )
        self.joint_limit = RewardTermSpec(
            func=rewards.joint_pos_limits,
            weight=-10.0,
            level=Requirement.REQUIRED,
        )
        self.undesired_contacts = RewardTermSpec(
            func=shadowing_terms.undesired_contacts,
            weight=-0.1,
            params={
                "threshold": 1.0,
                "sensor": ContactSensorRef(
                    name="contact_forces",
                    elements=(
                        r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
                        r"(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                    ),
                    track_air_time=True,
                    air_time_force_threshold=1.0,
                    engine_air_time_force_thresholds={"isaacsim": 10.0},
                    history_length=3,
                ),
            },
            level=Requirement.REQUIRED,
        )
        self.applied_torque_limits_by_ratio = RewardTermSpec(
            func=rewards.applied_torque_limits_by_ratio,
            weight=-0.05,
            params={
                "asset_cfg": EntityRef("robot", joints=(".*ankle.*", ".*wrist.*")),
                "limit_ratio": 0.8,
            },
            level=Requirement.REQUIRED,
        )


class PerceptiveTerminationsCfg:
    def __init__(self, motion_reference: MotionReferenceRef) -> None:
        self.time_out = DoneTermSpec(func=terminations.time_out, time_out=True)
        self.illegal_reset_contact = DoneTermSpec(
            func=shadowing_terms.IllegalResetContact,
            params={
                "sensor": ContactSensorRef(
                    name="contact_forces",
                    elements=(
                        r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
                        r"(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                    ),
                    track_air_time=True,
                    air_time_force_threshold=1.0,
                    engine_air_time_force_thresholds={"isaacsim": 10.0},
                    history_length=3,
                ),
                "threshold": 500.0,
                "episode_length_threshold": 2,
            },
            time_out=True,
        )
        self.base_pos_too_far = DoneTermSpec(
            func=shadowing_terms.base_position_too_far,
            params={
                "reference_cfg": motion_reference,
                "asset_cfg": EntityRef(
                    "robot",
                    bodies=motion_reference.links,
                    preserve_order=True,
                ),
                "distance_threshold": 0.25,
                "check_at_keyframe_threshold": -1,
                "print_reason": False,
                "height_only": True,
            },
        )
        self.base_pg_too_far = DoneTermSpec(
            func=shadowing_terms.projected_gravity_too_far,
            params={
                "reference_cfg": motion_reference,
                "asset_cfg": EntityRef(
                    "robot",
                    bodies=motion_reference.links,
                    preserve_order=True,
                ),
                "projected_gravity_threshold": 0.8,
                "check_at_keyframe_threshold": -1,
                "z_only": False,
                "print_reason": False,
            },
        )
        self.link_pos_too_far = DoneTermSpec(
            func=shadowing_terms.link_position_too_far,
            params={
                "reference_cfg": motion_reference,
                "asset_cfg": EntityRef(
                    "robot",
                    bodies=motion_reference.links,
                    preserve_order=True,
                ),
                "link_ids": (
                    motion_reference.links.index("left_ankle_roll_link"),
                    motion_reference.links.index("right_ankle_roll_link"),
                    motion_reference.links.index("left_wrist_yaw_link"),
                    motion_reference.links.index("right_wrist_yaw_link"),
                ),
                "distance_threshold": 0.25,
                "in_base_frame": False,
                "check_at_keyframe_threshold": -1,
                "height_only": True,
                "print_reason": False,
            },
        )
        self.dataset_exhausted = DoneTermSpec(
            func=terminations.dataset_exhausted,
            params={"sensor": motion_reference, "reset_without_notice": False},
            time_out=True,
        )
        self.out_of_border = DoneTermSpec(
            func=terminations.terrain_out_of_bounds,
            params={"distance_buffer": 0.1},
            time_out=True,
        )


class PerceptivePlayTerminationsCfg:
    def __init__(self, motion_reference: MotionReferenceRef) -> None:
        self.time_out = DoneTermSpec(func=terminations.time_out, time_out=True)
        self.illegal_reset_contact = DoneTermSpec(
            func=shadowing_terms.IllegalResetContact,
            params={
                "sensor": ContactSensorRef(
                    name="contact_forces",
                    elements=(
                        r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
                        r"(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                    ),
                    track_air_time=True,
                    air_time_force_threshold=1.0,
                    engine_air_time_force_thresholds={"isaacsim": 10.0},
                    history_length=3,
                ),
                "threshold": 500.0,
                "episode_length_threshold": 2,
            },
            time_out=True,
        )
        self.dataset_exhausted = DoneTermSpec(
            func=terminations.dataset_exhausted,
            params={"sensor": motion_reference, "reset_without_notice": False},
            time_out=True,
        )
        self.out_of_border = DoneTermSpec(
            func=terminations.terrain_out_of_bounds,
            params={"distance_buffer": 0.1},
            time_out=True,
        )


class PerceptiveVaePlayTerminationsCfg:
    def __init__(self, motion_reference: MotionReferenceRef) -> None:
        self.time_out = DoneTermSpec(func=terminations.time_out, time_out=True)
        self.illegal_reset_contact = DoneTermSpec(
            func=shadowing_terms.IllegalResetContact,
            params={
                "sensor": ContactSensorRef(
                    name="contact_forces",
                    elements=(
                        r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
                        r"(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                    ),
                    track_air_time=True,
                    air_time_force_threshold=1.0,
                    engine_air_time_force_thresholds={"isaacsim": 10.0},
                    history_length=3,
                ),
                "threshold": 500.0,
                "episode_length_threshold": 2,
            },
            time_out=True,
        )
        self.dataset_exhausted = DoneTermSpec(
            func=terminations.dataset_exhausted,
            params={"sensor": motion_reference, "reset_without_notice": True},
            time_out=True,
        )
        self.out_of_border = DoneTermSpec(
            func=terminations.terrain_out_of_bounds,
            params={"distance_buffer": 0.1},
            time_out=True,
        )


class PerceptiveCurriculumCfg:
    def __init__(self) -> None:
        self.beyond_adaptive_sampling = CurriculumTermSpec(
            func=shadowing_events.adaptive_sampling
        )


class PerceptiveShadowingEnvCfg:
    def __init__(
        self,
        robot: RobotSpec,
        motion_reference: MotionReferenceRef,
        motion_paths: dict[str, str],
        ray_casters: tuple[RayCasterRef, ...],
        env_spacing: float,
        simulation: SimSpec,
        policy_observations: object,
        critic_observations: object,
        events: dict[str, EventTermSpec],
        curriculum: dict[str, CurriculumTermSpec],
        task_terminations: dict[str, DoneTermSpec],
        agent: AgentSpec,
    ) -> None:
        self.robot = robot
        self.scene = SceneSpec(
            terrain=TerrainSpec(
                kind="motion_matched",
                params={
                    "engine_paths": motion_paths,
                    "metadata_yaml": "metadata.yaml",
                },
            ),
            contact_sensors=(
                ContactSensorRef(
                    name="contact_forces",
                    elements=".*",
                    track_air_time=True,
                    air_time_force_threshold=1.0,
                    engine_air_time_force_thresholds={"isaacsim": 10.0},
                    history_length=3,
                ),
            ),
            ray_casters=ray_casters,
            motion_references=(motion_reference,),
            env_spacing=env_spacing,
        )
        self.sim = simulation
        self.observations = {
            "policy": ObsGroupSpec(
                terms=dict(vars(policy_observations)),
                enable_corruption=True,
                concatenate_terms=False,
            ),
            "critic": ObsGroupSpec(
                terms=dict(vars(critic_observations)),
                enable_corruption=False,
                concatenate_terms=False,
            ),
        }
        self.actions = dict(vars(PerceptiveActionsCfg(robot)))
        self.commands = dict(vars(PerceptiveCommandsCfg()))
        self.rewards = {"rewards": dict(vars(PerceptiveRewardsCfg(motion_reference)))}
        self.events = events
        self.curriculum = curriculum
        self.terminations = task_terminations
        self.agent = agent
