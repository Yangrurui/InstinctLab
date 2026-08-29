"""Explicit G1 configurations for Perceptive Shadowing."""

from __future__ import annotations

from instinctlab.engines.assets import RobotSpec
from instinctlab.spec import (
    AgentSpec,
    MdpSpec,
    MotionReferenceRef,
    RayCasterRef,
    RayPatternRef,
    SimSpec,
    TaskSpec,
)
from instinctlab.tasks.shadowing.perceptive.perceptive_env_cfg import (
    PerceptiveAdaptiveEventsCfg,
    PerceptiveCriticObsCfg,
    PerceptiveCurriculumCfg,
    PerceptiveEventsCfg,
    PerceptivePlayEventsCfg,
    PerceptivePlayTerminationsCfg,
    PerceptivePolicyObsCfg,
    PerceptiveShadowingEnvCfg,
    PerceptiveTerminationsCfg,
)


class G1PerceptiveShadowingEnvCfg(PerceptiveShadowingEnvCfg):
    def __init__(self, robot: RobotSpec) -> None:
        motion_reference = MotionReferenceRef(
            name="motion_reference",
            clip="~/Datasets/deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
            engine_clips={
                "isaacsim": "~/Datasets/deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
                "mjlab": "~/Datasets/deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
            },
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
            frame_interval_s=0.1,
            update_period=0.02,
            data_start_from="current_time",
            clip_target_fps=50.0,
            velocity_method="frontbackward",
            start_range=(0.0, 0.0),
            dataset_kind="terrain",
            metadata_yaml="metadata.yaml",
            first_motion_only=False,
            sampling_strategy="concat_motion_bins",
            motion_bin_length_s=1.0,
            ensure_link_below_zero_ground=True,
            motion_start_height_offset=0.1,
            engine_overrides={
                "isaacsim": {
                    "ensure_link_below_zero_ground": False,
                    "motion_start_height_offset": 0.0,
                }
            },
            exhaustion="freeze_last_and_flag",
            quaternion="wxyz",
            symmetric_augmentation=None,
        )
        camera = RayCasterRef(
            name="camera",
            attach="torso_link",
            offset=(0.0487988662332928, 0.015, 0.4378029937970051),
            offset_rot=(
                0.9135367613482678,
                0.004363309284746571,
                0.4067366430758002,
                0.0,
            ),
            offset_convention="world",
            pattern=RayPatternRef(
                kind="pinhole",
                width=48,
                height=27,
                horizontal_fov_deg=87.0,
                vertical_fov_deg=58.0,
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
            max_distance=1.0e6,
            min_distance=0.05,
            crop=(2, 2, 2, 2),
            update_period=1.0 / 60.0,
        )
        height_scanner = RayCasterRef(
            name="height_scanner",
            mode="terrain_height",
            attach="torso_link",
            offset=(0.0, 0.0, 20.0),
            pattern=RayPatternRef(
                kind="grid",
                resolution=0.1,
                size=(1.6, 1.0),
            ),
            hit="terrain",
            ray_alignment="yaw",
            miss="infinity",
            max_distance=1.0e6,
            engine_max_distances={"isaacsim": 1.0e6, "mjlab": 5.0},
            update_period=0.02,
        )
        super().__init__(
            robot=robot,
            motion_reference=motion_reference,
            motion_paths={
                "isaacsim": "~/Datasets/deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
                "mjlab": "~/Datasets/deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
            },
            ray_casters=(camera, height_scanner),
            env_spacing=4.0,
            simulation=SimSpec(
                physics_dt=0.005,
                decimation=4,
                episode_length_s=10.0,
                profiles={
                    "isaacsim": {
                        "use_terrain_physics_material": True,
                        "gpu_max_rigid_patch_count": 10 * 2**15,
                        "gpu_max_rigid_contact_count": 2**27,
                        "gpu_collision_stack_size": 2**27,
                    },
                    "mjlab": {
                        "iterations": 10,
                        "ls_iterations": 20,
                        "njmax": 1200,
                        "nconmax": None,
                        "pinhole_cameras": {
                            "camera": {
                                "include_geom_groups": (0, 2),
                                "exclude_parent_body": False,
                                "mesh_filter_max_hops": 24,
                                "mesh_filter_epsilon": 1.0e-4,
                                "update_period": 1.0 / 60.0,
                            }
                        },
                    },
                },
            ),
            policy_observations=PerceptivePolicyObsCfg(robot, camera),
            critic_observations=PerceptiveCriticObsCfg(
                robot,
                motion_reference,
                height_scanner,
            ),
            events=dict(vars(PerceptiveAdaptiveEventsCfg())),
            curriculum=dict(vars(PerceptiveCurriculumCfg())),
            task_terminations=dict(vars(PerceptiveTerminationsCfg(motion_reference))),
            agent=AgentSpec(
                runner=(
                    "instinctlab.tasks.shadowing.perceptive.config.g1.agents."
                    "instinct_rl_ppo_cfg:G1PerceptiveShadowingPPORunnerCfg"
                )
            ),
        )


class G1PerceptiveShadowingEnvCfg_PLAY(PerceptiveShadowingEnvCfg):
    def __init__(self, robot: RobotSpec) -> None:
        motion_reference = MotionReferenceRef(
            name="motion_reference",
            clip="~/Datasets/deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
            engine_clips={
                "isaacsim": "~/Datasets/deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
                "mjlab": "~/Datasets/deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
            },
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
            frame_interval_s=0.1,
            update_period=0.02,
            data_start_from="current_time",
            clip_target_fps=50.0,
            velocity_method="frontbackward",
            start_range=(0.0, 0.0),
            dataset_kind="terrain",
            metadata_yaml="metadata.yaml",
            first_motion_only=False,
            sampling_strategy="independent",
            motion_bin_length_s=None,
            ensure_link_below_zero_ground=True,
            motion_start_height_offset=0.1,
            engine_overrides={
                "isaacsim": {
                    "ensure_link_below_zero_ground": False,
                    "motion_start_height_offset": 0.0,
                }
            },
            exhaustion="freeze_last_and_flag",
            quaternion="wxyz",
            symmetric_augmentation=None,
        )
        camera = RayCasterRef(
            name="camera",
            attach="torso_link",
            offset=(0.0487988662332928, 0.015, 0.4378029937970051),
            offset_rot=(
                0.9135367613482678,
                0.004363309284746571,
                0.4067366430758002,
                0.0,
            ),
            offset_convention="world",
            pattern=RayPatternRef(
                kind="pinhole",
                width=48,
                height=27,
                horizontal_fov_deg=87.0,
                vertical_fov_deg=58.0,
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
            max_distance=1.0e6,
            min_distance=0.05,
            crop=(2, 2, 2, 2),
            update_period=1.0 / 60.0,
        )
        height_scanner = RayCasterRef(
            name="height_scanner",
            mode="terrain_height",
            attach="torso_link",
            offset=(0.0, 0.0, 20.0),
            pattern=RayPatternRef(
                kind="grid",
                resolution=0.1,
                size=(1.6, 1.0),
            ),
            hit="terrain",
            ray_alignment="yaw",
            miss="infinity",
            max_distance=1.0e6,
            engine_max_distances={"isaacsim": 1.0e6, "mjlab": 5.0},
            update_period=0.02,
        )
        super().__init__(
            robot=robot,
            motion_reference=motion_reference,
            motion_paths={
                "isaacsim": "~/Datasets/deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
                "mjlab": "~/Datasets/deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
            },
            ray_casters=(camera, height_scanner),
            env_spacing=2.5,
            simulation=SimSpec(
                physics_dt=0.005,
                decimation=4,
                episode_length_s=10.0,
                profiles={
                    "isaacsim": {
                        "use_terrain_physics_material": True,
                        "gpu_max_rigid_patch_count": 10 * 2**15,
                        "gpu_max_rigid_contact_count": 2**27,
                        "gpu_collision_stack_size": 2**27,
                    },
                    "mjlab": {
                        "iterations": 10,
                        "ls_iterations": 20,
                        "njmax": 1200,
                        "nconmax": None,
                        "pinhole_cameras": {
                            "camera": {
                                "include_geom_groups": (0, 2),
                                "exclude_parent_body": False,
                                "mesh_filter_max_hops": 24,
                                "mesh_filter_epsilon": 1.0e-4,
                                "update_period": 1.0 / 60.0,
                            }
                        },
                    },
                },
            ),
            policy_observations=PerceptivePolicyObsCfg(robot, camera),
            critic_observations=PerceptiveCriticObsCfg(
                robot,
                motion_reference,
                height_scanner,
            ),
            events=dict(vars(PerceptivePlayEventsCfg())),
            curriculum={},
            task_terminations=dict(
                vars(PerceptivePlayTerminationsCfg(motion_reference))
            ),
            agent=AgentSpec(
                runner=(
                    "instinctlab.tasks.shadowing.perceptive.config.g1.agents."
                    "instinct_rl_ppo_cfg:G1PerceptiveShadowingPPORunnerCfg"
                )
            ),
        )


class G1PerceptiveShadowingOneMotionEnvCfg(PerceptiveShadowingEnvCfg):
    def __init__(self, robot: RobotSpec) -> None:
        motion_reference = MotionReferenceRef(
            name="motion_reference",
            clip="~/Datasets/deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
            engine_clips={
                "isaacsim": "~/Datasets/deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
                "mjlab": "~/Datasets/deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
            },
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
            frame_interval_s=0.1,
            update_period=0.02,
            data_start_from="current_time",
            clip_target_fps=50.0,
            velocity_method="frontbackward",
            start_range=(0.0, 0.0),
            dataset_kind="terrain",
            metadata_yaml="metadata.yaml",
            first_motion_only=True,
            sampling_strategy="independent",
            motion_bin_length_s=None,
            ensure_link_below_zero_ground=True,
            motion_start_height_offset=0.1,
            engine_overrides={
                "isaacsim": {
                    "ensure_link_below_zero_ground": False,
                    "motion_start_height_offset": 0.0,
                }
            },
            exhaustion="freeze_last_and_flag",
            quaternion="wxyz",
            symmetric_augmentation=None,
        )
        camera = RayCasterRef(
            name="camera",
            attach="torso_link",
            offset=(0.0487988662332928, 0.015, 0.4378029937970051),
            offset_rot=(
                0.9135367613482678,
                0.004363309284746571,
                0.4067366430758002,
                0.0,
            ),
            offset_convention="world",
            pattern=RayPatternRef(
                kind="pinhole",
                width=48,
                height=27,
                horizontal_fov_deg=87.0,
                vertical_fov_deg=58.0,
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
            max_distance=1.0e6,
            min_distance=0.05,
            crop=(2, 2, 2, 2),
            update_period=1.0 / 60.0,
        )
        height_scanner = RayCasterRef(
            name="height_scanner",
            mode="terrain_height",
            attach="torso_link",
            offset=(0.0, 0.0, 20.0),
            pattern=RayPatternRef(
                kind="grid",
                resolution=0.1,
                size=(1.6, 1.0),
            ),
            hit="terrain",
            ray_alignment="yaw",
            miss="infinity",
            max_distance=1.0e6,
            engine_max_distances={"isaacsim": 1.0e6, "mjlab": 5.0},
            update_period=0.02,
        )
        super().__init__(
            robot=robot,
            motion_reference=motion_reference,
            motion_paths={
                "isaacsim": "~/Datasets/deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
                "mjlab": "~/Datasets/deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
            },
            ray_casters=(camera, height_scanner),
            env_spacing=4.0,
            simulation=SimSpec(
                physics_dt=0.005,
                decimation=4,
                episode_length_s=10.0,
                profiles={
                    "isaacsim": {
                        "use_terrain_physics_material": True,
                        "gpu_max_rigid_patch_count": 10 * 2**15,
                        "gpu_max_rigid_contact_count": 2**27,
                        "gpu_collision_stack_size": 2**27,
                    },
                    "mjlab": {
                        "iterations": 10,
                        "ls_iterations": 20,
                        "njmax": 1200,
                        "nconmax": None,
                        "pinhole_cameras": {
                            "camera": {
                                "include_geom_groups": (0, 2),
                                "exclude_parent_body": False,
                                "mesh_filter_max_hops": 24,
                                "mesh_filter_epsilon": 1.0e-4,
                                "update_period": 1.0 / 60.0,
                            }
                        },
                    },
                },
            ),
            policy_observations=PerceptivePolicyObsCfg(robot, camera),
            critic_observations=PerceptiveCriticObsCfg(
                robot,
                motion_reference,
                height_scanner,
            ),
            events=dict(vars(PerceptiveEventsCfg())),
            curriculum={},
            task_terminations=dict(vars(PerceptiveTerminationsCfg(motion_reference))),
            agent=AgentSpec(
                runner=(
                    "instinctlab.tasks.shadowing.perceptive.config.g1.agents."
                    "instinct_rl_ppo_cfg:G1PerceptiveShadowingPPORunnerCfg"
                ),
                overrides={"experiment_name": "g1_perceptive_shadowing_one_motion"},
            ),
        )


class G1PerceptiveShadowingOneMotionEnvCfg_PLAY(PerceptiveShadowingEnvCfg):
    def __init__(self, robot: RobotSpec) -> None:
        motion_reference = MotionReferenceRef(
            name="motion_reference",
            clip="~/Datasets/deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
            engine_clips={
                "isaacsim": "~/Datasets/deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
                "mjlab": "~/Datasets/deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
            },
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
            frame_interval_s=0.1,
            update_period=0.02,
            data_start_from="current_time",
            clip_target_fps=50.0,
            velocity_method="frontbackward",
            start_range=(0.0, 0.0),
            dataset_kind="terrain",
            metadata_yaml="metadata.yaml",
            first_motion_only=True,
            sampling_strategy="independent",
            motion_bin_length_s=None,
            ensure_link_below_zero_ground=True,
            motion_start_height_offset=0.1,
            engine_overrides={
                "isaacsim": {
                    "ensure_link_below_zero_ground": False,
                    "motion_start_height_offset": 0.0,
                }
            },
            exhaustion="freeze_last_and_flag",
            quaternion="wxyz",
            symmetric_augmentation=None,
        )
        camera = RayCasterRef(
            name="camera",
            attach="torso_link",
            offset=(0.0487988662332928, 0.015, 0.4378029937970051),
            offset_rot=(
                0.9135367613482678,
                0.004363309284746571,
                0.4067366430758002,
                0.0,
            ),
            offset_convention="world",
            pattern=RayPatternRef(
                kind="pinhole",
                width=48,
                height=27,
                horizontal_fov_deg=87.0,
                vertical_fov_deg=58.0,
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
            max_distance=1.0e6,
            min_distance=0.05,
            crop=(2, 2, 2, 2),
            update_period=1.0 / 60.0,
        )
        height_scanner = RayCasterRef(
            name="height_scanner",
            mode="terrain_height",
            attach="torso_link",
            offset=(0.0, 0.0, 20.0),
            pattern=RayPatternRef(
                kind="grid",
                resolution=0.1,
                size=(1.6, 1.0),
            ),
            hit="terrain",
            ray_alignment="yaw",
            miss="infinity",
            max_distance=1.0e6,
            engine_max_distances={"isaacsim": 1.0e6, "mjlab": 5.0},
            update_period=0.02,
        )
        super().__init__(
            robot=robot,
            motion_reference=motion_reference,
            motion_paths={
                "isaacsim": "~/Datasets/deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
                "mjlab": "~/Datasets/deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
            },
            ray_casters=(camera, height_scanner),
            env_spacing=2.5,
            simulation=SimSpec(
                physics_dt=0.005,
                decimation=4,
                episode_length_s=10.0,
                profiles={
                    "isaacsim": {
                        "use_terrain_physics_material": True,
                        "gpu_max_rigid_patch_count": 10 * 2**15,
                        "gpu_max_rigid_contact_count": 2**27,
                        "gpu_collision_stack_size": 2**27,
                    },
                    "mjlab": {
                        "iterations": 10,
                        "ls_iterations": 20,
                        "njmax": 1200,
                        "nconmax": None,
                        "pinhole_cameras": {
                            "camera": {
                                "include_geom_groups": (0, 2),
                                "exclude_parent_body": False,
                                "mesh_filter_max_hops": 24,
                                "mesh_filter_epsilon": 1.0e-4,
                                "update_period": 1.0 / 60.0,
                            }
                        },
                    },
                },
            ),
            policy_observations=PerceptivePolicyObsCfg(robot, camera),
            critic_observations=PerceptiveCriticObsCfg(
                robot,
                motion_reference,
                height_scanner,
            ),
            events=dict(vars(PerceptivePlayEventsCfg())),
            curriculum={},
            task_terminations=dict(
                vars(PerceptivePlayTerminationsCfg(motion_reference))
            ),
            agent=AgentSpec(
                runner=(
                    "instinctlab.tasks.shadowing.perceptive.config.g1.agents."
                    "instinct_rl_ppo_cfg:G1PerceptiveShadowingPPORunnerCfg"
                ),
                overrides={"experiment_name": "g1_perceptive_shadowing_one_motion"},
            ),
        )


def g1_perceptive_shadowing(robot: RobotSpec) -> TaskSpec:
    config = G1PerceptiveShadowingEnvCfg(robot)
    return TaskSpec(
        task_id="Instinct-Perceptive-Shadowing-G1-v0",
        robot=config.robot,
        scene=config.scene,
        sim=config.sim,
        mdp=MdpSpec(
            observations=config.observations,
            actions=config.actions,
            commands=config.commands,
            rewards=config.rewards,
            events=config.events,
            curriculum=config.curriculum,
            terminations=config.terminations,
        ),
        agent=config.agent,
        engines=("isaacsim", "mjlab"),
    )


def g1_perceptive_shadowing_play(robot: RobotSpec) -> TaskSpec:
    config = G1PerceptiveShadowingEnvCfg_PLAY(robot)
    return TaskSpec(
        task_id="Instinct-Perceptive-Shadowing-G1-Play-v0",
        robot=config.robot,
        scene=config.scene,
        sim=config.sim,
        mdp=MdpSpec(
            observations=config.observations,
            actions=config.actions,
            commands=config.commands,
            rewards=config.rewards,
            events=config.events,
            curriculum=config.curriculum,
            terminations=config.terminations,
        ),
        agent=config.agent,
        engines=("isaacsim", "mjlab"),
    )


def g1_perceptive_shadowing_one_motion(robot: RobotSpec) -> TaskSpec:
    config = G1PerceptiveShadowingOneMotionEnvCfg(robot)
    return TaskSpec(
        task_id="Instinct-Perceptive-Shadowing-G1-OneMotion-v0",
        robot=config.robot,
        scene=config.scene,
        sim=config.sim,
        mdp=MdpSpec(
            observations=config.observations,
            actions=config.actions,
            commands=config.commands,
            rewards=config.rewards,
            events=config.events,
            curriculum=config.curriculum,
            terminations=config.terminations,
        ),
        agent=config.agent,
        engines=("isaacsim", "mjlab"),
    )


def g1_perceptive_shadowing_one_motion_play(robot: RobotSpec) -> TaskSpec:
    config = G1PerceptiveShadowingOneMotionEnvCfg_PLAY(robot)
    return TaskSpec(
        task_id="Instinct-Perceptive-Shadowing-G1-OneMotion-Play-v0",
        robot=config.robot,
        scene=config.scene,
        sim=config.sim,
        mdp=MdpSpec(
            observations=config.observations,
            actions=config.actions,
            commands=config.commands,
            rewards=config.rewards,
            events=config.events,
            curriculum=config.curriculum,
            terminations=config.terminations,
        ),
        agent=config.agent,
        engines=("isaacsim", "mjlab"),
    )
