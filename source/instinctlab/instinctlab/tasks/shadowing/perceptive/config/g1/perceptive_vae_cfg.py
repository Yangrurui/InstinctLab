"""Explicit G1 configurations for Perceptive VAE."""

from __future__ import annotations

from instinctlab_engine.spec import (
    AgentSpec,
    CollisionExclusionRef,
    MdpSpec,
    MotionReferenceRef,
    RayCasterRef,
    RayPatternRef,
    SimSpec,
    TaskSpec,
)
from instinctlab_engine.spec.robot import RobotSpec
from instinctlab.tasks.shadowing.perceptive.perceptive_env_cfg import (
    PerceptiveAdaptiveEventsCfg,
    PerceptiveCurriculumCfg,
    PerceptivePlayEventsCfg,
    PerceptiveShadowingEnvCfg,
    PerceptiveTerminationsCfg,
    PerceptiveVaeCriticObsCfg,
    PerceptiveVaePlayTerminationsCfg,
    PerceptiveVaePolicyObsCfg,
)


class G1PerceptiveVaeEnvCfg(PerceptiveShadowingEnvCfg):
    def __init__(self, robot: RobotSpec) -> None:
        motion_reference = MotionReferenceRef(
            name="motion_reference",
            clip="dataset://deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
            engine_clips={
                "isaacsim": "dataset://deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
                "mjlab": "dataset://deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
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
            sampling_strategy="concat_motion_bins",
            motion_bin_length_s=1.0,
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
        super().__init__(
            robot=robot,
            collision_exclusions=(
                CollisionExclusionRef(
                    body_a="left_elbow_link",
                    body_b="left_wrist_pitch_link",
                ),
                CollisionExclusionRef(
                    body_a="right_elbow_link",
                    body_b="right_wrist_pitch_link",
                ),
                CollisionExclusionRef(
                    body_a="pelvis",
                    body_b="right_hip_roll_link",
                ),
                CollisionExclusionRef(
                    body_a="pelvis",
                    body_b="left_hip_roll_link",
                ),
            ),
            motion_reference=motion_reference,
            motion_paths={
                "isaacsim": "dataset://deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
                "mjlab": "dataset://deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
            },
            ray_casters=(camera,),
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
                        "njmax": 512,
                        "nconmax": 128,
                        "contact_sensor_maxmatch": 128,
                        "ccd_iterations": 128,
                        "jacobian": "sparse",
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
            policy_observations=PerceptiveVaePolicyObsCfg(robot, camera),
            critic_observations=PerceptiveVaeCriticObsCfg(robot, camera),
            events=dict(vars(PerceptiveAdaptiveEventsCfg())),
            curriculum=dict(vars(PerceptiveCurriculumCfg())),
            task_terminations=dict(vars(PerceptiveTerminationsCfg(motion_reference))),
            agent=AgentSpec(
                runner=(
                    "instinctlab.tasks.shadowing.perceptive.config.g1.agents."
                    "instinct_rl_vae_cfg:G1PerceptiveVaePPORunnerCfg"
                )
            ),
        )


class G1PerceptiveVaeEnvCfg_PLAY(PerceptiveShadowingEnvCfg):
    def __init__(self, robot: RobotSpec) -> None:
        motion_reference = MotionReferenceRef(
            name="motion_reference",
            clip="dataset://deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
            engine_clips={
                "isaacsim": "dataset://deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
                "mjlab": "dataset://deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
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
            sampling_strategy="independent",
            motion_bin_length_s=None,
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
        super().__init__(
            robot=robot,
            collision_exclusions=(
                CollisionExclusionRef(
                    body_a="left_elbow_link",
                    body_b="left_wrist_pitch_link",
                ),
                CollisionExclusionRef(
                    body_a="right_elbow_link",
                    body_b="right_wrist_pitch_link",
                ),
                CollisionExclusionRef(
                    body_a="pelvis",
                    body_b="right_hip_roll_link",
                ),
                CollisionExclusionRef(
                    body_a="pelvis",
                    body_b="left_hip_roll_link",
                ),
            ),
            motion_reference=motion_reference,
            motion_paths={
                "isaacsim": "dataset://deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
                "mjlab": "dataset://deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
            },
            ray_casters=(camera,),
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
                        "njmax": 512,
                        "nconmax": 128,
                        "contact_sensor_maxmatch": 128,
                        "ccd_iterations": 128,
                        "jacobian": "sparse",
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
            policy_observations=PerceptiveVaePolicyObsCfg(robot, camera),
            critic_observations=PerceptiveVaeCriticObsCfg(robot, camera),
            events=dict(vars(PerceptivePlayEventsCfg())),
            curriculum={},
            task_terminations=dict(
                vars(PerceptiveVaePlayTerminationsCfg(motion_reference))
            ),
            agent=AgentSpec(
                runner=(
                    "instinctlab.tasks.shadowing.perceptive.config.g1.agents."
                    "instinct_rl_vae_cfg:G1PerceptiveVaePPORunnerCfg"
                )
            ),
        )


def g1_perceptive_vae(robot: RobotSpec) -> TaskSpec:
    config = G1PerceptiveVaeEnvCfg(robot)
    return TaskSpec(
        task_id="Instinct-Perceptive-Vae-G1-v0",
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


def g1_perceptive_vae_play(robot: RobotSpec) -> TaskSpec:
    config = G1PerceptiveVaeEnvCfg_PLAY(robot)
    return TaskSpec(
        task_id="Instinct-Perceptive-Vae-G1-Play-v0",
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
