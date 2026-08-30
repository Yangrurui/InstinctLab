"""Explicit G1 configurations for Perceptive HOI Shadowing."""

from __future__ import annotations

from instinctlab_engine.spec import (
    AgentSpec,
    CollisionExclusionRef,
    MdpSpec,
    MotionReferenceRef,
    RayCasterRef,
    RayPatternRef,
    RigidObjectRef,
    TaskSpec,
)
from instinctlab_engine.spec.robot import RobotSpec
from instinctlab.tasks.shadowing.perceptive_hoi.perceptive_env_cfg import (
    PerceptiveHoiCurriculumCfg,
    PerceptiveHoiEventsCfg,
    PerceptiveHoiPlayEventsCfg,
    PerceptiveHoiPlayTerminationsCfg,
    PerceptiveHoiShadowingEnvCfg,
    PerceptiveHoiTerminationsCfg,
)


class G1PerceptiveHoiShadowingEnvCfg(PerceptiveHoiShadowingEnvCfg):
    def __init__(self, robot: RobotSpec) -> None:
        objects = (
            RigidObjectRef(
                name="floorlamp",
                mesh="/localhdd/Datasets/OMOMO/data/captured_objects/floorlamp_cleaned_simplified.obj",
                engine_meshes={
                    "isaacsim": "/localhdd/Datasets/OMOMO/data/captured_objects/floorlamp_cleaned_simplified.obj",
                    "mjlab": "~/Datasets/OMOMO/data/captured_objects/floorlamp_cleaned_simplified.obj",
                },
                scale=(1.55 * 0.3793, 1.55 * 0.3793, 1.55 * 0.3793),
            ),
            RigidObjectRef(
                name="largebox",
                mesh="/localhdd/Datasets/OMOMO/data/captured_objects/largebox_cleaned_simplified.obj",
                engine_meshes={
                    "isaacsim": "/localhdd/Datasets/OMOMO/data/captured_objects/largebox_cleaned_simplified.obj",
                    "mjlab": "~/Datasets/OMOMO/data/captured_objects/largebox_cleaned_simplified.obj",
                },
                scale=(1.55 * 0.3486, 1.55 * 0.3486, 1.55 * 0.3486),
            ),
            RigidObjectRef(
                name="whitechair",
                mesh="/localhdd/Datasets/OMOMO/data/captured_objects/whitechair_cleaned_simplified.obj",
                engine_meshes={
                    "isaacsim": "/localhdd/Datasets/OMOMO/data/captured_objects/whitechair_cleaned_simplified.obj",
                    "mjlab": "~/Datasets/OMOMO/data/captured_objects/whitechair_cleaned_simplified.obj",
                },
                scale=(1.55 * 0.3129, 1.55 * 0.3129, 1.55 * 0.3129),
            ),
            RigidObjectRef(
                name="trashcan",
                mesh="/localhdd/Datasets/OMOMO/data/captured_objects/trashcan_cleaned_simplified.obj",
                engine_meshes={
                    "isaacsim": "/localhdd/Datasets/OMOMO/data/captured_objects/trashcan_cleaned_simplified.obj",
                    "mjlab": "~/Datasets/OMOMO/data/captured_objects/trashcan_cleaned_simplified.obj",
                },
                scale=(1.55 * 0.2326, 1.55 * 0.2326, 1.55 * 0.2326),
            ),
            RigidObjectRef(
                name="smalltable",
                mesh="/localhdd/Datasets/OMOMO/data/captured_objects/smalltable_cleaned_simplified.obj",
                engine_meshes={
                    "isaacsim": "/localhdd/Datasets/OMOMO/data/captured_objects/smalltable_cleaned_simplified.obj",
                    "mjlab": "~/Datasets/OMOMO/data/captured_objects/smalltable_cleaned_simplified.obj",
                },
                scale=(1.55 * 0.0162, 1.55 * 0.0162, 1.55 * 0.0162),
            ),
            RigidObjectRef(
                name="suitcase",
                mesh="/localhdd/Datasets/OMOMO/data/captured_objects/suitcase_cleaned_simplified.obj",
                engine_meshes={
                    "isaacsim": "/localhdd/Datasets/OMOMO/data/captured_objects/suitcase_cleaned_simplified.obj",
                    "mjlab": "~/Datasets/OMOMO/data/captured_objects/suitcase_cleaned_simplified.obj",
                },
                scale=(1.55 * 0.3672, 1.55 * 0.3672, 1.55 * 0.3672),
            ),
        )
        motion_reference = MotionReferenceRef(
            name="motion_reference",
            clip="/localhdd/Datasets/OMOMO/retargeted",
            engine_clips={
                "isaacsim": "/localhdd/Datasets/OMOMO/retargeted",
                "mjlab": "~/Datasets/OMOMO/retargeted_omniretarget_instinctmj_torso_v10_object_xy_align_foot_lock",
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
            scene_objects=(
                "floorlamp",
                "largebox",
                "whitechair",
                "trashcan",
                "smalltable",
                "suitcase",
            ),
            num_frames=10,
            frame_interval_s=0.1,
            update_period=0.02,
            data_start_from="current_time",
            clip_target_fps=50.0,
            velocity_method="frontbackward",
            start_range=(0.0, 0.0),
            dataset_kind="omomo",
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
                "floorlamp",
                "largebox",
                "whitechair",
                "trashcan",
                "smalltable",
                "suitcase",
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
            objects=objects,
            camera=camera,
            height_scanner=height_scanner,
            env_spacing=4.0,
            events=dict(vars(PerceptiveHoiEventsCfg())),
            curriculum=dict(vars(PerceptiveHoiCurriculumCfg())),
            task_terminations=dict(
                vars(PerceptiveHoiTerminationsCfg(motion_reference))
            ),
            agent=AgentSpec(
                runner=(
                    "instinctlab.tasks.shadowing.perceptive_hoi.config.g1.agents."
                    "instinct_rl_ppo_cfg:G1PerceptiveHoiShadowingPPORunnerCfg"
                )
            ),
        )


class G1PerceptiveHoiShadowingEnvCfg_PLAY(PerceptiveHoiShadowingEnvCfg):
    def __init__(self, robot: RobotSpec) -> None:
        objects = (
            RigidObjectRef(
                name="floorlamp",
                mesh="/localhdd/Datasets/OMOMO/data/captured_objects/floorlamp_cleaned_simplified.obj",
                engine_meshes={
                    "isaacsim": "/localhdd/Datasets/OMOMO/data/captured_objects/floorlamp_cleaned_simplified.obj",
                    "mjlab": "~/Datasets/OMOMO/data/captured_objects/floorlamp_cleaned_simplified.obj",
                },
                scale=(1.55 * 0.3793, 1.55 * 0.3793, 1.55 * 0.3793),
            ),
            RigidObjectRef(
                name="largebox",
                mesh="/localhdd/Datasets/OMOMO/data/captured_objects/largebox_cleaned_simplified.obj",
                engine_meshes={
                    "isaacsim": "/localhdd/Datasets/OMOMO/data/captured_objects/largebox_cleaned_simplified.obj",
                    "mjlab": "~/Datasets/OMOMO/data/captured_objects/largebox_cleaned_simplified.obj",
                },
                scale=(1.55 * 0.3486, 1.55 * 0.3486, 1.55 * 0.3486),
            ),
            RigidObjectRef(
                name="whitechair",
                mesh="/localhdd/Datasets/OMOMO/data/captured_objects/whitechair_cleaned_simplified.obj",
                engine_meshes={
                    "isaacsim": "/localhdd/Datasets/OMOMO/data/captured_objects/whitechair_cleaned_simplified.obj",
                    "mjlab": "~/Datasets/OMOMO/data/captured_objects/whitechair_cleaned_simplified.obj",
                },
                scale=(1.55 * 0.3129, 1.55 * 0.3129, 1.55 * 0.3129),
            ),
            RigidObjectRef(
                name="trashcan",
                mesh="/localhdd/Datasets/OMOMO/data/captured_objects/trashcan_cleaned_simplified.obj",
                engine_meshes={
                    "isaacsim": "/localhdd/Datasets/OMOMO/data/captured_objects/trashcan_cleaned_simplified.obj",
                    "mjlab": "~/Datasets/OMOMO/data/captured_objects/trashcan_cleaned_simplified.obj",
                },
                scale=(1.55 * 0.2326, 1.55 * 0.2326, 1.55 * 0.2326),
            ),
            RigidObjectRef(
                name="smalltable",
                mesh="/localhdd/Datasets/OMOMO/data/captured_objects/smalltable_cleaned_simplified.obj",
                engine_meshes={
                    "isaacsim": "/localhdd/Datasets/OMOMO/data/captured_objects/smalltable_cleaned_simplified.obj",
                    "mjlab": "~/Datasets/OMOMO/data/captured_objects/smalltable_cleaned_simplified.obj",
                },
                scale=(1.55 * 0.0162, 1.55 * 0.0162, 1.55 * 0.0162),
            ),
            RigidObjectRef(
                name="suitcase",
                mesh="/localhdd/Datasets/OMOMO/data/captured_objects/suitcase_cleaned_simplified.obj",
                engine_meshes={
                    "isaacsim": "/localhdd/Datasets/OMOMO/data/captured_objects/suitcase_cleaned_simplified.obj",
                    "mjlab": "~/Datasets/OMOMO/data/captured_objects/suitcase_cleaned_simplified.obj",
                },
                scale=(1.55 * 0.3672, 1.55 * 0.3672, 1.55 * 0.3672),
            ),
        )
        motion_reference = MotionReferenceRef(
            name="motion_reference",
            clip="/localhdd/Datasets/OMOMO/retargeted",
            engine_clips={
                "isaacsim": "/localhdd/Datasets/OMOMO/retargeted",
                "mjlab": "~/Datasets/OMOMO/retargeted_omniretarget_instinctmj_torso_v10_object_xy_align_foot_lock",
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
            scene_objects=(
                "floorlamp",
                "largebox",
                "whitechair",
                "trashcan",
                "smalltable",
                "suitcase",
            ),
            num_frames=10,
            frame_interval_s=0.1,
            update_period=0.02,
            data_start_from="current_time",
            clip_target_fps=50.0,
            velocity_method="frontbackward",
            start_range=(0.0, 0.0),
            dataset_kind="omomo",
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
                "floorlamp",
                "largebox",
                "whitechair",
                "trashcan",
                "smalltable",
                "suitcase",
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
            objects=objects,
            camera=camera,
            height_scanner=height_scanner,
            env_spacing=2.5,
            events=dict(vars(PerceptiveHoiPlayEventsCfg())),
            curriculum={},
            task_terminations=dict(
                vars(PerceptiveHoiPlayTerminationsCfg(motion_reference))
            ),
            agent=AgentSpec(
                runner=(
                    "instinctlab.tasks.shadowing.perceptive_hoi.config.g1.agents."
                    "instinct_rl_ppo_cfg:G1PerceptiveHoiShadowingPPORunnerCfg"
                )
            ),
        )


def g1_perceptive_hoi_shadowing(robot: RobotSpec) -> TaskSpec:
    config = G1PerceptiveHoiShadowingEnvCfg(robot)
    return TaskSpec(
        task_id="Instinct-Perceptive-HOI-Shadowing-G1-v0",
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


def g1_perceptive_hoi_shadowing_play(robot: RobotSpec) -> TaskSpec:
    config = G1PerceptiveHoiShadowingEnvCfg_PLAY(robot)
    return TaskSpec(
        task_id="Instinct-Perceptive-HOI-Shadowing-G1-Play-v0",
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
