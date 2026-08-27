"""G1 Perceptive Shadowing task configuration."""

from instinctlab.assets.unitree_g1.catalog import (
    make_g1_29dof_robot_spec,
    make_g1_29dof_shadowing_robot_spec,
)
from instinctlab.spec import MotionReferenceRef, TaskSpec
from instinctlab.tasks.shadowing.perceptive.perceptive_env_cfg import make_task

TASK_ID = "Instinct-Perceptive-Shadowing-G1-v0"
PLAY_TASK_ID = "Instinct-Perceptive-Shadowing-G1-Play-v0"
ONE_MOTION_TASK_ID = "Instinct-Perceptive-Shadowing-G1-OneMotion-v0"
ONE_MOTION_PLAY_TASK_ID = "Instinct-Perceptive-Shadowing-G1-OneMotion-Play-v0"
MOTION_PATH = (
    "~/Datasets/deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1"
)
MOTION_PATHS = {"isaacsim": MOTION_PATH, "mjlab": MOTION_PATH}
RUNNER = (
    "instinctlab.tasks.shadowing.perceptive.config.g1.agents.instinct_rl_ppo_cfg:"
    "G1PerceptiveShadowingPPORunnerCfg"
)
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


def make_motion_reference(play: bool, one_motion: bool) -> MotionReferenceRef:
    robot = make_g1_29dof_robot_spec()
    binned = not play and not one_motion
    return MotionReferenceRef(
        name="motion_reference",
        clip=MOTION_PATH,
        engine_clips=MOTION_PATHS,
        joints=tuple(robot.joint_names),
        links=MOTION_LINKS,
        num_frames=10,
        frame_interval_s=0.1,
        update_period=0.02,
        data_start_from="current_time",
        clip_target_fps=50.0,
        velocity_method="frontbackward",
        start_range=(0.0, 0.0),
        dataset_kind="terrain",
        metadata_yaml="metadata.yaml",
        first_motion_only=one_motion,
        sampling_strategy="concat_motion_bins" if binned else "independent",
        motion_bin_length_s=1.0 if binned else None,
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


def make_robot():
    return make_g1_29dof_shadowing_robot_spec()


def g1_perceptive_shadowing() -> TaskSpec:
    return make_task(
        TASK_ID,
        make_robot(),
        make_motion_reference(False, False),
        MOTION_PATHS,
        RUNNER,
        False,
        False,
        {},
    )


def g1_perceptive_shadowing_play() -> TaskSpec:
    return make_task(
        PLAY_TASK_ID,
        make_robot(),
        make_motion_reference(True, False),
        MOTION_PATHS,
        RUNNER,
        True,
        False,
        {},
    )


def g1_perceptive_shadowing_one_motion() -> TaskSpec:
    return make_task(
        ONE_MOTION_TASK_ID,
        make_robot(),
        make_motion_reference(False, True),
        MOTION_PATHS,
        RUNNER,
        False,
        False,
        {"experiment_name": "g1_perceptive_shadowing_one_motion"},
    )


def g1_perceptive_shadowing_one_motion_play() -> TaskSpec:
    return make_task(
        ONE_MOTION_PLAY_TASK_ID,
        make_robot(),
        make_motion_reference(True, True),
        MOTION_PATHS,
        RUNNER,
        True,
        False,
        {"experiment_name": "g1_perceptive_shadowing_one_motion"},
    )


__all__ = [
    "g1_perceptive_shadowing",
    "g1_perceptive_shadowing_one_motion",
    "g1_perceptive_shadowing_one_motion_play",
    "g1_perceptive_shadowing_play",
]
