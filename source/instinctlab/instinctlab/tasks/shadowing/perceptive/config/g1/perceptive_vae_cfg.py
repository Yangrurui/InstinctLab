"""G1 Perceptive VAE task configuration."""

from instinctlab.assets.unitree_g1.catalog import (
    make_g1_29dof_robot_spec,
    make_g1_29dof_shadowing_robot_spec,
)
from instinctlab.spec import MotionReferenceRef, TaskSpec
from instinctlab.tasks.shadowing.perceptive.perceptive_env_cfg import make_task

TASK_ID = "Instinct-Perceptive-Vae-G1-v0"
PLAY_TASK_ID = "Instinct-Perceptive-Vae-G1-Play-v0"
ISAAC_MOTION_PATH = (
    "~/Datasets/NoKov-Marslab-Motions-instinctnpz/20251116_50cm_kneeClimbStep1"
)
MJLAB_MOTION_PATH = "~/Xyk/Datasets/20260317_50cm_kneeClimbStep1_projectInstinct"
MOTION_PATHS = {"isaacsim": ISAAC_MOTION_PATH, "mjlab": MJLAB_MOTION_PATH}
RUNNER = (
    "instinctlab.tasks.shadowing.perceptive.config.g1.agents.instinct_rl_vae_cfg:"
    "G1PerceptiveVaePPORunnerCfg"
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


def make_motion_reference(play: bool) -> MotionReferenceRef:
    robot = make_g1_29dof_robot_spec()
    return MotionReferenceRef(
        name="motion_reference",
        clip=ISAAC_MOTION_PATH,
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
        sampling_strategy="independent" if play else "concat_motion_bins",
        motion_bin_length_s=None if play else 1.0,
        exhaustion="freeze_last_and_flag",
        quaternion="wxyz",
        symmetric_augmentation=None,
    )


def make_robot():
    return make_g1_29dof_shadowing_robot_spec()


def g1_perceptive_vae() -> TaskSpec:
    return make_task(
        TASK_ID,
        make_robot(),
        make_motion_reference(False),
        MOTION_PATHS,
        RUNNER,
        False,
        True,
        {},
    )


def g1_perceptive_vae_play() -> TaskSpec:
    return make_task(
        PLAY_TASK_ID,
        make_robot(),
        make_motion_reference(True),
        MOTION_PATHS,
        RUNNER,
        True,
        True,
        {},
    )


__all__ = ["g1_perceptive_vae", "g1_perceptive_vae_play"]
