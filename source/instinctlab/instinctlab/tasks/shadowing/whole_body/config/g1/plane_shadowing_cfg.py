"""G1 Whole Body Shadowing task configuration."""

from instinctlab.assets.unitree_g1.catalog import (
    make_g1_29dof_robot_spec,
    make_g1_29dof_shadowing_robot_spec,
)
from instinctlab.spec import MotionReferenceRef, TaskSpec
from instinctlab.tasks.shadowing.whole_body.shadowing_env_cfg import make_task

TASK_ID = "Instinct-Shadowing-WholeBody-Plane-G1-v0"
PLAY_TASK_ID = "Instinct-Shadowing-WholeBody-Plane-G1-Play-v0"
MOTION_PATH = "~/Datasets/NoKov-Marslab-Motions-instinctnpz/20251016_diveroll4_single"
RUNNER = (
    "instinctlab.tasks.shadowing.whole_body.config.g1.agents.instinct_rl_ppo_cfg:"
    "G1ShadowingPPORunnerCfg"
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


def make_motion_reference() -> MotionReferenceRef:
    robot = make_g1_29dof_robot_spec()
    return MotionReferenceRef(
        name="motion_reference",
        clip=MOTION_PATH,
        engine_clips={"isaacsim": MOTION_PATH, "mjlab": MOTION_PATH},
        joints=tuple(robot.joint_names),
        links=MOTION_LINKS,
        num_frames=10,
        frame_interval_s=0.02,
        update_period=0.02,
        data_start_from="current_time",
        clip_target_fps=50.0,
        velocity_method="frontbackward",
        start_range=(0.0, 0.8),
        dataset_kind="retargetted",
        sampling_strategy="independent",
        motion_bin_length_s=1.0,
        exhaustion="freeze_last_and_flag",
        quaternion="wxyz",
        symmetric_augmentation=None,
    )


def g1_plane_shadowing() -> TaskSpec:
    robot = make_g1_29dof_shadowing_robot_spec()
    return make_task(TASK_ID, robot, make_motion_reference(), RUNNER, False)


def g1_plane_shadowing_play() -> TaskSpec:
    robot = make_g1_29dof_shadowing_robot_spec()
    return make_task(PLAY_TASK_ID, robot, make_motion_reference(), RUNNER, True)


__all__ = ["g1_plane_shadowing", "g1_plane_shadowing_play"]
