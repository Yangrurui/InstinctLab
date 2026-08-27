"""G1 BeyondMimic task configuration."""

import instinctlab.tasks.shadowing.beyondmimic.beyondmimic_env_cfg as beyondmimic_cfg
from instinctlab.assets.unitree_g1.catalog import (
    make_g1_29dof_robot_spec,
    make_g1_29dof_shadowing_robot_spec,
)
from instinctlab.spec import MotionReferenceRef, TaskSpec

TASK_ID = "Instinct-BeyondMimic-Plane-G1-v0"
PLAY_TASK_ID = "Instinct-BeyondMimic-Plane-G1-Play-v0"
ISAAC_MOTION_PATH = "~/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz"
MJLAB_MOTION_PATH = "~/Xyk/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz"
MOTION_PATHS = {"isaacsim": ISAAC_MOTION_PATH, "mjlab": MJLAB_MOTION_PATH}
SELECTED_MOTION = "sprint1_subject2_retargetted.npz"
RUNNER = "instinctlab.tasks.shadowing.beyondmimic.config.g1.agents.beyondmimic_ppo_cfg:G1BeyondMimicPPORunnerCfg"
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
        num_frames=1,
        frame_interval_s=0.0,
        update_period=0.02,
        data_start_from="current_time",
        clip_target_fps=50.0,
        velocity_method="frontbackward",
        start_range=(0.0, 0.0) if play else (0.0, 0.8),
        dataset_kind="retargetted",
        selected_files=(SELECTED_MOTION,),
        sampling_strategy="independent",
        motion_bin_length_s=None if play else 1.0,
        exhaustion="freeze_last_and_flag",
        quaternion="wxyz",
        symmetric_augmentation=None,
    )


def make_robot():
    return make_g1_29dof_shadowing_robot_spec()


class G1BeyondMimicPlaneEnvCfg(beyondmimic_cfg.BeyondMimicEnvCfg):
    def __init__(self) -> None:
        super().__init__(
            robot=make_robot(),
            motion_reference=make_motion_reference(False),
            play=False,
        )


class G1BeyondMimicPlaneEnvCfg_PLAY(beyondmimic_cfg.BeyondMimicEnvCfg):
    def __init__(self) -> None:
        super().__init__(
            robot=make_robot(),
            motion_reference=make_motion_reference(True),
            play=True,
        )


def g1_beyondmimic_plane() -> TaskSpec:
    return G1BeyondMimicPlaneEnvCfg().to_task_spec(TASK_ID, RUNNER)


def g1_beyondmimic_plane_play() -> TaskSpec:
    return G1BeyondMimicPlaneEnvCfg_PLAY().to_task_spec(PLAY_TASK_ID, RUNNER)


__all__ = [
    "G1BeyondMimicPlaneEnvCfg",
    "G1BeyondMimicPlaneEnvCfg_PLAY",
    "g1_beyondmimic_plane",
    "g1_beyondmimic_plane_play",
]
