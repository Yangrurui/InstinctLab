"""Engine-neutral G1 Parkour task configuration."""

from instinctlab.sim.robot_spec import RobotSpec
from instinctlab.spec import MotionReferenceRef, SymmetricAugmentationSpec, TaskSpec
from instinctlab.tasks.parkour.config.parkour_env_cfg import (
    AMP_HISTORY,
    ParkourEnvCfg,
)

TASK_ID = "Instinct-Parkour-Target-G1"
PARKOUR_MOTION_CLIP = "~/Datasets/parkour_release/parkour_motion_reference/parkour_motion_without_run_retargetted.npz"
PARKOUR_MOTION_LINKS = (
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


def make_motion_reference(robot: RobotSpec) -> MotionReferenceRef:
    return MotionReferenceRef(
        name="motion_reference",
        clip=PARKOUR_MOTION_CLIP,
        joints=tuple(robot.joint_names),
        links=PARKOUR_MOTION_LINKS,
        num_frames=AMP_HISTORY,
        frame_interval_s=0.02,
        update_period=0.02,
        data_start_from="one_frame_interval",
        clip_target_fps=50.0,
        velocity_method="frontward",
        start_range=(0.0, 0.9),
        exhaustion="freeze_last_and_flag",
        quaternion="wxyz",
        symmetric_augmentation=SymmetricAugmentationSpec.from_left_right(
            tuple(robot.joint_names), PARKOUR_MOTION_LINKS
        ),
    )


class G1ParkourEnvCfg(ParkourEnvCfg):
    def __init__(self, robot: RobotSpec) -> None:
        super().__init__(robot=robot, motion_reference=make_motion_reference(robot))


def parkour_target_g1(robot: RobotSpec) -> TaskSpec:
    return G1ParkourEnvCfg(robot).to_task_spec(TASK_ID)


__all__ = [
    "G1ParkourEnvCfg",
    "parkour_target_g1",
]
