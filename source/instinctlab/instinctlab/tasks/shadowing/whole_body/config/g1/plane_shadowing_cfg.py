"""G1 Whole Body Shadowing task configuration."""

import instinctlab.tasks.shadowing.whole_body.shadowing_env_cfg as shadowing_cfg
from instinctlab.sim.robot_spec import RobotSpec
from instinctlab.spec import MotionReferenceRef, TaskSpec

TASK_ID = "Instinct-Shadowing-WholeBody-Plane-G1-v0"
PLAY_TASK_ID = "Instinct-Shadowing-WholeBody-Plane-G1-Play-v0"
MOTION_PATH = "~/Datasets/NoKov-Marslab-Motions-instinctnpz/20251016_diveroll4_single"
RUNNER = "instinctlab.tasks.shadowing.whole_body.config.g1.agents.instinct_rl_ppo_cfg:G1ShadowingPPORunnerCfg"
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


def make_motion_reference(robot: RobotSpec) -> MotionReferenceRef:
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


class G1PlaneShadowingEnvCfg(shadowing_cfg.ShadowingEnvCfg):
    def __init__(self, robot: RobotSpec) -> None:
        super().__init__(
            robot=robot,
            motion_reference=make_motion_reference(robot),
            play=False,
        )


class G1PlaneShadowingEnvCfg_PLAY(shadowing_cfg.ShadowingEnvCfg):
    def __init__(self, robot: RobotSpec) -> None:
        super().__init__(
            robot=robot,
            motion_reference=make_motion_reference(robot),
            play=True,
        )


def g1_plane_shadowing(robot: RobotSpec) -> TaskSpec:
    return G1PlaneShadowingEnvCfg(robot).to_task_spec(TASK_ID, RUNNER)


def g1_plane_shadowing_play(robot: RobotSpec) -> TaskSpec:
    return G1PlaneShadowingEnvCfg_PLAY(robot).to_task_spec(PLAY_TASK_ID, RUNNER)


__all__ = [
    "G1PlaneShadowingEnvCfg",
    "G1PlaneShadowingEnvCfg_PLAY",
    "g1_plane_shadowing",
    "g1_plane_shadowing_play",
]
