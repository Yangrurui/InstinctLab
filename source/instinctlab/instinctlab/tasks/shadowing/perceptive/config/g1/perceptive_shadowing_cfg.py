"""G1 Perceptive Shadowing task configuration."""

import instinctlab.tasks.shadowing.perceptive.perceptive_env_cfg as perceptual_cfg
from instinctlab.engines.assets import RobotSpec
from instinctlab.spec import MotionReferenceRef, TaskSpec

TASK_ID = "Instinct-Perceptive-Shadowing-G1-v0"
PLAY_TASK_ID = "Instinct-Perceptive-Shadowing-G1-Play-v0"
ONE_MOTION_TASK_ID = "Instinct-Perceptive-Shadowing-G1-OneMotion-v0"
ONE_MOTION_PLAY_TASK_ID = "Instinct-Perceptive-Shadowing-G1-OneMotion-Play-v0"
MOTION_PATH = (
    "~/Datasets/deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1"
)
MOTION_PATHS = {"isaacsim": MOTION_PATH, "mjlab": MOTION_PATH}
RUNNER = "instinctlab.tasks.shadowing.perceptive.config.g1.agents.instinct_rl_ppo_cfg:G1PerceptiveShadowingPPORunnerCfg"
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


def make_motion_reference(
    robot: RobotSpec, play: bool, one_motion: bool
) -> MotionReferenceRef:
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


class G1PerceptiveShadowingEnvCfg(perceptual_cfg.PerceptiveShadowingEnvCfg):
    def __init__(self, robot: RobotSpec) -> None:
        super().__init__(
            robot=robot,
            motion_reference=make_motion_reference(robot, False, False),
            motion_paths=MOTION_PATHS,
            play=False,
            vae=False,
        )


class G1PerceptiveShadowingEnvCfg_PLAY(perceptual_cfg.PerceptiveShadowingEnvCfg):
    def __init__(self, robot: RobotSpec) -> None:
        super().__init__(
            robot=robot,
            motion_reference=make_motion_reference(robot, True, False),
            motion_paths=MOTION_PATHS,
            play=True,
            vae=False,
        )
        self.terminations.pop("base_pos_too_far")
        self.terminations.pop("base_pg_too_far")
        self.terminations.pop("link_pos_too_far")


class G1PerceptiveShadowingOneMotionEnvCfg(perceptual_cfg.PerceptiveShadowingEnvCfg):
    def __init__(self, robot: RobotSpec) -> None:
        super().__init__(
            robot=robot,
            motion_reference=make_motion_reference(robot, False, True),
            motion_paths=MOTION_PATHS,
            play=False,
            vae=False,
        )
        self.agent_overrides = {
            "experiment_name": "g1_perceptive_shadowing_one_motion",
        }


class G1PerceptiveShadowingOneMotionEnvCfg_PLAY(
    perceptual_cfg.PerceptiveShadowingEnvCfg
):
    def __init__(self, robot: RobotSpec) -> None:
        super().__init__(
            robot=robot,
            motion_reference=make_motion_reference(robot, True, True),
            motion_paths=MOTION_PATHS,
            play=True,
            vae=False,
        )
        self.terminations.pop("base_pos_too_far")
        self.terminations.pop("base_pg_too_far")
        self.terminations.pop("link_pos_too_far")
        self.agent_overrides = {
            "experiment_name": "g1_perceptive_shadowing_one_motion",
        }


def g1_perceptive_shadowing(robot: RobotSpec) -> TaskSpec:
    return G1PerceptiveShadowingEnvCfg(robot).to_task_spec(TASK_ID, RUNNER)


def g1_perceptive_shadowing_play(robot: RobotSpec) -> TaskSpec:
    return G1PerceptiveShadowingEnvCfg_PLAY(robot).to_task_spec(PLAY_TASK_ID, RUNNER)


def g1_perceptive_shadowing_one_motion(robot: RobotSpec) -> TaskSpec:
    return G1PerceptiveShadowingOneMotionEnvCfg(robot).to_task_spec(
        ONE_MOTION_TASK_ID, RUNNER
    )


def g1_perceptive_shadowing_one_motion_play(robot: RobotSpec) -> TaskSpec:
    return G1PerceptiveShadowingOneMotionEnvCfg_PLAY(robot).to_task_spec(
        ONE_MOTION_PLAY_TASK_ID, RUNNER
    )


__all__ = [
    "G1PerceptiveShadowingEnvCfg",
    "G1PerceptiveShadowingEnvCfg_PLAY",
    "G1PerceptiveShadowingOneMotionEnvCfg",
    "G1PerceptiveShadowingOneMotionEnvCfg_PLAY",
    "g1_perceptive_shadowing",
    "g1_perceptive_shadowing_one_motion",
    "g1_perceptive_shadowing_one_motion_play",
    "g1_perceptive_shadowing_play",
]
