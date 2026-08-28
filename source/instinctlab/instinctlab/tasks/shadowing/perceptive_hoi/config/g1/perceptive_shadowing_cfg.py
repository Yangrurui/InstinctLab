"""G1 Perceptive HOI Shadowing task configuration."""

import instinctlab.tasks.shadowing.perceptive_hoi.perceptive_env_cfg as perceptual_cfg
from instinctlab.assets.unitree_g1 import (
    make_g1_29dof_robot_spec,
    make_g1_29dof_shadowing_robot_spec,
)
from instinctlab.spec import MotionReferenceRef, RigidObjectRef, TaskSpec

TASK_ID = "Instinct-Perceptive-HOI-Shadowing-G1-v0"
PLAY_TASK_ID = "Instinct-Perceptive-HOI-Shadowing-G1-Play-v0"
ISAAC_MOTION_PATH = "/localhdd/Datasets/OMOMO/retargeted"
MJLAB_MOTION_PATH = "~/Datasets/OMOMO/retargeted_omniretarget_instinctmj_torso_v10_object_xy_align_foot_lock"
MOTION_PATHS = {"isaacsim": ISAAC_MOTION_PATH, "mjlab": MJLAB_MOTION_PATH}
RUNNER = (
    "instinctlab.tasks.shadowing.perceptive_hoi.config.g1.agents.instinct_rl_ppo_cfg:"
    "G1PerceptiveHoiShadowingPPORunnerCfg"
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


def make_objects() -> tuple[RigidObjectRef, ...]:
    return (
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


def make_motion_reference(play: bool) -> MotionReferenceRef:
    robot = make_g1_29dof_robot_spec()
    return MotionReferenceRef(
        name="motion_reference",
        clip=ISAAC_MOTION_PATH,
        engine_clips=MOTION_PATHS,
        joints=tuple(robot.joint_names),
        links=MOTION_LINKS,
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
        sampling_strategy="independent" if play else "concat_motion_bins",
        motion_bin_length_s=None if play else 1.0,
        exhaustion="freeze_last_and_flag",
        quaternion="wxyz",
        symmetric_augmentation=None,
    )


def make_robot():
    return make_g1_29dof_shadowing_robot_spec()


class G1PerceptiveHoiShadowingEnvCfg(perceptual_cfg.PerceptiveHoiShadowingEnvCfg):
    def __init__(self) -> None:
        super().__init__(
            robot=make_robot(),
            motion_reference=make_motion_reference(False),
            objects=make_objects(),
            play=False,
        )


class G1PerceptiveHoiShadowingEnvCfg_PLAY(perceptual_cfg.PerceptiveHoiShadowingEnvCfg):
    def __init__(self) -> None:
        super().__init__(
            robot=make_robot(),
            motion_reference=make_motion_reference(True),
            objects=make_objects(),
            play=True,
        )
        self.terminations.pop("base_pos_too_far")
        self.terminations.pop("base_pg_too_far")
        self.terminations.pop("link_pos_too_far")


def g1_perceptive_hoi_shadowing() -> TaskSpec:
    return G1PerceptiveHoiShadowingEnvCfg().to_task_spec(TASK_ID, RUNNER)


def g1_perceptive_hoi_shadowing_play() -> TaskSpec:
    return G1PerceptiveHoiShadowingEnvCfg_PLAY().to_task_spec(PLAY_TASK_ID, RUNNER)


__all__ = [
    "G1PerceptiveHoiShadowingEnvCfg",
    "G1PerceptiveHoiShadowingEnvCfg_PLAY",
    "g1_perceptive_hoi_shadowing",
    "g1_perceptive_hoi_shadowing_play",
]
