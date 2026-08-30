"""Explicit G1 configuration for Whole Body Shadowing on a plane."""

from __future__ import annotations

from instinctlab_engine.spec import AgentSpec, MdpSpec, MotionReferenceRef, TaskSpec
from instinctlab_engine.spec.robot import RobotSpec
from instinctlab.tasks.shadowing.whole_body.shadowing_env_cfg import (
    ShadowingCurriculumCfg,
    ShadowingEnvCfg,
    ShadowingEventsCfg,
    ShadowingPlayEventsCfg,
)


class G1PlaneShadowingEnvCfg(ShadowingEnvCfg):
    def __init__(self, robot: RobotSpec) -> None:
        motion_reference = MotionReferenceRef(
            name="motion_reference",
            clip="~/Datasets/NoKov-Marslab-Motions-instinctnpz/20251016_diveroll4_single",
            engine_clips={
                "isaacsim": "~/Datasets/NoKov-Marslab-Motions-instinctnpz/20251016_diveroll4_single",
                "mjlab": "~/Datasets/NoKov-Marslab-Motions-instinctnpz/20251016_diveroll4_single",
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
        super().__init__(
            robot=robot,
            motion_reference=motion_reference,
            env_spacing=4.0,
            events=dict(vars(ShadowingEventsCfg())),
            curriculum=dict(vars(ShadowingCurriculumCfg())),
        )
        self.agent = AgentSpec(
            runner=(
                "instinctlab.tasks.shadowing.whole_body.config.g1.agents."
                "instinct_rl_ppo_cfg:G1ShadowingPPORunnerCfg"
            )
        )


class G1PlaneShadowingEnvCfg_PLAY(ShadowingEnvCfg):
    def __init__(self, robot: RobotSpec) -> None:
        motion_reference = MotionReferenceRef(
            name="motion_reference",
            clip="~/Datasets/NoKov-Marslab-Motions-instinctnpz/20251016_diveroll4_single",
            engine_clips={
                "isaacsim": "~/Datasets/NoKov-Marslab-Motions-instinctnpz/20251016_diveroll4_single",
                "mjlab": "~/Datasets/NoKov-Marslab-Motions-instinctnpz/20251016_diveroll4_single",
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
        super().__init__(
            robot=robot,
            motion_reference=motion_reference,
            env_spacing=2.5,
            events=dict(vars(ShadowingPlayEventsCfg())),
            curriculum={},
        )
        self.agent = AgentSpec(
            runner=(
                "instinctlab.tasks.shadowing.whole_body.config.g1.agents."
                "instinct_rl_ppo_cfg:G1ShadowingPPORunnerCfg"
            )
        )


def g1_plane_shadowing(robot: RobotSpec) -> TaskSpec:
    """Convert the complete training config at the registry boundary."""
    config = G1PlaneShadowingEnvCfg(robot)
    return TaskSpec(
        task_id="Instinct-Shadowing-WholeBody-Plane-G1-v0",
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


def g1_plane_shadowing_play(robot: RobotSpec) -> TaskSpec:
    """Convert the complete play config at the registry boundary."""
    config = G1PlaneShadowingEnvCfg_PLAY(robot)
    return TaskSpec(
        task_id="Instinct-Shadowing-WholeBody-Plane-G1-Play-v0",
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
