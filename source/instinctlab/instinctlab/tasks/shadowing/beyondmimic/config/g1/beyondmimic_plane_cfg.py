"""Explicit G1 configuration for BeyondMimic on a plane."""

from __future__ import annotations

from instinctlab_engine.spec import AgentSpec, MdpSpec, MotionReferenceRef, SimSpec, TaskSpec
from instinctlab_engine.spec.robot import RobotSpec
from instinctlab.tasks.shadowing.beyondmimic.beyondmimic_env_cfg import (
    BeyondMimicCurriculumCfg,
    BeyondMimicEnvCfg,
    BeyondMimicEventsCfg,
    BeyondMimicPlayEventsCfg,
)


class G1BeyondMimicPlaneEnvCfg(BeyondMimicEnvCfg):
    def __init__(self, robot: RobotSpec) -> None:
        motion_reference = MotionReferenceRef(
            name="motion_reference",
            clip=(
                "dataset://UbisoftLAFAN1_GMR_g1_29dof_torsoBase_"
                "retargetted_instinctnpz"
            ),
            engine_clips={
                "isaacsim": (
                    "dataset://UbisoftLAFAN1_GMR_g1_29dof_torsoBase_"
                    "retargetted_instinctnpz"
                ),
                "mjlab": (
                    "dataset://UbisoftLAFAN1_GMR_g1_29dof_torsoBase_"
                    "retargetted_instinctnpz"
                ),
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
            num_frames=1,
            frame_interval_s=0.0,
            update_period=0.02,
            data_start_from="current_time",
            clip_target_fps=50.0,
            velocity_method="frontbackward",
            start_range=(0.0, 0.8),
            dataset_kind="retargetted",
            selected_files=("sprint1_subject2_retargetted.npz",),
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
            simulation=SimSpec(
                physics_dt=0.005,
                decimation=4,
                episode_length_s=10.0,
                profiles={
                    "isaacsim": {
                        "use_terrain_physics_material": True,
                        "gpu_max_rigid_patch_count": 10 * 2**15,
                    },
                    "mjlab": {
                        "iterations": 10,
                        "ls_iterations": 20,
                        "njmax": 350,
                        "nconmax": 100,
                        "contact_sensor_maxmatch": 100,
                        "ccd_iterations": 80,
                    },
                },
            ),
            events=dict(vars(BeyondMimicEventsCfg())),
            curriculum=dict(vars(BeyondMimicCurriculumCfg())),
        )
        self.agent = AgentSpec(
            runner=(
                "instinctlab.tasks.shadowing.beyondmimic.config.g1.agents."
                "beyondmimic_ppo_cfg:G1BeyondMimicPPORunnerCfg"
            )
        )


class G1BeyondMimicPlaneEnvCfg_PLAY(BeyondMimicEnvCfg):
    def __init__(self, robot: RobotSpec) -> None:
        motion_reference = MotionReferenceRef(
            name="motion_reference",
            clip=(
                "dataset://UbisoftLAFAN1_GMR_g1_29dof_torsoBase_"
                "retargetted_instinctnpz"
            ),
            engine_clips={
                "isaacsim": (
                    "dataset://UbisoftLAFAN1_GMR_g1_29dof_torsoBase_"
                    "retargetted_instinctnpz"
                ),
                "mjlab": (
                    "dataset://UbisoftLAFAN1_GMR_g1_29dof_torsoBase_"
                    "retargetted_instinctnpz"
                ),
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
            num_frames=1,
            frame_interval_s=0.0,
            update_period=0.02,
            data_start_from="current_time",
            clip_target_fps=50.0,
            velocity_method="frontbackward",
            start_range=(0.0, 0.0),
            dataset_kind="retargetted",
            selected_files=("sprint1_subject2_retargetted.npz",),
            sampling_strategy="independent",
            motion_bin_length_s=None,
            exhaustion="freeze_last_and_flag",
            quaternion="wxyz",
            symmetric_augmentation=None,
        )
        super().__init__(
            robot=robot,
            motion_reference=motion_reference,
            env_spacing=2.5,
            simulation=SimSpec(
                physics_dt=0.005,
                decimation=4,
                episode_length_s=6000.0,
                profiles={
                    "isaacsim": {
                        "use_terrain_physics_material": True,
                        "gpu_max_rigid_patch_count": 10 * 2**15,
                    },
                    "mjlab": {
                        "iterations": 10,
                        "ls_iterations": 20,
                        "njmax": None,
                        "nconmax": None,
                        "contact_sensor_maxmatch": 500,
                        "ccd_iterations": 80,
                    },
                },
            ),
            events=dict(vars(BeyondMimicPlayEventsCfg())),
            curriculum={},
        )
        self.agent = AgentSpec(
            runner=(
                "instinctlab.tasks.shadowing.beyondmimic.config.g1.agents."
                "beyondmimic_ppo_cfg:G1BeyondMimicPPORunnerCfg"
            )
        )


def g1_beyondmimic_plane(robot: RobotSpec) -> TaskSpec:
    """Convert the complete training config at the registry boundary."""
    config = G1BeyondMimicPlaneEnvCfg(robot)
    return TaskSpec(
        task_id="Instinct-BeyondMimic-Plane-G1-v0",
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


def g1_beyondmimic_plane_play(robot: RobotSpec) -> TaskSpec:
    """Convert the complete play config at the registry boundary."""
    config = G1BeyondMimicPlaneEnvCfg_PLAY(robot)
    return TaskSpec(
        task_id="Instinct-BeyondMimic-Plane-G1-Play-v0",
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
