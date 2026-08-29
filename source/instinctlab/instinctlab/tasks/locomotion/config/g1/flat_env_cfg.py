"""Explicit G1 selections for the robot-independent locomotion task."""

from __future__ import annotations

from instinctlab.engines.assets import RobotSpec
from instinctlab.spec import (
    ContactSensorRef,
    EntityRef,
    MdpSpec,
    RewardTermSpec,
    TaskSpec,
)
from instinctlab.spec.capability import Requirement
from instinctlab.tasks.locomotion.config.g1.rl_cfgs import G1_LOCOMOTION_TRAINING_CFG
from instinctlab.tasks.locomotion.config.locomotion_env_cfg import (
    LocomotionEnvCfg,
    LocomotionRewardsCfg,
)
from instinctlab.tasks.locomotion.mdp import rewards


class G1LocomotionRewardsCfg:
    def __init__(self) -> None:
        self.dof_pos_limits = RewardTermSpec(
            func=rewards.joint_pos_limits,
            weight=-1.0,
            params={
                "asset_cfg": EntityRef(
                    "robot",
                    joints=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
                )
            },
        )
        self.joint_deviation_hip = RewardTermSpec(
            func=rewards.joint_deviation_l1,
            weight=-0.1,
            params={
                "asset_cfg": EntityRef(
                    "robot",
                    joints=(".*_hip_yaw_joint", ".*_hip_roll_joint"),
                )
            },
        )
        self.joint_deviation_arms = RewardTermSpec(
            func=rewards.joint_deviation_l1,
            weight=-0.1,
            params={
                "asset_cfg": EntityRef(
                    "robot",
                    joints=(
                        ".*_shoulder_pitch_joint",
                        ".*_shoulder_roll_joint",
                        ".*_shoulder_yaw_joint",
                        ".*_elbow_joint",
                        ".*_wrist_roll_joint",
                        ".*_wrist_pitch_joint",
                        ".*_wrist_yaw_joint",
                    ),
                )
            },
        )
        self.joint_deviation_torso = RewardTermSpec(
            func=rewards.joint_deviation_l1,
            weight=-0.1,
            params={"asset_cfg": EntityRef("robot", joints="waist_.*")},
        )
        self.joint_deviation_knee = RewardTermSpec(
            func=rewards.joint_deviation_l1,
            weight=-0.05,
            params={"asset_cfg": EntityRef("robot", joints=(".*_knee_joint",))},
        )
        self.dof_acc_l2 = RewardTermSpec(
            func=rewards.joint_acc_l2,
            weight=-2.0e-7,
            params={
                "asset_cfg": EntityRef(
                    "robot",
                    joints=(".*_hip_.*", ".*_knee_joint"),
                )
            },
            level=Requirement.REQUIRED,
        )
        self.dof_torques_l2 = RewardTermSpec(
            func=rewards.joint_torques_l2,
            weight=-4.0e-6,
            params={
                "asset_cfg": EntityRef(
                    "robot",
                    joints=(".*_hip_.*", ".*_knee_joint"),
                )
            },
            level=Requirement.REQUIRED,
        )


class G1LocomotionFlatEnvCfg(LocomotionEnvCfg):
    def __init__(self, robot: RobotSpec) -> None:
        locomotion_rewards = LocomotionRewardsCfg(
            feet_contact=ContactSensorRef(
                name="contact_forces",
                elements=".*_ankle_roll_link",
            ),
            feet=EntityRef("robot", bodies=".*_ankle_roll_link"),
        )
        g1_rewards = G1LocomotionRewardsCfg()
        super().__init__(
            robot=robot,
            rewards={
                "rewards": {
                    "termination_penalty": locomotion_rewards.termination_penalty,
                    "track_lin_vel_xy_exp": locomotion_rewards.track_lin_vel_xy_exp,
                    "track_ang_vel_z_exp": locomotion_rewards.track_ang_vel_z_exp,
                    "feet_air_time": locomotion_rewards.feet_air_time,
                    "feet_slide": locomotion_rewards.feet_slide,
                    "flat_orientation_l2": locomotion_rewards.flat_orientation_l2,
                    "stand_still": locomotion_rewards.stand_still,
                    "dof_pos_limits": g1_rewards.dof_pos_limits,
                    "joint_deviation_hip": g1_rewards.joint_deviation_hip,
                    "joint_deviation_arms": g1_rewards.joint_deviation_arms,
                    "joint_deviation_torso": g1_rewards.joint_deviation_torso,
                    "joint_deviation_knee": g1_rewards.joint_deviation_knee,
                    "lin_vel_z_l2": locomotion_rewards.lin_vel_z_l2,
                    "action_rate_l2": locomotion_rewards.action_rate_l2,
                    "dof_acc_l2": g1_rewards.dof_acc_l2,
                    "dof_torques_l2": g1_rewards.dof_torques_l2,
                }
            },
            illegal_contact=ContactSensorRef(
                name="contact_forces",
                elements=(
                    "torso_link",
                    ".*_shoulder_.*",
                    ".*_elbow_.*",
                    ".*_wrist_.*",
                    ".*_hip_.*",
                    ".*_knee_.*",
                ),
            ),
            agent=G1_LOCOMOTION_TRAINING_CFG,
        )


def flat_g1(robot: RobotSpec) -> TaskSpec:
    """Convert the explicit Flat config at the task registry boundary."""
    config = G1LocomotionFlatEnvCfg(robot)
    return TaskSpec(
        task_id="Instinct-Velocity-Flat-G1",
        robot=config.robot,
        scene=config.scene,
        sim=config.sim,
        mdp=MdpSpec(
            observations=config.observations,
            actions=config.actions,
            commands=config.commands,
            rewards=config.rewards,
            terminations=config.terminations,
            events=config.events,
            curriculum=config.curriculum,
        ),
        agent=config.agent,
        engines=("isaacsim", "mjlab"),
    )
