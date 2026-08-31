"""Explicit locked-arm G1 selections for the 15-DoF flat locomotion task."""

from __future__ import annotations

from instinctlab_engine.spec import (
    ActionTermSpec,
    ContactSensorRef,
    DoneTermSpec,
    EntityRef,
    EventTermSpec,
    MdpSpec,
    NoiseSpec,
    ObsGroupSpec,
    ObsTermSpec,
    RewardTermSpec,
    SceneSpec,
    TaskSpec,
)
from instinctlab_engine.spec.capability import Requirement
from instinctlab_engine.spec.robot import RobotSpec

from instinctlab.tasks.locomotion.config.g1.rl_cfgs import G1_LOCOMOTION_TRAINING_CFG
from instinctlab.tasks.locomotion.config.locomotion_env_cfg import (
    LocomotionCriticObsCfg,
    LocomotionEnvCfg,
    LocomotionPolicyObsCfg,
    LocomotionRewardsCfg,
)
from instinctlab.tasks.locomotion.mdp import observations, rewards, terminations


class G115DofLocomotionFlatPolicyObsCfg(LocomotionPolicyObsCfg):
    def __init__(self, robot: RobotSpec) -> None:
        super().__init__()
        self.joint_pos = ObsTermSpec(
            func=observations.joint_pos_rel,
            noise=NoiseSpec("uniform", -0.01, 0.01),
            params={
                "asset_cfg": EntityRef(
                    "robot",
                    joints=tuple(robot.joint_names),
                    preserve_order=True,
                )
            },
        )
        self.joint_vel = ObsTermSpec(
            func=observations.joint_vel,
            noise=NoiseSpec("uniform", -1.5, 1.5),
            params={
                "asset_cfg": EntityRef(
                    "robot",
                    joints=tuple(robot.joint_names),
                    preserve_order=True,
                )
            },
        )
        self.actions = ObsTermSpec(func=observations.last_action)


class G115DofLocomotionFlatCriticObsCfg(LocomotionCriticObsCfg):
    def __init__(self, robot: RobotSpec) -> None:
        super().__init__()
        self.joint_pos = ObsTermSpec(
            func=observations.joint_pos_rel,
            params={
                "asset_cfg": EntityRef(
                    "robot",
                    joints=tuple(robot.joint_names),
                    preserve_order=True,
                )
            },
        )
        self.joint_vel = ObsTermSpec(
            func=observations.joint_vel,
            params={
                "asset_cfg": EntityRef(
                    "robot",
                    joints=tuple(robot.joint_names),
                    preserve_order=True,
                )
            },
        )
        self.actions = ObsTermSpec(func=observations.last_action)


class G115DofLocomotionFlatRewardsCfg(LocomotionRewardsCfg):
    def __init__(self) -> None:
        super().__init__()
        self.feet_air_time = RewardTermSpec(
            func=rewards.feet_air_time_positive_biped,
            weight=1.0,
            params={
                "command_name": "base_velocity",
                "sensor": ContactSensorRef(
                    name="contact_forces",
                    elements=".*_ankle_roll_link",
                ),
                "threshold": 0.5,
            },
        )
        self.feet_slide = RewardTermSpec(
            func=rewards.contact_slide,
            weight=-0.1,
            params={
                "sensor_cfg": ContactSensorRef(
                    name="contact_forces",
                    elements=".*_ankle_roll_link",
                ),
                "asset_cfg": EntityRef(
                    "robot",
                    bodies=".*_ankle_roll_link",
                ),
                "threshold": 0.1,
            },
            level=Requirement.REQUIRED,
        )
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


class G115DofLocomotionFlatEnvCfg(LocomotionEnvCfg):
    def __init__(self, robot: RobotSpec) -> None:
        super().__init__()
        self.robot = robot
        self.scene = SceneSpec(
            contact_sensors=(
                ContactSensorRef(
                    name="contact_forces",
                    elements=".*",
                    track_air_time=True,
                    history_length=3,
                ),
            ),
            env_spacing=2.5,
        )

        policy = G115DofLocomotionFlatPolicyObsCfg(robot)
        critic = G115DofLocomotionFlatCriticObsCfg(robot)
        self.observations = {
            "policy": ObsGroupSpec(
                terms=dict(vars(policy)),
                enable_corruption=True,
                concatenate_terms=False,
            ),
            "critic": ObsGroupSpec(
                terms=dict(vars(critic)),
                enable_corruption=False,
                concatenate_terms=False,
            ),
        }
        self.actions = {
            "joint_pos": ActionTermSpec(
                kind="joint_position",
                target=EntityRef(
                    "robot",
                    joints=tuple(robot.joint_names),
                    preserve_order=True,
                ),
                params={
                    "scale": {
                        joint.name: joint.action_scale
                        for joint in robot.joint_properties
                    },
                    "use_default_offset": True,
                },
            )
        }

        locomotion_rewards = G115DofLocomotionFlatRewardsCfg()
        self.rewards = {
            "rewards": {
                "termination_penalty": locomotion_rewards.termination_penalty,
                "track_lin_vel_xy_exp": locomotion_rewards.track_lin_vel_xy_exp,
                "track_ang_vel_z_exp": locomotion_rewards.track_ang_vel_z_exp,
                "feet_air_time": locomotion_rewards.feet_air_time,
                "feet_slide": locomotion_rewards.feet_slide,
                "flat_orientation_l2": locomotion_rewards.flat_orientation_l2,
                "stand_still": locomotion_rewards.stand_still,
                "dof_pos_limits": locomotion_rewards.dof_pos_limits,
                "joint_deviation_hip": locomotion_rewards.joint_deviation_hip,
                "joint_deviation_torso": locomotion_rewards.joint_deviation_torso,
                "joint_deviation_knee": locomotion_rewards.joint_deviation_knee,
                "lin_vel_z_l2": locomotion_rewards.lin_vel_z_l2,
                "action_rate_l2": locomotion_rewards.action_rate_l2,
                "dof_acc_l2": locomotion_rewards.dof_acc_l2,
                "dof_torques_l2": locomotion_rewards.dof_torques_l2,
            }
        }
        self.terminations = {
            "time_out": DoneTermSpec(func=terminations.time_out, time_out=True),
            "base_contact": DoneTermSpec(
                func=terminations.illegal_contact,
                time_out=False,
                params={
                    "sensor": ContactSensorRef(
                        name="contact_forces",
                        elements=(
                            "torso_link",
                            ".*_shoulder_.*",
                            ".*_elbow_.*",
                            ".*_wrist_.*",
                            ".*_hip_.*",
                            ".*_knee_.*",
                        ),
                    )
                },
            ),
        }
        self.events = {
            "physics_material": EventTermSpec(
                kind="randomize_friction",
                mode="startup",
                target=EntityRef(
                    "robot",
                    bodies=tuple(robot.body_names),
                    preserve_order=True,
                ),
                engine_params={
                    "isaacsim": {
                        "static_friction_range": (0.25, 0.8),
                        "dynamic_friction_range": (0.2, 0.6),
                        "restitution_range": (0.0, 0.8),
                        "num_buckets": 64,
                    },
                    "mjlab": {
                        "ranges": (0.2, 0.8),
                        "operation": "abs",
                        "shared_random": True,
                    },
                },
            ),
            "add_base_mass": EventTermSpec(
                kind="randomize_body_mass",
                mode="startup",
                target=EntityRef("robot", bodies=robot.root_body),
                params={"add_range": (-5.0, 5.0), "operation": "add"},
            ),
            "base_external_force_torque": EventTermSpec(
                kind="apply_external_force_torque",
                mode="reset",
                target=EntityRef("robot", bodies=robot.root_body),
                params={
                    "force_range": (0.0, 0.0),
                    "torque_range": (-0.0, 0.0),
                },
                level=Requirement.OPTIONAL,
            ),
            "reset_base": EventTermSpec(
                kind="reset_root_state_uniform",
                mode="reset",
                target=EntityRef("robot"),
                params={
                    "pose_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "yaw": (-3.14, 3.14),
                    },
                    "velocity_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "z": (-0.1, 0.1),
                        "roll": (-0.5, 0.5),
                        "pitch": (-0.5, 0.5),
                        "yaw": (-0.5, 0.5),
                    },
                },
            ),
            "reset_robot_joints": EventTermSpec(
                kind="reset_joints_by_scale",
                mode="reset",
                target=EntityRef(
                    "robot",
                    joints=tuple(robot.joint_names),
                    preserve_order=True,
                ),
                params={
                    "position_range": (0.8, 1.2),
                    "velocity_range": (-1.0, 1.0),
                },
            ),
            "push_robot": EventTermSpec(
                kind="push_by_setting_velocity",
                mode="interval",
                target=EntityRef("robot"),
                interval_range_s=(10.0, 15.0),
                params={
                    "velocity_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                    }
                },
            ),
        }
        self.curriculum = {}
        self.agent = G1_LOCOMOTION_TRAINING_CFG


def flat_g1_15dof(robot: RobotSpec) -> TaskSpec:
    """Convert the explicit locked-arm Flat config at the registry boundary."""
    config = G115DofLocomotionFlatEnvCfg(robot)
    return TaskSpec(
        task_id="Instinct-Velocity-Flat-G1-15DoF",
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
