"""Explicit engine-neutral G1 locomotion configuration.

The declaration order follows InstinctMJ: observations, rewards, then the
complete environment configuration. Native engine types remain in the engine
adapters; this file states every task value directly.
"""

from __future__ import annotations

import math

from instinctlab import mdp
from instinctlab.engines.assets import RobotSpec
from instinctlab.spec import (
    ActionTermSpec,
    CommandTermSpec,
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
    SimSpec,
    TaskSpec,
)
from instinctlab.spec.capability import Requirement
from instinctlab.tasks.locomotion.config.g1.rl_cfgs import G1_LOCOMOTION_TRAINING_CFG

COMMAND = "base_velocity"
ROBOT = EntityRef("robot", bodies=".*")
FEET_CONTACT = ContactSensorRef(name="contact_forces", elements=".*_ankle_roll_link")
UPPER_BODY_CONTACT = ContactSensorRef(
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


class G1FlatPolicyObsCfg:
    def __init__(self, joints: EntityRef) -> None:
        self.base_ang_vel = ObsTermSpec(
            func=mdp.base_ang_vel,
            noise=NoiseSpec("uniform", -0.2, 0.2),
        )
        self.projected_gravity = ObsTermSpec(
            func=mdp.projected_gravity,
            noise=NoiseSpec("uniform", -0.05, 0.05),
        )
        self.velocity_commands = ObsTermSpec(
            func=mdp.generated_commands,
            params={"command_name": COMMAND},
        )
        self.joint_pos = ObsTermSpec(
            func=mdp.joint_pos_rel,
            noise=NoiseSpec("uniform", -0.01, 0.01),
            params={"asset_cfg": joints},
        )
        self.joint_vel = ObsTermSpec(
            func=mdp.joint_vel,
            noise=NoiseSpec("uniform", -1.5, 1.5),
            params={"asset_cfg": joints},
        )
        self.actions = ObsTermSpec(func=mdp.last_action)


class G1FlatCriticObsCfg:
    def __init__(self, joints: EntityRef) -> None:
        self.base_lin_vel = ObsTermSpec(func=mdp.base_lin_vel)
        self.base_ang_vel = ObsTermSpec(func=mdp.base_ang_vel)
        self.projected_gravity = ObsTermSpec(func=mdp.projected_gravity)
        self.velocity_commands = ObsTermSpec(
            func=mdp.generated_commands,
            params={"command_name": COMMAND},
        )
        self.joint_pos = ObsTermSpec(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": joints},
        )
        self.joint_vel = ObsTermSpec(
            func=mdp.joint_vel,
            params={"asset_cfg": joints},
        )
        self.actions = ObsTermSpec(func=mdp.last_action)


class G1FlatRewardsCfg:
    def __init__(self) -> None:
        self.termination_penalty = RewardTermSpec(func=mdp.is_terminated, weight=-200.0)
        self.track_lin_vel_xy_exp = RewardTermSpec(
            func=mdp.track_lin_vel_xy_yaw_frame_exp,
            weight=1.0,
            params={"command_name": COMMAND, "std": 0.5},
        )
        self.track_ang_vel_z_exp = RewardTermSpec(
            func=mdp.track_ang_vel_z_world_exp,
            weight=1.0,
            params={"command_name": COMMAND, "std": 0.5},
        )
        self.feet_air_time = RewardTermSpec(
            func=mdp.feet_air_time_positive_biped,
            weight=1.0,
            params={"command_name": COMMAND, "sensor": FEET_CONTACT, "threshold": 0.5},
        )
        self.feet_slide = RewardTermSpec(
            kind="contact_slide",
            weight=-0.1,
            params={
                "sensor_cfg": FEET_CONTACT,
                "asset_cfg": EntityRef("robot", bodies=".*_ankle_roll_link"),
            },
            level=Requirement.REQUIRED,
        )
        self.flat_orientation_l2 = RewardTermSpec(func=mdp.flat_orientation_l2, weight=-1.0)
        self.stand_still = RewardTermSpec(
            func=mdp.stand_still,
            weight=-0.8,
            params={"command_name": COMMAND},
        )
        self.dof_pos_limits = RewardTermSpec(
            func=mdp.joint_pos_limits,
            weight=-1.0,
            params={
                "asset_cfg": EntityRef(
                    "robot",
                    joints=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
                )
            },
        )
        self.joint_deviation_hip = RewardTermSpec(
            func=mdp.joint_deviation_l1,
            weight=-0.1,
            params={
                "asset_cfg": EntityRef(
                    "robot",
                    joints=(".*_hip_yaw_joint", ".*_hip_roll_joint"),
                )
            },
        )
        self.joint_deviation_arms = RewardTermSpec(
            func=mdp.joint_deviation_l1,
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
            func=mdp.joint_deviation_l1,
            weight=-0.1,
            params={"asset_cfg": EntityRef("robot", joints="waist_.*")},
        )
        self.joint_deviation_knee = RewardTermSpec(
            func=mdp.joint_deviation_l1,
            weight=-0.05,
            params={"asset_cfg": EntityRef("robot", joints=(".*_knee_joint",))},
        )
        self.lin_vel_z_l2 = RewardTermSpec(func=mdp.lin_vel_z_l2, weight=-0.1)
        self.action_rate_l2 = RewardTermSpec(func=mdp.action_rate_l2, weight=-0.05)
        self.dof_acc_l2 = RewardTermSpec(
            kind="joint_acc_l2",
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
            kind="joint_torques_l2",
            weight=-4.0e-6,
            params={
                "asset_cfg": EntityRef(
                    "robot",
                    joints=(".*_hip_.*", ".*_knee_joint"),
                )
            },
            level=Requirement.REQUIRED,
        )


class G1LocomotionFlatEnvCfg:
    def __init__(self, robot: RobotSpec) -> None:
        self.robot = robot
        joints = EntityRef("robot", joints=tuple(robot.joint_names), preserve_order=True)

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
        self.sim = SimSpec(physics_dt=0.005, decimation=4, episode_length_s=20.0)

        policy = G1FlatPolicyObsCfg(joints)
        critic = G1FlatCriticObsCfg(joints)
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
                target=joints,
                params={
                    "scale": {
                        joint.name: joint.action_scale
                        for joint in robot.joint_properties
                    },
                    "use_default_offset": True,
                },
            )
        }
        self.commands = {
            COMMAND: CommandTermSpec(
                kind="uniform_velocity",
                params={
                    "entity": "robot",
                    "resampling_time_range": (10.0, 10.0),
                    "rel_standing_envs": 0.2,
                    "rel_heading_envs": 0.5,
                    "heading_command": True,
                    "heading_control_stiffness": 0.5,
                    "debug_vis": True,
                    "lin_vel_x": (-0.5, 1.0),
                    "lin_vel_y": (-0.5, 0.5),
                    "ang_vel_z": (-1.5, 1.5),
                    "heading": (-math.pi, math.pi),
                },
            )
        }

        rewards = G1FlatRewardsCfg()
        self.rewards = {"rewards": dict(vars(rewards))}
        self.terminations = {
            "time_out": DoneTermSpec(func=mdp.time_out, time_out=True),
            "base_contact": DoneTermSpec(
                func=mdp.illegal_contact,
                time_out=False,
                params={"sensor": UPPER_BODY_CONTACT},
            ),
        }
        self.events = {
            "physics_material": EventTermSpec(
                kind="randomize_friction",
                mode="startup",
                target=ROBOT,
            ),
            "add_base_mass": EventTermSpec(
                kind="randomize_body_mass",
                mode="startup",
                target=EntityRef("robot", bodies="torso_link"),
                params={"add_range": (-5.0, 5.0), "operation": "add"},
            ),
            "base_external_force_torque": EventTermSpec(
                kind="apply_external_force_torque",
                mode="reset",
                target=EntityRef("robot", bodies="torso_link"),
                params={
                    "force_range": (0.0, 0.0),
                    "torque_range": (-0.0, 0.0),
                },
                level=Requirement.OPTIONAL,
            ),
            "reset_base": EventTermSpec(
                kind="reset_root_state_uniform",
                mode="reset",
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
                params={
                    "position_range": (0.8, 1.2),
                    "velocity_range": (-1.0, 1.0),
                },
            ),
            "push_robot": EventTermSpec(
                kind="push_by_setting_velocity",
                mode="interval",
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


__all__ = [
    "COMMAND",
    "G1FlatCriticObsCfg",
    "G1FlatPolicyObsCfg",
    "G1FlatRewardsCfg",
    "G1LocomotionFlatEnvCfg",
    "flat_g1",
]
