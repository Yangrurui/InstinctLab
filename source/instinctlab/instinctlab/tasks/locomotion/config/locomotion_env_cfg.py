"""Robot-independent velocity-tracking locomotion configuration."""

from __future__ import annotations

import math

from instinctlab.engines.assets import RobotSpec
from instinctlab.spec import (
    ActionTermSpec,
    AgentSpec,
    CommandTermSpec,
    ContactSensorRef,
    DoneTermSpec,
    EntityRef,
    EventTermSpec,
    NoiseSpec,
    ObsGroupSpec,
    ObsTermSpec,
    RewardTermSpec,
    SceneSpec,
    SimSpec,
)
from instinctlab.spec.capability import Requirement
from instinctlab.tasks.locomotion.mdp import observations, rewards, terminations


class LocomotionPolicyObsCfg:
    def __init__(self, joints: EntityRef) -> None:
        self.base_ang_vel = ObsTermSpec(
            func=observations.base_ang_vel,
            noise=NoiseSpec("uniform", -0.2, 0.2),
        )
        self.projected_gravity = ObsTermSpec(
            func=observations.projected_gravity,
            noise=NoiseSpec("uniform", -0.05, 0.05),
        )
        self.velocity_commands = ObsTermSpec(
            func=observations.generated_commands,
            params={"command_name": "base_velocity"},
        )
        self.joint_pos = ObsTermSpec(
            func=observations.joint_pos_rel,
            noise=NoiseSpec("uniform", -0.01, 0.01),
            params={"asset_cfg": joints},
        )
        self.joint_vel = ObsTermSpec(
            func=observations.joint_vel,
            noise=NoiseSpec("uniform", -1.5, 1.5),
            params={"asset_cfg": joints},
        )
        self.actions = ObsTermSpec(func=observations.last_action)


class LocomotionCriticObsCfg:
    def __init__(self, joints: EntityRef) -> None:
        self.base_lin_vel = ObsTermSpec(func=observations.base_lin_vel)
        self.base_ang_vel = ObsTermSpec(func=observations.base_ang_vel)
        self.projected_gravity = ObsTermSpec(func=observations.projected_gravity)
        self.velocity_commands = ObsTermSpec(
            func=observations.generated_commands,
            params={"command_name": "base_velocity"},
        )
        self.joint_pos = ObsTermSpec(
            func=observations.joint_pos_rel,
            params={"asset_cfg": joints},
        )
        self.joint_vel = ObsTermSpec(
            func=observations.joint_vel,
            params={"asset_cfg": joints},
        )
        self.actions = ObsTermSpec(func=observations.last_action)


class LocomotionRewardsCfg:
    def __init__(self, feet_contact: ContactSensorRef, feet: EntityRef) -> None:
        self.termination_penalty = RewardTermSpec(
            func=rewards.is_terminated, weight=-200.0
        )
        self.track_lin_vel_xy_exp = RewardTermSpec(
            func=rewards.track_lin_vel_xy_yaw_frame_exp,
            weight=1.0,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        self.track_ang_vel_z_exp = RewardTermSpec(
            func=rewards.track_ang_vel_z_world_exp,
            weight=1.0,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        self.feet_air_time = RewardTermSpec(
            func=rewards.feet_air_time_positive_biped,
            weight=1.0,
            params={
                "command_name": "base_velocity",
                "sensor": feet_contact,
                "threshold": 0.5,
            },
        )
        self.feet_slide = RewardTermSpec(
            kind="contact_slide",
            weight=-0.1,
            params={
                "sensor_cfg": feet_contact,
                "asset_cfg": feet,
                "threshold": 0.1,
            },
            level=Requirement.REQUIRED,
        )
        self.flat_orientation_l2 = RewardTermSpec(
            func=rewards.flat_orientation_l2, weight=-1.0
        )
        self.stand_still = RewardTermSpec(
            func=rewards.stand_still,
            weight=-0.8,
            params={"command_name": "base_velocity"},
        )
        self.lin_vel_z_l2 = RewardTermSpec(func=rewards.lin_vel_z_l2, weight=-0.1)
        self.action_rate_l2 = RewardTermSpec(func=rewards.action_rate_l2, weight=-0.05)


class LocomotionEnvCfg:
    def __init__(
        self,
        robot: RobotSpec,
        rewards: dict[str, dict[str, RewardTermSpec]],
        illegal_contact: ContactSensorRef,
        agent: AgentSpec,
    ) -> None:
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
        self.sim = SimSpec(physics_dt=0.005, decimation=4, episode_length_s=20.0)

        policy = LocomotionPolicyObsCfg(
            EntityRef("robot", joints=tuple(robot.joint_names), preserve_order=True)
        )
        critic = LocomotionCriticObsCfg(
            EntityRef("robot", joints=tuple(robot.joint_names), preserve_order=True)
        )
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
        self.commands = {
            "base_velocity": CommandTermSpec(
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
        self.rewards = rewards
        self.terminations = {
            "time_out": DoneTermSpec(func=terminations.time_out, time_out=True),
            "base_contact": DoneTermSpec(
                func=terminations.illegal_contact,
                time_out=False,
                params={"sensor": illegal_contact},
            ),
        }
        self.events = {
            "physics_material": EventTermSpec(
                kind="randomize_friction",
                mode="startup",
                target=EntityRef("robot", bodies=".*"),
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
                target=EntityRef("robot", joints=".*"),
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
        self.agent = agent
