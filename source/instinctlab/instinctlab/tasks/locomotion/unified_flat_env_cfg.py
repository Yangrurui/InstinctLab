"""Unified G1 locomotion-flat task, ported from ``config/g1/flat_env_cfg.py``."""

from __future__ import annotations

import math

from instinctlab.assets import ASSETS
from instinctlab.assets.unitree_g1 import G1_29DOF_DFS_JOINT_NAMES
from instinctlab.envs import UnifiedManagerBasedRLEnvCfg
from instinctlab.managers import (
    CommandTermCfg,
    EventTermCfg,
    JointPositionActionCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardGroupCfg,
    RewardTermCfg,
    TerminationGroupCfg,
    TerminationTermCfg,
    UniformNoiseCfg,
)
from instinctlab.rl import OnPolicyRunnerCfg
from instinctlab.sim.backend import JOINT_ACC_SOURCES, RuntimeRequirements
from instinctlab.sim.capabilities import Capability
from instinctlab.sim.scene import ContactSensorSpec, SceneSpec, SimulationSpec, TerrainSpec
from instinctlab.sim.schema import EnvSchema, locomotion_flat_schema
from instinctlab.tasks.locomotion import commands
from instinctlab.tasks.locomotion.mdp import unified as mdp

_FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
_ANKLE_JOINTS = (
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
_HIP_ROLL_YAW_JOINTS = (
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
)
_HIP_AND_KNEE_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
)
_KNEE_JOINTS = ("left_knee_joint", "right_knee_joint")
_WAIST_JOINTS = ("waist_pitch_joint", "waist_roll_joint", "waist_yaw_joint")
_ARM_JOINTS = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
LOCOMOTION_MATERIAL_BACKEND_PARAMS = {
    "default": {
        "static_friction_range": (0.25, 0.8),
        "dynamic_friction_range": (0.2, 0.6),
        "restitution_range": (0.0, 0.8),
        "shared_random": False,
        "separate_dynamic_friction": True,
    },
    "mjlab": {
        "static_friction_range": (0.25, 0.8),
        "dynamic_friction_range": (0.2, 0.6),
        "restitution_range": None,
        "shared_random": True,
        "separate_dynamic_friction": False,
    },
    "isaacsim": {
        "static_friction_range": (0.25, 0.8),
        "dynamic_friction_range": (0.2, 0.6),
        "restitution_range": (0.0, 0.8),
        "shared_random": False,
        "separate_dynamic_friction": True,
        "num_buckets": 64,
        "assign_per_shape": True,
    },
}

_ILLEGAL_CONTACT_BODIES = (
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
)


def locomotion_flat_env_cfg(*, num_envs: int = 4096) -> UnifiedManagerBasedRLEnvCfg:
    robot = ASSETS.make("unitree_g1_29dof")
    num_joints = len(robot.joint_names)
    policy_terms = {
        "base_ang_vel": ObservationTermCfg(
            mdp.base_ang_vel, noise=UniformNoiseCfg(-0.2, 0.2), shape=(3,), semantic="rad/s"
        ),
        "projected_gravity": ObservationTermCfg(
            mdp.projected_gravity, noise=UniformNoiseCfg(-0.05, 0.05), shape=(3,), semantic="unit_vector"
        ),
        "velocity_commands": ObservationTermCfg(
            mdp.velocity_commands, params={"command_name": "base_velocity"}, shape=(3,), semantic="m/s,m/s,rad/s"
        ),
        "joint_pos": ObservationTermCfg(
            mdp.joint_pos_rel, noise=UniformNoiseCfg(-0.01, 0.01), shape=(num_joints,), semantic="rad_dfs_v1"
        ),
        "joint_vel": ObservationTermCfg(
            mdp.joint_vel, noise=UniformNoiseCfg(-1.5, 1.5), shape=(num_joints,), semantic="rad/s_dfs_v1"
        ),
        "actions": ObservationTermCfg(mdp.last_action, shape=(num_joints,), semantic="joint_position_action_dfs_v1"),
    }
    policy_order = (
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos",
        "joint_vel",
        "actions",
    )
    critic_terms = {
        "base_lin_vel": ObservationTermCfg(mdp.base_lin_vel, shape=(3,), semantic="m/s"),
        **policy_terms,
    }
    critic_order = ("base_lin_vel", *policy_order)

    return UnifiedManagerBasedRLEnvCfg(
        scene=SceneSpec(
            num_envs=num_envs,
            env_spacing=2.5,
            robot=robot,
            terrain=TerrainSpec(terrain_type="plane", sliding_friction=1.0, restitution=0.0),
            contact_sensors=(
                ContactSensorSpec(
                    name="feet_contact_forces",
                    entity_name="robot",
                    body_names=_FEET,
                    history_length=3,
                    force_threshold=1.0,
                    track_air_time=True,
                ),
                ContactSensorSpec(
                    name="base_contact_forces",
                    entity_name="robot",
                    body_names=_ILLEGAL_CONTACT_BODIES,
                    history_length=3,
                    force_threshold=1.0,
                    track_air_time=False,
                ),
            ),
            backend_options={
                "isaacsim": {
                    "scene": {
                        "lazy_sensor_update": True,
                        "replicate_physics": True,
                        "filter_collisions": True,
                    },
                    "robot_spawn": {
                        "self_collision": True,
                        "rigid_props": {
                            "disable_gravity": False,
                            "retain_accelerations": False,
                            "linear_damping": 0.0,
                            "angular_damping": 0.0,
                            "max_linear_velocity": 1000.0,
                            "max_angular_velocity": 1000.0,
                            "max_depenetration_velocity": 1.0,
                        },
                        "articulation_props": {
                            "enabled_self_collisions": True,
                            "solver_position_iteration_count": 8,
                            "solver_velocity_iteration_count": 4,
                        },
                    },
                }
            },
        ),
        simulation=SimulationSpec(
            sim_dt=0.005,
            decimation=4,
            engine_options={
                "mjlab": {
                    "njmax": 300,
                    "solver": "newton",
                    "iterations": 10,
                    "ls_iterations": 20,
                    "ccd_iterations": 500,
                }
            },
        ),
        actions={"joint_pos": JointPositionActionCfg()},
        observations={
            "policy": ObservationGroupCfg(policy_terms, policy_order, enable_corruption=True),
            "critic": ObservationGroupCfg(critic_terms, critic_order, enable_corruption=False),
        },
        rewards={
            "default": RewardGroupCfg(
                terms={
                    "termination_penalty": RewardTermCfg(mdp.is_terminated, weight=-200.0),
                    "track_lin_vel_xy": RewardTermCfg(
                        mdp.track_lin_vel_xy_yaw_frame_exp,
                        weight=1.0,
                        params={"std": 0.5, "command_name": "base_velocity"},
                    ),
                    "track_ang_vel_z": RewardTermCfg(
                        mdp.track_ang_vel_z_world_exp,
                        weight=1.0,
                        params={"std": 0.5, "command_name": "base_velocity"},
                    ),
                    "feet_air_time": RewardTermCfg(
                        mdp.feet_air_time_positive_biped,
                        weight=1.0,
                        params={
                            "command_name": "base_velocity",
                            "threshold": 0.5,
                            "sensor_name": "feet_contact_forces",
                            "body_names": _FEET,
                        },
                    ),
                    "feet_slide": RewardTermCfg(
                        mdp.contact_slide,
                        weight=-0.1,
                        params={
                            "sensor_name": "feet_contact_forces",
                            "body_names": _FEET,
                            "threshold": 0.1,
                        },
                    ),
                    "flat_orientation": RewardTermCfg(mdp.flat_orientation_l2, weight=-1.0),
                    "stand_still": RewardTermCfg(
                        mdp.stand_still, weight=-0.8, params={"command_name": "base_velocity"}
                    ),
                    "joint_pos_limits": RewardTermCfg(
                        mdp.joint_pos_limits, weight=-1.0, params={"joint_names": _ANKLE_JOINTS}
                    ),
                    "hip_deviation": RewardTermCfg(
                        mdp.joint_deviation_l1, weight=-0.1, params={"joint_names": _HIP_ROLL_YAW_JOINTS}
                    ),
                    "arm_deviation": RewardTermCfg(
                        mdp.joint_deviation_l1, weight=-0.1, params={"joint_names": _ARM_JOINTS}
                    ),
                    "torso_deviation": RewardTermCfg(
                        mdp.joint_deviation_l1, weight=-0.1, params={"joint_names": _WAIST_JOINTS}
                    ),
                    "knee_deviation": RewardTermCfg(
                        mdp.joint_deviation_l1, weight=-0.05, params={"joint_names": _KNEE_JOINTS}
                    ),
                    "lin_vel_z": RewardTermCfg(mdp.lin_vel_z_l2, weight=-0.1),
                    "action_rate": RewardTermCfg(mdp.action_rate_l2, weight=-0.05),
                    "joint_acc": RewardTermCfg(
                        mdp.joint_acc_l2, weight=-2.0e-7, params={"joint_names": _HIP_AND_KNEE_JOINTS}
                    ),
                    "joint_torques": RewardTermCfg(
                        mdp.joint_torques_l2, weight=-4.0e-6, params={"joint_names": _HIP_AND_KNEE_JOINTS}
                    ),
                },
                term_order=(
                    "termination_penalty",
                    "track_lin_vel_xy",
                    "track_ang_vel_z",
                    "feet_air_time",
                    "feet_slide",
                    "flat_orientation",
                    "stand_still",
                    "joint_pos_limits",
                    "hip_deviation",
                    "arm_deviation",
                    "torso_deviation",
                    "knee_deviation",
                    "lin_vel_z",
                    "action_rate",
                    "joint_acc",
                    "joint_torques",
                ),
            )
        },
        terminations=TerminationGroupCfg(
            terms={
                "time_out": TerminationTermCfg(mdp.time_out, time_out=True),
                "base_contact": TerminationTermCfg(
                    mdp.illegal_contact,
                    params={"sensor_name": "base_contact_forces", "body_names": _ILLEGAL_CONTACT_BODIES},
                ),
            },
            term_order=("time_out", "base_contact"),
        ),
        events={
            "randomize_material": EventTermCfg(
                mdp.randomize_sliding_friction,
                mode="startup",
                params={
                    "body_names": (".*",),
                    "backend_params": LOCOMOTION_MATERIAL_BACKEND_PARAMS,
                },
            ),
            "add_base_mass": EventTermCfg(
                mdp.randomize_body_mass,
                mode="startup",
                params={
                    "body_names": ("torso_link",),
                    "mass_range": (-5.0, 5.0),
                    "operation": "add",
                },
            ),
            "reset_root": EventTermCfg(
                mdp.reset_root_state_uniform,
                mode="reset",
                writes_state=True,
                params={
                    "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
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
            "reset_joints": EventTermCfg(
                mdp.reset_joints_by_scale,
                mode="reset",
                writes_state=True,
                params={"position_range": (0.8, 1.2), "velocity_range": (-1.0, 1.0)},
            ),
            "push": EventTermCfg(
                mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(10.0, 15.0),
                writes_state=True,
                params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
            ),
        },
        commands={
            "base_velocity": CommandTermCfg(
                commands.UniformVelocityCommand,
                params={
                    "resampling_time_range": (10.0, 10.0),
                    "rel_standing_envs": 0.2,
                    "rel_heading_envs": 0.5,
                    "rel_world_envs": 0.0,
                    "rel_forward_envs": 0.0,
                    "init_velocity_prob": 0.0,
                    "heading_command": True,
                    "heading_control_stiffness": 0.5,
                    "ranges": {
                        "lin_vel_x": (-0.5, 1.0),
                        "lin_vel_y": (-0.5, 0.5),
                        "ang_vel_z": (-1.5, 1.5),
                        "heading": (-math.pi, math.pi),
                    },
                },
            )
        },
        requirements=RuntimeRequirements(
            capabilities=frozenset(
                {
                    Capability.BATCHED_SIMULATION,
                    Capability.PLANE_TERRAIN,
                    Capability.ROOT_STATE,
                    Capability.JOINT_STATE,
                    Capability.BODY_STATE,
                    Capability.IMPLICIT_POSITION_CONTROL,
                    Capability.CONTACT_ACTIVE,
                    Capability.CONTACT_HISTORY,
                    Capability.CONTACT_AIR_TIME,
                    Capability.CONTACT_FORCE_VECTOR,
                    Capability.DR_SLIDING_FRICTION,
                    Capability.BODY_MASS_PROPERTIES,
                    Capability.ROOT_VELOCITY_WRITE,
                }
            ),
            optional_capabilities=frozenset({Capability.DR_RESTITUTION}),
            randomization_fields=frozenset({"sliding_friction", "mass", "root_pose", "root_velocity", "joint_state"}),
            accepted_joint_acc_sources=JOINT_ACC_SOURCES,
        ),
        episode_length_s=20.0,
        is_finite_horizon=False,
        action_order=("joint_pos",),
        observation_group_order=("policy", "critic"),
        reward_group_order=("default",),
        event_order=("randomize_material", "add_base_mass", "reset_root", "reset_joints", "push"),
        command_order=("base_velocity",),
    )


def locomotion_flat_agent_cfg(**overrides) -> OnPolicyRunnerCfg:
    cfg = OnPolicyRunnerCfg(experiment_name="g1_locomotion_flat")
    for name, value in overrides.items():
        if not hasattr(cfg, name):
            raise TypeError(f"unknown runner configuration field {name!r}")
        setattr(cfg, name, value)
    return cfg


def locomotion_flat_env_schema() -> EnvSchema:
    """Stable observation/action/reward schema for checkpoint compatibility."""
    return locomotion_flat_schema(len(G1_29DOF_DFS_JOINT_NAMES))


__all__ = [
    "LOCOMOTION_MATERIAL_BACKEND_PARAMS",
    "locomotion_flat_agent_cfg",
    "locomotion_flat_env_cfg",
    "locomotion_flat_env_schema",
]
