"""Engine-neutral G1 locomotion-flat task configuration."""

from __future__ import annotations

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
from instinctlab.sim.backend import RuntimeRequirements
from instinctlab.sim.capabilities import Capability
from instinctlab.sim.robot_spec import G1_29DOF_DFS_BODY_NAMES, make_g1_29dof_robot_spec
from instinctlab.sim.scene import ContactSensorSpec, SceneSpec, SimulationSpec, TerrainSpec
from instinctlab.tasks.locomotion import commands
from instinctlab.tasks.locomotion.mdp import unified as mdp

_FEET = ("LL_FOOT", "LR_FOOT")
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


def locomotion_flat_env_cfg(*, num_envs: int = 4096) -> UnifiedManagerBasedRLEnvCfg:
    robot = make_g1_29dof_robot_spec()
    policy_terms = {
        "base_ang_vel": ObservationTermCfg(
            mdp.base_ang_vel, noise=UniformNoiseCfg(-0.2, 0.2), scale=0.25, shape=(3,), semantic="rad/s"
        ),
        "projected_gravity": ObservationTermCfg(
            mdp.projected_gravity, noise=UniformNoiseCfg(-0.05, 0.05), shape=(3,), semantic="unit_vector"
        ),
        "velocity_commands": ObservationTermCfg(
            mdp.velocity_commands, params={"command_name": "base_velocity"}, shape=(3,), semantic="m/s,m/s,rad/s"
        ),
        "joint_pos": ObservationTermCfg(
            mdp.joint_pos_rel, noise=UniformNoiseCfg(-0.01, 0.01), shape=(29,), semantic="rad_dfs_v1"
        ),
        "joint_vel": ObservationTermCfg(
            mdp.joint_vel, noise=UniformNoiseCfg(-1.5, 1.5), scale=0.05, shape=(29,), semantic="rad/s_dfs_v1"
        ),
        "actions": ObservationTermCfg(mdp.last_action, shape=(29,), semantic="joint_position_action_dfs_v1"),
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
                    name="contact_forces",
                    entity_name="robot",
                    body_names=("torso_link", *_FEET),
                    history_length=3,
                    force_threshold=1.0,
                    track_air_time=True,
                ),
            ),
        ),
        simulation=SimulationSpec(sim_dt=0.005, decimation=4),
        actions={"joint_pos": JointPositionActionCfg()},
        observations={
            "policy": ObservationGroupCfg(policy_terms, policy_order, enable_corruption=True),
            "critic": ObservationGroupCfg(critic_terms, critic_order, enable_corruption=False),
        },
        rewards={
            "default": RewardGroupCfg(
                terms={
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
                            "threshold": 0.4,
                            "sensor_name": "contact_forces",
                            "body_names": _FEET,
                        },
                    ),
                    "feet_slide": RewardTermCfg(
                        mdp.contact_slide,
                        weight=-0.25,
                        params={"sensor_name": "contact_forces", "body_names": _FEET},
                    ),
                    "lin_vel_z": RewardTermCfg(mdp.lin_vel_z_l2, weight=-2.0),
                    "ang_vel_xy": RewardTermCfg(
                        lambda env: mdp.base_ang_vel(env)[:, :2].square().sum(dim=1), weight=-0.05
                    ),
                    "flat_orientation": RewardTermCfg(mdp.flat_orientation_l2, weight=-1.0),
                    "action_rate": RewardTermCfg(mdp.action_rate_l2, weight=-0.01),
                    "joint_acc": RewardTermCfg(
                        mdp.joint_acc_l2, weight=-2.5e-7, params={"joint_names": robot.joint_names}
                    ),
                    "joint_torques": RewardTermCfg(
                        mdp.joint_torques_l2, weight=-2.0e-6, params={"joint_names": robot.joint_names}
                    ),
                    "joint_pos_limits": RewardTermCfg(
                        mdp.joint_pos_limits, weight=-1.0, params={"joint_names": robot.joint_names}
                    ),
                    "stand_still": RewardTermCfg(
                        mdp.stand_still, weight=-1.0, params={"command_name": "base_velocity"}
                    ),
                    "ankle_deviation": RewardTermCfg(
                        mdp.joint_deviation_l1, weight=-0.1, params={"joint_names": _ANKLE_JOINTS}
                    ),
                    "hip_deviation": RewardTermCfg(
                        mdp.joint_deviation_l1, weight=-0.1, params={"joint_names": _HIP_ROLL_YAW_JOINTS}
                    ),
                    "arm_deviation": RewardTermCfg(
                        mdp.joint_deviation_l1, weight=-0.1, params={"joint_names": _ARM_JOINTS}
                    ),
                },
                term_order=(
                    "track_lin_vel_xy",
                    "track_ang_vel_z",
                    "feet_air_time",
                    "feet_slide",
                    "lin_vel_z",
                    "ang_vel_xy",
                    "flat_orientation",
                    "action_rate",
                    "joint_acc",
                    "joint_torques",
                    "joint_pos_limits",
                    "stand_still",
                    "ankle_deviation",
                    "hip_deviation",
                    "arm_deviation",
                ),
            )
        },
        terminations=TerminationGroupCfg(
            terms={
                "time_out": TerminationTermCfg(mdp.time_out, time_out=True),
                "base_contact": TerminationTermCfg(
                    mdp.illegal_contact,
                    params={"sensor_name": "contact_forces", "body_names": ("torso_link",)},
                ),
            },
            term_order=("time_out", "base_contact"),
        ),
        events={
            "randomize_material": EventTermCfg(
                mdp.randomize_sliding_friction,
                mode="startup",
                params={
                    "body_names": G1_29DOF_DFS_BODY_NAMES,
                    "friction_range": (0.2, 1.25),
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
                        "z": (-0.5, 0.5),
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
                params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
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
                    "rel_standing_envs": 0.02,
                    "rel_heading_envs": 1.0,
                    "heading_control_stiffness": 0.5,
                    "ranges": {
                        "lin_vel_x": (-1.0, 1.0),
                        "lin_vel_y": (-0.5, 0.5),
                        "ang_vel_z": (-1.0, 1.0),
                        "heading": (-3.14, 3.14),
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
                    Capability.ROOT_VELOCITY_WRITE,
                }
            ),
            randomization_fields=frozenset({"sliding_friction", "root_pose", "root_velocity", "joint_state"}),
        ),
        episode_length_s=20.0,
        is_finite_horizon=False,
        action_order=("joint_pos",),
        observation_group_order=("policy", "critic"),
        reward_group_order=("default",),
        event_order=("randomize_material", "reset_root", "reset_joints", "push"),
        command_order=("base_velocity",),
    )


def locomotion_flat_agent_cfg(**overrides) -> OnPolicyRunnerCfg:
    cfg = OnPolicyRunnerCfg()
    for name, value in overrides.items():
        if not hasattr(cfg, name):
            raise TypeError(f"unknown runner configuration field {name!r}")
        setattr(cfg, name, value)
    return cfg


__all__ = ["locomotion_flat_agent_cfg", "locomotion_flat_env_cfg"]
